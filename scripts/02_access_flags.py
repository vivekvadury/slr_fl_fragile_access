#!/usr/bin/env python
"""
Exploratory block-level centroid access flags for the South Florida tri-county
study area across multiple sea-level-rise (SLR) scenarios.

This script is intentionally a first-pass exploratory workflow.

What it does
- Builds a simple undirected drivable road graph from the retained tri-county
  OSM PBF.
- Merges public + private primary schools and filtered fire stations into a
  combined essential-services layer.
- Snaps block centroids and services to the retained network.
- Creates one baseline graph and one "dry" graph per requested NOAA SLR layer by
  removing road segments that intersect the inundation polygon.
- Computes centroid-based exploratory access diagnostics in long format, with
  one row per block x SLR level.

What it does NOT do
- It does not replicate a final paper-ready block accessibility workflow.
- It does not model directed traffic rules; the graph is intentionally
  undirected for this first pass.
- It does not split roads at polygon boundaries or handle bridges/tunnels; a
  road segment is treated as flooded if the segment geometry intersects the SLR
  inundation polygon.
- It does not estimate final paper-ready exposure metrics or run regressions.

Important interpretation note
- These are block-centroid exploratory measures, not final replication-
  equivalent access variables.
- Flooded centroids are treated as inaccessible origins for the exploratory
  access flags even if a snapped network node would otherwise remain connected.

Required inputs
- data/processed/census/blocks/fl_tricounty_blocks_2020.gpkg
- data/raw/noaa/FL_SE_slr_final_dist.gpkg
- data/processed/services/primary_schools/fl_private_schools.shp
- data/processed/services/primary_schools/fl_public_schools.shp
- data/raw/services/fire_stations/Critical_Community_and_Emergency_Facilities_2_4652348443233868839.geojson
- data/processed/road/tri_county_slr_network.osm.pbf
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyogrio
from networkx.algorithms.connectivity import k_edge_components
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point, box


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BLOCKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "census"
    / "blocks"
    / "fl_tricounty_blocks_2020.gpkg"
)
NOAA_GPKG_PATH = PROJECT_ROOT / "data" / "raw" / "noaa" / "FL_SE_slr_final_dist.gpkg"
PRIVATE_SCHOOLS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "services"
    / "primary_schools"
    / "fl_private_schools.shp"
)
PUBLIC_SCHOOLS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "services"
    / "primary_schools"
    / "fl_public_schools.shp"
)
FIRE_STATIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "services"
    / "fire_stations"
    / "Critical_Community_and_Emergency_Facilities_2_4652348443233868839.geojson"
)
ROAD_PBF_PATH = (
    PROJECT_ROOT / "data" / "processed" / "road" / "tri_county_slr_network.osm.pbf"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "access" / "edited"
DEFAULT_OUTPUT_STEM = "block_access_flags_long"
DEFAULT_QA_STEM = "block_access_flags_qa_sample"

PROJECTED_CRS = "EPSG:32617"
NODE_COORD_ROUND_DECIMALS = 3

BASELINE_SLR_FT = 0
BASELINE_SLR_LAYER = "FL_SE_slr_0_0ft"

SLR_LAYER_MAP: dict[int, str] = {
    1: "FL_SE_slr_1_0ft",
    2: "FL_SE_slr_2_0ft",
    3: "FL_SE_slr_3_0ft",
    4: "FL_SE_slr_4_0ft",
    5: "FL_SE_slr_5_0ft",
    6: "FL_SE_slr_6_0ft",
}

COUNTY_NAME_MAP = {
    "011": "Broward",
    "086": "Miami-Dade",
    "099": "Palm Beach",
}

NETWORK_BOUNDARY_BBOX = (-80.90, 25.10, -79.90, 27.00)

SERVICE_BUFFER_M = 10_000
MAX_SERVICE_SNAP_M = 1_000
MAX_ORIGIN_SNAP_M = 2_000
BOUNDARY_FLAG_DISTANCE_M = 2_000
SLR_CLIP_BUFFER_M = 5_000
MAX_CANDIDATE_SERVICE_NODES = 8
MAX_EDGE_DISJOINT_PATHS_CAP = 2
QA_SAMPLE_PER_GROUP = 5

# Filtering to drivable highways for analysis.
DRIVABLE_HIGHWAYS = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "residential_link",
}
DROP_PRIVATE_ACCESS_EDGES = True


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def warn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def check_required_inputs(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required input(s):\n{missing_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute block-centroid access flags and compare each SLR scenario to the 0 ft baseline."
    )
    parser.add_argument(
        "--max-blocks",
        "--max-bg",
        dest="max_blocks",
        type=int,
        default=None,
        help="Optional smoke-test cap on number of blocks.",
    )
    parser.add_argument(
        "--slr-ft",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of positive SLR feet to run, e.g. --slr-ft 1 3 6. The 0 ft baseline is always included.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix appended to output filenames.",
    )
    return parser.parse_args()


def build_output_suffix(args: argparse.Namespace) -> str:
    if args.output_suffix:
        return args.output_suffix
    if args.max_blocks is not None or args.slr_ft:
        return "__subset"
    return ""


def read_vector(
    path: Path,
    *,
    layer: str | None = None,
    columns: list[str] | None = None,
) -> gpd.GeoDataFrame:
    kwargs: dict[str, object] = {"columns": columns}
    if layer is not None:
        kwargs["layer"] = layer
    return pyogrio.read_dataframe(path, **kwargs)


def list_layers(path: Path) -> set[str]:
    return {layer_name for layer_name, _ in pyogrio.list_layers(path)}

# Transforms CRS84 (Long/Lat) to projected CRS if needed
def maybe_to_projected(gdf: gpd.GeoDataFrame, crs: str = PROJECTED_CRS) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Input layer has no CRS; cannot continue safely.")
    try:
        epsg = gdf.crs.to_epsg()
    except Exception:
        epsg = None
    if epsg in {4269, 4326}:
        gdf = gdf.set_crs("OGC:CRS84", allow_override=True)
    if str(gdf.crs) == crs:
        return gdf
    return gdf.to_crs(crs)


def prepare_blocks_layer(blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    output = blocks.copy()
    rename_map: dict[str, str] = {}

    if "geoid" in output.columns:
        rename_map["geoid"] = "block_geoid"
    elif "GEOID20" in output.columns:
        rename_map["GEOID20"] = "block_geoid"
    elif "GEOID" in output.columns:
        rename_map["GEOID"] = "block_geoid"

    if "state_fips" not in output.columns:
        if "STATEFP20" in output.columns:
            rename_map["STATEFP20"] = "state_fips"
        elif "STATEFP" in output.columns:
            rename_map["STATEFP"] = "state_fips"

    if "county_fips" not in output.columns:
        if "COUNTYFP20" in output.columns:
            rename_map["COUNTYFP20"] = "county_fips"
        elif "COUNTYFP" in output.columns:
            rename_map["COUNTYFP"] = "county_fips"

    if "tract_geoid" not in output.columns:
        if "TRACTCE20" in output.columns:
            rename_map["TRACTCE20"] = "tract_code"
        elif "TRACTCE" in output.columns:
            rename_map["TRACTCE"] = "tract_code"

    if "block_group_geoid" not in output.columns and "BLKGRPCE" in output.columns:
        rename_map["BLKGRPCE"] = "block_group"

    if "block" not in output.columns and "BLOCKCE20" in output.columns:
        rename_map["BLOCKCE20"] = "block"

    output = output.rename(columns=rename_map)

    if "block_geoid" not in output.columns:
        raise ValueError("Blocks layer must include a block GEOID field.")

    output["block_geoid"] = output["block_geoid"].astype(str).str.zfill(15)

    if "state_fips" not in output.columns:
        output["state_fips"] = output["block_geoid"].str.slice(0, 2)
    else:
        output["state_fips"] = output["state_fips"].astype(str).str.zfill(2)

    if "county_fips" not in output.columns:
        output["county_fips"] = output["block_geoid"].str.slice(2, 5)
    else:
        output["county_fips"] = output["county_fips"].astype(str).str.zfill(3)

    if "tract_geoid" not in output.columns:
        output["tract_geoid"] = output["block_geoid"].str.slice(0, 11)
    else:
        output["tract_geoid"] = output["tract_geoid"].astype(str).str.zfill(11)

    if "block_group_geoid" not in output.columns:
        output["block_group_geoid"] = output["block_geoid"].str.slice(0, 12)
    else:
        output["block_group_geoid"] = output["block_group_geoid"].astype(str).str.zfill(12)

    if "block" not in output.columns:
        output["block"] = output["block_geoid"].str.slice(11, 15)
    else:
        output["block"] = output["block"].astype(str).str.zfill(4)

    if "county_name" not in output.columns:
        output["county_name"] = output["county_fips"].map(COUNTY_NAME_MAP).fillna("Unknown")
    else:
        output["county_name"] = output["county_name"].fillna(
            output["county_fips"].map(COUNTY_NAME_MAP).fillna("Unknown")
        )

    return output

# What does the margin of error?
def build_study_area_boundary(
    bounds: tuple[float, float, float, float],
    source_crs,
) -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = bounds
    transformer = Transformer.from_crs(source_crs, PROJECTED_CRS, always_xy=True)
    corners = [
        transformer.transform(xmin, ymin),
        transformer.transform(xmin, ymax),
        transformer.transform(xmax, ymin),
        transformer.transform(xmax, ymax),
    ]
    xs = [x for x, _ in corners]
    ys = [y for _, y in corners]
    projected_boundary = gpd.GeoDataFrame(
        {"name": ["retained_network_bbox"]},
        geometry=[box(min(xs), min(ys), max(xs), max(ys))],
        crs=PROJECTED_CRS,
    )
    return projected_boundary


def point_distance_to_boundary(points: gpd.GeoSeries, boundary_polygon: gpd.GeoDataFrame) -> pd.Series:
    boundary_line = boundary_polygon.geometry.iloc[0].boundary
    return points.distance(boundary_line)


def make_centroids(origins_layer: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    centroids = origins_layer.copy()
    centroid_geom = centroids.geometry.centroid
    finite_mask = np.isfinite(centroid_geom.x) & np.isfinite(centroid_geom.y)
    fallback_geom = centroids.geometry.representative_point()
    centroid_geom.loc[~finite_mask] = fallback_geom.loc[~finite_mask]
    centroids["geometry"] = centroid_geom
    return centroids


def canonicalize_service_ids(services: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    output = services.copy()
    output["service_id"] = output["service_id"].astype(str)
    output["service_type"] = output["service_type"].astype(str)
    return output


def load_services() -> gpd.GeoDataFrame:
    public = read_vector(
        PUBLIC_SCHOOLS_PATH,
        columns=["NCESSCH", "NAME", "CNTY", "geometry"],
    )
    public = maybe_to_projected(public)
    public["service_id"] = "public_school_" + public["NCESSCH"].astype(str)
    public["service_type"] = "school"
    public["service_source"] = "public_primary_school"
    public["service_name"] = public["NAME"].astype(str)

    private = read_vector(
        PRIVATE_SCHOOLS_PATH,
        columns=["PPIN", "NAME", "CNTY", "geometry"],
    )
    private = maybe_to_projected(private)
    private["service_id"] = "private_school_" + private["PPIN"].astype(str)
    private["service_type"] = "school"
    private["service_source"] = "private_primary_school"
    private["service_name"] = private["NAME"].astype(str)

    fire = read_vector(
        FIRE_STATIONS_PATH,
        columns=["FACILITY_T", "Asset_Type", "Asset_ID", "NAME", "COUNTY", "geometry"],
    )
    fire = maybe_to_projected(fire)
    fire_mask = (
        fire.get("FACILITY_T", pd.Series(index=fire.index, dtype=object)).fillna("").eq("FIRE STATION")
        | fire.get("Asset_Type", pd.Series(index=fire.index, dtype=object)).fillna("").eq("Fire Stations")
    )
    fire = fire.loc[fire_mask].copy()
    fire_id_fallback = pd.Series(fire.index.astype(str), index=fire.index)
    fire["service_id"] = "fire_station_" + fire["Asset_ID"].fillna(fire_id_fallback).astype(str)
    fire["service_type"] = "fire_station"
    fire["service_source"] = "fire_station"
    fire["service_name"] = fire["NAME"].astype(str)

    columns = ["service_id", "service_type", "service_source", "service_name", "geometry"]
    services = pd.concat(
        [public[columns], private[columns], fire[columns]],
        ignore_index=True,
    )
    services = gpd.GeoDataFrame(services, geometry="geometry", crs=PROJECTED_CRS)
    return canonicalize_service_ids(services)


def parse_other_tag(other_tags: str | None, key: str) -> str | None:
    if not other_tags or not isinstance(other_tags, str):
        return None
    match = re.search(rf'"{re.escape(key)}"=>"([^"]+)"', other_tags)
    if match:
        return match.group(1)
    return None


def filter_drivable_roads(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    output = roads.copy()
    output["highway"] = output["highway"].astype("string")
    output = output.loc[output["highway"].isin(DRIVABLE_HIGHWAYS)].copy()

    if DROP_PRIVATE_ACCESS_EDGES:
        output["access_tag"] = output["other_tags"].map(lambda value: parse_other_tag(value, "access"))
        output = output.loc[output["access_tag"].fillna("") != "private"].copy()
    else:
        output["access_tag"] = None

    output = output.loc[output.geometry.notnull() & ~output.geometry.is_empty].copy()
    return output


def load_roads() -> gpd.GeoDataFrame:
    roads = read_vector(
        ROAD_PBF_PATH,
        layer="lines",
        columns=["osm_id", "highway", "z_order", "other_tags", "geometry"],
    )
    roads = filter_drivable_roads(roads)
    roads["osm_id"] = roads["osm_id"].astype(str)
    return roads

# Build the road graph from consecutive line vertices rather than collapsing an
# entire OSM feature to one start-to-end edge. This reduces artificial network
# fragmentation and should make baseline isolation less sensitive to long road
# features that contain many interior bends/vertices.
#
# Important assumptions that still remain:
# - We still do not planarize crossings (no edges cross each other). If two lines cross but do not share a
#   vertex in the source OSM geometry, we do not create an intersection.
# - We still treat the graph as undirected and ignore turn restrictions.
# - We still rely on the OSM geometry being correctly noded where real
#   connectivity exists, and we may still over-connect grade-separated features
#   if they share the same explicit vertex coordinates.
def segmentize_roads(roads: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    node_lookup: dict[tuple[float, float], int] = {}
    node_records: list[dict[str, object]] = []
    edge_records: list[dict[str, object]] = []
    next_node_id = 0
    next_edge_id = 0
    transformer = Transformer.from_crs(roads.crs, PROJECTED_CRS, always_xy=True)

    def get_node_id(x: float, y: float) -> int:
        nonlocal next_node_id
        key = (round(x, NODE_COORD_ROUND_DECIMALS), round(y, NODE_COORD_ROUND_DECIMALS))
        if key not in node_lookup:
            node_lookup[key] = next_node_id
            node_records.append(
                {
                    "node_id": next_node_id,
                    "x": key[0],
                    "y": key[1],
                    "geometry": Point(key),
                }
            )
            next_node_id += 1
        return node_lookup[key]

    def iter_lines(geometry) -> Iterable[LineString]:
        if geometry is None or geometry.is_empty:
            return []
        if isinstance(geometry, LineString):
            return [geometry]
        if isinstance(geometry, MultiLineString):
            return list(geometry.geoms)
        return []

    for row in roads.itertuples(index=False):
        for line in iter_lines(row.geometry):
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            projected_coords = [transformer.transform(x, y) for x, y in coords]
            for idx in range(len(coords) - 1):
                start = coords[idx]
                end = coords[idx + 1]
                if start == end:
                    continue

                projected_start = projected_coords[idx]
                projected_end = projected_coords[idx + 1]
                projected_line = LineString([projected_start, projected_end])
                if projected_line.length == 0:
                    continue

                u = get_node_id(projected_start[0], projected_start[1])
                v = get_node_id(projected_end[0], projected_end[1])
                if u == v:
                    continue

                edge_records.append(
                    {
                        "edge_id": next_edge_id,
                        "u": u,
                        "v": v,
                        "osm_id": row.osm_id,
                        "highway": row.highway,
                        "z_order": row.z_order,
                        "geometry": LineString([start, end]),
                        "length_m": projected_line.length,
                    }
                )
                next_edge_id += 1

    nodes = gpd.GeoDataFrame(node_records, geometry="geometry", crs=PROJECTED_CRS)
    edges = gpd.GeoDataFrame(edge_records, geometry="geometry", crs=roads.crs)

    if edges.empty:
        raise RuntimeError("No drivable road segments were created from the retained network.")

    return nodes, edges


def build_graph(edges: gpd.GeoDataFrame) -> nx.Graph:
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        (int(row.u), int(row.v), float(row.length_m))
        for row in edges.itertuples(index=False)
    )
    return graph

# Build a spatial index (k-d tree) for the graph nodes to enable efficient snapping of points (centroids and services) to the nearest node in the graph.
def build_node_kdtree(nodes: gpd.GeoDataFrame) -> tuple[cKDTree, np.ndarray, np.ndarray]:
    coords = np.column_stack([nodes["x"].to_numpy(), nodes["y"].to_numpy()])
    node_ids = nodes["node_id"].to_numpy()
    tree = cKDTree(coords)
    return tree, coords, node_ids


def snap_points_to_nodes(
    points: gpd.GeoDataFrame,
    *,
    point_id_col: str,
    tree: cKDTree,
    node_ids: np.ndarray,
    nodes: gpd.GeoDataFrame,
    max_snap_m: float,
) -> pd.DataFrame:
    coords = np.column_stack([points.geometry.x.to_numpy(), points.geometry.y.to_numpy()])
    finite_mask = np.isfinite(coords).all(axis=1)
    distances = np.full(len(points), np.nan, dtype=float)
    snapped_node_ids = np.full(len(points), -1, dtype=int)

    if finite_mask.any():
        valid_distances, valid_indices = tree.query(coords[finite_mask], k=1)
        distances[finite_mask] = valid_distances.astype(float)
        snapped_node_ids[finite_mask] = node_ids[valid_indices].astype(int)

    result = pd.DataFrame(
        {
            point_id_col: points[point_id_col].to_numpy(),
            "node_id": snapped_node_ids.astype(int),
            "snap_distance_m": distances.astype(float),
            "snap_valid": finite_mask & (distances <= max_snap_m),
        }
    )

    node_xy = nodes.set_index("node_id")[["x", "y"]]
    result = result.join(node_xy, on="node_id")
    return result

# If a service is more than 10 kilometers away from the edge of your road network, it is too far to be considered a viable, everyday essential service for the people living inside the study area.
def filter_services_by_buffer(
    services: gpd.GeoDataFrame,
    boundary_polygon: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    buffered = boundary_polygon.buffer(SERVICE_BUFFER_M).iloc[0]
    keep_mask = services.geometry.intersects(buffered)
    return services.loc[keep_mask].copy()


def compute_origin_boundary_fields(
    centroids: gpd.GeoDataFrame,
    boundary_polygon: gpd.GeoDataFrame,
) -> pd.DataFrame:
    boundary_distance = point_distance_to_boundary(centroids.geometry, boundary_polygon)
    return pd.DataFrame(
        {
            "block_geoid": centroids["block_geoid"].to_numpy(),
            "boundary_distance_m": boundary_distance.to_numpy(),
            "boundary_flag": (boundary_distance <= BOUNDARY_FLAG_DISTANCE_M).to_numpy(),
        }
    )

# For each connected component in the graph, we determine which nodes belong to that component, count the number of services accessible from that component, check if any node in the component touches the boundary, and count how many nodes in the component have at least one service. This information is stored in dictionaries for later use in access diagnostics.
def build_component_maps(
    graph: nx.Graph,
    services: pd.DataFrame,
    boundary_node_ids: set[int],
) -> tuple[dict[int, int], dict[int, int], dict[int, bool], dict[int, int]]:
    node_to_component: dict[int, int] = {}
    component_service_counts: dict[int, int] = {}
    component_touches_boundary: dict[int, bool] = {}
    component_service_node_counts: dict[int, int] = {}

    services_per_node = (
        services.groupby("node_id", as_index=False)
        .agg(service_count=("service_id", "size"))
        .set_index("node_id")["service_count"]
        .to_dict()
    )

    for component_id, component_nodes in enumerate(nx.connected_components(graph)):
        component_nodes = set(component_nodes)
        for node in component_nodes:
            node_to_component[int(node)] = component_id
        component_service_counts[component_id] = int(
            sum(services_per_node.get(int(node), 0) for node in component_nodes)
        )
        component_service_node_counts[component_id] = int(
            sum(1 for node in component_nodes if services_per_node.get(int(node), 0) > 0)
        )
        component_touches_boundary[component_id] = any(
            int(node) in boundary_node_ids for node in component_nodes
        )

    return (
        node_to_component,
        component_service_counts,
        component_touches_boundary,
        component_service_node_counts,
    )

# Calculates the nearest service node and distance for each node in the graph
def build_nearest_service_lookup(
    graph: nx.Graph,
    services: pd.DataFrame,
) -> pd.DataFrame:
    if services.empty:
        return pd.DataFrame(
            columns=["node_id", "nearest_service_node_id", "nearest_service_id", "nearest_service_type", "distance_m"]
        )

    canonical_services = (
        services.sort_values(["snap_distance_m", "service_id"])
        .drop_duplicates("node_id")
        .loc[:, ["node_id", "service_id", "service_type"]]
        .rename(columns={"node_id": "nearest_service_node_id"})
    )
    canonical_services = canonical_services.loc[
        canonical_services["nearest_service_node_id"].isin(graph.nodes)
    ].copy()
    if canonical_services.empty:
        return pd.DataFrame(
            columns=["node_id", "nearest_service_node_id", "nearest_service_id", "nearest_service_type", "distance_m"]
        )
    service_nodes = canonical_services["nearest_service_node_id"].astype(int).tolist()

# Calculating the shortest path from each node in the graph to the nearest service node, and storing the distance and path information in a DataFrame for later use in access diagnostics.
    distances, paths = nx.multi_source_dijkstra(graph, service_nodes, weight="weight")
    records: list[dict[str, object]] = []
    service_lookup = canonical_services.set_index("nearest_service_node_id")

    for node_id, distance in distances.items():
        path = paths[node_id]
        nearest_service_node_id = int(path[0]) if path else int(node_id)
        service_meta = service_lookup.loc[nearest_service_node_id]
        records.append(
            {
                "node_id": int(node_id),
                "nearest_service_node_id": nearest_service_node_id,
                "nearest_service_id": service_meta["service_id"],
                "nearest_service_type": service_meta["service_type"],
                "distance_m": float(distance),
            }
        )

    return pd.DataFrame.from_records(records)


def build_two_edge_component_maps(
    graph: nx.Graph,
    services: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int]]:
    if graph.number_of_nodes() == 0:
        return {}, {}

    services_per_node = (
        services.loc[services["node_id"].isin(graph.nodes), ["node_id", "service_id"]]
        .groupby("node_id", as_index=False)
        .agg(service_count=("service_id", "size"))
        .set_index("node_id")["service_count"]
        .to_dict()
    )

    node_to_two_edge_component: dict[int, int] = {}
    two_edge_component_service_counts: dict[int, int] = {}

    for component_id, component_nodes in enumerate(k_edge_components(graph, 2)):
        component_nodes = set(component_nodes)
        for node in component_nodes:
            node_to_two_edge_component[int(node)] = component_id
        two_edge_component_service_counts[component_id] = int(
            sum(services_per_node.get(int(node), 0) for node in component_nodes)
        )

    return node_to_two_edge_component, two_edge_component_service_counts


def build_service_node_candidate_map(
    origins: pd.DataFrame,
    services: pd.DataFrame,
    nodes: gpd.GeoDataFrame,
) -> dict[int, list[int]]:
    unique_service_nodes = (
        services[["node_id"]]
        .drop_duplicates()
        .merge(nodes[["node_id", "x", "y"]], on="node_id", how="left")
        .dropna(subset=["x", "y"])
        .copy()
    )

    if unique_service_nodes.empty:
        return {int(node_id): [] for node_id in origins["node_id"].dropna().astype(int).unique()}

    service_coords = unique_service_nodes[["x", "y"]].to_numpy()
    service_node_ids = unique_service_nodes["node_id"].astype(int).to_numpy()
    service_tree = cKDTree(service_coords)

    origin_nodes = (
        origins.loc[:, ["node_id"]]
        .drop_duplicates()
        .loc[lambda frame: frame["node_id"] >= 0]
        .merge(nodes[["node_id", "x", "y"]], on="node_id", how="left")
        .dropna(subset=["x", "y"])
        .copy()
    )

    k = min(MAX_CANDIDATE_SERVICE_NODES, len(unique_service_nodes))
    origin_coords = origin_nodes[["x", "y"]].to_numpy()
    _, neighbor_indices = service_tree.query(origin_coords, k=k)
    if k == 1:
        neighbor_indices = neighbor_indices.reshape(-1, 1)

    candidate_map: dict[int, list[int]] = {}
    for row_index, origin_row in enumerate(origin_nodes.itertuples(index=False)):
        candidate_map[int(origin_row.node_id)] = [
            int(service_node_ids[idx])
            for idx in np.atleast_1d(neighbor_indices[row_index]).tolist()
        ]
    return candidate_map
# This builds: dry_two_edge_component_lookup, dry_two_edge_component_service_counts

# Computes the local edge connectivity between an origin node and a set of candidate service nodes, with an optional cap on the maximum number of edge-disjoint paths to consider. This function is used to assess the redundancy of access from the origin node to essential services in the graph, which can inform the classification of access flags for block centroids.  
def capped_local_edge_connectivity(
    graph: nx.Graph,
    origin_node: int,
    *,
    nearest_service_node: int | None,
    two_edge_component_lookup: dict[int, int],
    two_edge_component_service_counts: dict[int, int],
    cap: int = MAX_EDGE_DISJOINT_PATHS_CAP,
) -> int:
    if origin_node not in graph:
        return 0
    origin_degree = int(graph.degree(origin_node))
    if origin_degree == 0:
        return 0
    if nearest_service_node is not None and nearest_service_node == origin_node:
        return min(max(origin_degree, 1), cap)
    if origin_degree <= 1:
        return 1

    two_edge_component_id = two_edge_component_lookup.get(origin_node)
    if two_edge_component_id is None:
        return 1
    if two_edge_component_service_counts.get(two_edge_component_id, 0) > 0:
        return cap
    return 1


def classify_status(row: pd.Series) -> str:
    if row["block_centroid_inundated"] == 1:
        return "inundated"
    if row["block_centroid_isolated"] == 1:
        return "isolated"
    if row["block_centroid_redundant"] == 1:
        return "redundant"
    if row["block_centroid_fragile"] == 1:
        return "fragile"
    return "other"


def classify_status_columns(
    frame: pd.DataFrame,
    *,
    inundated_col: str,
    isolated_col: str,
    redundant_col: str,
    fragile_col: str,
) -> pd.Series:
    conditions = [
        frame[inundated_col].eq(1),
        frame[isolated_col].eq(1),
        frame[redundant_col].eq(1),
        frame[fragile_col].eq(1),
    ]
    choices = ["inundated", "isolated", "redundant", "fragile"]
    return pd.Series(np.select(conditions, choices, default="other"), index=frame.index, dtype="object")


def add_baseline_comparison_fields(results: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        results.loc[results["slr_ft"] == BASELINE_SLR_FT, [
            "block_geoid",
            "block_centroid_inundated",
            "block_centroid_isolated",
            "block_centroid_redundant",
            "block_centroid_fragile",
            "n_reachable_services",
            "n_reachable_service_nodes",
            "max_edge_disjoint_paths_any_service",
            "baseline_shortest_path_distance_m",
            "dry_shortest_path_distance_m",
            "detour_ratio",
        ]]
        .drop_duplicates("block_geoid")
        .rename(
            columns={
                "block_centroid_inundated": "baseline_block_centroid_inundated",
                "block_centroid_isolated": "baseline_block_centroid_isolated",
                "block_centroid_redundant": "baseline_block_centroid_redundant",
                "block_centroid_fragile": "baseline_block_centroid_fragile",
                "n_reachable_services": "baseline_n_reachable_services",
                "n_reachable_service_nodes": "baseline_n_reachable_service_nodes",
                "max_edge_disjoint_paths_any_service": "baseline_max_edge_disjoint_paths_any_service",
                "baseline_shortest_path_distance_m": "baseline_baseline_shortest_path_distance_m",
                "dry_shortest_path_distance_m": "baseline_dry_shortest_path_distance_m",
                "detour_ratio": "baseline_detour_ratio",
            }
        )
        .copy()
    )

    output = results.merge(
        baseline,
        on="block_geoid",
        how="left",
        validate="many_to_one",
    )

    output["baseline_status"] = classify_status_columns(
        output,
        inundated_col="baseline_block_centroid_inundated",
        isolated_col="baseline_block_centroid_isolated",
        redundant_col="baseline_block_centroid_redundant",
        fragile_col="baseline_block_centroid_fragile",
    )
    output["scenario_status"] = classify_status_columns(
        output,
        inundated_col="block_centroid_inundated",
        isolated_col="block_centroid_isolated",
        redundant_col="block_centroid_redundant",
        fragile_col="block_centroid_fragile",
    )

    positive_scenario = output["slr_ft"] != BASELINE_SLR_FT
    output["persistent_fragile"] = (
        positive_scenario
        & output["baseline_block_centroid_fragile"].eq(1)
        & output["block_centroid_fragile"].eq(1)
    ).astype(int)
    output["new_fragile_due_to_slr"] = (
        positive_scenario
        & output["baseline_block_centroid_redundant"].eq(1)
        & output["block_centroid_fragile"].eq(1)
    ).astype(int)
    output["new_isolated_due_to_slr"] = (
        positive_scenario
        & output["baseline_block_centroid_isolated"].eq(0)
        & output["block_centroid_isolated"].eq(1)
    ).astype(int)
    output["new_inundated_due_to_slr"] = (
        positive_scenario
        & output["baseline_block_centroid_inundated"].eq(0)
        & output["block_centroid_inundated"].eq(1)
    ).astype(int)

    return output


def safe_detour_ratio(baseline_distance: float | None, dry_distance: float | None) -> float | None:
    if baseline_distance is None or dry_distance is None:
        return np.nan
    if pd.isna(baseline_distance) or pd.isna(dry_distance):
        return np.nan
    if baseline_distance == 0 and dry_distance == 0:
        return 1.0
    if baseline_distance <= 0:
        return np.nan
    return float(dry_distance / baseline_distance)


def load_slr_layer(layer_name: str, clip_polygon) -> gpd.GeoDataFrame | None:
    slr = read_vector(NOAA_GPKG_PATH, layer=layer_name, columns=["Id", "gridcode", "geometry"])
    slr = slr.set_crs("OGC:CRS84", allow_override=True)
    slr = slr.loc[slr.geometry.intersects(clip_polygon)].copy()
    if slr.empty:
        return None
    return slr


def build_boundary_node_set(nodes: gpd.GeoDataFrame, boundary_polygon: gpd.GeoDataFrame) -> set[int]:
    boundary_distance = point_distance_to_boundary(nodes.geometry, boundary_polygon)
    return set(nodes.loc[boundary_distance <= BOUNDARY_FLAG_DISTANCE_M, "node_id"].astype(int).tolist())


def scenario_results_for_origins(
    *,
    slr_ft: int,
    slr_layer_name: str,
    slr_layer: gpd.GeoDataFrame | None,
    graph: nx.Graph,
    services: pd.DataFrame,
    origins: pd.DataFrame,
    centroid_boundary: pd.DataFrame,
    centroid_geometry_source: gpd.GeoDataFrame,
    baseline_nearest: pd.DataFrame,
    dry_boundary_node_ids: set[int],
) -> pd.DataFrame:
    (
        dry_component_lookup,
        dry_component_service_counts,
        dry_component_touches_boundary,
        dry_component_service_node_counts,
    ) = build_component_maps(graph, services, dry_boundary_node_ids)
    dry_nearest = build_nearest_service_lookup(graph, services)
    (
        dry_two_edge_component_lookup,
        dry_two_edge_component_service_counts,
    ) = build_two_edge_component_maps(graph, services)

    dry_nearest_map = dry_nearest.set_index("node_id") if not dry_nearest.empty else pd.DataFrame()
    baseline_nearest_map = baseline_nearest.set_index("node_id") if not baseline_nearest.empty else pd.DataFrame()

    centroid_inundated = centroid_geometry_source[["block_geoid", "geometry"]].copy()
    centroid_inundated["block_centroid_inundated"] = 0
    if slr_layer is not None and not slr_layer.empty:
        flooded_centroids = gpd.sjoin(
            centroid_inundated,
            slr_layer[["geometry"]],
            how="left",
            predicate="intersects",
        )
        centroid_inundated["block_centroid_inundated"] = (
            flooded_centroids["index_right"].notna().astype(int).to_numpy()
        )
    centroid_inundated = centroid_inundated.drop(columns="geometry")

    result = (
        origins.merge(centroid_boundary, on="block_geoid", how="left")
        .merge(centroid_inundated, on="block_geoid", how="left")
        .copy()
    )
    result["slr_ft"] = slr_ft
    result["slr_layer_name"] = slr_layer_name

    records: list[dict[str, object]] = []
    for row in result.itertuples(index=False):
        origin_node_id = int(row.node_id)
        origin_valid = bool(row.snap_valid)
        centroid_is_flooded = bool(row.block_centroid_inundated)

        baseline_distance = np.nan
        if origin_valid and origin_node_id in baseline_nearest_map.index:
            baseline_distance = float(baseline_nearest_map.loc[origin_node_id, "distance_m"])

        reachable_service_count = 0
        component_touches_boundary = False
        nearest_service_id = pd.NA
        nearest_service_type = pd.NA
        dry_distance = np.nan
        max_edge_disjoint_paths = 0
        nearest_service_node = None

        if origin_valid and origin_node_id in dry_component_lookup:
            component_id = dry_component_lookup[origin_node_id]
            reachable_service_count = int(dry_component_service_counts.get(component_id, 0))
            component_touches_boundary = bool(dry_component_touches_boundary.get(component_id, False))

            if origin_node_id in dry_nearest_map.index:
                nearest_service_row = dry_nearest_map.loc[origin_node_id]
                nearest_service_id = nearest_service_row["nearest_service_id"]
                nearest_service_type = nearest_service_row["nearest_service_type"]
                dry_distance = float(nearest_service_row["distance_m"])
                nearest_service_node = int(nearest_service_row["nearest_service_node_id"])

            if reachable_service_count > 0 and not centroid_is_flooded:
                max_edge_disjoint_paths = capped_local_edge_connectivity(
                    graph,
                    origin_node_id,
                    nearest_service_node=nearest_service_node,
                    two_edge_component_lookup=dry_two_edge_component_lookup,
                    two_edge_component_service_counts=dry_two_edge_component_service_counts,
                )

        access_failure = (not origin_valid) or reachable_service_count == 0
        is_inundated = int(centroid_is_flooded)
        is_isolated = int((not centroid_is_flooded) and access_failure)
        is_redundant = int(
            (not centroid_is_flooded)
            and (not access_failure)
            and max_edge_disjoint_paths >= 2
        )
        is_fragile = int(
            (not centroid_is_flooded)
            and (not access_failure)
            and max_edge_disjoint_paths == 1
        )

        if is_inundated or is_isolated:
            nearest_service_id = pd.NA
            nearest_service_type = pd.NA
            dry_distance = np.nan
            max_edge_disjoint_paths = 0

        records.append(
            {
                "block_geoid": row.block_geoid,
                "block_group_geoid": row.block_group_geoid,
                "tract_geoid": row.tract_geoid,
                "block": row.block,
                "county_fips": row.county_fips,
                "county_name": row.county_name,
                "slr_ft": slr_ft,
                "slr_layer_name": slr_layer_name,
                "origin_node_id": origin_node_id,
                "origin_snap_distance_m": float(row.snap_distance_m),
                "origin_snap_exceeds_threshold": int(not origin_valid),
                "boundary_flag": int(bool(row.boundary_flag)),
                "boundary_distance_m": float(row.boundary_distance_m),
                "component_touches_boundary": int(component_touches_boundary),
                "block_centroid_inundated": is_inundated,
                "block_centroid_isolated": is_isolated,
                "block_centroid_redundant": is_redundant,
                "block_centroid_fragile": is_fragile,
                "n_reachable_services": int(reachable_service_count),
                "n_reachable_service_nodes": int(
                    dry_component_service_node_counts.get(
                        dry_component_lookup.get(origin_node_id, -1),
                        0,
                    )
                ),
                "max_edge_disjoint_paths_any_service": int(max_edge_disjoint_paths),
                "nearest_reachable_service_type": nearest_service_type,
                "nearest_reachable_service_id": nearest_service_id,
                "baseline_shortest_path_distance_m": baseline_distance,
                "dry_shortest_path_distance_m": dry_distance,
                "detour_ratio": safe_detour_ratio(baseline_distance, dry_distance),
            }
        )

    return pd.DataFrame.from_records(records)


def save_main_output(results: pd.DataFrame, output_stem: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{output_stem}.csv"
    results.to_csv(csv_path, index=False)

    try:
        import pyarrow  # noqa: F401

        parquet_path = OUTPUT_DIR / f"{output_stem}.parquet"
        results.to_parquet(parquet_path, index=False)
        log(f"Saved long-format output to {parquet_path}")
        return parquet_path
    except Exception:
        log("`pyarrow` not available; saved CSV instead of parquet.")
        log(f"Saved long-format output to {csv_path}")
        return csv_path


def save_qa_sample(
    results: pd.DataFrame,
    centroids: gpd.GeoDataFrame,
    qa_stem: str,
) -> Path:
    qa = results.copy()
    qa["status_category"] = qa.apply(classify_status, axis=1)
    qa = qa.loc[qa["status_category"] != "other"].copy()

    if qa.empty:
        warn("No QA sample rows were available to save.")
        return OUTPUT_DIR / f"{qa_stem}.geojson"

    qa = (
        qa.groupby(["slr_ft", "county_name", "status_category"], group_keys=False)
        .apply(lambda frame: frame.sample(min(len(frame), QA_SAMPLE_PER_GROUP), random_state=42))
        .reset_index(drop=True)
    )

    qa_geo = centroids[["block_geoid", "geometry"]].merge(qa, on="block_geoid", how="inner")
    qa_geo = gpd.GeoDataFrame(qa_geo, geometry="geometry", crs=PROJECTED_CRS).to_crs("EPSG:4326")
    qa_path = OUTPUT_DIR / f"{qa_stem}.geojson"
    qa_geo.to_file(qa_path, driver="GeoJSON")
    log(f"Saved QA centroid sample to {qa_path}")
    return qa_path


def print_summary_tables(results: pd.DataFrame) -> None:
    summary = (
        results.groupby("slr_ft", as_index=False)
        .agg(
            n_blocks=("block_geoid", "size"),
            n_inundated=("block_centroid_inundated", "sum"),
            n_isolated=("block_centroid_isolated", "sum"),
            n_fragile=("block_centroid_fragile", "sum"),
            n_redundant=("block_centroid_redundant", "sum"),
            n_origin_snap_warnings=("origin_snap_exceeds_threshold", "sum"),
            n_boundary_flag=("boundary_flag", "sum"),
            n_component_touches_boundary=("component_touches_boundary", "sum"),
        )
    )
    print("\nSummary by SLR level")
    print(summary.to_string(index=False))

    county_summary = (
        results.groupby(["slr_ft", "county_name"], as_index=False)
        .agg(
            n_blocks=("block_geoid", "size"),
            n_inundated=("block_centroid_inundated", "sum"),
            n_isolated=("block_centroid_isolated", "sum"),
            n_fragile=("block_centroid_fragile", "sum"),
            n_redundant=("block_centroid_redundant", "sum"),
        )
    )
    print("\nSummary by SLR level and county")
    print(county_summary.to_string(index=False))


def print_transition_summaries(results: pd.DataFrame) -> None:
    positive_results = results.loc[results["slr_ft"] != BASELINE_SLR_FT].copy()
    if positive_results.empty:
        return

    summary = (
        positive_results.groupby("slr_ft", as_index=False)
        .agg(
            n_blocks=("block_geoid", "size"),
            baseline_fragile=("baseline_block_centroid_fragile", "sum"),
            baseline_redundant=("baseline_block_centroid_redundant", "sum"),
            scenario_fragile=("block_centroid_fragile", "sum"),
            scenario_isolated=("block_centroid_isolated", "sum"),
            scenario_inundated=("block_centroid_inundated", "sum"),
            persistent_fragile=("persistent_fragile", "sum"),
            new_fragile_due_to_slr=("new_fragile_due_to_slr", "sum"),
            new_isolated_due_to_slr=("new_isolated_due_to_slr", "sum"),
            new_inundated_due_to_slr=("new_inundated_due_to_slr", "sum"),
        )
    )
    print("\nTransition summary by SLR level")
    print(summary.to_string(index=False))

    county_summary = (
        positive_results.groupby(["slr_ft", "county_name"], as_index=False)
        .agg(
            n_blocks=("block_geoid", "size"),
            persistent_fragile=("persistent_fragile", "sum"),
            new_fragile_due_to_slr=("new_fragile_due_to_slr", "sum"),
            new_isolated_due_to_slr=("new_isolated_due_to_slr", "sum"),
            new_inundated_due_to_slr=("new_inundated_due_to_slr", "sum"),
        )
    )
    print("\nTransition summary by SLR level and county")
    print(county_summary.to_string(index=False))


def main() -> int:
    args = parse_args()
    output_suffix = build_output_suffix(args)
    output_stem = f"{DEFAULT_OUTPUT_STEM}{output_suffix}"
    qa_stem = f"{DEFAULT_QA_STEM}{output_suffix}"

    check_required_inputs(
        [
            BLOCKS_PATH,
            NOAA_GPKG_PATH,
            PRIVATE_SCHOOLS_PATH,
            PUBLIC_SCHOOLS_PATH,
            FIRE_STATIONS_PATH,
            ROAD_PBF_PATH,
        ]
    )

    available_layers = list_layers(NOAA_GPKG_PATH)
    requested_layers = SLR_LAYER_MAP.copy()
    if args.slr_ft:
        requested_layers = {
            slr_ft: SLR_LAYER_MAP[slr_ft]
            for slr_ft in args.slr_ft
            if slr_ft in SLR_LAYER_MAP
        }
        invalid_layers = sorted(set(args.slr_ft) - set(requested_layers))
        if invalid_layers:
            raise ValueError(f"Unsupported SLR level(s): {invalid_layers}. Supported values: {sorted(SLR_LAYER_MAP)}")
    requested_layers = {BASELINE_SLR_FT: BASELINE_SLR_LAYER, **requested_layers}

    missing_layers = [layer_name for layer_name in requested_layers.values() if layer_name not in available_layers]
    if missing_layers:
        raise ValueError(f"Missing requested NOAA layer(s): {missing_layers}")

    start_time = time.time()
    log("Loading blocks, services, and roads...")
    blocks = read_vector(BLOCKS_PATH)
    blocks = prepare_blocks_layer(blocks)
    blocks = maybe_to_projected(blocks)
    if args.max_blocks is not None:
        blocks = blocks.sort_values("block_geoid").head(args.max_blocks).copy()
        warn(f"Running smoke-test subset with max_blocks={args.max_blocks}")

    centroids = make_centroids(blocks)
    centroids_source = centroids.to_crs("OGC:CRS84")
    services = load_services()
    roads = load_roads()
    roads = roads.set_crs("OGC:CRS84", allow_override=True)

    boundary_polygon = build_study_area_boundary(tuple(roads.total_bounds), roads.crs)
    source_clip_polygon = box(*roads.total_bounds)
    centroid_boundary = compute_origin_boundary_fields(centroids, boundary_polygon)
    services = filter_services_by_buffer(services, boundary_polygon)
    log(f"Services inside the buffered retained-network boundary: {len(services):,}")

    log("Segmentizing roads and building baseline graph...")
    nodes, edges = segmentize_roads(roads)
    graph_baseline = build_graph(edges)
    tree, _, node_ids = build_node_kdtree(nodes)

    road_bounds = roads.total_bounds
    log(f"Configured clipping note bbox (lon/lat): {NETWORK_BOUNDARY_BBOX}")
    log(
        "Observed retained-network road bbox used for diagnostics/filtering (lon/lat): "
        f"{tuple(round(value, 4) for value in road_bounds)}"
    )
    log(f"Baseline graph nodes: {graph_baseline.number_of_nodes():,}")
    log(f"Baseline graph edges: {graph_baseline.number_of_edges():,}")
    log(f"Segmentized road edges: {len(edges):,}")

    origins = centroids[
        ["block_geoid", "block_group_geoid", "tract_geoid", "block", "county_fips", "county_name", "geometry"]
    ].copy()
    origin_snap = snap_points_to_nodes(
        origins,
        point_id_col="block_geoid",
        tree=tree,
        node_ids=node_ids,
        nodes=nodes,
        max_snap_m=MAX_ORIGIN_SNAP_M,
    )
    origins = origins.merge(origin_snap, on="block_geoid", how="left")

    services_snap = snap_points_to_nodes(
        services[["service_id", "geometry"]].copy(),
        point_id_col="service_id",
        tree=tree,
        node_ids=node_ids,
        nodes=nodes,
        max_snap_m=MAX_SERVICE_SNAP_M,
    )
    services = services.merge(services_snap, on="service_id", how="left")
    services = services.loc[services["snap_valid"]].copy()

    if services.empty:
        raise RuntimeError("No services remained after buffered-footprint filtering and service snap checks.")

    log(f"Snapped services retained: {len(services):,}")
    log(
        "Service counts by type after filtering/snap:\n"
        + services.groupby("service_type").size().sort_values(ascending=False).to_string()
    )

    if not origins["snap_valid"].all():
        warn(
            f"{int((~origins['snap_valid']).sum()):,} block centroids exceed "
            f"the origin snap threshold of {MAX_ORIGIN_SNAP_M:,} meters."
        )

    boundary_node_ids = build_boundary_node_set(nodes, boundary_polygon)
    (
        baseline_component_lookup,
        baseline_component_service_counts,
        baseline_component_touches_boundary,
        baseline_component_service_node_counts,
    ) = build_component_maps(graph_baseline, services, boundary_node_ids)
    baseline_nearest = build_nearest_service_lookup(graph_baseline, services)
    log(f"Baseline components: {len(set(baseline_component_lookup.values())):,}")
    log(
        "Baseline reachable-service summary across components: "
        f"{sum(value > 0 for value in baseline_component_service_counts.values()):,} components with >=1 service"
    )
    log(
        "Origins exceeding snap threshold by county:\n"
        + origins.groupby("county_name")["snap_valid"].apply(lambda s: int((~s).sum())).to_string()
    )

    scenario_outputs: list[pd.DataFrame] = []
    edges_sindex = edges.sindex

    for slr_ft, slr_layer_name in requested_layers.items():
        log(f"Processing SLR {slr_ft} ft ({slr_layer_name})...")
        slr_layer = load_slr_layer(slr_layer_name, source_clip_polygon)
        if slr_layer is None:
            warn(f"Layer {slr_layer_name} had no inundation polygons within the buffered retained network.")
            dry_edges = edges
        else:
            log(f"SLR polygons retained for {slr_ft} ft: {len(slr_layer):,}")
            query_matches = edges_sindex.query(slr_layer.geometry, predicate="intersects")
            if isinstance(query_matches, tuple):
                flooded_edge_indices = np.unique(np.asarray(query_matches[1], dtype=int))
            elif hasattr(query_matches, "shape") and len(query_matches.shape) == 2:
                flooded_edge_indices = np.unique(np.asarray(query_matches[1], dtype=int))
            else:
                flooded_edge_indices = np.unique(np.asarray(query_matches, dtype=int))
            dry_edges = edges.drop(index=edges.index[flooded_edge_indices]).copy()
            log(f"Flooded segment count at {slr_ft} ft: {len(flooded_edge_indices):,}")

        dry_graph = build_graph(dry_edges)
        scenario_output = scenario_results_for_origins(
            slr_ft=slr_ft,
            slr_layer_name=slr_layer_name,
            slr_layer=slr_layer,
            graph=dry_graph,
            services=services,
            origins=origins.drop(columns="geometry"),
            centroid_boundary=centroid_boundary,
            centroid_geometry_source=centroids_source[["block_geoid", "geometry"]],
            baseline_nearest=baseline_nearest,
            dry_boundary_node_ids=boundary_node_ids,
        )
        scenario_outputs.append(scenario_output)

        scenario_summary = (
            scenario_output.groupby("county_name", as_index=False)
            .agg(
                n_blocks=("block_geoid", "size"),
                n_inundated=("block_centroid_inundated", "sum"),
                n_isolated=("block_centroid_isolated", "sum"),
                n_fragile=("block_centroid_fragile", "sum"),
                n_redundant=("block_centroid_redundant", "sum"),
            )
        )
        print(f"\nCounty summary for {slr_ft} ft")
        print(scenario_summary.to_string(index=False))

    raw_results = pd.concat(scenario_outputs, ignore_index=True)
    results = add_baseline_comparison_fields(raw_results)
    main_output_path = save_main_output(results, output_stem)
    qa_output_path = save_qa_sample(results, centroids, qa_stem)
    print_summary_tables(results)
    print_transition_summaries(results)

    elapsed = time.time() - start_time
    log(f"Main output: {main_output_path}")
    log(f"QA sample output: {qa_output_path}")
    log(f"Finished in {elapsed / 60:.1f} minutes.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover - top-level fail-fast reporting
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        raise
