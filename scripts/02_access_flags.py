#!/usr/bin/env python
"""
Compute block-level road-network access states for South Florida census blocks.

What it does
- Builds a simple undirected drivable road graph from the retained tri-county
  OSM PBF.
- Merges public + private primary schools and filtered fire stations into a
  combined essential-services layer.
- Attaches services to redundancy-eligible public-road nodes on the raw graph.
- Places block origins inside their polygons and snaps them to the raw graph's
  largest connected component.
- Creates one baseline graph and one "dry" graph per requested NOAA SLR layer.
- Classifies every block as inundated, isolated, fragile, redundant, or
  unclassified, retaining ineligible rows with explicit exclusion flags.
- Writes a reproducibility manifest, service-snap audit, and cached segmentized
  graph artifacts under a configuration-specific run directory.

Important interpretation notes
- It does not model directed traffic rules; the graph is intentionally
  undirected.
- It does not split ordinary roads at polygon boundaries; non-bridge-like road
  segments are removed when their geometry intersects the SLR polygon.
- Road segments are removed when their geometry intersects the SLR polygon.
- Flooded centroids are treated as inaccessible origins even if a snapped
  network node would otherwise remain connected.
- Service access currently combines primary schools and fire stations. See
  docs/manuscript_feedback_todo.md for the planned service-specific extension.
- ``--legacy-mode`` restores all legacy behavioral choices for verification.

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
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEPENDENCY_INSTALL_VERSIONS = {
    "geopandas": "1.1.4",
    "shapely": "2.1.2",
    "pyogrio": "0.12.1",
    "pandas": "3.0.3",
    "numpy": "2.4.6",
    "networkx": "3.6.1",
    "scipy": "1.17.1",
    "pyproj": "3.7.2",
    "pyarrow": "25.0.0",
}


def _load_required_dependencies() -> dict[str, object]:
    modules: dict[str, object] = {}
    missing: list[tuple[str, str]] = []
    for package_name in DEPENDENCY_INSTALL_VERSIONS:
        try:
            modules[package_name] = importlib.import_module(package_name)
        except (ImportError, OSError) as exc:
            missing.append((package_name, str(exc)))
    if missing:
        print("[DEPENDENCY ERROR] Required Python dependencies are unavailable.", file=sys.stderr)
        for package_name, error in missing:
            print(
                f"[DEPENDENCY ERROR] {package_name}: {error}; proposed={package_name}=={DEPENDENCY_INSTALL_VERSIONS[package_name]}",
                file=sys.stderr,
            )
        print("[DEPENDENCY ERROR] packages_installed=0", file=sys.stderr)
        raise SystemExit(2)
    return modules


_DEPENDENCIES = _load_required_dependencies()
gpd = _DEPENDENCIES["geopandas"]
nx = _DEPENDENCIES["networkx"]
np = _DEPENDENCIES["numpy"]
pd = _DEPENDENCIES["pandas"]
pyogrio = _DEPENDENCIES["pyogrio"]
Transformer = _DEPENDENCIES["pyproj"].Transformer
cKDTree = _DEPENDENCIES["scipy"].spatial.cKDTree
k_edge_components = nx.algorithms.connectivity.k_edge_components
LineString = _DEPENDENCIES["shapely"].geometry.LineString
MultiLineString = _DEPENDENCIES["shapely"].geometry.MultiLineString
Point = _DEPENDENCIES["shapely"].geometry.Point
box = _DEPENDENCIES["shapely"].geometry.box


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
CENSUS_BLOCK_ATTRIBUTES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "census"
    / "blocks"
    / "2020"
    / "tl_2020_12_tabblock20.shp"
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

TRICOUNTY_FIPS = {"011", "086", "099"}
SMOKE_BBOX = (-80.35, 25.65, -80.10, 25.95)

SERVICE_BUFFER_M = 10_000
MAX_SERVICE_SNAP_M = 1_000
MAX_ORIGIN_SNAP_M = 2_000
BOUNDARY_FLAG_DISTANCE_M = 2_000
SLR_CLIP_BUFFER_M = 5_000
MAX_EDGE_DISJOINT_PATHS_CAP = 2
QA_SAMPLE_PER_GROUP = 5
CACHE_SCHEMA_VERSION = 2
DEFAULT_CONFIG_NAME = "corrected"

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
        "--scenarios",
        default=None,
        help="Comma-separated SLR levels to run, e.g. --scenarios 0 or --scenarios 0,1,2. Defaults to 0-6.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix appended to output filenames.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Parent output directory. The config name is appended to this path.",
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Short run label appended to --run-dir and stored in the manifest.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Graph-cache directory. Defaults to <run-dir>/cache and is shared across configurations.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Recompute and overwrite segmentized-graph and raw-membership caches.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load compatible segmentized and raw-graph membership caches when present.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Clip roads and blocks to the diagnostic SMOKE_BBOX.",
    )
    parser.add_argument(
        "--legacy-mode",
        action="store_true",
        help="Enable every legacy behavioral switch for published-run verification.",
    )
    parser.add_argument(
        "--legacy-service-snap",
        action="store_true",
        help="Snap services to the unconstrained nearest raw-graph node.",
    )
    parser.add_argument(
        "--legacy-origin-snap",
        action="store_true",
        help="Snap origins to all raw-graph nodes instead of LCC nodes only.",
    )
    parser.add_argument(
        "--legacy-centroid",
        action="store_true",
        help="Use polygon centroids as primary origins instead of representative points.",
    )
    parser.add_argument(
        "--legacy-origin-failure-status",
        action="store_true",
        help="Classify failed origin snaps as isolated instead of unclassified.",
    )
    parser.add_argument(
        "--legacy-collocated-rule",
        action="store_true",
        help="Use the legacy node-degree shortcut for collocated origins and services.",
    )
    parser.add_argument(
        "--legacy-centroid-inundation-join",
        action="store_true",
        help="Use the legacy positional spatial-join assignment for centroid inundation.",
    )
    return parser.parse_args()


def build_output_suffix(args: argparse.Namespace) -> str:
    if args.output_suffix:
        return args.output_suffix
    if args.max_blocks is not None or args.slr_ft:
        return "__subset"
    return ""


def apply_legacy_mode(args: argparse.Namespace) -> argparse.Namespace:
    if not args.legacy_mode:
        return args
    args.legacy_service_snap = True
    args.legacy_origin_snap = True
    args.legacy_centroid = True
    args.legacy_origin_failure_status = True
    args.legacy_collocated_rule = True
    args.legacy_centroid_inundation_join = True
    return args


def validate_config_name(config_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config_name):
        raise ValueError(
            "--config-name must start with an alphanumeric character and contain only letters, numbers, '.', '_', or '-'."
        )
    return config_name


def resolve_run_directories(args: argparse.Namespace) -> tuple[Path, Path]:
    config_name = validate_config_name(args.config_name)
    run_parent = Path(args.run_dir)
    run_dir = run_parent / config_name
    cache_dir = Path(args.cache_dir) if args.cache_dir is not None else run_parent / "cache"
    return run_dir, cache_dir


def resolve_requested_layers(args: argparse.Namespace) -> dict[int, str]:
    supported = {BASELINE_SLR_FT: BASELINE_SLR_LAYER, **SLR_LAYER_MAP}
    if args.scenarios is not None and args.slr_ft:
        raise ValueError("Use either --scenarios or --slr-ft, not both.")

    if args.scenarios is not None:
        tokens = [token.strip() for token in args.scenarios.split(",") if token.strip()]
        if not tokens:
            raise ValueError("--scenarios must contain at least one SLR level.")
        try:
            requested = {int(token) for token in tokens}
        except ValueError as exc:
            raise ValueError("--scenarios must be a comma-separated list of integers.") from exc
    elif args.slr_ft:
        requested = {BASELINE_SLR_FT, *args.slr_ft}
    else:
        requested = set(supported)

    invalid = sorted(requested - set(supported))
    if invalid:
        raise ValueError(
            f"Unsupported SLR level(s): {invalid}. Supported values: {sorted(supported)}"
        )
    if BASELINE_SLR_FT not in requested:
        warn("Adding the 0 ft baseline because positive scenarios require baseline comparison fields.")
        requested.add(BASELINE_SLR_FT)
    return {slr_ft: supported[slr_ft] for slr_ft in sorted(requested)}


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


def load_block_attributes() -> pd.DataFrame:
    attributes = pyogrio.read_dataframe(
        CENSUS_BLOCK_ATTRIBUTES_PATH,
        columns=["GEOID20", "COUNTYFP20", "POP20", "ALAND20"],
        read_geometry=False,
    )
    attributes["COUNTYFP20"] = attributes["COUNTYFP20"].astype("string").str.zfill(3)
    attributes = attributes.loc[attributes["COUNTYFP20"].isin(TRICOUNTY_FIPS)].copy()
    attributes["block_geoid"] = attributes["GEOID20"].astype("string").str.zfill(15)
    attributes["pop20"] = pd.to_numeric(attributes["POP20"], errors="raise").astype("int64")
    attributes["land_area_m2"] = pd.to_numeric(
        attributes["ALAND20"], errors="raise"
    ).astype("int64")
    attributes = attributes[["block_geoid", "pop20", "land_area_m2"]].copy()
    if attributes["block_geoid"].duplicated().any():
        raise ValueError("Duplicate GEOID20 values in tri-county Census block attributes.")
    return attributes


def highway_filter_hash(drivable_highways: set[str]) -> str:
    payload = "|".join(sorted(drivable_highways)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def graph_cache_paths(
    cache_dir: Path,
    drivable_highways: set[str],
    *,
    smoke: bool,
) -> dict[str, Path]:
    filter_hash = highway_filter_hash(drivable_highways)
    scope = "smoke" if smoke else "full"
    stem = f"{filter_hash}_{scope}_v{CACHE_SCHEMA_VERSION}"
    return {
        "nodes": cache_dir / f"segmentized_nodes_{stem}.parquet",
        "edges": cache_dir / f"segmentized_edges_{stem}.parquet",
        "raw_membership": cache_dir / f"raw_graph_membership_{stem}.parquet",
        "metadata": cache_dir / f"graph_cache_{stem}.json",
    }


def load_or_build_segmentized_roads(
    roads: gpd.GeoDataFrame,
    *,
    cache_dir: Path,
    drivable_highways: set[str],
    smoke: bool,
    rebuild_cache: bool,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Path]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = graph_cache_paths(cache_dir, drivable_highways, smoke=smoke)
    cache_available = paths["nodes"].exists() and paths["edges"].exists()
    if cache_available and not rebuild_cache:
        log(f"Loaded segmentized nodes cache: {paths['nodes']}")
        log(f"Loaded segmentized edges cache: {paths['edges']}")
        nodes = gpd.read_parquet(paths["nodes"])
        edges = gpd.read_parquet(paths["edges"])
        required_edge_columns = {
            "edge_id",
            "u",
            "v",
            "osm_id",
            "highway",
            "bridge_tag_present",
            "layer_value",
            "bridge_like",
            "length_m",
            "geometry",
        }
        missing_columns = required_edge_columns - set(edges.columns)
        if missing_columns:
            raise ValueError(
                f"Segmentized edge cache is incompatible; missing columns: {sorted(missing_columns)}. Use --rebuild-cache."
            )
        return nodes, edges, paths

    log("Computing segmentized road cache.")
    nodes, edges = segmentize_roads(roads)
    nodes.to_parquet(paths["nodes"], index=False)
    edges.to_parquet(paths["edges"], index=False)
    cache_metadata = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "drivable_highways": sorted(drivable_highways),
        "highway_filter_hash": highway_filter_hash(drivable_highways),
        "smoke": bool(smoke),
        "road_source_size": ROAD_PBF_PATH.stat().st_size,
        "road_source_mtime": ROAD_PBF_PATH.stat().st_mtime,
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
    }
    paths["metadata"].write_text(
        json.dumps(cache_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    log(f"Computed and saved segmentized nodes cache: {paths['nodes']}")
    log(f"Computed and saved segmentized edges cache: {paths['edges']}")
    return nodes, edges, paths


def compute_raw_graph_membership(graph: nx.Graph) -> dict[str, object]:
    node_to_component_id: dict[int, int] = {}
    node_to_component_size: dict[int, int] = {}
    largest_component_id = -1
    largest_component_size = -1
    for component_id, component_nodes in enumerate(nx.connected_components(graph)):
        component_size = len(component_nodes)
        if component_size > largest_component_size:
            largest_component_id = int(component_id)
            largest_component_size = int(component_size)
        for node_id in component_nodes:
            node_to_component_id[int(node_id)] = int(component_id)
            node_to_component_size[int(node_id)] = int(component_size)

    node_to_2ecc_id: dict[int, int] = {}
    node_to_2ecc_size: dict[int, int] = {}
    for component_id, component_nodes in enumerate(k_edge_components(graph, 2)):
        component_size = len(component_nodes)
        for node_id in component_nodes:
            node_to_2ecc_id[int(node_id)] = int(component_id)
            node_to_2ecc_size[int(node_id)] = int(component_size)

    largest_component_nodes = {
        node_id
        for node_id, component_id in node_to_component_id.items()
        if component_id == largest_component_id
    }
    return {
        "node_to_component_id": node_to_component_id,
        "node_to_component_size": node_to_component_size,
        "largest_component_id": largest_component_id,
        "largest_component_size": largest_component_size,
        "largest_component_nodes": largest_component_nodes,
        "node_to_2ecc_id": node_to_2ecc_id,
        "node_to_2ecc_size": node_to_2ecc_size,
    }


def raw_membership_to_frame(membership: dict[str, object]) -> pd.DataFrame:
    node_to_component_id = membership["node_to_component_id"]
    node_to_component_size = membership["node_to_component_size"]
    node_to_2ecc_id = membership["node_to_2ecc_id"]
    node_to_2ecc_size = membership["node_to_2ecc_size"]
    largest_component_id = membership["largest_component_id"]
    records = []
    for node_id, component_id in node_to_component_id.items():
        records.append(
            {
                "node_id": int(node_id),
                "raw_component_id": int(component_id),
                "raw_component_size": int(node_to_component_size[node_id]),
                "raw_in_lcc": bool(component_id == largest_component_id),
                "raw_2ecc_id": int(node_to_2ecc_id.get(node_id, -1)),
                "raw_2ecc_size": int(node_to_2ecc_size.get(node_id, 0)),
            }
        )
    return pd.DataFrame.from_records(records)


def raw_membership_from_frame(frame: pd.DataFrame) -> dict[str, object]:
    largest_rows = frame.loc[frame["raw_in_lcc"]]
    largest_component_id = (
        int(largest_rows["raw_component_id"].iloc[0]) if not largest_rows.empty else -1
    )
    return {
        "node_to_component_id": frame.set_index("node_id")["raw_component_id"].astype(int).to_dict(),
        "node_to_component_size": frame.set_index("node_id")["raw_component_size"].astype(int).to_dict(),
        "largest_component_id": largest_component_id,
        "largest_component_size": int(len(largest_rows)),
        "largest_component_nodes": set(largest_rows["node_id"].astype(int)),
        "node_to_2ecc_id": frame.set_index("node_id")["raw_2ecc_id"].astype(int).to_dict(),
        "node_to_2ecc_size": frame.set_index("node_id")["raw_2ecc_size"].astype(int).to_dict(),
    }


def load_or_compute_raw_graph_membership(
    graph: nx.Graph,
    *,
    cache_path: Path,
    resume: bool,
    rebuild_cache: bool,
) -> dict[str, object]:
    if resume and cache_path.exists() and not rebuild_cache:
        log(f"Loaded raw-graph connected-component and 2ecc cache: {cache_path}")
        return raw_membership_from_frame(pd.read_parquet(cache_path))

    log("Computing raw-graph connected components and 2-edge-connected components.")
    membership = compute_raw_graph_membership(graph)
    raw_membership_to_frame(membership).to_parquet(cache_path, index=False)
    log(f"Computed and saved raw-graph membership cache: {cache_path}")
    return membership


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


def make_centroids(
    origins_layer: gpd.GeoDataFrame,
    *,
    use_representative_point: bool = False,
) -> gpd.GeoDataFrame:
    centroids = origins_layer.copy()
    if use_representative_point:
        primary_geom = centroids.geometry.representative_point()
        fallback_geom = centroids.geometry.centroid
        primary_method = "representative_point"
        fallback_method = "centroid_fallback"
    else:
        primary_geom = centroids.geometry.centroid
        fallback_geom = centroids.geometry.representative_point()
        primary_method = "centroid"
        fallback_method = "representative_point_fallback"

    valid_primary = (
        primary_geom.notna()
        & ~primary_geom.is_empty
        & np.isfinite(primary_geom.x)
        & np.isfinite(primary_geom.y)
    )
    primary_geom.loc[~valid_primary] = fallback_geom.loc[~valid_primary]
    centroids["geometry"] = primary_geom
    centroids["origin_geometry_method"] = np.where(
        valid_primary, primary_method, fallback_method
    )
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


def filter_drivable_roads(
    roads: gpd.GeoDataFrame,
    *,
    drivable_highways: set[str] | None = None,
) -> gpd.GeoDataFrame:
    output = roads.copy()
    highway_filter = DRIVABLE_HIGHWAYS if drivable_highways is None else drivable_highways
    output["highway"] = output["highway"].astype("string")
    output = output.loc[output["highway"].isin(highway_filter)].copy()

    if DROP_PRIVATE_ACCESS_EDGES:
        output["access_tag"] = output["other_tags"].map(lambda value: parse_other_tag(value, "access"))
        output = output.loc[output["access_tag"].fillna("") != "private"].copy()
    else:
        output["access_tag"] = None

    output = output.loc[output.geometry.notnull() & ~output.geometry.is_empty].copy()
    return output


def load_roads(
    *,
    drivable_highways: set[str] | None = None,
    smoke: bool = False,
) -> gpd.GeoDataFrame:
    roads = read_vector(
        ROAD_PBF_PATH,
        layer="lines",
        columns=["osm_id", "highway", "z_order", "other_tags", "geometry"],
    )
    roads = filter_drivable_roads(roads, drivable_highways=drivable_highways)
    roads["osm_id"] = roads["osm_id"].astype(str)
    if smoke:
        roads = roads.set_crs("OGC:CRS84", allow_override=True)
        roads = gpd.clip(roads, box(*SMOKE_BBOX), keep_geom_type=True)
        roads = roads.loc[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
        if roads.empty:
            raise RuntimeError("Smoke bbox retained no drivable road geometries.")
    return roads


def is_nonzero_layer_value(value: str | None) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    try:
        return float(text) != 0.0
    except ValueError:
        return True

def segmentize_roads(roads: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split retained road geometries into consecutive-vertex graph edges."""
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
        bridge_value = parse_other_tag(row.other_tags, "bridge")
        layer_value = parse_other_tag(row.other_tags, "layer")
        bridge_tag_present = bridge_value is not None
        bridge_like = bridge_tag_present or is_nonzero_layer_value(layer_value)
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
                        "bridge_tag_present": bool(bridge_tag_present),
                        "layer_value": layer_value,
                        "bridge_like": bool(bridge_like),
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


def attach_services_to_raw_graph(
    services: gpd.GeoDataFrame,
    *,
    nodes: gpd.GeoDataFrame,
    unconstrained_tree: cKDTree,
    unconstrained_node_ids: np.ndarray,
    raw_membership: dict[str, object],
    use_eligible_service_nodes: bool,
) -> pd.DataFrame:
    unconstrained = snap_points_to_nodes(
        services[["service_id", "geometry"]].copy(),
        point_id_col="service_id",
        tree=unconstrained_tree,
        node_ids=unconstrained_node_ids,
        nodes=nodes,
        max_snap_m=MAX_SERVICE_SNAP_M,
    ).rename(
        columns={
            "node_id": "unconstrained_node_id",
            "snap_distance_m": "unconstrained_snap_distance_m",
            "snap_valid": "unconstrained_snap_valid",
            "x": "unconstrained_x",
            "y": "unconstrained_y",
        }
    )
    attached = services.merge(unconstrained, on="service_id", how="left")
    attached = attached.reset_index(drop=True)
    attached.insert(0, "service_record_id", np.arange(len(attached), dtype=int))

    attached["node_id"] = attached["unconstrained_node_id"].astype(int)
    attached["snap_distance_m"] = attached["unconstrained_snap_distance_m"].astype(float)
    attached["snap_valid"] = attached["unconstrained_snap_valid"].astype(bool)
    attached["service_snap_rule"] = "legacy_unconstrained"

    if use_eligible_service_nodes:
        largest_component_nodes = raw_membership["largest_component_nodes"]
        node_to_2ecc_size = raw_membership["node_to_2ecc_size"]
        eligible_node_ids = {
            int(node_id)
            for node_id in largest_component_nodes
            if int(node_to_2ecc_size.get(int(node_id), 0)) >= 2
        }
        eligible_nodes = nodes.loc[nodes["node_id"].isin(eligible_node_ids)].copy()
        if eligible_nodes.empty:
            warn("No raw-graph LCC nodes with 2ecc size >= 2; all services use fallback snaps.")
            attached["service_snap_rule"] = "fallback_unconstrained"
        else:
            eligible_tree, _, eligible_ids = build_node_kdtree(eligible_nodes)
            eligible_snap = snap_points_to_nodes(
                attached[["service_record_id", "geometry"]].copy(),
                point_id_col="service_record_id",
                tree=eligible_tree,
                node_ids=eligible_ids,
                nodes=eligible_nodes,
                max_snap_m=MAX_SERVICE_SNAP_M,
            ).rename(
                columns={
                    "node_id": "eligible_node_id",
                    "snap_distance_m": "eligible_snap_distance_m",
                    "snap_valid": "eligible_snap_valid",
                }
            )
            attached = attached.merge(
                eligible_snap[
                    [
                        "service_record_id",
                        "eligible_node_id",
                        "eligible_snap_distance_m",
                        "eligible_snap_valid",
                    ]
                ],
                on="service_record_id",
                how="left",
                validate="one_to_one",
            )
            use_eligible = attached["eligible_snap_valid"].fillna(False)
            attached.loc[use_eligible, "node_id"] = attached.loc[
                use_eligible, "eligible_node_id"
            ].astype(int)
            attached.loc[use_eligible, "snap_distance_m"] = attached.loc[
                use_eligible, "eligible_snap_distance_m"
            ].astype(float)
            attached.loc[use_eligible, "snap_valid"] = True
            attached.loc[use_eligible, "service_snap_rule"] = "raw_lcc_2ecc"
            attached.loc[~use_eligible, "service_snap_rule"] = "fallback_unconstrained"

    node_to_2ecc_size = raw_membership["node_to_2ecc_size"]
    attached["service_raw_2ecc_size"] = (
        attached["node_id"].map(node_to_2ecc_size).fillna(0).astype(int)
    )
    attached["service_snap_distance_penalty_m"] = (
        attached["snap_distance_m"] - attached["unconstrained_snap_distance_m"]
    )
    attached["service_node_moved"] = attached["node_id"].ne(
        attached["unconstrained_node_id"]
    )
    chosen_xy = nodes.set_index("node_id")[["x", "y"]]
    attached = attached.drop(columns=["x", "y"], errors="ignore").join(
        chosen_xy, on="node_id"
    )
    return attached


def snap_origins_to_raw_graph(
    origins: gpd.GeoDataFrame,
    *,
    nodes: gpd.GeoDataFrame,
    raw_membership: dict[str, object],
    restrict_to_lcc: bool,
) -> pd.DataFrame:
    snap_nodes = nodes
    if restrict_to_lcc:
        snap_nodes = nodes.loc[
            nodes["node_id"].isin(raw_membership["largest_component_nodes"])
        ].copy()
    tree, _, node_ids = build_node_kdtree(snap_nodes)
    snapped = snap_points_to_nodes(
        origins,
        point_id_col="block_geoid",
        tree=tree,
        node_ids=node_ids,
        nodes=snap_nodes,
        max_snap_m=MAX_ORIGIN_SNAP_M,
    )
    node_to_component_id = raw_membership["node_to_component_id"]
    largest_component_id = raw_membership["largest_component_id"]
    snapped["origin_in_lcc"] = snapped["node_id"].map(node_to_component_id).eq(
        largest_component_id
    )
    return snapped

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

def build_nearest_service_lookup(
    graph: nx.Graph,
    services: pd.DataFrame,
) -> pd.DataFrame:
    if services.empty:
        return pd.DataFrame(
            columns=["node_id", "nearest_service_node_id", "nearest_service_id", "nearest_service_type", "nearest_service_snap_rule", "distance_m"]
        )

    service_columns = ["node_id", "service_id", "service_type"]
    if "service_snap_rule" in services.columns:
        service_columns.append("service_snap_rule")
    canonical_services = (
        services.sort_values(["snap_distance_m", "service_id"])
        .drop_duplicates("node_id")
        .loc[:, service_columns]
        .rename(columns={"node_id": "nearest_service_node_id"})
    )
    canonical_services = canonical_services.loc[
        canonical_services["nearest_service_node_id"].isin(graph.nodes)
    ].copy()
    if canonical_services.empty:
        return pd.DataFrame(
            columns=["node_id", "nearest_service_node_id", "nearest_service_id", "nearest_service_type", "nearest_service_snap_rule", "distance_m"]
        )
    service_nodes = canonical_services["nearest_service_node_id"].astype(int).tolist()

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
                "nearest_service_snap_rule": service_meta.get(
                    "service_snap_rule", pd.NA
                ),
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


def capped_local_edge_connectivity(
    graph: nx.Graph,
    origin_node: int,
    *,
    nearest_service_node: int | None,
    two_edge_component_lookup: dict[int, int],
    two_edge_component_service_counts: dict[int, int],
    cap: int = MAX_EDGE_DISJOINT_PATHS_CAP,
    legacy_collocated_rule: bool = True,
) -> int:
    if origin_node not in graph:
        return 0
    origin_degree = int(graph.degree(origin_node))
    if origin_degree == 0:
        return 0
    if (
        legacy_collocated_rule
        and nearest_service_node is not None
        and nearest_service_node == origin_node
    ):
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
    if row.get("block_centroid_unclassified", 0) == 1:
        return "unclassified"
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
    unclassified_col: str | None = None,
) -> pd.Series:
    conditions = [
        frame[unclassified_col].eq(1) if unclassified_col is not None else pd.Series(False, index=frame.index),
        frame[inundated_col].eq(1),
        frame[isolated_col].eq(1),
        frame[redundant_col].eq(1),
        frame[fragile_col].eq(1),
    ]
    choices = ["unclassified", "inundated", "isolated", "redundant", "fragile"]
    return pd.Series(np.select(conditions, choices, default="other"), index=frame.index, dtype="object")


def add_baseline_comparison_fields(results: pd.DataFrame) -> pd.DataFrame:
    baseline_columns = [
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
    ]
    if "block_centroid_unclassified" in results.columns:
        baseline_columns.append("block_centroid_unclassified")
    baseline = (
        results.loc[results["slr_ft"] == BASELINE_SLR_FT, baseline_columns]
        .drop_duplicates("block_geoid")
        .rename(
            columns={
                "block_centroid_inundated": "baseline_block_centroid_inundated",
                "block_centroid_isolated": "baseline_block_centroid_isolated",
                "block_centroid_redundant": "baseline_block_centroid_redundant",
                "block_centroid_fragile": "baseline_block_centroid_fragile",
                "block_centroid_unclassified": "baseline_block_centroid_unclassified",
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
        unclassified_col=(
            "baseline_block_centroid_unclassified"
            if "baseline_block_centroid_unclassified" in output.columns
            else None
        ),
    )
    output["scenario_status"] = classify_status_columns(
        output,
        inundated_col="block_centroid_inundated",
        isolated_col="block_centroid_isolated",
        redundant_col="block_centroid_redundant",
        fragile_col="block_centroid_fragile",
        unclassified_col=(
            "block_centroid_unclassified"
            if "block_centroid_unclassified" in output.columns
            else None
        ),
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
    unclassify_failed_origins: bool = False,
    legacy_collocated_rule: bool = True,
    legacy_centroid_inundation_join: bool = True,
    bridge_rule_applied: str = "intersect",
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
        if legacy_centroid_inundation_join:
            flooded_centroids = gpd.sjoin(
                centroid_inundated,
                slr_layer[["geometry"]],
                how="left",
                predicate="intersects",
            )
            centroid_inundated["block_centroid_inundated"] = (
                flooded_centroids["index_right"].notna().astype(int).to_numpy()
            )
        else:
            inundation_union = slr_layer.geometry.union_all()
            centroid_inundated["block_centroid_inundated"] = (
                centroid_inundated.geometry.intersects(inundation_union).astype(int)
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
        nearest_service_snap_rule = pd.NA
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
                nearest_service_snap_rule = nearest_service_row.get(
                    "nearest_service_snap_rule", pd.NA
                )
                dry_distance = float(nearest_service_row["distance_m"])
                nearest_service_node = int(nearest_service_row["nearest_service_node_id"])

            if reachable_service_count > 0 and not centroid_is_flooded:
                max_edge_disjoint_paths = capped_local_edge_connectivity(
                    graph,
                    origin_node_id,
                    nearest_service_node=nearest_service_node,
                    two_edge_component_lookup=dry_two_edge_component_lookup,
                    two_edge_component_service_counts=dry_two_edge_component_service_counts,
                    legacy_collocated_rule=legacy_collocated_rule,
                )

        access_failure = (not origin_valid) or reachable_service_count == 0
        is_unclassified = int(unclassify_failed_origins and not origin_valid)
        is_inundated = int(centroid_is_flooded)
        is_isolated = int(
            (not is_unclassified) and (not centroid_is_flooded) and access_failure
        )
        is_redundant = int(
            (not is_unclassified)
            and
            (not centroid_is_flooded)
            and (not access_failure)
            and max_edge_disjoint_paths >= 2
        )
        is_fragile = int(
            (not is_unclassified)
            and
            (not centroid_is_flooded)
            and (not access_failure)
            and max_edge_disjoint_paths == 1
        )

        if is_inundated or is_isolated or is_unclassified:
            nearest_service_id = pd.NA
            nearest_service_type = pd.NA
            nearest_service_snap_rule = pd.NA
            dry_distance = np.nan
            max_edge_disjoint_paths = 0

        pop20 = int(getattr(row, "pop20", 0))
        land_area_m2 = int(getattr(row, "land_area_m2", 1))
        origin_in_lcc = bool(getattr(row, "origin_in_lcc", True))
        origin_geometry_method = getattr(row, "origin_geometry_method", "centroid")
        zero_land_area = land_area_m2 == 0
        analysis_eligible = bool(origin_valid and not zero_land_area)
        if not origin_valid:
            exclusion_reason = "origin_snap_failed"
        elif zero_land_area:
            exclusion_reason = "zero_land_area"
        else:
            exclusion_reason = ""

        records.append(
            {
                "block_geoid": row.block_geoid,
                "block_group_geoid": row.block_group_geoid,
                "tract_geoid": row.tract_geoid,
                "block": row.block,
                "county_fips": row.county_fips,
                "county_name": row.county_name,
                "pop20": pop20,
                "land_area_m2": land_area_m2,
                "analysis_eligible": analysis_eligible,
                "exclusion_reason": exclusion_reason,
                "slr_ft": slr_ft,
                "slr_layer_name": slr_layer_name,
                "origin_node_id": origin_node_id,
                "origin_snap_distance_m": float(row.snap_distance_m),
                "origin_snap_exceeds_threshold": int(not origin_valid),
                "origin_in_lcc": origin_in_lcc,
                "origin_geometry_method": origin_geometry_method,
                "boundary_flag": int(bool(row.boundary_flag)),
                "boundary_distance_m": float(row.boundary_distance_m),
                "component_touches_boundary": int(component_touches_boundary),
                "block_centroid_inundated": is_inundated,
                "block_centroid_isolated": is_isolated,
                "block_centroid_redundant": is_redundant,
                "block_centroid_fragile": is_fragile,
                "block_centroid_unclassified": is_unclassified,
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
                "service_snap_rule": nearest_service_snap_rule,
                "bridge_rule_applied": bridge_rule_applied,
                "baseline_shortest_path_distance_m": baseline_distance,
                "dry_shortest_path_distance_m": dry_distance,
                "detour_ratio": safe_detour_ratio(baseline_distance, dry_distance),
            }
        )

    return pd.DataFrame.from_records(records)


def save_main_output(
    results: pd.DataFrame,
    output_stem: str,
    *,
    output_dir: Path | None = None,
) -> Path:
    target_dir = OUTPUT_DIR if output_dir is None else output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / f"{output_stem}.csv"
    results.to_csv(csv_path, index=False)
    parquet_path = target_dir / f"{output_stem}.parquet"
    results.to_parquet(parquet_path, index=False)
    log(f"Saved long-format CSV to {csv_path}")
    log(f"Saved long-format Parquet to {parquet_path}")
    return parquet_path


def save_qa_sample(
    results: pd.DataFrame,
    centroids: gpd.GeoDataFrame,
    qa_stem: str,
    *,
    output_dir: Path | None = None,
) -> Path:
    target_dir = OUTPUT_DIR if output_dir is None else output_dir
    qa = results.copy()
    qa["status_category"] = qa.apply(classify_status, axis=1)
    qa = qa.loc[qa["status_category"] != "other"].copy()

    if qa.empty:
        warn("No QA sample rows were available to save.")
        return target_dir / f"{qa_stem}.geojson"

    qa = (
        qa.groupby(["slr_ft", "county_name", "status_category"], group_keys=False)
        .apply(lambda frame: frame.sample(min(len(frame), QA_SAMPLE_PER_GROUP), random_state=42))
        .reset_index(drop=True)
    )

    qa_geo = centroids[["block_geoid", "geometry"]].merge(qa, on="block_geoid", how="inner")
    qa_geo = gpd.GeoDataFrame(qa_geo, geometry="geometry", crs=PROJECTED_CRS).to_crs("EPSG:4326")
    qa_path = target_dir / f"{qa_stem}.geojson"
    qa_geo.to_file(qa_path, driver="GeoJSON")
    log(f"Saved QA centroid sample to {qa_path}")
    return qa_path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def expand_input_paths(paths: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.suffix.lower() == ".shp":
            expanded.extend(sorted(path.parent.glob(f"{path.stem}.*")))
        else:
            expanded.append(path)
    unique: dict[str, Path] = {}
    for path in expanded:
        unique[str(path.resolve())] = path
    return list(unique.values())


def input_file_manifest(paths: Iterable[Path]) -> list[dict[str, object]]:
    records = []
    for path in expand_input_paths(paths):
        stat = path.stat()
        records.append(
            {
                "path": str(path.resolve()),
                "size_bytes": int(stat.st_size),
                "mtime_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                "sha256": sha256_file(path),
            }
        )
    return records


def manifest_json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(manifest_json_value(item) for item in value)
    if isinstance(value, tuple):
        return [manifest_json_value(item) for item in value]
    if isinstance(value, list):
        return [manifest_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): manifest_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def module_parameter_manifest() -> dict[str, object]:
    return {
        name: manifest_json_value(value)
        for name, value in sorted(globals().items())
        if name.isupper() and not name.startswith("_")
    }


def git_manifest() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "working_tree_dirty": bool(status.strip())}


def resolved_version_manifest() -> dict[str, str]:
    return {
        package_name: importlib.metadata.version(package_name)
        for package_name in [
            "geopandas",
            "shapely",
            "pyogrio",
            "pandas",
            "numpy",
            "networkx",
            "scipy",
        ]
    }


def build_run_manifest(
    *,
    args: argparse.Namespace,
    results: pd.DataFrame,
    service_audit: pd.DataFrame,
    origins: pd.DataFrame,
    input_paths: Iterable[Path],
    bridge_structure_summary: pd.DataFrame | None = None,
    include_input_hashes: bool = True,
) -> dict[str, object]:
    baseline = results.loc[results["slr_ft"].eq(BASELINE_SLR_FT)].drop_duplicates(
        "block_geoid"
    )
    eligibility_counts = (
        baseline["exclusion_reason"].replace("", "eligible").value_counts().to_dict()
    )
    service_rule_counts = service_audit["service_snap_rule"].value_counts().to_dict()
    service_rule_counts["moved_total"] = int(service_audit["service_node_moved"].sum())
    bridge_counts: list[dict[str, object]] = []
    if bridge_structure_summary is not None and not bridge_structure_summary.empty:
        bridge_counts = bridge_structure_summary.to_dict(orient="records")

    cli_flags = {
        key: manifest_json_value(value) for key, value in sorted(vars(args).items())
    }
    manifest = {
        "git": git_manifest(),
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "module_parameters": module_parameter_manifest(),
        "cli_flags": cli_flags,
        "input_files": input_file_manifest(input_paths) if include_input_hashes else [],
        "versions": resolved_version_manifest(),
        "row_counts_by_scenario": {
            str(int(slr_ft)): int(count)
            for slr_ft, count in results.groupby("slr_ft").size().items()
        },
        "service_snap_counts": {
            str(key): int(value) for key, value in service_rule_counts.items()
        },
        "origin_snap_failures": int((~origins["snap_valid"]).sum()),
        "eligibility_reason_counts": {
            str(key): int(value) for key, value in eligibility_counts.items()
        },
        "bridge_structure_counts": bridge_counts,
    }
    return manifest


def write_run_manifest(manifest: dict[str, object], path: Path) -> None:
    required_keys = {
        "git",
        "utc_timestamp",
        "module_parameters",
        "cli_flags",
        "input_files",
        "versions",
        "row_counts_by_scenario",
        "service_snap_counts",
        "origin_snap_failures",
        "eligibility_reason_counts",
        "bridge_structure_counts",
    }
    missing = required_keys - set(manifest)
    if missing:
        raise ValueError(f"Run manifest is missing required keys: {sorted(missing)}")
    serialized = json.dumps(manifest, indent=2, sort_keys=True)
    json.loads(serialized)
    path.write_text(serialized + "\n", encoding="utf-8")
    log(f"Saved run manifest to {path}")


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
    args = apply_legacy_mode(parse_args())
    output_suffix = build_output_suffix(args)
    output_stem = f"{DEFAULT_OUTPUT_STEM}{output_suffix}"
    qa_stem = f"{DEFAULT_QA_STEM}{output_suffix}"
    run_dir, cache_dir = resolve_run_directories(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    args.resolved_run_dir = run_dir
    args.resolved_cache_dir = cache_dir

    check_required_inputs(
        [
            BLOCKS_PATH,
            CENSUS_BLOCK_ATTRIBUTES_PATH,
            NOAA_GPKG_PATH,
            PRIVATE_SCHOOLS_PATH,
            PUBLIC_SCHOOLS_PATH,
            FIRE_STATIONS_PATH,
            ROAD_PBF_PATH,
        ]
    )

    available_layers = list_layers(NOAA_GPKG_PATH)
    requested_layers = resolve_requested_layers(args)

    missing_layers = [layer_name for layer_name in requested_layers.values() if layer_name not in available_layers]
    if missing_layers:
        raise ValueError(f"Missing requested NOAA layer(s): {missing_layers}")

    start_time = time.time()
    log("Loading blocks, services, and roads...")
    blocks = read_vector(BLOCKS_PATH)
    blocks = prepare_blocks_layer(blocks)
    blocks = maybe_to_projected(blocks)
    block_attributes = load_block_attributes()
    blocks = blocks.merge(
        block_attributes,
        on="block_geoid",
        how="left",
        validate="one_to_one",
    )
    if blocks[["pop20", "land_area_m2"]].isna().any(axis=None):
        missing_count = int(blocks[["pop20", "land_area_m2"]].isna().any(axis=1).sum())
        raise ValueError(f"Blocks missing Census POP20/ALAND20 attributes: {missing_count}")
    if args.max_blocks is not None:
        blocks = blocks.sort_values("block_geoid").head(args.max_blocks).copy()
        warn(f"Running smoke-test subset with max_blocks={args.max_blocks}")

    centroids = make_centroids(
        blocks, use_representative_point=not args.legacy_centroid
    )
    services = load_services()
    roads = load_roads(smoke=args.smoke)
    roads = roads.set_crs("OGC:CRS84", allow_override=True)

    if args.smoke:
        # Keep the smoke-test block universe invariant across origin-geometry
        # variants.  Selecting on the corrected representative point would
        # exchange boundary blocks with the legacy centroid selection and
        # defeat row-for-row configuration comparisons.
        smoke_selector = make_centroids(
            blocks, use_representative_point=False
        ).to_crs("OGC:CRS84")
        smoke_block_geoids = set(
            smoke_selector.loc[
                smoke_selector.geometry.intersects(box(*SMOKE_BBOX)),
                "block_geoid",
            ]
        )
        centroids = centroids.loc[
            centroids["block_geoid"].isin(smoke_block_geoids)
        ].copy()
        warn(
            f"Running bbox smoke subset with {len(centroids):,} blocks and bbox={SMOKE_BBOX}."
        )
    centroids_source = centroids.to_crs("OGC:CRS84")

    boundary_polygon = build_study_area_boundary(tuple(roads.total_bounds), roads.crs)
    source_clip_polygon = box(*roads.total_bounds)
    centroid_boundary = compute_origin_boundary_fields(centroids, boundary_polygon)
    services = filter_services_by_buffer(services, boundary_polygon)
    log(f"Services inside the buffered retained-network boundary: {len(services):,}")

    log("Segmentizing roads and building baseline graph...")
    highway_filter = set(DRIVABLE_HIGHWAYS)
    nodes, edges, cache_paths = load_or_build_segmentized_roads(
        roads,
        cache_dir=cache_dir,
        drivable_highways=highway_filter,
        smoke=args.smoke,
        rebuild_cache=args.rebuild_cache,
    )
    graph_baseline = build_graph(edges)
    tree, _, node_ids = build_node_kdtree(nodes)
    raw_membership = load_or_compute_raw_graph_membership(
        graph_baseline,
        cache_path=cache_paths["raw_membership"],
        resume=args.resume,
        rebuild_cache=args.rebuild_cache,
    )

    road_bounds = roads.total_bounds
    log(
        "Observed retained-network road bbox used for diagnostics/filtering (lon/lat): "
        f"{tuple(round(value, 4) for value in road_bounds)}"
    )
    log(f"Baseline graph nodes: {graph_baseline.number_of_nodes():,}")
    log(f"Baseline graph edges: {graph_baseline.number_of_edges():,}")
    log(f"Segmentized road edges: {len(edges):,}")
    log(f"Raw graph largest connected component nodes: {raw_membership['largest_component_size']:,}")

    origins = centroids[
        [
            "block_geoid",
            "block_group_geoid",
            "tract_geoid",
            "block",
            "county_fips",
            "county_name",
            "pop20",
            "land_area_m2",
            "origin_geometry_method",
            "geometry",
        ]
    ].copy()
    origin_snap = snap_origins_to_raw_graph(
        origins,
        nodes=nodes,
        raw_membership=raw_membership,
        restrict_to_lcc=not args.legacy_origin_snap,
    )
    origins = origins.merge(origin_snap, on="block_geoid", how="left")

    services = attach_services_to_raw_graph(
        services,
        nodes=nodes,
        unconstrained_tree=tree,
        unconstrained_node_ids=node_ids,
        raw_membership=raw_membership,
        use_eligible_service_nodes=not args.legacy_service_snap,
    )
    service_audit_columns = [
        "service_record_id",
        "service_id",
        "service_type",
        "service_source",
        "service_name",
        "unconstrained_node_id",
        "unconstrained_snap_distance_m",
        "node_id",
        "snap_distance_m",
        "snap_valid",
        "service_snap_rule",
        "service_raw_2ecc_size",
        "service_snap_distance_penalty_m",
        "service_node_moved",
    ]
    service_audit = services[service_audit_columns].copy()
    service_audit_path = run_dir / "service_snapping_audit.csv"
    service_audit.to_csv(service_audit_path, index=False)
    log(f"Saved service snapping audit to {service_audit_path}")
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
            unclassify_failed_origins=not args.legacy_origin_failure_status,
            legacy_collocated_rule=args.legacy_collocated_rule,
            legacy_centroid_inundation_join=args.legacy_centroid_inundation_join,
            bridge_rule_applied="intersect",
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
    main_output_path = save_main_output(results, output_stem, output_dir=run_dir)
    qa_output_path = save_qa_sample(
        results, centroids, qa_stem, output_dir=run_dir
    )
    manifest = build_run_manifest(
        args=args,
        results=results,
        service_audit=service_audit,
        origins=origins,
        input_paths=[
            BLOCKS_PATH,
            CENSUS_BLOCK_ATTRIBUTES_PATH,
            NOAA_GPKG_PATH,
            PRIVATE_SCHOOLS_PATH,
            PUBLIC_SCHOOLS_PATH,
            FIRE_STATIONS_PATH,
            ROAD_PBF_PATH,
        ],
    )
    write_run_manifest(manifest, run_dir / "run_manifest.json")
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
