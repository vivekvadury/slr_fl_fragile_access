#!/usr/bin/env python
"""
Prepare shared inputs for the hybrid access workflow.

This script creates the common origin, service, county-boundary, and OSRM-edge
inputs used by the split workflow:

- OSRM-based isolation
- graph-based fragility / redundancy

Design choices
- Origins use `representative_point()`, not geometric centroids.
- Services are filtered to buffered county footprints so nearby out-of-county
  facilities can still count.
- A lightweight road-adjacent access-point file is created for origins. This
  is meant for diagnostics and flood-exposure checks; OSRM still performs the
  final routing snap internally.
- An OSRM edge catalog with `u`/`v` node ids is extracted from the retained
  PBF via `pyrosm` so SLR-closure update files can be created later.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyrosm import OSM

import hybrid_access_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare shared inputs for the hybrid access workflow.")
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=None,
        help="Optional smoke-test cap on number of blocks.",
    )
    return parser.parse_args()


def nearest_road_access_points(
    origins: gpd.GeoDataFrame,
    roads_projected: gpd.GeoDataFrame,
    county_bounds: gpd.GeoDataFrame,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []

    for county_fips, county_origins in origins.groupby("county_fips", sort=True):
        county_name = county_origins["county_name"].iloc[0]
        county_boundary = county_bounds.loc[county_bounds["county_fips"] == county_fips]
        if county_boundary.empty:
            common.warn(f"No buffered county boundary found for county_fips={county_fips}; skipping.")
            continue

        polygon = county_boundary.geometry.iloc[0]
        county_roads = roads_projected.loc[roads_projected.geometry.intersects(polygon)].copy()
        if county_roads.empty:
            common.warn(f"No drivable roads intersect county buffer for {county_name}; skipping origin access points.")
            continue

        joined = gpd.sjoin_nearest(
            county_origins[["block_geoid", "county_fips", "county_name", "geometry"]],
            county_roads[["osm_id", "geometry"]],
            how="left",
            distance_col="origin_to_road_distance_m",
        )
        joined = (
            joined.sort_values(["block_geoid", "origin_to_road_distance_m", "osm_id"], na_position="last")
            .drop_duplicates("block_geoid", keep="first")
            .copy()
        )
        road_lookup = county_roads.geometry
        joined["road_geometry"] = joined["index_right"].map(road_lookup)

        def snap_to_road(row: pd.Series):
            road_geometry = row["road_geometry"]
            if road_geometry is None or pd.isna(road_geometry):
                return None
            return road_geometry.interpolate(road_geometry.project(row.geometry))

        access_points = gpd.GeoSeries(joined.apply(snap_to_road, axis=1), crs=roads_projected.crs)
        fallback_mask = access_points.isna()
        if fallback_mask.any():
            common.warn(
                f"{int(fallback_mask.sum()):,} origins in {county_name} could not be assigned a road access point; "
                "falling back to the representative point for diagnostics."
            )
            access_points.loc[fallback_mask] = joined.loc[fallback_mask, "geometry"]

        access_points_lonlat = access_points.to_crs("OGC:CRS84")

        county_output = pd.DataFrame(
            {
                "block_geoid": joined["block_geoid"].astype(str),
                "county_fips": joined["county_fips"].astype(str).str.zfill(3),
                "county_name": joined["county_name"].astype(str),
                "access_road_osm_id": joined["osm_id"].astype(str),
                "origin_to_road_distance_m": joined["origin_to_road_distance_m"].astype(float),
                "access_point_x_m": access_points.apply(lambda geom: geom.x if geom is not None else np.nan).astype(float),
                "access_point_y_m": access_points.apply(lambda geom: geom.y if geom is not None else np.nan).astype(float),
                "access_point_lon": access_points_lonlat.apply(lambda geom: geom.x if geom is not None else np.nan).astype(float),
                "access_point_lat": access_points_lonlat.apply(lambda geom: geom.y if geom is not None else np.nan).astype(float),
                "access_point_is_origin_fallback": fallback_mask.astype(int),
            }
        )
        records.append(county_output)
        common.log(
            f"Prepared origin access points for {county_name}: "
            f"{len(county_output):,} origins matched to {len(county_roads):,} retained roads"
        )

    if not records:
        raise RuntimeError("Origin access-point preparation did not produce any records.")
    return pd.concat(records, ignore_index=True)


def build_osrm_edge_catalog() -> gpd.GeoDataFrame:
    common.log("Extracting OSRM-ready driving edges from the retained PBF via pyrosm...")
    osm = OSM(str(common.BASE.ROAD_PBF_PATH))
    nodes, edges = osm.get_network(network_type="driving", nodes=True)
    if edges is None or edges.empty:
        raise RuntimeError("pyrosm did not return any driving edges from the retained road network.")

    keep_columns = [column for column in ["id", "u", "v", "highway", "oneway", "bridge", "tunnel", "length", "geometry"] if column in edges.columns]
    edges = edges[keep_columns].copy()
    edges = gpd.GeoDataFrame(edges, geometry="geometry", crs="OGC:CRS84")
    edges = edges.loc[edges.geometry.notnull() & ~edges.geometry.is_empty].copy()
    edges["u"] = edges["u"].astype("int64")
    edges["v"] = edges["v"].astype("int64")
    if "id" in edges.columns:
        edges["edge_osm_id"] = edges["id"].astype(str)
        edges = edges.drop(columns="id")
    else:
        edges["edge_osm_id"] = pd.Series(range(len(edges)), index=edges.index).astype(str)
    return edges


def main() -> int:
    args = parse_args()
    start = time.time()

    common.require_inputs(
        [
            common.BASE.BLOCKS_PATH,
            common.BASE.ROAD_PBF_PATH,
            common.BASE.NOAA_GPKG_PATH,
            common.BASE.PRIVATE_SCHOOLS_PATH,
            common.BASE.PUBLIC_SCHOOLS_PATH,
            common.BASE.FIRE_STATIONS_PATH,
        ]
    )

    common.ensure_dir(common.PREP_DIR)

    common.log("Loading blocks and shared services from the existing Adroit workflow...")
    blocks = common.BASE.read_vector(common.BASE.BLOCKS_PATH)
    blocks = common.BASE.prepare_blocks_layer(blocks)
    blocks = common.BASE.maybe_to_projected(blocks)
    if args.max_blocks is not None:
        blocks = blocks.sort_values("block_geoid").head(args.max_blocks).copy()
        common.warn(f"Running prep smoke subset with max_blocks={args.max_blocks}")

    origins = common.prepare_representative_point_origins(blocks)
    county_bounds = common.build_county_buffered_bounds(blocks)
    services = common.BASE.load_services()
    services_filtered, service_membership = common.filter_services_to_county_buffers(services, county_bounds)
    services_filtered = common.add_lon_lat_columns(
        services_filtered,
        lon_col="service_lon",
        lat_col="service_lat",
    )

    common.log(
        f"Prepared shared layers: {len(origins):,} origins, "
        f"{len(services_filtered):,} retained services, {len(service_membership):,} service-county memberships"
    )

    common.log("Building road-adjacent access-point diagnostics for origins...")
    roads = common.BASE.load_roads()
    roads = roads.set_crs("OGC:CRS84", allow_override=True)
    roads_projected = common.BASE.maybe_to_projected(roads)
    origin_access_points = nearest_road_access_points(origins, roads_projected, county_bounds)

    osrm_edges = build_osrm_edge_catalog()

    common.log("Writing prepared hybrid inputs...")
    origins.to_file(common.PREP_ORIGINS_GPKG, driver="GPKG")
    county_bounds.to_file(common.PREP_COUNTY_BOUNDS_GPKG, driver="GPKG")
    services_filtered.to_file(common.PREP_SERVICES_GPKG, driver="GPKG")
    osrm_edges.to_file(common.PREP_OSRM_EDGES_GPKG, driver="GPKG")
    common.write_table(origin_access_points, common.PREP_ORIGIN_ACCESS_CSV)
    common.write_table(service_membership, common.PREP_SERVICE_COUNTY_CSV)

    summary = pd.DataFrame(
        {
            "dataset": [
                "origins_rep_points",
                "services_filtered",
                "service_county_membership",
                "origin_access_points",
                "osrm_edge_catalog",
            ],
            "n_rows": [
                len(origins),
                len(services_filtered),
                len(service_membership),
                len(origin_access_points),
                len(osrm_edges),
            ],
            "path": [
                str(common.PREP_ORIGINS_GPKG),
                str(common.PREP_SERVICES_GPKG),
                str(common.PREP_SERVICE_COUNTY_CSV.with_suffix(".csv")),
                str(common.PREP_ORIGIN_ACCESS_CSV.with_suffix(".csv")),
                str(common.PREP_OSRM_EDGES_GPKG),
            ],
        }
    )
    common.write_table(summary, common.PREP_DIR / "hybrid_prep_summary")

    common.log(f"Prepared hybrid inputs in {time.time() - start:.1f} seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
