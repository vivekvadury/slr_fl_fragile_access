#!/usr/bin/env python
"""
OSRM-based isolation workflow for the hybrid access pipeline.

This script treats OSRM as the authoritative engine for service access
questions, while the custom graph is reserved for fragility / redundancy.

What it produces
- block-level long output with one row per block x SLR level
- service-type access flags:
  - has_school_access
  - has_fire_station_access
  - has_required_service_access
  - has_any_essential_access
- isolation flags with more descriptive names than `isolated_tom_like`
- per-scenario OSRM `update.csv` closure files derived from SLR-intersecting
  edges in the retained network

Important execution note
- The script can generate all scenario closure files by itself.
- To query positive SLR scenarios, it also needs an OSRM server whose network is
  refreshed between scenarios.
- If `--refresh-command` is not supplied, the script can only query the
  scenario already loaded by the current OSRM server.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from shapely.geometry import Point, box

import hybrid_access_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OSRM-based block access isolation for the hybrid workflow.")
    parser.add_argument("--osrm-url", default="http://localhost:5000", help="Base URL for the running OSRM server.")
    parser.add_argument("--transport-mode", default="driving", help="OSRM profile name, e.g. driving.")
    parser.add_argument(
        "--slr-ft",
        type=int,
        nargs="*",
        default=None,
        help="Positive SLR levels to run. The dry baseline (0 ft) is always included.",
    )
    parser.add_argument("--max-blocks", type=int, default=None, help="Optional smoke-test cap on number of blocks.")
    parser.add_argument("--output-suffix", default="", help="Optional suffix appended to output filenames.")
    parser.add_argument(
        "--server-scenario-ft",
        type=int,
        default=0,
        help="Scenario currently loaded in the OSRM server if no refresh command is supplied.",
    )
    parser.add_argument(
        "--refresh-command",
        default="",
        help=(
            "Optional command template to refresh the OSRM server for each scenario. "
            "Use {update_csv} and {scenario_ft} placeholders."
        ),
    )
    parser.add_argument(
        "--write-updates-only",
        action="store_true",
        help="Only write scenario update files and manifests; do not query OSRM.",
    )
    parser.add_argument("--request-timeout-seconds", type=int, default=120, help="HTTP timeout for OSRM requests.")
    return parser.parse_args()


def requests_retry_session(retries: int = 8, backoff_factor: float = 0.3) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def write_update_csv(flooded_edges: gpd.GeoDataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    update_path = output_dir / "update.csv"

    if flooded_edges.empty:
        pd.DataFrame(columns=["from_osmid", "to_osmid", "edge_speed"]).to_csv(update_path, index=False, header=False)
        return update_path

    update = flooded_edges.loc[:, ["u", "v"]].copy()
    update = update.rename(columns={"u": "from_osmid", "v": "to_osmid"})
    reverse = update.rename(columns={"from_osmid": "to_osmid", "to_osmid": "from_osmid"})
    update = pd.concat([update, reverse], ignore_index=True).drop_duplicates()
    update["edge_speed"] = 0
    update.to_csv(update_path, index=False, header=False)
    return update_path


def maybe_refresh_osrm(refresh_command: str, update_csv: Path, scenario_ft: int) -> None:
    if not refresh_command:
        return
    command = refresh_command.format(update_csv=str(update_csv), scenario_ft=scenario_ft)
    common.log(f"Refreshing OSRM for scenario {scenario_ft} ft using: {command}")
    subprocess.run(command, shell=True, check=True)


def osrm_table_query(
    session: requests.Session,
    *,
    osrm_url: str,
    transport_mode: str,
    origin_coords: list[tuple[float, float]],
    dest_coords: list[tuple[float, float]],
    timeout_seconds: int,
) -> list[list[float | None]]:
    all_coords = origin_coords + dest_coords
    coord_string = ";".join(f"{lon:.8f},{lat:.8f}" for lon, lat in all_coords)
    source_idx = ";".join(str(i) for i in range(len(origin_coords)))
    dest_offset = len(origin_coords)
    dest_idx = ";".join(str(dest_offset + i) for i in range(len(dest_coords)))
    url = (
        f"{osrm_url.rstrip('/')}/table/v1/{transport_mode}/{coord_string}"
        f"?annotations=distance&sources={source_idx}&destinations={dest_idx}"
    )
    response = session.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != "Ok":
        raise RuntimeError(f"OSRM table query failed: {payload}")
    return payload["distances"]


def build_candidate_service_map(
    origins: gpd.GeoDataFrame,
    services: gpd.GeoDataFrame,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    if origins.empty:
        return {}, {}
    if services.empty:
        empty = {str(block_geoid): [] for block_geoid in origins["block_geoid"]}
        return empty, {key: 0 for key in empty}

    service_coords = np.column_stack([services.geometry.x.to_numpy(), services.geometry.y.to_numpy()])
    origin_coords = np.column_stack([origins.geometry.x.to_numpy(), origins.geometry.y.to_numpy()])
    tree = common.BASE.cKDTree(service_coords)
    k = min(common.OSRM_CANDIDATE_K, len(services))
    distances, indices = tree.query(origin_coords, k=k)
    if k == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    service_ids = services["service_id"].astype(str).to_numpy()
    candidate_map: dict[str, list[str]] = {}
    candidate_counts: dict[str, int] = {}

    for row_idx, block_geoid in enumerate(origins["block_geoid"].astype(str).to_numpy()):
        within_radius = [
            service_ids[int(candidate_idx)]
            for distance, candidate_idx in zip(distances[row_idx], indices[row_idx], strict=False)
            if np.isfinite(distance) and distance <= common.OSRM_CANDIDATE_RADIUS_M
        ]
        nearest_fallback = [
            service_ids[int(candidate_idx)]
            for candidate_idx in indices[row_idx]
            if int(candidate_idx) >= 0
        ]
        ordered_unique = list(dict.fromkeys(within_radius + nearest_fallback))
        candidate_map[block_geoid] = ordered_unique
        candidate_counts[block_geoid] = len(ordered_unique)

    return candidate_map, candidate_counts


def build_batches(
    origin_candidates: pd.DataFrame,
    *,
    max_origins_per_batch: int,
    max_coords_per_request: int,
) -> list[tuple[pd.DataFrame, list[str]]]:
    batches: list[tuple[pd.DataFrame, list[str]]] = []
    start = 0

    while start < len(origin_candidates):
        batch_size = min(max_origins_per_batch, len(origin_candidates) - start)
        while True:
            batch = origin_candidates.iloc[start : start + batch_size].copy()
            dest_union = list(dict.fromkeys(service_id for ids in batch["candidate_service_ids"] for service_id in ids))
            total_coords = batch_size + len(dest_union)
            if total_coords <= max_coords_per_request or batch_size == 1:
                batches.append((batch, dest_union))
                start += batch_size
                break
            batch_size = max(1, math.floor(batch_size / 2))

    return batches


def nearest_distance_by_service_type(
    session: requests.Session,
    *,
    osrm_url: str,
    transport_mode: str,
    origins: gpd.GeoDataFrame,
    services: gpd.GeoDataFrame,
    timeout_seconds: int,
) -> pd.DataFrame:
    if origins.empty:
        return pd.DataFrame(columns=["block_geoid", "service_type", "has_access", "nearest_distance_m", "nearest_service_id", "candidate_service_count"])

    if services.empty:
        empty_records = []
        for row in origins.itertuples(index=False):
            empty_records.append(
                {
                    "block_geoid": row.block_geoid,
                    "service_type": row.service_type,
                    "has_access": 0,
                    "nearest_distance_m": np.nan,
                    "nearest_service_id": pd.NA,
                    "candidate_service_count": 0,
                }
            )
        return pd.DataFrame.from_records(empty_records)

    candidate_map, candidate_counts = build_candidate_service_map(origins, services)
    candidate_frame = origins.copy()
    candidate_frame["candidate_service_ids"] = candidate_frame["block_geoid"].astype(str).map(candidate_map)
    candidate_frame["candidate_service_count"] = candidate_frame["block_geoid"].astype(str).map(candidate_counts).fillna(0).astype(int)

    service_lookup = services.set_index("service_id")[["service_lon", "service_lat"]]
    results: list[dict[str, object]] = []

    for batch, dest_union in build_batches(
        candidate_frame,
        max_origins_per_batch=common.OSRM_MAX_ORIGINS_PER_BATCH,
        max_coords_per_request=common.OSRM_MAX_COORDS_PER_REQUEST,
    ):
        if not dest_union:
            for row in batch.itertuples(index=False):
                results.append(
                    {
                        "block_geoid": row.block_geoid,
                        "service_type": row.service_type,
                        "has_access": 0,
                        "nearest_distance_m": np.nan,
                        "nearest_service_id": pd.NA,
                        "candidate_service_count": int(row.candidate_service_count),
                    }
                )
            continue

        dest_frame = service_lookup.loc[dest_union].reset_index()
        origin_coords = list(zip(batch["origin_lon"].astype(float), batch["origin_lat"].astype(float), strict=False))
        dest_coords = list(zip(dest_frame["service_lon"].astype(float), dest_frame["service_lat"].astype(float), strict=False))
        distance_matrix = osrm_table_query(
            session,
            osrm_url=osrm_url,
            transport_mode=transport_mode,
            origin_coords=[(lon, lat) for lon, lat in origin_coords],
            dest_coords=[(lon, lat) for lon, lat in dest_coords],
            timeout_seconds=timeout_seconds,
        )
        dest_index_lookup = {service_id: idx for idx, service_id in enumerate(dest_frame["service_id"].astype(str))}

        for row_idx, row in enumerate(batch.itertuples(index=False)):
            best_distance = np.nan
            best_service_id = pd.NA
            for service_id in row.candidate_service_ids:
                dest_idx = dest_index_lookup.get(str(service_id))
                if dest_idx is None:
                    continue
                distance = distance_matrix[row_idx][dest_idx]
                if distance is None:
                    continue
                if pd.isna(best_distance) or float(distance) < float(best_distance):
                    best_distance = float(distance)
                    best_service_id = str(service_id)
            results.append(
                {
                    "block_geoid": row.block_geoid,
                    "service_type": row.service_type,
                    "has_access": int(pd.notna(best_service_id)),
                    "nearest_distance_m": best_distance,
                    "nearest_service_id": best_service_id,
                    "candidate_service_count": int(row.candidate_service_count),
                }
            )

    return pd.DataFrame.from_records(results)


def scenario_manifest_records(
    osrm_edges: gpd.GeoDataFrame,
    origins: gpd.GeoDataFrame,
    origin_access_points: gpd.GeoDataFrame,
    services: gpd.GeoDataFrame,
    scenarios: list[tuple[int, str | None]],
) -> tuple[dict[int, dict[str, object]], pd.DataFrame]:
    clip_polygon = box(*osrm_edges.total_bounds)
    edges_sindex = osrm_edges.sindex
    per_scenario: dict[int, dict[str, object]] = {}
    manifest_rows: list[dict[str, object]] = []

    for slr_ft, layer_name in scenarios:
        if slr_ft == 0:
            flooded_edges = osrm_edges.iloc[0:0].copy()
            origin_point_mask = pd.Series(False, index=origins.index)
            access_point_mask = pd.Series(False, index=origin_access_points.index)
            flooded_service_mask = pd.Series(False, index=services.index)
        else:
            slr_projected = common.load_projected_slr_layer(slr_ft, clip_polygon)
            slr_source = common.BASE.load_slr_layer(layer_name, clip_polygon)
            if slr_source is None or slr_source.empty:
                flooded_edges = osrm_edges.iloc[0:0].copy()
            else:
                query_matches = edges_sindex.query(slr_source.geometry, predicate="intersects")
                if isinstance(query_matches, tuple):
                    flooded_edge_indices = np.unique(np.asarray(query_matches[1], dtype=int))
                elif hasattr(query_matches, "shape") and len(query_matches.shape) == 2:
                    flooded_edge_indices = np.unique(np.asarray(query_matches[1], dtype=int))
                else:
                    flooded_edge_indices = np.unique(np.asarray(query_matches, dtype=int))
                flooded_edges = osrm_edges.iloc[flooded_edge_indices].copy()
            origin_point_mask = common.geometries_intersecting_slr(origins[["geometry"]], slr_projected)
            access_point_mask = common.geometries_intersecting_slr(origin_access_points[["geometry"]], slr_projected)
            flooded_service_mask = common.geometries_intersecting_slr(services[["geometry"]], slr_projected)

        update_dir = common.OSRM_UPDATE_DIR / f"slr_{slr_ft}ft"
        update_csv = write_update_csv(flooded_edges, update_dir)
        per_scenario[slr_ft] = {
            "flooded_edges": flooded_edges,
            "origin_point_mask": origin_point_mask.to_numpy(),
            "access_point_mask": access_point_mask.to_numpy(),
            "flooded_service_mask": flooded_service_mask.to_numpy(),
            "update_csv": update_csv,
            "slr_layer_name": "dry_baseline" if slr_ft == 0 else str(layer_name),
        }
        manifest_rows.append(
            {
                "slr_ft": slr_ft,
                "slr_layer_name": "dry_baseline" if slr_ft == 0 else str(layer_name),
                "update_csv": str(update_csv),
                "n_flooded_edges": int(len(flooded_edges)),
                "n_flooded_origin_points": int(origin_point_mask.sum()),
                "n_flooded_origin_access_points": int(access_point_mask.sum()),
                "n_flooded_services": int(flooded_service_mask.sum()),
            }
        )

    return per_scenario, pd.DataFrame.from_records(manifest_rows)


def main() -> int:
    args = parse_args()
    start = time.time()
    output_suffix = common.build_output_suffix(args.max_blocks, args.output_suffix)
    output_stem = common.OSRM_DIR / f"{common.DEFAULT_OSRM_OUTPUT_STEM}{output_suffix}"

    common.require_inputs(
        [
            common.PREP_ORIGINS_GPKG,
            common.PREP_SERVICES_GPKG,
            common.PREP_SERVICE_COUNTY_CSV.with_suffix(".csv"),
            common.PREP_COUNTY_BOUNDS_GPKG,
            common.PREP_ORIGIN_ACCESS_CSV.with_suffix(".csv"),
            common.PREP_OSRM_EDGES_GPKG,
        ]
    )

    common.ensure_dir(common.OSRM_DIR)
    common.ensure_dir(common.OSRM_UPDATE_DIR)

    origins = common.load_prepared_origins()
    if args.max_blocks is not None:
        origins = origins.sort_values("block_geoid").head(args.max_blocks).copy()
        common.warn(f"Running OSRM isolation smoke subset with max_blocks={args.max_blocks}")
    services = common.load_prepared_services()
    service_membership = common.load_prepared_service_county_membership()
    origin_access = common.load_prepared_origin_access_points()
    osrm_edges = common.load_prepared_osrm_edges().set_crs("OGC:CRS84", allow_override=True)

    origin_access_gdf = pd.DataFrame(origin_access).copy()
    origin_access_gdf = gpd.GeoDataFrame(
        origin_access_gdf,
        geometry=gpd.points_from_xy(origin_access_gdf["access_point_x_m"], origin_access_gdf["access_point_y_m"]),
        crs=common.PROJECTED_CRS,
    )
    origin_access_gdf = origin_access_gdf.loc[origin_access_gdf["block_geoid"].isin(origins["block_geoid"])].copy()
    origins = origins.merge(origin_access.drop(columns=["county_fips", "county_name"]), on="block_geoid", how="left")

    scenarios = common.scenario_records(args.slr_ft)
    scenario_meta, manifest = scenario_manifest_records(osrm_edges, origins, origin_access_gdf, services, scenarios)
    common.write_table(manifest, common.OSRM_DIR / f"osrm_scenario_manifest{output_suffix}")

    if args.write_updates_only:
        common.log("Scenario update files and manifest written. Skipping OSRM queries by request.")
        common.log(f"Finished in {time.time() - start:.1f} seconds.")
        return 0

    requested_flooded = [slr_ft for slr_ft, _ in scenarios if slr_ft != args.server_scenario_ft]
    if requested_flooded and not args.refresh_command:
        raise ValueError(
            "Requested scenarios do not match the currently loaded OSRM server scenario. "
            "Either run one scenario at a time with --server-scenario-ft or provide --refresh-command."
        )

    session = requests_retry_session()
    results: list[pd.DataFrame] = []

    service_membership["service_id"] = service_membership["service_id"].astype(str)
    services["service_id"] = services["service_id"].astype(str)

    for slr_ft, _ in scenarios:
        scenario = scenario_meta[slr_ft]
        maybe_refresh_osrm(args.refresh_command, scenario["update_csv"], slr_ft)
        slr_layer_name = scenario["slr_layer_name"]
        origin_point_mask = pd.Series(scenario["origin_point_mask"], index=origins.index)
        access_point_mask = pd.Series(scenario["access_point_mask"], index=origins.index)
        flooded_service_mask = pd.Series(scenario["flooded_service_mask"], index=services.index)
        open_services = services.loc[~flooded_service_mask].copy()

        common.log(
            f"Querying OSRM for {slr_ft} ft ({slr_layer_name}) with "
            f"{len(origins):,} origins and {len(open_services):,} open services..."
        )

        scenario_rows: list[pd.DataFrame] = []
        for county_fips, county_origins in origins.groupby("county_fips", sort=True):
            county_name = county_origins["county_name"].iloc[0]
            county_membership = service_membership.loc[service_membership["county_fips"] == county_fips].copy()
            if county_membership.empty:
                common.warn(f"No county-buffered services found for county {county_name}.")
                county_required = county_origins.copy()
                county_required["has_school_access"] = 0
                county_required["has_fire_station_access"] = 0
                county_required["nearest_school_distance_m"] = np.nan
                county_required["nearest_fire_station_distance_m"] = np.nan
                county_required["nearest_school_service_id"] = pd.NA
                county_required["nearest_fire_station_service_id"] = pd.NA
                county_required["school_candidate_service_count"] = 0
                county_required["fire_station_candidate_service_count"] = 0
            else:
                county_service_ids = county_membership["service_id"].astype(str).unique()
                county_services = open_services.loc[open_services["service_id"].isin(county_service_ids)].copy()
                county_frames: list[pd.DataFrame] = []
                for service_type in common.OSRM_REQUIRED_SERVICE_TYPES:
                    county_services_type = county_services.loc[county_services["service_type"] == service_type].copy()
                    county_origin_type = county_origins[["block_geoid", "origin_lon", "origin_lat", "geometry"]].copy()
                    county_origin_type["service_type"] = service_type
                    nearest = nearest_distance_by_service_type(
                        session,
                        osrm_url=args.osrm_url,
                        transport_mode=args.transport_mode,
                        origins=county_origin_type,
                        services=county_services_type,
                        timeout_seconds=args.request_timeout_seconds,
                    )
                    county_frames.append(nearest)

                county_required = county_origins.copy()
                county_required = county_required.merge(
                    county_frames[0][["block_geoid", "has_access", "nearest_distance_m", "nearest_service_id", "candidate_service_count"]].rename(
                        columns={
                            "has_access": "has_school_access",
                            "nearest_distance_m": "nearest_school_distance_m",
                            "nearest_service_id": "nearest_school_service_id",
                            "candidate_service_count": "school_candidate_service_count",
                        }
                    ),
                    on="block_geoid",
                    how="left",
                )
                county_required = county_required.merge(
                    county_frames[1][["block_geoid", "has_access", "nearest_distance_m", "nearest_service_id", "candidate_service_count"]].rename(
                        columns={
                            "has_access": "has_fire_station_access",
                            "nearest_distance_m": "nearest_fire_station_distance_m",
                            "nearest_service_id": "nearest_fire_station_service_id",
                            "candidate_service_count": "fire_station_candidate_service_count",
                        }
                    ),
                    on="block_geoid",
                    how="left",
                )

            county_required["has_school_access"] = county_required["has_school_access"].fillna(0).astype(int)
            county_required["has_fire_station_access"] = county_required["has_fire_station_access"].fillna(0).astype(int)
            county_required["origin_point_inundated"] = origin_point_mask.loc[county_required.index].astype(int).to_numpy()
            county_required["origin_access_point_inundated"] = access_point_mask.loc[county_required.index].astype(int).to_numpy()
            blocked_mask = (
                county_required["origin_point_inundated"].astype(bool)
                | county_required["origin_access_point_inundated"].astype(bool)
            )
            county_required.loc[blocked_mask, ["has_school_access", "has_fire_station_access"]] = 0
            county_required["has_required_service_access"] = (
                county_required["has_school_access"].astype(bool)
                & county_required["has_fire_station_access"].astype(bool)
            ).astype(int)
            county_required["has_any_essential_access"] = (
                county_required["has_school_access"].astype(bool)
                | county_required["has_fire_station_access"].astype(bool)
            ).astype(int)
            county_required["isolated_missing_required_service_access"] = (
                ~county_required["has_required_service_access"].astype(bool)
            ).astype(int)
            county_required["isolated_missing_any_essential_access"] = (
                ~county_required["has_any_essential_access"].astype(bool)
            ).astype(int)
            county_required["slr_ft"] = slr_ft
            county_required["slr_layer_name"] = slr_layer_name

            scenario_rows.append(
                county_required[
                    [
                        "block_geoid",
                        "block_group_geoid",
                        "tract_geoid",
                        "block",
                        "county_fips",
                        "county_name",
                        "slr_ft",
                        "slr_layer_name",
                        "origin_lon",
                        "origin_lat",
                        "origin_to_road_distance_m",
                        "origin_point_inundated",
                        "origin_access_point_inundated",
                        "has_school_access",
                        "has_fire_station_access",
                        "has_required_service_access",
                        "has_any_essential_access",
                        "isolated_missing_required_service_access",
                        "isolated_missing_any_essential_access",
                        "nearest_school_distance_m",
                        "nearest_fire_station_distance_m",
                        "nearest_school_service_id",
                        "nearest_fire_station_service_id",
                        "school_candidate_service_count",
                        "fire_station_candidate_service_count",
                    ]
                ].copy()
            )

        scenario_output = pd.concat(scenario_rows, ignore_index=True)
        results.append(scenario_output)

        summary = (
            scenario_output.groupby("county_name", as_index=False)
            .agg(
                n_blocks=("block_geoid", "size"),
                n_origin_point_inundated=("origin_point_inundated", "sum"),
                n_origin_access_point_inundated=("origin_access_point_inundated", "sum"),
                n_isolated_missing_required_service_access=("isolated_missing_required_service_access", "sum"),
                n_isolated_missing_any_essential_access=("isolated_missing_any_essential_access", "sum"),
            )
        )
        print(f"\nOSRM summary for {slr_ft} ft")
        print(summary.to_string(index=False))

    final_output = pd.concat(results, ignore_index=True)
    common.write_table(final_output, output_stem)
    common.log(f"Saved OSRM isolation output to {output_stem}")
    common.log(f"Finished in {time.time() - start:.1f} seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
