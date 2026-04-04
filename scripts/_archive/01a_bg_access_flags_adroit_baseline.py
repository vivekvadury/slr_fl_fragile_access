#!/usr/bin/env python
"""
Compare block-level access outcomes under each SLR scenario to a 0 ft baseline.

This script reuses the Adroit block-access workflow, but always computes a
baseline run using `FL_SE_slr_0_0ft` and then compares each positive SLR
scenario to that baseline.

Primary outputs
- One long-format comparison table with one row per block x positive-SLR
  scenario.
- Baseline flag columns prefixed with `baseline_`.
- Transition indicators such as:
  - `persistent_fragile`
  - `new_fragile_due_to_slr`
  - `new_isolated_due_to_slr`

Important interpretation note
- `0 ft` here means the 0 ft NOAA SLR layer in the same product family,
  not observed erosion or a full current-hazard model.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "01_bg_access_flags_adroit.py"

DEFAULT_OUTPUT_STEM = "block_access_flags_vs_baseline_long"

BASELINE_LAYER_MAP = {0: "FL_SE_slr_0_0ft"}


def load_base_module():
    spec = importlib.util.spec_from_file_location("bg_access_flags_adroit_base", BASE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base script from {BASE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare block access outcomes under SLR scenarios to the 0 ft baseline layer."
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
        help="Optional subset of positive SLR feet to compare against baseline, e.g. --slr-ft 1 3 6.",
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
        results.loc[results["slr_ft"] == 0, [
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

    scenario_results = results.loc[results["slr_ft"] != 0].copy()
    scenario_results = scenario_results.merge(
        baseline,
        on="block_geoid",
        how="left",
        validate="many_to_one",
    )

    scenario_results["baseline_status"] = classify_status_columns(
        scenario_results,
        inundated_col="baseline_block_centroid_inundated",
        isolated_col="baseline_block_centroid_isolated",
        redundant_col="baseline_block_centroid_redundant",
        fragile_col="baseline_block_centroid_fragile",
    )
    scenario_results["scenario_status"] = classify_status_columns(
        scenario_results,
        inundated_col="block_centroid_inundated",
        isolated_col="block_centroid_isolated",
        redundant_col="block_centroid_redundant",
        fragile_col="block_centroid_fragile",
    )

    scenario_results["persistent_fragile"] = (
        scenario_results["baseline_block_centroid_fragile"].eq(1)
        & scenario_results["block_centroid_fragile"].eq(1)
    ).astype(int)
    scenario_results["new_fragile_due_to_slr"] = (
        scenario_results["baseline_block_centroid_redundant"].eq(1)
        & scenario_results["block_centroid_fragile"].eq(1)
    ).astype(int)
    scenario_results["new_isolated_due_to_slr"] = (
        scenario_results["baseline_block_centroid_isolated"].eq(0)
        & scenario_results["block_centroid_isolated"].eq(1)
    ).astype(int)
    scenario_results["new_inundated_due_to_slr"] = (
        scenario_results["baseline_block_centroid_inundated"].eq(0)
        & scenario_results["block_centroid_inundated"].eq(1)
    ).astype(int)

    return scenario_results


def print_transition_summaries(results: pd.DataFrame) -> None:
    summary = (
        results.groupby("slr_ft", as_index=False)
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
        results.groupby(["slr_ft", "county_name"], as_index=False)
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

    compare_layers = base.SLR_LAYER_MAP.copy()
    if args.slr_ft:
        compare_layers = {
            slr_ft: base.SLR_LAYER_MAP[slr_ft]
            for slr_ft in args.slr_ft
            if slr_ft in base.SLR_LAYER_MAP
        }
        invalid_layers = sorted(set(args.slr_ft) - set(compare_layers))
        if invalid_layers:
            raise ValueError(
                f"Unsupported SLR level(s): {invalid_layers}. Supported positive values: {sorted(base.SLR_LAYER_MAP)}"
            )

    requested_layers = {**BASELINE_LAYER_MAP, **compare_layers}

    base.check_required_inputs(
        [
            base.BLOCKS_PATH,
            base.NOAA_GPKG_PATH,
            base.PRIVATE_SCHOOLS_PATH,
            base.PUBLIC_SCHOOLS_PATH,
            base.FIRE_STATIONS_PATH,
            base.ROAD_PBF_PATH,
            BASE_SCRIPT_PATH,
        ]
    )

    available_layers = base.list_layers(base.NOAA_GPKG_PATH)
    missing_layers = [layer_name for layer_name in requested_layers.values() if layer_name not in available_layers]
    if missing_layers:
        raise ValueError(f"Missing requested NOAA layer(s): {missing_layers}")

    start_time = time.time()
    base.log("Loading blocks, services, and roads for baseline comparison...")
    blocks = base.read_vector(base.BLOCKS_PATH)
    blocks = base.prepare_blocks_layer(blocks)
    blocks = base.maybe_to_projected(blocks)
    if args.max_blocks is not None:
        blocks = blocks.sort_values("block_geoid").head(args.max_blocks).copy()
        base.warn(f"Running smoke-test subset with max_blocks={args.max_blocks}")

    centroids = base.make_centroids(blocks)
    centroids_source = centroids.to_crs("OGC:CRS84")
    services = base.load_services()
    roads = base.load_roads()
    roads = roads.set_crs("OGC:CRS84", allow_override=True)

    boundary_polygon = base.build_study_area_boundary(tuple(roads.total_bounds), roads.crs)
    source_clip_polygon = box(*roads.total_bounds)
    centroid_boundary = base.compute_origin_boundary_fields(centroids, boundary_polygon)
    services = base.filter_services_by_buffer(services, boundary_polygon)
    base.log(f"Services inside the buffered retained-network boundary: {len(services):,}")

    base.log("Segmentizing roads and building baseline graph...")
    nodes, edges = base.segmentize_roads(roads)
    graph_baseline = base.build_graph(edges)
    tree, _, node_ids = base.build_node_kdtree(nodes)

    base.log(f"Baseline graph nodes: {graph_baseline.number_of_nodes():,}")
    base.log(f"Baseline graph edges: {graph_baseline.number_of_edges():,}")
    base.log(f"Segmentized road edges: {len(edges):,}")

    origins = centroids[
        ["block_geoid", "block_group_geoid", "tract_geoid", "block", "county_fips", "county_name", "geometry"]
    ].copy()
    origin_snap = base.snap_points_to_nodes(
        origins,
        point_id_col="block_geoid",
        tree=tree,
        node_ids=node_ids,
        nodes=nodes,
        max_snap_m=base.MAX_ORIGIN_SNAP_M,
    )
    origins = origins.merge(origin_snap, on="block_geoid", how="left")

    services_snap = base.snap_points_to_nodes(
        services[["service_id", "geometry"]].copy(),
        point_id_col="service_id",
        tree=tree,
        node_ids=node_ids,
        nodes=nodes,
        max_snap_m=base.MAX_SERVICE_SNAP_M,
    )
    services = services.merge(services_snap, on="service_id", how="left")
    services = services.loc[services["snap_valid"]].copy()

    if services.empty:
        raise RuntimeError("No services remained after buffered-footprint filtering and service snap checks.")

    if not origins["snap_valid"].all():
        base.warn(
            f"{int((~origins['snap_valid']).sum()):,} block centroids exceed "
            f"the origin snap threshold of {base.MAX_ORIGIN_SNAP_M:,} meters."
        )

    boundary_node_ids = base.build_boundary_node_set(nodes, boundary_polygon)
    (
        baseline_component_lookup,
        baseline_component_service_counts,
        _,
        _,
    ) = base.build_component_maps(graph_baseline, services, boundary_node_ids)
    baseline_nearest = base.build_nearest_service_lookup(graph_baseline, services)

    base.log(f"Baseline components: {len(set(baseline_component_lookup.values())):,}")
    base.log(
        "Baseline reachable-service summary across components: "
        f"{sum(value > 0 for value in baseline_component_service_counts.values()):,} components with >=1 service"
    )

    scenario_outputs: list[pd.DataFrame] = []
    edges_sindex = edges.sindex

    for slr_ft, slr_layer_name in requested_layers.items():
        base.log(f"Processing SLR {slr_ft} ft ({slr_layer_name}) for baseline comparison...")
        slr_layer = base.load_slr_layer(slr_layer_name, source_clip_polygon)
        if slr_layer is None:
            base.warn(f"Layer {slr_layer_name} had no inundation polygons within the buffered retained network.")
            dry_edges = edges
        else:
            base.log(f"SLR polygons retained for {slr_ft} ft: {len(slr_layer):,}")
            query_matches = edges_sindex.query(slr_layer.geometry, predicate="intersects")
            if isinstance(query_matches, tuple):
                flooded_edge_indices = np.unique(np.asarray(query_matches[1], dtype=int))
            elif hasattr(query_matches, "shape") and len(query_matches.shape) == 2:
                flooded_edge_indices = np.unique(np.asarray(query_matches[1], dtype=int))
            else:
                flooded_edge_indices = np.unique(np.asarray(query_matches, dtype=int))
            dry_edges = edges.drop(index=edges.index[flooded_edge_indices]).copy()
            base.log(f"Flooded segment count at {slr_ft} ft: {len(flooded_edge_indices):,}")

        dry_graph = base.build_graph(dry_edges)
        scenario_output = base.scenario_results_for_origins(
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

    raw_results = pd.concat(scenario_outputs, ignore_index=True)
    comparison_results = add_baseline_comparison_fields(raw_results)
    output_path = base.save_main_output(comparison_results, output_stem)

    print_transition_summaries(comparison_results)

    elapsed = time.time() - start_time
    base.log(f"Comparison output: {output_path}")
    base.log(f"Finished in {elapsed / 60:.1f} minutes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
