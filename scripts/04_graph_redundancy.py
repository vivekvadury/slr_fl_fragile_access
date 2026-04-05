#!/usr/bin/env python
"""
Graph-based fragility / redundancy workflow for the hybrid access pipeline.

This script intentionally keeps the custom graph only for the questions that
need graph structure:
- does the block retain any graph-reachable essential service?
- is that access fragile (1 path) or redundant (2+ paths)?

Isolation is left to the OSRM branch of the hybrid workflow.
"""

from __future__ import annotations

import argparse
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

import hybrid_access_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run graph-based fragility / redundancy for the hybrid workflow.")
    parser.add_argument(
        "--slr-ft",
        type=int,
        nargs="*",
        default=None,
        help="Positive SLR levels to run. The dry baseline (0 ft) is always included.",
    )
    parser.add_argument("--max-blocks", type=int, default=None, help="Optional smoke-test cap on number of blocks.")
    parser.add_argument("--output-suffix", default="", help="Optional suffix appended to output filenames.")
    return parser.parse_args()


def add_graph_baseline_fields(results: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        results.loc[results["slr_ft"] == 0, ["block_geoid", "graph_fragile", "graph_redundant", "graph_has_any_essential_access"]]
        .drop_duplicates("block_geoid")
        .rename(
            columns={
                "graph_fragile": "baseline_graph_fragile",
                "graph_redundant": "baseline_graph_redundant",
                "graph_has_any_essential_access": "baseline_graph_has_any_essential_access",
            }
        )
    )
    output = results.merge(baseline, on="block_geoid", how="left")
    output["persistent_graph_fragile"] = (
        (output["slr_ft"] != 0)
        & output["baseline_graph_fragile"].astype(bool)
        & output["graph_fragile"].astype(bool)
    ).astype(int)
    output["new_graph_fragile_due_to_slr"] = (
        (output["slr_ft"] != 0)
        & output["baseline_graph_redundant"].astype(bool)
        & output["graph_fragile"].astype(bool)
    ).astype(int)
    output["new_graph_unreachable_due_to_slr"] = (
        (output["slr_ft"] != 0)
        & output["baseline_graph_has_any_essential_access"].astype(bool)
        & ~output["graph_has_any_essential_access"].astype(bool)
    ).astype(int)
    return output


def main() -> int:
    args = parse_args()
    start = time.time()
    output_suffix = common.build_output_suffix(args.max_blocks, args.output_suffix)
    output_stem = common.GRAPH_DIR / f"{common.DEFAULT_GRAPH_OUTPUT_STEM}{output_suffix}"

    common.require_inputs(
        [
            common.PREP_ORIGINS_GPKG,
            common.PREP_SERVICES_GPKG,
            common.BASE.ROAD_PBF_PATH,
            common.BASE.NOAA_GPKG_PATH,
        ]
    )
    common.ensure_dir(common.GRAPH_DIR)

    origins = common.load_prepared_origins()
    if args.max_blocks is not None:
        origins = origins.sort_values("block_geoid").head(args.max_blocks).copy()
        common.warn(f"Running graph redundancy smoke subset with max_blocks={args.max_blocks}")
    services = common.load_prepared_services()

    common.log("Loading retained roads and building the graph from the working Adroit logic...")
    roads = common.BASE.load_roads()
    roads = roads.set_crs("OGC:CRS84", allow_override=True)
    boundary_polygon = common.BASE.build_study_area_boundary(tuple(roads.total_bounds), roads.crs)
    source_clip_polygon = box(*roads.total_bounds)
    origin_boundary = common.BASE.compute_origin_boundary_fields(origins, boundary_polygon)

    nodes, edges = common.BASE.segmentize_roads(roads)
    graph_baseline = common.BASE.build_graph(edges)
    tree, _, node_ids = common.BASE.build_node_kdtree(nodes)

    origin_snap = common.BASE.snap_points_to_nodes(
        origins[["block_geoid", "geometry"]].copy(),
        point_id_col="block_geoid",
        tree=tree,
        node_ids=node_ids,
        nodes=nodes,
        max_snap_m=common.BASE.MAX_ORIGIN_SNAP_M,
    )
    origins_for_graph = origins.merge(origin_snap, on="block_geoid", how="left")

    service_snap = common.BASE.snap_points_to_nodes(
        services[["service_id", "geometry"]].copy(),
        point_id_col="service_id",
        tree=tree,
        node_ids=node_ids,
        nodes=nodes,
        max_snap_m=common.BASE.MAX_SERVICE_SNAP_M,
    )
    services = services.merge(service_snap, on="service_id", how="left")
    services = services.loc[services["snap_valid"]].copy()
    if services.empty:
        raise RuntimeError("No services remained after snapping to the graph.")

    boundary_node_ids = common.BASE.build_boundary_node_set(nodes, boundary_polygon)
    baseline_nearest = common.BASE.build_nearest_service_lookup(graph_baseline, services)
    origin_points_source = origins[["block_geoid", "geometry"]].to_crs("OGC:CRS84")

    common.log(
        f"Graph baseline ready: {graph_baseline.number_of_nodes():,} nodes, "
        f"{graph_baseline.number_of_edges():,} edges, {len(services):,} snapped services"
    )

    outputs: list[pd.DataFrame] = []
    edges_sindex = edges.sindex
    for slr_ft, layer_name in common.scenario_records(args.slr_ft):
        if slr_ft == 0:
            slr_layer = None
            dry_edges = edges
            slr_layer_name = "dry_baseline"
        else:
            slr_layer = common.BASE.load_slr_layer(layer_name, source_clip_polygon)
            if slr_layer is None or slr_layer.empty:
                dry_edges = edges
            else:
                query_matches = edges_sindex.query(slr_layer.geometry, predicate="intersects")
                if isinstance(query_matches, tuple):
                    flooded_edge_indices = np.unique(np.asarray(query_matches[1], dtype=int))
                elif hasattr(query_matches, "shape") and len(query_matches.shape) == 2:
                    flooded_edge_indices = np.unique(np.asarray(query_matches[1], dtype=int))
                else:
                    flooded_edge_indices = np.unique(np.asarray(query_matches, dtype=int))
                dry_edges = edges.drop(index=edges.index[flooded_edge_indices]).copy()
            slr_layer_name = str(layer_name)

        dry_graph = common.BASE.build_graph(dry_edges)
        scenario = common.BASE.scenario_results_for_origins(
            slr_ft=slr_ft,
            slr_layer_name=slr_layer_name,
            slr_layer=slr_layer,
            graph=dry_graph,
            services=services,
            origins=origins_for_graph.drop(columns="geometry"),
            centroid_boundary=origin_boundary,
            centroid_geometry_source=origin_points_source,
            baseline_nearest=baseline_nearest,
            dry_boundary_node_ids=boundary_node_ids,
        )
        outputs.append(scenario)

    results = pd.concat(outputs, ignore_index=True).rename(
        columns={
            "origin_node_id": "graph_origin_node_id",
            "origin_snap_distance_m": "graph_origin_snap_distance_m",
            "origin_snap_exceeds_threshold": "graph_origin_snap_exceeds_threshold",
            "boundary_flag": "graph_boundary_flag",
            "boundary_distance_m": "graph_boundary_distance_m",
            "component_touches_boundary": "graph_component_touches_boundary",
            "block_centroid_inundated": "origin_rep_point_inundated",
            "block_centroid_isolated": "graph_unreachable_any_essential",
            "block_centroid_redundant": "graph_redundant",
            "block_centroid_fragile": "graph_fragile",
            "n_reachable_services": "graph_n_reachable_services",
            "n_reachable_service_nodes": "graph_n_reachable_service_nodes",
            "max_edge_disjoint_paths_any_service": "graph_max_edge_disjoint_paths_any_service",
            "nearest_reachable_service_type": "graph_nearest_reachable_service_type",
            "nearest_reachable_service_id": "graph_nearest_reachable_service_id",
            "baseline_shortest_path_distance_m": "graph_baseline_shortest_path_distance_m",
            "dry_shortest_path_distance_m": "graph_dry_shortest_path_distance_m",
            "detour_ratio": "graph_detour_ratio",
        }
    )
    results["graph_has_any_essential_access"] = (
        (~results["origin_rep_point_inundated"].astype(bool))
        & (~results["graph_unreachable_any_essential"].astype(bool))
        & (results["graph_n_reachable_services"] > 0)
    ).astype(int)
    results = add_graph_baseline_fields(results)

    common.write_table(results, output_stem)
    common.log(f"Saved graph redundancy output to {output_stem}")
    common.log(f"Finished in {time.time() - start:.1f} seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
