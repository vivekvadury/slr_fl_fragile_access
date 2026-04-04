#!/usr/bin/env python
"""
Diagnose graph connected-component structure for the current Adroit access workflow.

This script rebuilds the current road graph used by `01_bg_access_flags_adroit.py`
and quantifies connected-component structure for:
- the raw drivable graph
- the 0 ft dry graph after removing edges intersecting `FL_SE_slr_0_0ft`

It also optionally joins those graph diagnostics back to one completed run
directory so we can see whether baseline fragile / isolated blocks are mostly
inside or outside the largest connected component.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "01_bg_access_flags_adroit.py"

RESULT_DTYPE = {
    "block_geoid": "string",
    "block_group_geoid": "string",
    "tract_geoid": "string",
    "block": "string",
    "county_fips": "string",
    "county_name": "string",
    "slr_layer_name": "string",
    "nearest_reachable_service_type": "string",
    "nearest_reachable_service_id": "string",
    "baseline_status": "string",
    "scenario_status": "string",
}

COMPONENT_BINS = [
    (1, 1, "1"),
    (2, 5, "2-5"),
    (6, 10, "6-10"),
    (11, 50, "11-50"),
    (51, 100, "51-100"),
    (101, 500, "101-500"),
    (501, 1000, "501-1000"),
    (1001, None, "1001+"),
]


def log(message: str) -> None:
    print(f"[graph_component_diagnostics] {message}", flush=True)


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
        description="Diagnose connected-component structure for the current Adroit road graph and 0 ft dry graph."
    )
    parser.add_argument(
        "--run-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "access" / "edited" / "2026-04-03_run"),
        help="Optional completed run directory used to join baseline block statuses back to component membership.",
    )
    parser.add_argument(
        "--tiny-max-nodes",
        type=int,
        default=10,
        help="Upper bound used for the 'tiny component' count in summary outputs.",
    )
    return parser.parse_args()


def ensure_output_dir(run_dir: Path) -> Path:
    output_dir = run_dir / "diagnostics" / "graph_components"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    log(f"Saved {path}")


def load_results(run_dir: Path) -> pd.DataFrame:
    files = sorted(
        path
        for path in run_dir.glob("*.csv")
        if path.name.startswith("block_access_flags_long")
    )
    if not files:
        raise FileNotFoundError(f"No block_access_flags_long*.csv files found in {run_dir}")

    frames = [pd.read_csv(path, dtype=RESULT_DTYPE, low_memory=False) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    combined["slr_ft"] = combined["slr_ft"].astype(int)
    combined = combined.sort_values(["slr_ft", "block_geoid"]).copy()
    deduped = combined.drop_duplicates(subset=["block_geoid", "slr_ft"], keep="first").reset_index(drop=True)
    return deduped


def build_inputs():
    blocks = base.read_vector(base.BLOCKS_PATH)
    blocks = base.prepare_blocks_layer(blocks)
    blocks = base.maybe_to_projected(blocks)
    centroids = base.make_centroids(blocks)

    services = base.load_services()
    roads = base.load_roads()
    roads = roads.set_crs("OGC:CRS84", allow_override=True)

    boundary_polygon = base.build_study_area_boundary(tuple(roads.total_bounds), roads.crs)
    services = base.filter_services_by_buffer(services, boundary_polygon)

    nodes, edges = base.segmentize_roads(roads)
    graph_raw = base.build_graph(edges)
    tree, _, node_ids = base.build_node_kdtree(nodes)

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

    source_clip_polygon = box(*roads.total_bounds)
    slr0 = base.load_slr_layer(base.BASELINE_SLR_LAYER, source_clip_polygon)
    if slr0 is None or slr0.empty:
        dry_edges = edges.copy()
    else:
        edges_sindex = edges.sindex
        query_matches = edges_sindex.query(slr0.geometry, predicate="intersects")
        if isinstance(query_matches, tuple):
            flooded_edge_indices = np.unique(np.asarray(query_matches[1], dtype=int))
        elif hasattr(query_matches, "shape") and len(query_matches.shape) == 2:
            flooded_edge_indices = np.unique(np.asarray(query_matches[1], dtype=int))
        else:
            flooded_edge_indices = np.unique(np.asarray(query_matches, dtype=int))
        dry_edges = edges.drop(index=edges.index[flooded_edge_indices]).copy()

    graph_dry0 = base.build_graph(dry_edges)
    return {
        "blocks": blocks,
        "centroids": centroids,
        "origins": origins,
        "services": services,
        "roads": roads,
        "nodes": nodes,
        "edges": edges,
        "graph_raw": graph_raw,
        "graph_dry0": graph_dry0,
    }


def component_assignments(graph) -> tuple[pd.DataFrame, int]:
    records: list[dict[str, int]] = []
    largest_component_id = -1
    largest_component_size = -1

    for component_id, component_nodes in enumerate(base.nx.connected_components(graph)):
        component_nodes = list(component_nodes)
        component_size = len(component_nodes)
        if component_size > largest_component_size:
            largest_component_size = component_size
            largest_component_id = component_id
        for node_id in component_nodes:
            records.append(
                {
                    "node_id": int(node_id),
                    "component_id": component_id,
                    "component_size_nodes": component_size,
                }
            )

    assignment = pd.DataFrame.from_records(records)
    return assignment, largest_component_id


def summarize_graph(
    *,
    graph_name: str,
    graph,
    node_assignment: pd.DataFrame,
    largest_component_id: int,
    origins: pd.DataFrame,
    services: pd.DataFrame,
    tiny_max_nodes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    component_stats = (
        node_assignment.groupby(["component_id", "component_size_nodes"], as_index=False)
        .agg(n_nodes=("node_id", "size"))
        .sort_values(["component_size_nodes", "component_id"], ascending=[False, True])
        .reset_index(drop=True)
    )

    block_component = (
        origins.loc[origins["snap_valid"] & origins["node_id"].isin(graph.nodes), ["block_geoid", "node_id"]]
        .merge(node_assignment[["node_id", "component_id"]], on="node_id", how="left")
        .groupby("component_id", as_index=False)
        .agg(n_blocks=("block_geoid", "size"))
    )
    service_component = (
        services.loc[services["node_id"].isin(graph.nodes), ["service_id", "node_id"]]
        .merge(node_assignment[["node_id", "component_id"]], on="node_id", how="left")
        .groupby("component_id", as_index=False)
        .agg(n_services=("service_id", "size"))
    )

    component_stats = component_stats.merge(block_component, on="component_id", how="left")
    component_stats = component_stats.merge(service_component, on="component_id", how="left")
    component_stats[["n_blocks", "n_services"]] = component_stats[["n_blocks", "n_services"]].fillna(0).astype(int)
    component_stats["graph_name"] = graph_name
    component_stats["is_largest_component"] = component_stats["component_id"].eq(largest_component_id).astype(int)

    total_blocks = len(origins)
    valid_snapped_blocks = int(origins["snap_valid"].sum())
    total_services = len(services)
    valid_graph_services = int(services["node_id"].isin(graph.nodes).sum())

    largest_row = component_stats.loc[component_stats["component_id"] == largest_component_id].iloc[0]
    singleton_components = int((component_stats["component_size_nodes"] == 1).sum())
    tiny_components = int((component_stats["component_size_nodes"] <= tiny_max_nodes).sum())
    tiny_components_excl_singletons = int(
        ((component_stats["component_size_nodes"] >= 2) & (component_stats["component_size_nodes"] <= tiny_max_nodes)).sum()
    )

    summary = pd.DataFrame(
        {
            "graph_name": [graph_name],
            "n_nodes": [graph.number_of_nodes()],
            "n_edges": [graph.number_of_edges()],
            "n_components": [len(component_stats)],
            "largest_component_id": [largest_component_id],
            "largest_component_nodes": [int(largest_row["component_size_nodes"])],
            "largest_component_node_share": [float(largest_row["component_size_nodes"]) / float(graph.number_of_nodes())],
            "largest_component_blocks": [int(largest_row["n_blocks"])],
            "largest_component_block_share_of_all_blocks": [float(largest_row["n_blocks"]) / float(total_blocks)],
            "largest_component_block_share_of_valid_snapped_blocks": [
                float(largest_row["n_blocks"]) / float(valid_snapped_blocks) if valid_snapped_blocks else np.nan
            ],
            "largest_component_services": [int(largest_row["n_services"])],
            "largest_component_service_share_of_valid_graph_services": [
                float(largest_row["n_services"]) / float(valid_graph_services) if valid_graph_services else np.nan
            ],
            "singleton_components": [singleton_components],
            f"tiny_components_le_{tiny_max_nodes}": [tiny_components],
            f"tiny_components_2_to_{tiny_max_nodes}": [tiny_components_excl_singletons],
            "valid_snapped_blocks": [valid_snapped_blocks],
            "total_blocks": [total_blocks],
            "valid_graph_services": [valid_graph_services],
            "total_services": [total_services],
        }
    )
    return summary, component_stats


def summarize_component_bins(component_stats: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    graph_name = component_stats["graph_name"].iloc[0]
    for lower, upper, label in COMPONENT_BINS:
        if upper is None:
            mask = component_stats["component_size_nodes"] >= lower
        else:
            mask = component_stats["component_size_nodes"].between(lower, upper)
        subset = component_stats.loc[mask]
        records.append(
            {
                "graph_name": graph_name,
                "component_size_bin": label,
                "n_components": int(len(subset)),
                "n_nodes_in_bin": int(subset["component_size_nodes"].sum()),
                "n_blocks_in_bin": int(subset["n_blocks"].sum()),
                "n_services_in_bin": int(subset["n_services"].sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def plot_component_histogram(component_stats_list: list[pd.DataFrame], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(component_stats_list), figsize=(14, 5), constrained_layout=True)
    if len(component_stats_list) == 1:
        axes = [axes]

    for ax, component_stats in zip(axes, component_stats_list):
        counts = component_stats["component_size_nodes"]
        ax.hist(counts, bins=[1, 2, 6, 11, 51, 101, 501, 1001, counts.max() + 1], color="#4C78A8", edgecolor="white")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Component size (nodes)")
        ax.set_ylabel("Count of components")
        ax.set_title(component_stats["graph_name"].iloc[0])

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved {output_path}")


def plot_component_rank(component_stats_list: list[pd.DataFrame], output_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    colors = {"raw_graph": "#1b9e77", "dry0_graph": "#d95f02"}
    for component_stats in component_stats_list:
        graph_name = component_stats["graph_name"].iloc[0]
        sizes = component_stats["component_size_nodes"].sort_values(ascending=False).to_numpy()
        ranks = np.arange(1, len(sizes) + 1)
        ax.plot(ranks, sizes, label=graph_name, color=colors.get(graph_name, None))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Component rank")
    ax.set_ylabel("Component size (nodes)")
    ax.set_title("Connected-component size rank plot")
    ax.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved {output_path}")


def join_baseline_to_components(
    baseline_results: pd.DataFrame,
    origins: pd.DataFrame,
    raw_assignment: pd.DataFrame,
    raw_largest_component_id: int,
    dry0_assignment: pd.DataFrame,
    dry0_largest_component_id: int,
) -> pd.DataFrame:
    snapped = origins[
        ["block_geoid", "snap_valid", "node_id", "snap_distance_m", "county_name", "county_fips"]
    ].copy()
    output = baseline_results.merge(snapped, on="block_geoid", how="left", validate="one_to_one")

    raw_lookup = raw_assignment[["node_id", "component_id", "component_size_nodes"]].rename(
        columns={
            "component_id": "raw_component_id",
            "component_size_nodes": "raw_component_size_nodes",
        }
    )
    dry_lookup = dry0_assignment[["node_id", "component_id", "component_size_nodes"]].rename(
        columns={
            "component_id": "dry0_component_id",
            "component_size_nodes": "dry0_component_size_nodes",
        }
    )

    output = output.merge(raw_lookup, on="node_id", how="left")
    output = output.merge(dry_lookup, on="node_id", how="left")
    output["raw_in_largest_component"] = output["raw_component_id"].eq(raw_largest_component_id).astype(int)
    output["dry0_in_largest_component"] = output["dry0_component_id"].eq(dry0_largest_component_id).astype(int)
    return output


def summarize_baseline_membership(baseline_joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    county_col = "county_name"
    if county_col not in baseline_joined.columns:
        for candidate in ("county_name_x", "county_name_y"):
            if candidate in baseline_joined.columns:
                county_col = candidate
                break

    by_status = (
        baseline_joined.groupby(
            ["scenario_status", "snap_valid", "raw_in_largest_component", "dry0_in_largest_component"],
            as_index=False,
        )
        .agg(n_blocks=("block_geoid", "size"))
        .sort_values(["scenario_status", "snap_valid", "raw_in_largest_component", "dry0_in_largest_component"])
        .reset_index(drop=True)
    )

    isolated_fragile = (
        baseline_joined.loc[baseline_joined["scenario_status"].isin(["isolated", "fragile"])]
        .groupby(
            ["scenario_status", county_col, "raw_in_largest_component", "dry0_in_largest_component"],
            as_index=False,
        )
        .agg(n_blocks=("block_geoid", "size"))
        .rename(columns={county_col: "county_name"})
        .sort_values(["scenario_status", "county_name", "raw_in_largest_component", "dry0_in_largest_component"])
        .reset_index(drop=True)
    )
    return by_status, isolated_fragile


def print_key_results(
    graph_summary: pd.DataFrame,
    transition_like_summary: pd.DataFrame | None,
    baseline_membership_summary: pd.DataFrame | None,
) -> None:
    print("\nGraph summary")
    print(graph_summary.to_string(index=False))

    if transition_like_summary is not None:
        print("\nBaseline status by largest-component membership")
        print(transition_like_summary.to_string(index=False))

    if baseline_membership_summary is not None:
        print("\nIsolated/fragile blocks by county and largest-component membership")
        print(baseline_membership_summary.to_string(index=False))


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    output_dir = ensure_output_dir(run_dir)
    inputs = build_inputs()

    raw_assignment, raw_largest_component_id = component_assignments(inputs["graph_raw"])
    dry0_assignment, dry0_largest_component_id = component_assignments(inputs["graph_dry0"])

    raw_summary, raw_component_stats = summarize_graph(
        graph_name="raw_graph",
        graph=inputs["graph_raw"],
        node_assignment=raw_assignment,
        largest_component_id=raw_largest_component_id,
        origins=inputs["origins"],
        services=inputs["services"],
        tiny_max_nodes=args.tiny_max_nodes,
    )
    dry0_summary, dry0_component_stats = summarize_graph(
        graph_name="dry0_graph",
        graph=inputs["graph_dry0"],
        node_assignment=dry0_assignment,
        largest_component_id=dry0_largest_component_id,
        origins=inputs["origins"],
        services=inputs["services"],
        tiny_max_nodes=args.tiny_max_nodes,
    )

    graph_summary = pd.concat([raw_summary, dry0_summary], ignore_index=True)
    component_stats = pd.concat([raw_component_stats, dry0_component_stats], ignore_index=True)
    component_bins = pd.concat(
        [
            summarize_component_bins(raw_component_stats),
            summarize_component_bins(dry0_component_stats),
        ],
        ignore_index=True,
    )

    write_dataframe(graph_summary, output_dir / "graph_summary.csv")
    write_dataframe(component_stats, output_dir / "component_stats.csv")
    write_dataframe(component_bins, output_dir / "component_size_bins.csv")
    plot_component_histogram([raw_component_stats, dry0_component_stats], output_dir / "component_size_histogram.png")
    plot_component_rank([raw_component_stats, dry0_component_stats], output_dir / "component_size_rank_plot.png")

    results = load_results(run_dir)
    baseline_results = results.loc[results["slr_ft"] == 0].copy()
    baseline_joined = join_baseline_to_components(
        baseline_results,
        inputs["origins"],
        raw_assignment,
        raw_largest_component_id,
        dry0_assignment,
        dry0_largest_component_id,
    )
    baseline_membership_summary, isolated_fragile_summary = summarize_baseline_membership(baseline_joined)

    write_dataframe(baseline_joined, output_dir / "baseline_blocks_with_component_membership.csv")
    write_dataframe(baseline_membership_summary, output_dir / "baseline_status_by_component_membership.csv")
    write_dataframe(isolated_fragile_summary, output_dir / "baseline_isolated_fragile_by_component_membership.csv")

    print_key_results(graph_summary, baseline_membership_summary, isolated_fragile_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
