#!/usr/bin/env python
"""
Measurement-validity diagnostics for the South Florida access workflow.

Outputs are written to <run-dir>/diagnostics/validity/. The block-universe
diagnostics run and print before any road graph is built. Graph sensitivity
variants are disabled unless --sensitivity is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "02_access_flags.py"
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "access"
    / "edited"
    / "2026-04-03_run"
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

TRICOUNTY_FIPS = {"011", "086", "099"}
SMOKE_BBOX = (-80.35, 25.65, -80.10, 25.95)

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

DEPENDENCY_SPECS = {
    "geopandas": ("geopandas", "1.1.4"),
    "shapely": ("shapely", "2.1.2"),
    "pyogrio": ("pyogrio", "0.12.1"),
    "pandas": ("pandas", "3.0.3"),
    "numpy": ("numpy", "2.4.6"),
    "networkx": ("networkx", "3.6.1"),
    "scipy": ("scipy", "1.17.1"),
    "pyproj": ("pyproj", "3.7.2"),
}
SENSITIVITY_DEPENDENCY_SPECS = {
    "pyarrow": ("pyarrow", "25.0.0"),
}
SUMMARY_VERSION_PACKAGES = [
    "geopandas",
    "shapely",
    "pyogrio",
    "pandas",
    "numpy",
    "networkx",
    "scipy",
]


# Set by configure_dependencies() before any data work or base-module loading.
gpd = None
np = None
nx = None
pd = None
pyogrio = None
box = None
cKDTree = None
k_edge_components = None
base = None


def log(message: str) -> None:
    print(f"[measurement_validity] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure service-snap validity, bridge-tag removals, and optional baseline sensitivity."
    )
    parser.add_argument(
        "--run-dir",
        default=str(DEFAULT_RUN_DIR),
        help="Run directory containing block_access_flags_long*.csv files.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run graph items on a bbox-clipped road subset.",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run the optional 0 ft fragility-sensitivity variants.",
    )
    return parser.parse_args()


def configure_dependencies(*, sensitivity: bool) -> dict[str, str] | None:
    specs = dict(DEPENDENCY_SPECS)
    if sensitivity:
        specs.update(SENSITIVITY_DEPENDENCY_SPECS)

    modules: dict[str, ModuleType] = {}
    missing: list[tuple[str, str, str]] = []
    for module_name, (package_name, install_version) in specs.items():
        try:
            modules[module_name] = importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            missing.append((package_name, install_version, str(exc)))

    if missing:
        print("DEPENDENCY CHECK FAILED", file=sys.stderr, flush=True)
        for package_name, install_version, error in missing:
            print(
                f"{package_name}: missing_or_unavailable; install_version={install_version}; error={error}",
                file=sys.stderr,
                flush=True,
            )
        print("packages_installed=0", file=sys.stderr, flush=True)
        return None

    global gpd, np, nx, pd, pyogrio, box, cKDTree, k_edge_components
    gpd = modules["geopandas"]
    np = modules["numpy"]
    nx = modules["networkx"]
    pd = modules["pandas"]
    pyogrio = modules["pyogrio"]
    box = modules["shapely"].geometry.box
    cKDTree = modules["scipy"].spatial.cKDTree
    k_edge_components = modules["networkx"].algorithms.connectivity.k_edge_components

    return {
        package_name: importlib.metadata.version(package_name)
        for package_name in SUMMARY_VERSION_PACKAGES
    }


def load_base_module():
    spec = importlib.util.spec_from_file_location("access_flags_base", BASE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base script from {BASE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_result_files(run_dir: Path) -> list[Path]:
    files = sorted(
        path
        for path in run_dir.glob("*.csv")
        if path.name.startswith("block_access_flags_long")
    )
    if not files:
        raise FileNotFoundError(
            f"No block_access_flags_long*.csv files found in {run_dir}"
        )
    return files


def load_and_combine_results(files: list[Path]) -> tuple[object, object]:
    """Combine and dedupe exactly as 02b.load_and_combine_results."""
    frames = [
        pd.read_csv(path, dtype=RESULT_DTYPE, low_memory=False) for path in files
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined["slr_ft"] = combined["slr_ft"].astype(int)

    duplicate_mask = combined.duplicated(
        subset=["block_geoid", "slr_ft"], keep="first"
    )
    duplicate_rows = combined.loc[
        duplicate_mask, ["block_geoid", "slr_ft"]
    ].copy()
    deduped = combined.loc[~duplicate_mask].copy()
    deduped = deduped.sort_values(["slr_ft", "block_geoid"]).reset_index(
        drop=True
    )
    return deduped, duplicate_rows


def write_dataframe(frame, path: Path) -> None:
    frame.to_csv(path, index=False)
    log(f"saved={path}")


def make_crosstab(frame, flag_col: str):
    table = pd.crosstab(
        frame["scenario_status"],
        frame[flag_col],
        margins=True,
        margins_name="all",
        dropna=False,
    )
    table = table.rename(columns={False: "0", True: "1"})
    table.columns.name = None
    return table.reset_index()


def run_block_universe(results, output_dir: Path) -> dict[str, object]:
    log("item=block_universe; state=started")
    attributes = pyogrio.read_dataframe(
        CENSUS_BLOCK_ATTRIBUTES_PATH,
        columns=["GEOID20", "COUNTYFP20", "POP20", "ALAND20"],
        read_geometry=False,
    )
    attributes["COUNTYFP20"] = (
        attributes["COUNTYFP20"].astype("string").str.zfill(3)
    )
    attributes = attributes.loc[
        attributes["COUNTYFP20"].isin(TRICOUNTY_FIPS)
    ].copy()
    attributes["block_geoid"] = (
        attributes["GEOID20"].astype("string").str.zfill(15)
    )
    attributes["POP20"] = pd.to_numeric(attributes["POP20"], errors="raise")
    attributes["ALAND20"] = pd.to_numeric(
        attributes["ALAND20"], errors="raise"
    )
    attributes = attributes[
        ["block_geoid", "COUNTYFP20", "POP20", "ALAND20"]
    ].copy()
    if attributes["block_geoid"].duplicated().any():
        raise ValueError("Duplicate GEOID20 values in tri-county Census attributes.")

    baseline = results.loc[results["slr_ft"] == 0].copy()
    if baseline["block_geoid"].duplicated().any():
        raise ValueError("Duplicate baseline block_geoid values after run deduplication.")
    joined = baseline.merge(
        attributes,
        on="block_geoid",
        how="left",
        validate="one_to_one",
    )
    missing_attributes = joined[["POP20", "ALAND20"]].isna().any(axis=1)
    if missing_attributes.any():
        raise ValueError(
            f"Baseline rows missing Census attributes: {int(missing_attributes.sum())}"
        )

    joined["pop20_zero"] = joined["POP20"].eq(0)
    joined["aland20_zero"] = joined["ALAND20"].eq(0)
    pop_crosstab = make_crosstab(joined, "pop20_zero")
    aland_crosstab = make_crosstab(joined, "aland20_zero")

    populated = joined.loc[joined["POP20"] > 0]
    fragile_summary = pd.DataFrame.from_records(
        [
            {
                "universe": "all_baseline_blocks",
                "n_blocks": int(len(joined)),
                "n_fragile": int(joined["scenario_status"].eq("fragile").sum()),
                "fragile_share": float(
                    joined["scenario_status"].eq("fragile").mean()
                ),
            },
            {
                "universe": "pop20_gt_0",
                "n_blocks": int(len(populated)),
                "n_fragile": int(
                    populated["scenario_status"].eq("fragile").sum()
                ),
                "fragile_share": float(
                    populated["scenario_status"].eq("fragile").mean()
                ),
            },
        ]
    )

    write_dataframe(pop_crosstab, output_dir / "baseline_status_by_pop20_zero.csv")
    write_dataframe(
        aland_crosstab, output_dir / "baseline_status_by_aland20_zero.csv"
    )
    write_dataframe(
        fragile_summary, output_dir / "baseline_fragile_shares.csv"
    )

    print("\nBLOCK UNIVERSE: STATUS x POP20_ZERO", flush=True)
    print(pop_crosstab.to_string(index=False), flush=True)
    print("\nBLOCK UNIVERSE: STATUS x ALAND20_ZERO", flush=True)
    print(aland_crosstab.to_string(index=False), flush=True)
    print("\nBLOCK UNIVERSE: FRAGILE SHARES", flush=True)
    print(fragile_summary.to_string(index=False), flush=True)
    return {
        "joined": joined,
        "fragile_summary": fragile_summary,
    }


def cache_paths(
    output_dir: Path,
    highway_filter: set[str],
    *,
    smoke: bool,
) -> tuple[Path, Path]:
    filter_key = "|".join(sorted(highway_filter))
    digest = hashlib.sha256(filter_key.encode("utf-8")).hexdigest()[:16]
    mode = "smoke" if smoke else "full"
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"segmentized_{digest}_{mode}"
    return cache_dir / f"{stem}_nodes.parquet", cache_dir / f"{stem}_edges.parquet"


def segmentize_with_optional_cache(
    roads,
    *,
    highway_filter: set[str],
    output_dir: Path,
    smoke: bool,
    cache_enabled: bool,
):
    if not cache_enabled:
        return base.segmentize_roads(roads)

    nodes_path, edges_path = cache_paths(
        output_dir, highway_filter, smoke=smoke
    )
    if nodes_path.exists() and edges_path.exists():
        log(f"segment_cache=hit; highway_count={len(highway_filter)}")
        return gpd.read_parquet(nodes_path), gpd.read_parquet(edges_path)

    log(f"segment_cache=miss; highway_count={len(highway_filter)}")
    nodes, edges = base.segmentize_roads(roads)
    nodes.to_parquet(nodes_path, index=False)
    edges.to_parquet(edges_path, index=False)
    log(f"saved={nodes_path}")
    log(f"saved={edges_path}")
    return nodes, edges


def load_graph_roads(*, smoke: bool):
    roads = base.load_roads()
    roads = roads.set_crs("OGC:CRS84", allow_override=True)
    if smoke:
        smoke_polygon = box(*SMOKE_BBOX)
        roads = gpd.clip(roads, smoke_polygon, keep_geom_type=True)
        roads = roads.loc[
            roads.geometry.notna() & ~roads.geometry.is_empty
        ].copy()
        if roads.empty:
            raise RuntimeError("Smoke bbox retained no road geometries.")
    return roads


def query_removed_edge_positions(edges, slr_layer) -> object:
    if slr_layer is None or slr_layer.empty:
        return np.asarray([], dtype=int)
    query_matches = edges.sindex.query(
        slr_layer.geometry, predicate="intersects"
    )
    if isinstance(query_matches, tuple):
        flooded_edge_indices = np.unique(
            np.asarray(query_matches[1], dtype=int)
        )
    elif hasattr(query_matches, "shape") and len(query_matches.shape) == 2:
        flooded_edge_indices = np.unique(
            np.asarray(query_matches[1], dtype=int)
        )
    else:
        flooded_edge_indices = np.unique(
            np.asarray(query_matches, dtype=int)
        )
    return flooded_edge_indices


def load_zero_and_one_ft_layers(roads):
    source_clip_polygon = box(*roads.total_bounds)
    slr0 = base.load_slr_layer(base.BASELINE_SLR_LAYER, source_clip_polygon)
    slr1 = base.load_slr_layer(base.SLR_LAYER_MAP[1], source_clip_polygon)
    return slr0, slr1


def prepare_services(services_source, roads, nodes):
    boundary_polygon = base.build_study_area_boundary(
        tuple(roads.total_bounds), roads.crs
    )
    services = base.filter_services_by_buffer(
        services_source, boundary_polygon
    )
    tree, _, node_ids = base.build_node_kdtree(nodes)
    service_snap = base.snap_points_to_nodes(
        services[["service_id", "geometry"]].copy(),
        point_id_col="service_id",
        tree=tree,
        node_ids=node_ids,
        nodes=nodes,
        max_snap_m=base.MAX_SERVICE_SNAP_M,
    )
    services = services.merge(service_snap, on="service_id", how="left")
    services = services.loc[services["snap_valid"]].reset_index(drop=True).copy()
    if services.empty:
        raise RuntimeError("No services remained after filtering and snap checks.")
    services.insert(0, "service_record_id", np.arange(len(services), dtype=int))
    return services, boundary_polygon


def collapse_two_edge_components(graph) -> tuple[dict[int, int], dict[int, int]]:
    node_to_component: dict[int, int] = {}
    node_to_component_size: dict[int, int] = {}
    for component_id, component_nodes in enumerate(
        k_edge_components(graph, 2)
    ):
        component_size = len(component_nodes)
        for node_id in component_nodes:
            node_to_component[int(node_id)] = int(component_id)
            node_to_component_size[int(node_id)] = int(component_size)
    return node_to_component, node_to_component_size


def service_component_outputs(
    services,
    node_to_component: dict[int, int],
    node_to_component_size: dict[int, int],
    output_dir: Path,
) -> tuple[object, object, object]:
    assignments = services[
        [
            "service_record_id",
            "service_id",
            "service_type",
            "service_source",
            "service_name",
            "node_id",
            "snap_distance_m",
        ]
    ].copy()
    assignments["two_edge_component_id"] = assignments["node_id"].map(
        node_to_component
    )
    assignments["two_edge_component_size"] = (
        assignments["node_id"]
        .map(node_to_component_size)
        .fillna(0)
        .astype(int)
    )
    assignments["singleton_2ecc"] = assignments[
        "two_edge_component_size"
    ].eq(1).astype(int)

    distribution_frames = []
    for service_type, group in [("all", assignments), *assignments.groupby("service_type")]:
        distribution = (
            group.groupby("two_edge_component_size", as_index=False)
            .agg(n_services=("service_record_id", "size"))
            .sort_values("two_edge_component_size")
        )
        distribution.insert(0, "service_type", service_type)
        distribution["share_within_service_type"] = (
            distribution["n_services"] / len(group)
        )
        distribution_frames.append(distribution)
    distribution = pd.concat(distribution_frames, ignore_index=True)

    singleton_records = []
    for service_type, group in [("all", assignments), *assignments.groupby("service_type")]:
        n_singleton = int(group["singleton_2ecc"].sum())
        singleton_records.append(
            {
                "service_type": service_type,
                "n_services": int(len(group)),
                "n_singleton_2ecc": n_singleton,
                "singleton_2ecc_share": float(n_singleton / len(group)),
            }
        )
    singleton_summary = pd.DataFrame.from_records(singleton_records)

    write_dataframe(
        assignments, output_dir / "service_2ecc_assignments.csv"
    )
    write_dataframe(
        distribution, output_dir / "service_2ecc_size_distribution.csv"
    )
    write_dataframe(
        singleton_summary, output_dir / "service_singleton_2ecc_summary.csv"
    )
    return assignments, distribution, singleton_summary


def resnap_services_to_eligible_nodes(
    services,
    nodes,
    node_to_component: dict[int, int],
    node_to_component_size: dict[int, int],
):
    eligible_node_ids = {
        int(node_id)
        for node_id, component_size in node_to_component_size.items()
        if component_size >= 2
    }
    eligible_nodes = nodes.loc[
        nodes["node_id"].isin(eligible_node_ids)
    ].copy()

    if eligible_nodes.empty:
        candidate = pd.DataFrame(
            {
                "service_record_id": services["service_record_id"].to_numpy(),
                "candidate_new_node_id": pd.array(
                    [pd.NA] * len(services), dtype="Int64"
                ),
                "new_snap_distance_m": np.full(len(services), np.nan),
                "new_snap_valid": np.zeros(len(services), dtype=bool),
            }
        )
    else:
        eligible_tree, _, eligible_node_id_array = base.build_node_kdtree(
            eligible_nodes
        )
        candidate = base.snap_points_to_nodes(
            services[["service_record_id", "geometry"]].copy(),
            point_id_col="service_record_id",
            tree=eligible_tree,
            node_ids=eligible_node_id_array,
            nodes=eligible_nodes,
            max_snap_m=base.MAX_SERVICE_SNAP_M,
        ).rename(
            columns={
                "node_id": "candidate_new_node_id",
                "snap_distance_m": "new_snap_distance_m",
                "snap_valid": "new_snap_valid",
            }
        )

    output = services[
        [
            "service_record_id",
            "service_id",
            "service_type",
            "service_source",
            "service_name",
            "node_id",
            "snap_distance_m",
        ]
    ].rename(
        columns={
            "node_id": "old_node_id",
            "snap_distance_m": "old_snap_distance_m",
        }
    )
    output = output.merge(
        candidate[
            [
                "service_record_id",
                "candidate_new_node_id",
                "new_snap_distance_m",
                "new_snap_valid",
            ]
        ],
        on="service_record_id",
        how="left",
        validate="one_to_one",
    )
    output["old_2ecc_component_id"] = output["old_node_id"].map(
        node_to_component
    )
    output["old_2ecc_size"] = (
        output["old_node_id"]
        .map(node_to_component_size)
        .fillna(0)
        .astype(int)
    )
    output["new_node_id"] = output["candidate_new_node_id"].astype("Int64")
    output.loc[~output["new_snap_valid"], "new_node_id"] = pd.NA
    output["new_2ecc_component_id"] = output["new_node_id"].map(
        node_to_component
    )
    output["new_2ecc_size"] = (
        output["new_node_id"]
        .map(node_to_component_size)
        .fillna(0)
        .astype(int)
    )
    output["recovery_needed"] = output["old_2ecc_size"].lt(2)
    output["singleton_recovery_needed"] = output["old_2ecc_size"].eq(1)
    output["recovered"] = (
        output["recovery_needed"] & output["new_snap_valid"]
    ).astype(int)
    output["singleton_recovered"] = (
        output["singleton_recovery_needed"] & output["new_snap_valid"]
    ).astype(int)
    output["moved"] = (
        output["new_snap_valid"]
        & output["new_node_id"].ne(output["old_node_id"]).fillna(False)
    ).astype(int)
    output["added_snap_distance_m"] = np.where(
        output["moved"].eq(1),
        output["new_snap_distance_m"] - output["old_snap_distance_m"],
        np.nan,
    )
    output = output.drop(columns="candidate_new_node_id")
    return output


def summarize_resnapping(recovery):
    moved_added_distance = recovery.loc[
        recovery["moved"].eq(1), "added_snap_distance_m"
    ]
    n_recovery_needed = int(recovery["recovery_needed"].sum())
    n_recovered = int(recovery["recovered"].sum())
    n_singleton = int(recovery["singleton_recovery_needed"].sum())
    n_singleton_recovered = int(recovery["singleton_recovered"].sum())
    return pd.DataFrame.from_records(
        [
            {
                "n_services": int(len(recovery)),
                "n_services_moved": int(recovery["moved"].sum()),
                "n_recovery_needed_2ecc_lt_2": n_recovery_needed,
                "n_recovered_2ecc_lt_2": n_recovered,
                "recovered_share_of_2ecc_lt_2": (
                    float(n_recovered / n_recovery_needed)
                    if n_recovery_needed
                    else np.nan
                ),
                "n_unrecoverable_2ecc_lt_2": n_recovery_needed - n_recovered,
                "n_singleton_2ecc": n_singleton,
                "n_singleton_recovered": n_singleton_recovered,
                "singleton_recovered_share": (
                    float(n_singleton_recovered / n_singleton)
                    if n_singleton
                    else np.nan
                ),
                "n_singleton_unrecoverable": n_singleton
                - n_singleton_recovered,
                "added_snap_distance_median": (
                    float(moved_added_distance.median())
                    if len(moved_added_distance)
                    else np.nan
                ),
                "added_snap_distance_p90": (
                    float(moved_added_distance.quantile(0.90))
                    if len(moved_added_distance)
                    else np.nan
                ),
                "added_snap_distance_max": (
                    float(moved_added_distance.max())
                    if len(moved_added_distance)
                    else np.nan
                ),
            }
        ]
    )


def nonzero_layer(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    try:
        return float(text) != 0.0
    except ValueError:
        return True


def run_bridge_audit(
    roads,
    edges,
    removed_positions_by_slr: dict[int, object],
    output_dir: Path,
) -> dict[str, object]:
    removed_frames = []
    for slr_ft in [0, 1]:
        positions = removed_positions_by_slr[slr_ft]
        removed = edges.iloc[positions][
            ["edge_id", "u", "v", "osm_id", "highway", "length_m"]
        ].copy()
        removed.insert(0, "slr_ft", slr_ft)
        removed_frames.append(removed)
    removed_segments = pd.concat(removed_frames, ignore_index=True)

    road_tags = roads[["osm_id", "other_tags"]].copy()
    road_tags["osm_id"] = road_tags["osm_id"].astype(str)
    road_tags = road_tags.drop_duplicates("osm_id", keep="first")
    removed_segments["osm_id"] = removed_segments["osm_id"].astype(str)
    removed_segments = removed_segments.merge(
        road_tags,
        on="osm_id",
        how="left",
        validate="many_to_one",
    )
    removed_segments["bridge_tag"] = removed_segments["other_tags"].map(
        lambda value: base.parse_other_tag(value, "bridge")
    )
    removed_segments["layer_tag"] = removed_segments["other_tags"].map(
        lambda value: base.parse_other_tag(value, "layer")
    )
    removed_segments["bridge_tag_present"] = (
        removed_segments["bridge_tag"].notna().astype(int)
    )
    removed_segments["nonzero_layer"] = (
        removed_segments["layer_tag"].map(nonzero_layer).astype(int)
    )
    removed_segments["bridge_or_nonzero_layer"] = (
        removed_segments["bridge_tag_present"].eq(1)
        | removed_segments["nonzero_layer"].eq(1)
    ).astype(int)

    highway_counts = (
        removed_segments.groupby(
            [
                "slr_ft",
                "highway",
                "bridge_tag_present",
                "nonzero_layer",
                "bridge_or_nonzero_layer",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_removed_segments=("edge_id", "size"),
            n_parent_ways=("osm_id", "nunique"),
            removed_length_m=("length_m", "sum"),
        )
        .sort_values(
            [
                "slr_ft",
                "highway",
                "bridge_tag_present",
                "nonzero_layer",
            ]
        )
        .reset_index(drop=True)
    )
    audit_summary = (
        removed_segments.groupby("slr_ft", as_index=False)
        .agg(
            n_removed_segments=("edge_id", "size"),
            n_bridge_tagged_segments=("bridge_tag_present", "sum"),
            n_nonzero_layer_segments=("nonzero_layer", "sum"),
            n_bridge_or_nonzero_layer_segments=(
                "bridge_or_nonzero_layer",
                "sum",
            ),
        )
        .sort_values("slr_ft")
        .reset_index(drop=True)
    )

    largest_bridge_segments = (
        removed_segments.loc[removed_segments["bridge_tag_present"].eq(1)]
        .sort_values(["slr_ft", "edge_id"])
        .drop_duplicates("edge_id", keep="first")
        .sort_values(
            ["length_m", "osm_id", "edge_id"],
            ascending=[False, True, True],
        )
        .head(20)
        [
            [
                "osm_id",
                "edge_id",
                "slr_ft",
                "length_m",
                "highway",
                "bridge_tag",
                "layer_tag",
            ]
        ]
        .reset_index(drop=True)
    )

    write_dataframe(
        removed_segments.drop(columns="other_tags"),
        output_dir / "removed_segments_0ft_1ft.csv",
    )
    write_dataframe(
        highway_counts,
        output_dir / "removed_segments_by_highway_and_parent_tags.csv",
    )
    write_dataframe(
        audit_summary, output_dir / "removed_bridge_audit_summary.csv"
    )
    write_dataframe(
        largest_bridge_segments,
        output_dir / "largest_20_removed_bridge_segments.csv",
    )
    return {
        "summary": audit_summary,
        "highway_counts": highway_counts,
        "largest_bridge_segments": largest_bridge_segments,
    }


def services_from_recovery(services, recovery):
    replacements = recovery[
        [
            "service_record_id",
            "new_node_id",
            "new_snap_distance_m",
            "new_snap_valid",
        ]
    ].copy()
    output = services.drop(
        columns=["node_id", "snap_distance_m", "snap_valid", "x", "y"],
        errors="ignore",
    ).merge(
        replacements,
        on="service_record_id",
        how="inner",
        validate="one_to_one",
    )
    output = output.loc[output["new_snap_valid"]].copy()
    output = output.rename(
        columns={
            "new_node_id": "node_id",
            "new_snap_distance_m": "snap_distance_m",
            "new_snap_valid": "snap_valid",
        }
    )
    output["node_id"] = output["node_id"].astype(int)
    return output


def prepare_sensitivity_origins(selected_centroids, nodes, boundary_polygon):
    origins = selected_centroids[
        [
            "block_geoid",
            "block_group_geoid",
            "tract_geoid",
            "block",
            "county_fips",
            "county_name",
            "geometry",
        ]
    ].copy()
    tree, _, node_ids = base.build_node_kdtree(nodes)
    origin_snap = base.snap_points_to_nodes(
        origins,
        point_id_col="block_geoid",
        tree=tree,
        node_ids=node_ids,
        nodes=nodes,
        max_snap_m=base.MAX_ORIGIN_SNAP_M,
    )
    origins = origins.merge(origin_snap, on="block_geoid", how="left")
    centroid_boundary = base.compute_origin_boundary_fields(
        selected_centroids, boundary_polygon
    )
    centroids_source = selected_centroids.to_crs("OGC:CRS84")
    return origins, centroid_boundary, centroids_source


def recompute_zero_ft_status(
    *,
    edges,
    dry_graph,
    services,
    origins,
    centroid_boundary,
    centroids_source,
    slr0,
    boundary_node_ids: set[int],
):
    baseline_graph = base.build_graph(edges)
    baseline_nearest = base.build_nearest_service_lookup(
        baseline_graph, services
    )
    result = base.scenario_results_for_origins(
        slr_ft=0,
        slr_layer_name=base.BASELINE_SLR_LAYER,
        slr_layer=slr0,
        graph=dry_graph,
        services=services,
        origins=origins.drop(columns="geometry"),
        centroid_boundary=centroid_boundary,
        centroid_geometry_source=centroids_source[["block_geoid", "geometry"]],
        baseline_nearest=baseline_nearest,
        dry_boundary_node_ids=boundary_node_ids,
    )
    result["scenario_status"] = base.classify_status_columns(
        result,
        inundated_col="block_centroid_inundated",
        isolated_col="block_centroid_isolated",
        redundant_col="block_centroid_redundant",
        fragile_col="block_centroid_fragile",
    )
    return result


def status_summary_row(variant: str, statuses) -> dict[str, object]:
    row: dict[str, object] = {
        "variant": variant,
        "n_blocks": int(len(statuses)),
    }
    for status in ["inundated", "isolated", "fragile", "redundant", "other"]:
        count = int(statuses["scenario_status"].eq(status).sum())
        row[f"n_{status}"] = count
        row[f"share_{status}"] = float(count / len(statuses)) if len(statuses) else np.nan
    return row


def run_sensitivity(
    *,
    results,
    roads,
    nodes,
    edges,
    dry_graph,
    slr0,
    services_source,
    services,
    boundary_polygon,
    recovery,
    output_dir: Path,
    smoke: bool,
):
    log("item=fragility_sensitivity; state=started")
    blocks = base.read_vector(base.BLOCKS_PATH)
    blocks = base.prepare_blocks_layer(blocks)
    blocks = base.maybe_to_projected(blocks)
    centroids = base.make_centroids(blocks)

    analysis_centroids = centroids
    if smoke:
        analysis_centroids = centroids.loc[
            centroids.geometry.intersects(boundary_polygon.geometry.iloc[0])
        ].copy()
    default_origins, default_centroid_boundary, default_centroids_source = (
        prepare_sensitivity_origins(
            analysis_centroids,
            nodes,
            boundary_polygon,
        )
    )

    default_boundary_nodes = base.build_boundary_node_set(
        nodes, boundary_polygon
    )
    current = recompute_zero_ft_status(
        edges=edges,
        dry_graph=dry_graph,
        services=services,
        origins=default_origins,
        centroid_boundary=default_centroid_boundary,
        centroids_source=default_centroids_source,
        slr0=slr0,
        boundary_node_ids=default_boundary_nodes,
    )
    resnapped_services = services_from_recovery(services, recovery)
    resnapped = recompute_zero_ft_status(
        edges=edges,
        dry_graph=dry_graph,
        services=resnapped_services,
        origins=default_origins,
        centroid_boundary=default_centroid_boundary,
        centroids_source=default_centroids_source,
        slr0=slr0,
        boundary_node_ids=default_boundary_nodes,
    )

    no_service_highways = set(base.DRIVABLE_HIGHWAYS) - {"service"}
    no_service_roads = roads.loc[roads["highway"].ne("service")].copy()
    no_service_nodes, no_service_edges = segmentize_with_optional_cache(
        no_service_roads,
        highway_filter=no_service_highways,
        output_dir=output_dir,
        smoke=smoke,
        cache_enabled=True,
    )
    no_service_services, no_service_boundary = prepare_services(
        services_source, no_service_roads, no_service_nodes
    )
    no_service_slr0 = base.load_slr_layer(
        base.BASELINE_SLR_LAYER, box(*no_service_roads.total_bounds)
    )
    no_service_removed = query_removed_edge_positions(
        no_service_edges, no_service_slr0
    )
    no_service_dry_edges = no_service_edges.drop(
        index=no_service_edges.index[no_service_removed]
    ).copy()
    no_service_dry_graph = base.build_graph(no_service_dry_edges)

    no_service_origins, no_service_centroid_boundary, no_service_centroids_source = (
        prepare_sensitivity_origins(
            analysis_centroids,
            no_service_nodes,
            no_service_boundary,
        )
    )
    no_service_boundary_nodes = base.build_boundary_node_set(
        no_service_nodes, no_service_boundary
    )
    service_removed = recompute_zero_ft_status(
        edges=no_service_edges,
        dry_graph=no_service_dry_graph,
        services=no_service_services,
        origins=no_service_origins,
        centroid_boundary=no_service_centroid_boundary,
        centroids_source=no_service_centroids_source,
        slr0=no_service_slr0,
        boundary_node_ids=no_service_boundary_nodes,
    )

    no_service_component, no_service_component_size = (
        collapse_two_edge_components(no_service_dry_graph)
    )
    no_service_recovery = resnap_services_to_eligible_nodes(
        no_service_services,
        no_service_nodes,
        no_service_component,
        no_service_component_size,
    )
    both_services = services_from_recovery(
        no_service_services, no_service_recovery
    )
    both = recompute_zero_ft_status(
        edges=no_service_edges,
        dry_graph=no_service_dry_graph,
        services=both_services,
        origins=no_service_origins,
        centroid_boundary=no_service_centroid_boundary,
        centroids_source=no_service_centroids_source,
        slr0=no_service_slr0,
        boundary_node_ids=no_service_boundary_nodes,
    )

    selected_ids = set(default_origins["block_geoid"].astype(str))
    reported = results.loc[
        results["slr_ft"].eq(0)
        & results["block_geoid"].astype(str).isin(selected_ids)
    ].copy()
    comparison = pd.DataFrame.from_records(
        [
            status_summary_row("reported_baseline", reported),
            status_summary_row("recomputed_current", current),
            status_summary_row("service_highway_removed", service_removed),
            status_summary_row("services_resnapped", resnapped),
            status_summary_row("service_highway_removed_and_resnapped", both),
        ]
    )
    write_dataframe(
        comparison, output_dir / "fragility_sensitivity_comparison.csv"
    )
    return comparison


def print_final_summary(
    *,
    versions: dict[str, str],
    block_fragile_summary,
    singleton_summary,
    recovery_summary,
    bridge_summary,
    sensitivity_summary,
) -> None:
    print("\nFINAL SUMMARY", flush=True)
    print("\nRESOLVED VERSIONS", flush=True)
    for package_name in SUMMARY_VERSION_PACKAGES:
        print(f"{package_name} {versions[package_name]}", flush=True)

    print("\nBASELINE FRAGILE SHARE", flush=True)
    print(block_fragile_summary.to_string(index=False), flush=True)
    print("\nSERVICES IN SINGLETON 2ECC", flush=True)
    print(singleton_summary.to_string(index=False), flush=True)
    print("\nSERVICE RE-SNAPPING", flush=True)
    print(recovery_summary.to_string(index=False), flush=True)
    print("\nREMOVED BRIDGE-TAGGED SEGMENTS", flush=True)
    print(
        bridge_summary[
            ["slr_ft", "n_bridge_tagged_segments"]
        ].to_string(index=False),
        flush=True,
    )
    if sensitivity_summary is not None:
        print("\nFRAGILITY SENSITIVITY", flush=True)
        print(sensitivity_summary.to_string(index=False), flush=True)


def main() -> int:
    args = parse_args()
    versions = configure_dependencies(sensitivity=args.sensitivity)
    if versions is None:
        return 2

    global base
    base = load_base_module()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not CENSUS_BLOCK_ATTRIBUTES_PATH.exists():
        raise FileNotFoundError(
            f"Missing Census block attributes: {CENSUS_BLOCK_ATTRIBUTES_PATH}"
        )
    base.check_required_inputs(
        [
            base.NOAA_GPKG_PATH,
            base.PRIVATE_SCHOOLS_PATH,
            base.PUBLIC_SCHOOLS_PATH,
            base.FIRE_STATIONS_PATH,
            base.ROAD_PBF_PATH,
        ]
    )
    if args.sensitivity:
        base.check_required_inputs([base.BLOCKS_PATH])

    output_dir = run_dir / "diagnostics" / "validity"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_result_files(run_dir)
    results, duplicate_rows = load_and_combine_results(files)
    combine_summary = pd.DataFrame.from_records(
        [
            {
                "n_files": len(files),
                "n_input_rows": int(len(results) + len(duplicate_rows)),
                "n_duplicate_block_slr_rows_removed": int(len(duplicate_rows)),
                "n_unique_block_slr_rows": int(len(results)),
            }
        ]
    )
    write_dataframe(combine_summary, output_dir / "combined_run_summary.csv")
    block_output = run_block_universe(results, output_dir)

    log("item=graph_inputs; state=started")
    roads = load_graph_roads(smoke=args.smoke)
    highway_filter = set(base.DRIVABLE_HIGHWAYS)
    nodes, edges = segmentize_with_optional_cache(
        roads,
        highway_filter=highway_filter,
        output_dir=output_dir,
        smoke=args.smoke,
        cache_enabled=args.sensitivity,
    )
    services_source = base.load_services()
    services, boundary_polygon = prepare_services(
        services_source, roads, nodes
    )
    slr0, slr1 = load_zero_and_one_ft_layers(roads)
    removed_positions_by_slr = {
        0: query_removed_edge_positions(edges, slr0),
        1: query_removed_edge_positions(edges, slr1),
    }
    dry0_edges = edges.drop(
        index=edges.index[removed_positions_by_slr[0]]
    ).copy()
    dry0_graph = base.build_graph(dry0_edges)
    log(
        "item=dry0_graph; "
        f"nodes={dry0_graph.number_of_nodes()}; edges={dry0_graph.number_of_edges()}"
    )

    node_to_component, node_to_component_size = (
        collapse_two_edge_components(dry0_graph)
    )
    _, _, singleton_summary = service_component_outputs(
        services,
        node_to_component,
        node_to_component_size,
        output_dir,
    )

    recovery = resnap_services_to_eligible_nodes(
        services,
        nodes,
        node_to_component,
        node_to_component_size,
    )
    recovery_summary = summarize_resnapping(recovery)
    write_dataframe(recovery, output_dir / "service_resnapping.csv")
    write_dataframe(
        recovery_summary, output_dir / "service_resnapping_summary.csv"
    )

    bridge_output = run_bridge_audit(
        roads,
        edges,
        removed_positions_by_slr,
        output_dir,
    )

    sensitivity_summary = None
    if args.sensitivity:
        sensitivity_summary = run_sensitivity(
            results=results,
            roads=roads,
            nodes=nodes,
            edges=edges,
            dry_graph=dry0_graph,
            slr0=slr0,
            services_source=services_source,
            services=services,
            boundary_polygon=boundary_polygon,
            recovery=recovery,
            output_dir=output_dir,
            smoke=args.smoke,
        )

    versions_frame = pd.DataFrame.from_records(
        [
            {"package": package_name, "version": versions[package_name]}
            for package_name in SUMMARY_VERSION_PACKAGES
        ]
    )
    write_dataframe(versions_frame, output_dir / "resolved_versions.csv")
    print_final_summary(
        versions=versions,
        block_fragile_summary=block_output["fragile_summary"],
        singleton_summary=singleton_summary,
        recovery_summary=recovery_summary,
        bridge_summary=bridge_output["summary"],
        sensitivity_summary=sensitivity_summary,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        raise
