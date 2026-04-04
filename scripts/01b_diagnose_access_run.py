#!/usr/bin/env python
"""
Diagnose a completed block-access run directory.

This script is intentionally read-only with respect to the underlying access
workflow. It ingests one completed run folder, combines the per-scenario output
files, produces summary diagnostics, and joins one selected SLR scenario back
to block geometry for spatial QA.

Primary outputs
- diagnostics/file_inventory.csv
- diagnostics/status_summary_by_slr.csv
- diagnostics/status_summary_by_slr_county.csv
- diagnostics/baseline_status_summary_by_county.csv
- diagnostics/baseline_isolated_reason_summary.csv
- diagnostics/transition_summary_by_slr.csv
- diagnostics/transition_summary_by_slr_county.csv
- diagnostics/baseline_vs_scenario_crosstab.csv
- diagnostics/slr_<ft>_block_status_join.gpkg
- diagnostics/slr_<ft>_status_map.png
- diagnostics/slr_<ft>_affected_only_map.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "census"
    / "blocks"
    / "fl_tricounty_blocks_2020.gpkg"
)

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

STATUS_ORDER = ["inundated", "isolated", "fragile", "redundant", "other"]
STATUS_COLORS = {
    "inundated": "#1f78b4",
    "isolated": "#d7301f",
    "fragile": "#fdae61",
    "redundant": "#bdbdbd",
    "other": "#f7f7f7",
}
TRANSITION_COLORS = {
    "new_inundated_due_to_slr": "#2166ac",
    "new_isolated_due_to_slr": "#b2182b",
    "new_fragile_due_to_slr": "#ef8a62",
    "persistent_fragile": "#fddbc7",
    "other": "#d9d9d9",
}


def log(message: str) -> None:
    print(f"[diagnose_access_run] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose one completed block-access run directory and produce summary tables plus QA maps."
    )
    parser.add_argument(
        "--run-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "access" / "edited" / "2026-04-03_run"),
        help="Path to the completed run directory containing block_access_flags_long*.csv outputs.",
    )
    parser.add_argument(
        "--map-slr-ft",
        type=int,
        default=1,
        help="Positive SLR scenario to join back to block geometry for QA maps.",
    )
    return parser.parse_args()


def find_result_files(run_dir: Path) -> list[Path]:
    files = sorted(
        path
        for path in run_dir.glob("*.csv")
        if path.name.startswith("block_access_flags_long")
    )
    if not files:
        raise FileNotFoundError(f"No block_access_flags_long*.csv files found in {run_dir}")
    return files


def load_file_inventory(files: list[Path]) -> pd.DataFrame:
    inventory_records: list[dict[str, object]] = []
    for path in files:
        slim = pd.read_csv(path, usecols=["slr_ft"], dtype={"slr_ft": "Int64"})
        inventory_records.append(
            {
                "file_name": path.name,
                "rows": int(len(slim)),
                "unique_slr_ft": ",".join(str(int(x)) for x in sorted(slim["slr_ft"].dropna().unique().tolist())),
            }
        )
    return pd.DataFrame.from_records(inventory_records)


def load_and_combine_results(files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [pd.read_csv(path, dtype=RESULT_DTYPE, low_memory=False) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    combined["slr_ft"] = combined["slr_ft"].astype(int)

    duplicate_mask = combined.duplicated(subset=["block_geoid", "slr_ft"], keep="first")
    duplicate_rows = combined.loc[duplicate_mask, ["block_geoid", "slr_ft"]].copy()
    deduped = combined.loc[~duplicate_mask].copy()
    deduped = deduped.sort_values(["slr_ft", "block_geoid"]).reset_index(drop=True)
    return deduped, duplicate_rows


def ensure_output_dir(run_dir: Path) -> Path:
    output_dir = run_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    log(f"Saved {path}")


def summarize_status_by_slr(results: pd.DataFrame) -> pd.DataFrame:
    output = (
        results.groupby(["slr_ft", "scenario_status"], as_index=False)
        .agg(n_blocks=("block_geoid", "size"))
        .sort_values(["slr_ft", "scenario_status"])
        .reset_index(drop=True)
    )
    total_by_slr = output.groupby("slr_ft")["n_blocks"].transform("sum")
    output["share_of_blocks"] = output["n_blocks"] / total_by_slr
    return output


def summarize_status_by_slr_county(results: pd.DataFrame) -> pd.DataFrame:
    output = (
        results.groupby(["slr_ft", "county_name", "scenario_status"], as_index=False)
        .agg(n_blocks=("block_geoid", "size"))
        .sort_values(["slr_ft", "county_name", "scenario_status"])
        .reset_index(drop=True)
    )
    total_by_group = output.groupby(["slr_ft", "county_name"])["n_blocks"].transform("sum")
    output["share_of_blocks"] = output["n_blocks"] / total_by_group
    return output


def summarize_baseline_status_by_county(results: pd.DataFrame) -> pd.DataFrame:
    baseline = results.loc[results["slr_ft"] == 0].copy()
    output = (
        baseline.groupby(["county_name", "scenario_status"], as_index=False)
        .agg(n_blocks=("block_geoid", "size"))
        .sort_values(["county_name", "scenario_status"])
        .reset_index(drop=True)
    )
    total_by_county = output.groupby("county_name")["n_blocks"].transform("sum")
    output["share_of_blocks"] = output["n_blocks"] / total_by_county
    return output


def summarize_baseline_isolated_reasons(results: pd.DataFrame) -> pd.DataFrame:
    baseline = results.loc[results["slr_ft"] == 0].copy()
    baseline["isolated_reason"] = "not_isolated"
    isolated_mask = baseline["block_centroid_isolated"].eq(1)
    baseline.loc[isolated_mask & baseline["origin_snap_exceeds_threshold"].eq(1) & baseline["n_reachable_services"].eq(0), "isolated_reason"] = "snap_warning_and_no_service"
    baseline.loc[isolated_mask & baseline["origin_snap_exceeds_threshold"].eq(1) & baseline["n_reachable_services"].gt(0), "isolated_reason"] = "snap_warning_only"
    baseline.loc[isolated_mask & baseline["origin_snap_exceeds_threshold"].eq(0) & baseline["n_reachable_services"].eq(0), "isolated_reason"] = "no_reachable_service"

    output = (
        baseline.groupby(
            [
                "county_name",
                "isolated_reason",
                "boundary_flag",
                "component_touches_boundary",
            ],
            as_index=False,
        )
        .agg(n_blocks=("block_geoid", "size"))
        .sort_values(["county_name", "isolated_reason", "boundary_flag", "component_touches_boundary"])
        .reset_index(drop=True)
    )
    return output


def summarize_transitions(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positive = results.loc[results["slr_ft"] != 0].copy()

    by_slr = (
        positive.groupby("slr_ft", as_index=False)
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
        .sort_values("slr_ft")
        .reset_index(drop=True)
    )

    by_slr_county = (
        positive.groupby(["slr_ft", "county_name"], as_index=False)
        .agg(
            n_blocks=("block_geoid", "size"),
            persistent_fragile=("persistent_fragile", "sum"),
            new_fragile_due_to_slr=("new_fragile_due_to_slr", "sum"),
            new_isolated_due_to_slr=("new_isolated_due_to_slr", "sum"),
            new_inundated_due_to_slr=("new_inundated_due_to_slr", "sum"),
        )
        .sort_values(["slr_ft", "county_name"])
        .reset_index(drop=True)
    )

    crosstab = (
        positive.groupby(["slr_ft", "baseline_status", "scenario_status"], as_index=False)
        .agg(n_blocks=("block_geoid", "size"))
        .sort_values(["slr_ft", "baseline_status", "scenario_status"])
        .reset_index(drop=True)
    )
    return by_slr, by_slr_county, crosstab


def build_status_join(results: pd.DataFrame, map_slr_ft: int) -> gpd.GeoDataFrame:
    scenario_rows = results.loc[results["slr_ft"] == map_slr_ft].copy()
    if scenario_rows.empty:
        raise ValueError(f"No rows found for slr_ft={map_slr_ft}")

    blocks = gpd.read_file(BLOCKS_PATH)
    if "geoid" in blocks.columns and "block_geoid" not in blocks.columns:
        blocks = blocks.rename(columns={"geoid": "block_geoid"})
    blocks["block_geoid"] = blocks["block_geoid"].astype(str)

    joined = blocks.merge(
        scenario_rows,
        on="block_geoid",
        how="inner",
        validate="one_to_one",
    )
    joined = gpd.GeoDataFrame(joined, geometry="geometry", crs=blocks.crs)
    return joined


def add_transition_label(joined: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    output = joined.copy()
    output["transition_label"] = "other"
    output.loc[output["new_inundated_due_to_slr"].eq(1), "transition_label"] = "new_inundated_due_to_slr"
    output.loc[output["new_isolated_due_to_slr"].eq(1), "transition_label"] = "new_isolated_due_to_slr"
    output.loc[output["new_fragile_due_to_slr"].eq(1), "transition_label"] = "new_fragile_due_to_slr"
    output.loc[output["persistent_fragile"].eq(1), "transition_label"] = "persistent_fragile"
    return output


def plot_status_maps(joined: gpd.GeoDataFrame, output_dir: Path, map_slr_ft: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)

    for ax, column, title in [
        (axes[0], "baseline_status", f"Baseline Status (0 ft)"),
        (axes[1], "scenario_status", f"Scenario Status ({map_slr_ft} ft)"),
    ]:
        joined.plot(color="#f5f5f5", linewidth=0, ax=ax)
        for status in STATUS_ORDER:
            subset = joined.loc[joined[column] == status]
            if subset.empty:
                continue
            subset.plot(
                ax=ax,
                color=STATUS_COLORS[status],
                linewidth=0,
            )
        ax.set_title(title)
        ax.set_axis_off()

    status_path = output_dir / f"slr_{map_slr_ft}ft_status_map.png"
    fig.savefig(status_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved {status_path}")


def plot_affected_only_map(joined: gpd.GeoDataFrame, output_dir: Path, map_slr_ft: int) -> None:
    plot_frame = add_transition_label(joined)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    plot_frame.plot(color="#f2f2f2", linewidth=0, ax=ax)
    for label, color in TRANSITION_COLORS.items():
        subset = plot_frame.loc[plot_frame["transition_label"] == label]
        if subset.empty:
            continue
        subset.plot(ax=ax, color=color, linewidth=0)
    ax.set_title(f"Transition Diagnostics ({map_slr_ft} ft)")
    ax.set_axis_off()

    transition_path = output_dir / f"slr_{map_slr_ft}ft_affected_only_map.png"
    fig.savefig(transition_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved {transition_path}")


def save_spatial_join(joined: gpd.GeoDataFrame, output_dir: Path, map_slr_ft: int) -> Path:
    gpkg_path = output_dir / f"slr_{map_slr_ft}ft_block_status_join.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()
    joined.to_file(gpkg_path, driver="GPKG")
    log(f"Saved {gpkg_path}")
    return gpkg_path


def print_key_findings(
    file_inventory: pd.DataFrame,
    results: pd.DataFrame,
    transition_by_slr: pd.DataFrame,
    map_slr_ft: int,
) -> None:
    baseline = results.loc[results["slr_ft"] == 0].copy()
    scenario = results.loc[results["slr_ft"] == map_slr_ft].copy()

    print("\nFile inventory")
    print(file_inventory.to_string(index=False))

    print("\nBaseline status counts (all counties)")
    print(
        baseline.groupby("scenario_status")
        .size()
        .reset_index(name="n_blocks")
        .sort_values("scenario_status")
        .to_string(index=False)
    )

    print(f"\nScenario status counts for {map_slr_ft} ft")
    print(
        scenario.groupby("scenario_status")
        .size()
        .reset_index(name="n_blocks")
        .sort_values("scenario_status")
        .to_string(index=False)
    )

    positive_transition = transition_by_slr.loc[transition_by_slr["slr_ft"] == map_slr_ft].copy()
    if not positive_transition.empty:
        print(f"\nTransition summary for {map_slr_ft} ft")
        print(positive_transition.to_string(index=False))


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    output_dir = ensure_output_dir(run_dir)
    files = find_result_files(run_dir)
    file_inventory = load_file_inventory(files)
    results, duplicate_rows = load_and_combine_results(files)

    duplicate_summary = pd.DataFrame(
        {
            "n_input_rows": [len(results) + len(duplicate_rows)],
            "n_duplicate_block_slr_rows_removed": [len(duplicate_rows)],
            "n_unique_block_slr_rows": [len(results)],
            "n_unique_blocks": [results["block_geoid"].nunique()],
            "unique_slr_ft": [",".join(str(x) for x in sorted(results["slr_ft"].unique().tolist()))],
        }
    )

    status_by_slr = summarize_status_by_slr(results)
    status_by_slr_county = summarize_status_by_slr_county(results)
    baseline_by_county = summarize_baseline_status_by_county(results)
    baseline_isolated_reasons = summarize_baseline_isolated_reasons(results)
    transition_by_slr, transition_by_slr_county, baseline_vs_scenario = summarize_transitions(results)

    write_dataframe(file_inventory, output_dir / "file_inventory.csv")
    write_dataframe(duplicate_summary, output_dir / "combined_run_summary.csv")
    write_dataframe(status_by_slr, output_dir / "status_summary_by_slr.csv")
    write_dataframe(status_by_slr_county, output_dir / "status_summary_by_slr_county.csv")
    write_dataframe(baseline_by_county, output_dir / "baseline_status_summary_by_county.csv")
    write_dataframe(baseline_isolated_reasons, output_dir / "baseline_isolated_reason_summary.csv")
    write_dataframe(transition_by_slr, output_dir / "transition_summary_by_slr.csv")
    write_dataframe(transition_by_slr_county, output_dir / "transition_summary_by_slr_county.csv")
    write_dataframe(baseline_vs_scenario, output_dir / "baseline_vs_scenario_crosstab.csv")

    joined = build_status_join(results, args.map_slr_ft)
    save_spatial_join(joined, output_dir, args.map_slr_ft)
    plot_status_maps(joined, output_dir, args.map_slr_ft)
    plot_affected_only_map(joined, output_dir, args.map_slr_ft)

    print_key_findings(file_inventory, results, transition_by_slr, args.map_slr_ft)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
