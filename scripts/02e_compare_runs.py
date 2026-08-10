#!/usr/bin/env python
"""Compare two access-classification run directories.

The utility preserves each run's block universe, emits correction transition
matrices by SLR scenario, reports baseline fragile shares under four analysis
universes, and writes every baseline block whose status changed.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
PROPOSED_DEPENDENCIES = {"pandas": "3.0.3", "pyogrio": "0.12.1"}


def load_dependencies():
    modules = {}
    missing = []
    for package_name, version in PROPOSED_DEPENDENCIES.items():
        try:
            modules[package_name] = importlib.import_module(package_name)
        except (ImportError, OSError) as exc:
            missing.append((package_name, version, str(exc)))
    if missing:
        for package_name, version, error in missing:
            print(
                f"[DEPENDENCY ERROR] {package_name}: {error}; proposed={package_name}=={version}",
                file=sys.stderr,
            )
        print("[DEPENDENCY ERROR] packages_installed=0", file=sys.stderr)
        raise SystemExit(2)
    return modules["pandas"], modules["pyogrio"]


pd, pyogrio = load_dependencies()


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
    "exclusion_reason": "string",
    "origin_geometry_method": "string",
    "service_snap_rule": "string",
    "bridge_rule_applied": "string",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two access run directories.")
    parser.add_argument("old_run", type=Path, help="Legacy/reference run directory.")
    parser.add_argument("new_run", type=Path, help="Corrected/comparison run directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to <new-run>/comparison_vs_<old-run-name>.",
    )
    return parser.parse_args()


def find_result_files(run_dir: Path) -> list[Path]:
    files = sorted(
        path
        for path in run_dir.glob("block_access_flags_long*.csv")
        if path.is_file()
    )
    if not files:
        raise FileNotFoundError(f"No block_access_flags_long*.csv files in {run_dir}")
    return files


def load_results(run_dir: Path):
    frames = [
        pd.read_csv(path, dtype=RESULT_DTYPE, low_memory=False)
        for path in find_result_files(run_dir)
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined["slr_ft"] = combined["slr_ft"].astype(int)
    duplicate_mask = combined.duplicated(["block_geoid", "slr_ft"], keep="first")
    results = combined.loc[~duplicate_mask].copy()
    return results.sort_values(["slr_ft", "block_geoid"]).reset_index(drop=True)


def load_census_attributes():
    attributes = pyogrio.read_dataframe(
        CENSUS_BLOCK_ATTRIBUTES_PATH,
        columns=["GEOID20", "COUNTYFP20", "POP20", "ALAND20"],
        read_geometry=False,
    )
    attributes["COUNTYFP20"] = attributes["COUNTYFP20"].astype("string").str.zfill(3)
    attributes = attributes.loc[
        attributes["COUNTYFP20"].isin(TRICOUNTY_FIPS)
    ].copy()
    attributes["block_geoid"] = attributes["GEOID20"].astype("string").str.zfill(15)
    attributes["census_pop20"] = pd.to_numeric(attributes["POP20"], errors="raise").astype(int)
    attributes["census_land_area_m2"] = pd.to_numeric(
        attributes["ALAND20"], errors="raise"
    ).astype(int)
    return attributes[
        ["block_geoid", "census_pop20", "census_land_area_m2"]
    ].drop_duplicates("block_geoid")


def add_analysis_fields(results, census):
    output = results.merge(census, on="block_geoid", how="left", validate="many_to_one")
    if "pop20" not in output:
        output["pop20"] = output["census_pop20"]
    else:
        output["pop20"] = output["pop20"].fillna(output["census_pop20"])
    if "land_area_m2" not in output:
        output["land_area_m2"] = output["census_land_area_m2"]
    else:
        output["land_area_m2"] = output["land_area_m2"].fillna(
            output["census_land_area_m2"]
        )
    if "analysis_eligible" not in output:
        origin_valid = ~output.get(
            "origin_snap_exceeds_threshold", pd.Series(0, index=output.index)
        ).eq(1)
        output["analysis_eligible"] = output["land_area_m2"].ne(0) & origin_valid
    output["analysis_eligible"] = output["analysis_eligible"].astype(bool)
    return output.drop(columns=["census_pop20", "census_land_area_m2"])


def correction_crosstab(old, new):
    merged = old[["block_geoid", "slr_ft", "scenario_status"]].merge(
        new[["block_geoid", "slr_ft", "scenario_status"]],
        on=["block_geoid", "slr_ft"],
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
        validate="one_to_one",
    )
    mismatched = merged["_merge"].ne("both")
    if mismatched.any():
        raise ValueError(
            f"Run universes differ for {int(mismatched.sum())} block-scenario rows."
        )
    return (
        merged.groupby(
            ["slr_ft", "scenario_status_old", "scenario_status_new"],
            dropna=False,
            as_index=False,
        )
        .agg(n_blocks=("block_geoid", "size"))
        .sort_values(["slr_ft", "scenario_status_old", "scenario_status_new"])
        .reset_index(drop=True)
    )


def fragile_share_rows(run_label: str, baseline):
    masks = {
        "all_blocks": pd.Series(True, index=baseline.index),
        "pop20_gt_0": baseline["pop20"].gt(0),
        "analysis_eligible": baseline["analysis_eligible"],
        "analysis_eligible_pop20_gt_0": (
            baseline["analysis_eligible"] & baseline["pop20"].gt(0)
        ),
    }
    rows = []
    for universe, mask in masks.items():
        subset = baseline.loc[mask]
        n_fragile = int(subset["scenario_status"].eq("fragile").sum())
        rows.append(
            {
                "run": run_label,
                "universe": universe,
                "n_blocks": int(len(subset)),
                "n_fragile": n_fragile,
                "fragile_share": n_fragile / len(subset) if len(subset) else float("nan"),
            }
        )
    return rows


def changed_baseline_blocks(old, new):
    old_base = old.loc[old["slr_ft"].eq(0)].copy()
    new_base = new.loc[new["slr_ft"].eq(0)].copy()
    optional_new_columns = [
        "service_snap_rule",
        "bridge_rule_applied",
        "analysis_eligible",
        "exclusion_reason",
        "nearby_bridge_structure_id",
        "nearby_bridge_structure_distance_m",
        "nearby_bridge_structure_retained",
    ]
    new_columns = [
        "block_geoid",
        "scenario_status",
        "pop20",
        "county_name",
        *[column for column in optional_new_columns if column in new_base.columns],
    ]
    changed = old_base[["block_geoid", "scenario_status"]].merge(
        new_base[new_columns],
        on="block_geoid",
        how="inner",
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    changed = changed.loc[
        changed["scenario_status_old"].ne(changed["scenario_status_new"])
    ].copy()
    return changed.sort_values(["county_name", "block_geoid"]).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    old_run = args.old_run.resolve()
    new_run = args.new_run.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else new_run / f"comparison_vs_{old_run.name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    census = load_census_attributes()
    old = add_analysis_fields(load_results(old_run), census)
    new = add_analysis_fields(load_results(new_run), census)

    crosstab = correction_crosstab(old, new)
    fragile_shares = pd.DataFrame.from_records(
        [
            *fragile_share_rows("old", old.loc[old["slr_ft"].eq(0)]),
            *fragile_share_rows("new", new.loc[new["slr_ft"].eq(0)]),
        ]
    )
    changed = changed_baseline_blocks(old, new)

    crosstab.to_csv(output_dir / "status_correction_crosstab_by_slr.csv", index=False)
    fragile_shares.to_csv(output_dir / "baseline_fragile_share_comparison.csv", index=False)
    changed.to_csv(output_dir / "baseline_status_changed_blocks.csv", index=False)

    print(crosstab.to_string(index=False))
    print("\nBaseline fragile shares")
    print(fragile_shares.to_string(index=False))
    print(f"\nChanged baseline blocks: {len(changed):,}")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
