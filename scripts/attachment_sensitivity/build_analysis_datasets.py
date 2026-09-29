#!/usr/bin/env python
"""Rebuild block-level and block-group analysis datasets for each attachment arm.

Reproduces the eligible-universe filter, derived variables (notebook 03, cell 12),
and block-group aggregation (cell 18) of
``scripts/03_build_extension_dataset_and_memo.ipynb`` without plotting or API
calls. ACS covariates are reused by block-group GEOID from the production
``data/processed/analysis/block_group_analysis_dataset_approach.csv`` (one value
per GEOID, asserted), so every arm uses identical social covariates.

Writes, per arm, under data/processed/access/attachment_sensitivity/<exp>/<arm>/:
  block_level_long_dataset.parquet     eligible rows + derived variables
  block_group_analysis_dataset.csv     model input for 04_transition_models.R
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attachment_lib as al  # noqa: E402

PROJECT_ROOT = al.PROJECT_ROOT
PRODUCTION_BG = PROJECT_ROOT / "data/processed/analysis/block_group_analysis_dataset_approach.csv"
STATUS_ORDER = ["unclassified", "inundated", "isolated", "fragile", "redundant"]
ACS_COLUMNS = [
    "total_pop", "median_income", "median_age", "hh_no_vehicle", "no_vehicle_share",
    "pop_age_65plus", "pct_age_65plus", "pct_white_nh", "pct_black_nh", "pct_hispanic",
    "pct_nonwhite", "renter_share", "poverty_rate", "log_median_income",
]
GROUP_KEYS = ["block_group_geoid", "county_fips", "county_name", "tract_geoid", "slr_ft"]
GEOID_WIDTHS = {"block_geoid": 15, "block_group_geoid": 12, "tract_geoid": 11, "county_fips": 3}


def derive_block_level(source: pd.DataFrame) -> pd.DataFrame:
    for column, width in GEOID_WIDTHS.items():
        source[column] = source[column].astype("string").str.zfill(width)
    source["slr_ft"] = pd.to_numeric(source["slr_ft"], errors="raise").astype(int)
    assert not source.duplicated(["block_geoid", "slr_ft"]).any()
    assert source.groupby("block_geoid")["analysis_eligible"].nunique(dropna=False).eq(1).all()
    long_df = source.loc[source["analysis_eligible"].eq(True)].copy()
    assert long_df.groupby("block_geoid")["slr_ft"].nunique().eq(7).all()

    status = long_df["scenario_status"]
    assert set(status.dropna().unique()) <= set(STATUS_ORDER)
    worse = status.isin(["fragile", "isolated", "inundated"])
    long_df["fragile_or_worse"] = worse.astype("Int8")
    long_df["any_loss_of_redundancy"] = (long_df["baseline_block_centroid_redundant"].eq(1) & worse).astype("Int8")
    pairs = {
        "baseline_redundant_to_fragile": ("baseline_block_centroid_redundant", "block_centroid_fragile"),
        "baseline_redundant_to_isolated": ("baseline_block_centroid_redundant", "block_centroid_isolated"),
        "baseline_redundant_to_inundated": ("baseline_block_centroid_redundant", "block_centroid_inundated"),
        "baseline_fragile_to_isolated": ("baseline_block_centroid_fragile", "block_centroid_isolated"),
        "baseline_fragile_to_inundated": ("baseline_block_centroid_fragile", "block_centroid_inundated"),
        "baseline_isolated_to_inundated": ("baseline_block_centroid_isolated", "block_centroid_inundated"),
    }
    for name, (origin_col, dest_col) in pairs.items():
        long_df[name] = (long_df[origin_col].eq(1) & long_df[dest_col].eq(1)).astype("Int8")
    long_df["baseline_fragile_persisted"] = long_df["persistent_fragile"].astype("Int8")
    long_df["became_fragile"] = long_df["new_fragile_due_to_slr"].astype("Int8")
    long_df["became_isolated"] = long_df["new_isolated_due_to_slr"].astype("Int8")
    long_df["became_inundated"] = long_df["new_inundated_due_to_slr"].astype("Int8")
    long_df["nearby_bridge_structure_retained_flag"] = long_df["nearby_bridge_structure_retained"].eq(True).astype("Int8")
    long_df["origin_in_lcc_false"] = (~long_df["origin_in_lcc"].astype(bool)).astype("Int8")
    pir = pd.to_numeric(long_df["detour_ratio"], errors="coerce")
    pir[long_df["block_centroid_isolated"].eq(1) | long_df["block_centroid_inundated"].eq(1)] = np.nan
    long_df["path_inflation_ratio"] = pir.replace([np.inf, -np.inf], np.nan)
    return long_df.sort_values(["slr_ft", "block_geoid"]).reset_index(drop=True)


def aggregate_block_groups(long_df: pd.DataFrame) -> pd.DataFrame:
    scen = ["block_centroid_unclassified", "block_centroid_inundated", "block_centroid_isolated",
            "block_centroid_fragile", "block_centroid_redundant"]
    base_cols = ["baseline_" + c for c in scen]
    method = long_df["origin_geometry_method"].fillna("missing").astype(str)
    method_share = {}
    for m in sorted(method.unique()):
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", m).strip("_").lower() or "missing"
        long_df[f"n_origin_geometry_method_{slug}"] = method.eq(m).astype("Int8")
        method_share[f"n_origin_geometry_method_{slug}"] = f"share_origin_geometry_method_{slug}"
    count_cols = scen + base_cols + [
        "fragile_or_worse", "any_loss_of_redundancy",
        "new_fragile_due_to_slr", "new_isolated_due_to_slr", "new_inundated_due_to_slr",
        "baseline_redundant_to_fragile", "baseline_redundant_to_isolated", "baseline_redundant_to_inundated",
        "baseline_fragile_to_isolated", "baseline_fragile_to_inundated", "baseline_isolated_to_inundated",
        "nearby_bridge_structure_retained_flag", "origin_in_lcc_false",
    ] + list(method_share)
    agg = {c: "sum" for c in count_cols}
    agg.update({"block_geoid": "size", "pop20": "sum"})
    bg = (long_df.groupby(GROUP_KEYS, dropna=False).agg(agg).reset_index()
          .rename(columns={"block_geoid": "total_blocks", "pop20": "eligible_pop20"}))
    for c in count_cols + ["total_blocks", "eligible_pop20"]:
        bg[c] = bg[c].astype(int)
    assert bg[scen].sum(axis=1).eq(bg["total_blocks"]).all()
    assert bg[base_cols].sum(axis=1).eq(bg["total_blocks"]).all()
    meta = long_df.groupby(GROUP_KEYS, dropna=False, as_index=False).agg(bridge_rule_applied=("bridge_rule_applied", "first"))
    paths = (long_df.groupby(GROUP_KEYS, dropna=False)["max_edge_disjoint_paths_any_service"].agg(["mean", "median"])
             .rename(columns={"mean": "mean_max_edge_disjoint_paths", "median": "median_max_edge_disjoint_paths"}).reset_index())
    pir = (long_df.groupby(["block_group_geoid", "slr_ft"], dropna=False)["path_inflation_ratio"].agg(["mean", "median"])
           .rename(columns={"mean": "mean_path_inflation_ratio", "median": "median_path_inflation_ratio"}).reset_index())
    bg = (bg.merge(meta, on=GROUP_KEYS, validate="one_to_one").merge(paths, on=GROUP_KEYS, validate="one_to_one")
          .merge(pir, on=["block_group_geoid", "slr_ft"], validate="one_to_one"))
    share_map = {
        "block_centroid_unclassified": "share_unclassified", "block_centroid_inundated": "share_inundated",
        "block_centroid_isolated": "share_isolated", "block_centroid_fragile": "share_fragile",
        "block_centroid_redundant": "share_redundant", "fragile_or_worse": "share_fragile_or_worse",
        "any_loss_of_redundancy": "share_lost_redundancy", "new_fragile_due_to_slr": "share_new_fragile",
        "new_isolated_due_to_slr": "share_new_isolated", "new_inundated_due_to_slr": "share_new_inundated",
        "nearby_bridge_structure_retained_flag": "share_nearby_bridge_structure_retained",
        "origin_in_lcc_false": "share_origin_in_lcc_false", **method_share,
    }
    for count_col, share_col in share_map.items():
        bg[share_col] = bg[count_col] / bg["total_blocks"]
    return bg


def load_acs() -> pd.DataFrame:
    prod = pd.read_csv(PRODUCTION_BG, dtype={"block_group_geoid": str})
    prod["block_group_geoid"] = prod["block_group_geoid"].str.zfill(12)
    acs = prod[["block_group_geoid"] + ACS_COLUMNS]
    per_geoid = acs.groupby("block_group_geoid").nunique(dropna=False)
    assert per_geoid.le(1).all(axis=None), "ACS values vary within a GEOID in the production dataset."
    return acs.drop_duplicates("block_group_geoid").reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--arms", default=",".join(al.ARM_ORDER))
    p.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "data/processed/access/attachment_sensitivity")
    args = p.parse_args()
    acs = load_acs()
    for arm in [a for a in args.arms.split(",") if a]:
        arm_dir = args.out_root / args.experiment_id / arm
        if not (arm_dir / "_COMPLETE.json").exists():
            raise SystemExit(f"Arm {arm} has no completed access run in {arm_dir}.")
        source = pd.read_parquet(arm_dir / "block_access_flags_long.parquet")
        long_df = derive_block_level(source)
        bg = aggregate_block_groups(long_df.copy())
        missing = sorted(set(bg["block_group_geoid"]) - set(acs["block_group_geoid"]))
        if missing:
            raise SystemExit(f"{arm}: {len(missing)} block groups lack reused ACS rows (e.g. {missing[:5]}).")
        bg = bg.merge(acs, on="block_group_geoid", how="left", validate="many_to_one")
        al.atomic_to_parquet(long_df, arm_dir / "block_level_long_dataset.parquet")
        al.atomic_to_csv(bg, arm_dir / "block_group_analysis_dataset.csv")
        al.atomic_write_json(arm_dir / "_DATASETS_COMPLETE.json", {
            "eligible_blocks": int(long_df["block_geoid"].nunique()),
            "block_groups": int(bg["block_group_geoid"].nunique()),
            "bg_rows": int(len(bg)),
        })
        print(f"{arm}: {long_df['block_geoid'].nunique():,} eligible blocks; {bg['block_group_geoid'].nunique():,} block groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
