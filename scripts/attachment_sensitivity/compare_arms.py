#!/usr/bin/env python
"""Access-level comparison tables for the attachment arms (no models).

Outputs under outputs/attachment_sensitivity/<exp>/:
  attachment_audit.csv              one row per (arm, facility candidate)
  attachment_audit_summary.csv      distance distributions and change counts
  origin_attachment_summary.csv     origin distance distributions and exclusions
  eligibility_comparison.csv        eligible blocks/population and exclusion reasons
  state_comparison.csv              state counts/pop/shares by arm x SLR, deltas vs reference
  state_crosstab_vs_reference.csv   reference-vs-arm state by GEOID (common eligible origins)
  transition_population_comparison.csv  five transitions + 05-style cumulative totals
  changed_unit_composition.csv      descriptive composition of changed/excluded origins
  spatial_review/attachment_review.gpkg  changed/distant attachments for map review
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attachment_lib as al  # noqa: E402

PROJECT_ROOT = al.PROJECT_ROOT
STATES = ["redundant", "fragile", "isolated", "inundated", "unclassified"]
TRANSITIONS = ["baseline_redundant_to_fragile", "baseline_redundant_to_isolated",
               "baseline_redundant_to_inundated", "baseline_fragile_to_isolated",
               "baseline_fragile_to_inundated"]
COVARS = ["pct_black_nh", "pct_hispanic", "renter_share", "log_median_income", "pct_age_65plus", "no_vehicle_share"]
PCTS = [0.5, 0.9, 0.95, 0.99]


def dist_stats(series: pd.Series, prefix: str) -> dict:
    s = series.dropna().astype(float)
    out = {f"{prefix}_n": int(len(s))}
    if len(s):
        for q in PCTS:
            out[f"{prefix}_p{int(q * 100)}"] = float(s.quantile(q))
        out[f"{prefix}_max"] = float(s.max())
    return out


def facility_audit(data_dir: Path, arms: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = pd.read_csv(data_dir / "reference" / "service_attachment_audit.csv")
    ref_cols = ref[["service_record_id", "node_id", "snap_valid", "snap_distance_m", "service_snap_rule", "x", "y"]].rename(
        columns={"node_id": "ref_node_id", "snap_valid": "ref_valid", "snap_distance_m": "ref_distance_m",
                 "service_snap_rule": "ref_rule", "x": "ref_node_x", "y": "ref_node_y"})
    rows, summary = [], []
    for arm in arms:
        a = pd.read_csv(data_dir / arm / "service_attachment_audit.csv").merge(ref_cols, on="service_record_id", validate="one_to_one")
        a["valid"] = a["snap_valid"].astype(str).eq("True")
        a["ref_valid"] = a["ref_valid"].astype(str).eq("True")
        a["change_type"] = np.select(
            [a["valid"] & ~a["ref_valid"], ~a["valid"] & a["ref_valid"],
             a["valid"] & a["ref_valid"] & a["node_id"].ne(a["ref_node_id"])],
            ["added", "excluded", "reattached"], default="unchanged")
        a["added_snap_distance_m"] = np.where(a["valid"] & a["ref_valid"], a["snap_distance_m"] - a["ref_distance_m"], np.nan)
        a["node_displacement_m"] = np.where(
            a["change_type"].eq("reattached"), np.hypot(a["x"] - a["ref_node_x"], a["y"] - a["ref_node_y"]), np.nan)
        rows.append(a)
        rec = {"attachment_arm": arm, "candidates": int(len(a)), "valid": int(a["valid"].sum()),
               "fallback_valid": int((a["valid"] & a["service_snap_rule"].eq("fallback_unconstrained")).sum()),
               **{f"n_{t}": int(a["change_type"].eq(t).sum()) for t in ["added", "excluded", "reattached", "unchanged"]}}
        for t in ["school", "fire_station"]:
            sub = a.loc[a["service_type"].eq(t)]
            rec[f"valid_{t}"] = int(sub["valid"].sum())
            rec[f"changed_{t}"] = int(sub["change_type"].ne("unchanged").sum())
        rec.update(dist_stats(a.loc[a["valid"], "snap_distance_m"], "valid_distance_m"))
        rec.update(dist_stats(a.loc[a["change_type"].eq("reattached"), "node_displacement_m"], "reattach_displacement_m"))
        rec.update(dist_stats(a.loc[a["change_type"].eq("reattached"), "added_snap_distance_m"], "reattach_added_distance_m"))
        rec.update(dist_stats(a.loc[a["change_type"].eq("added"), "snap_distance_m"], "added_distance_m"))
        summary.append(rec)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summary)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/access/attachment_sensitivity")
    p.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "outputs/attachment_sensitivity")
    args = p.parse_args()
    data_dir = args.data_root / args.experiment_id
    out = args.out_root / args.experiment_id
    out.mkdir(parents=True, exist_ok=True)
    arms = [a for a in al.ARM_ORDER if (data_dir / a / "_DATASETS_COMPLETE.json").exists()]

    fac, fac_summary = facility_audit(data_dir, arms)
    al.atomic_to_csv(fac, out / "attachment_audit.csv")
    al.atomic_to_csv(fac_summary, out / "attachment_audit_summary.csv")

    full = {a: pd.read_parquet(data_dir / a / "block_access_flags_long.parquet") for a in arms}
    long = {a: pd.read_parquet(data_dir / a / "block_level_long_dataset.parquet") for a in arms}
    bg = {a: pd.read_csv(data_dir / a / "block_group_analysis_dataset.csv", dtype={"block_group_geoid": str}) for a in arms}
    acs = bg["reference"].drop_duplicates("block_group_geoid").set_index("block_group_geoid")[COVARS]

    # Origin attachment and eligibility.
    orig_rows, elig_rows = [], []
    ref0 = full["reference"].loc[full["reference"]["slr_ft"].eq(0)].set_index("block_geoid")
    for a in arms:
        b0 = full[a].loc[full[a]["slr_ft"].eq(0)]
        el = b0.loc[b0["analysis_eligible"]]
        rec = {"attachment_arm": a, **dist_stats(el["origin_snap_distance_m"], "eligible_origin_distance_m")}
        rec.update(dist_stats(el.loc[el["pop20"].gt(0), "origin_snap_distance_m"], "eligible_populated_origin_distance_m"))
        orig_rows.append(rec)
        e = {"attachment_arm": a, "input_blocks": int(len(b0)), "eligible_blocks": int(len(el)),
             "eligible_populated_blocks": int(el["pop20"].gt(0).sum()), "eligible_pop20": int(el["pop20"].sum()),
             "eligible_block_groups": int(el["block_group_geoid"].nunique())}
        for reason, g in b0.loc[~b0["analysis_eligible"]].groupby("exclusion_reason"):
            e[f"excluded_{reason}_blocks"] = int(len(g))
            e[f"excluded_{reason}_pop20"] = int(g["pop20"].sum())
        ref_el = ref0.loc[ref0["analysis_eligible"]]
        e["change_vs_reference_blocks"] = e["eligible_blocks"] - int(len(ref_el))
        e["change_vs_reference_pop20"] = e["eligible_pop20"] - int(ref_el["pop20"].sum())
        elig_rows.append(e)
    al.atomic_to_csv(pd.DataFrame(orig_rows), out / "origin_attachment_summary.csv")
    al.atomic_to_csv(pd.DataFrame(elig_rows), out / "eligibility_comparison.csv")

    # States by arm x SLR (eligible universe of each arm).
    srows = []
    for a in arms:
        l = long[a]
        for s, g in l.groupby("slr_ft"):
            nb, npop = len(g), g["pop20"].sum()
            rec = {"attachment_arm": a, "slr_ft": int(s), "eligible_blocks": int(nb), "eligible_pop20": int(npop)}
            for st in STATES:
                m = g["scenario_status"].eq(st)
                rec[f"{st}_blocks"] = int(m.sum())
                rec[f"{st}_pop20"] = int(g.loc[m, "pop20"].sum())
                rec[f"{st}_share_blocks"] = m.sum() / nb
                rec[f"{st}_share_pop20"] = g.loc[m, "pop20"].sum() / npop
            srows.append(rec)
    states = pd.DataFrame(srows)
    ref_states = states.loc[states["attachment_arm"].eq("reference")].set_index("slr_ft")
    for st in STATES:
        for kind in ["blocks", "pop20"]:
            states[f"{st}_{kind}_diff"] = states[f"{st}_{kind}"] - states["slr_ft"].map(ref_states[f"{st}_{kind}"])
            states[f"{st}_share_{kind}_diff_pp"] = 100 * (states[f"{st}_share_{kind}"] - states["slr_ft"].map(ref_states[f"{st}_share_{kind}"]))
    al.atomic_to_csv(states, out / "state_comparison.csv")

    # Reference-vs-arm crosstab on common eligible origins.
    xrows = []
    ref_l = long["reference"][["block_geoid", "slr_ft", "scenario_status", "baseline_status", "pop20"]]
    for a in arms:
        if a == "reference":
            continue
        m = ref_l.merge(long[a][["block_geoid", "slr_ft", "scenario_status", "baseline_status"]],
                        on=["block_geoid", "slr_ft"], suffixes=("_ref", "_arm"))
        x = (m.groupby(["slr_ft", "scenario_status_ref", "scenario_status_arm"])
             .agg(n_blocks=("block_geoid", "size"), pop20=("pop20", "sum")).reset_index())
        x.insert(0, "attachment_arm", a)
        xrows.append(x)
    al.atomic_to_csv(pd.concat(xrows, ignore_index=True), out / "state_crosstab_vs_reference.csv")

    # Transition and 05-style population totals (baseline-connected origins).
    trows = []
    for a in arms:
        for s, g in long[a].loc[long[a]["slr_ft"].gt(0)].groupby("slr_ft"):
            rec = {"attachment_arm": a, "slr_ft": int(s)}
            for t in TRANSITIONS:
                m = g[t].eq(1)
                rec[f"{t}_blocks"] = int(m.sum())
                rec[f"{t}_pop20"] = int(g.loc[m, "pop20"].sum())
            new_inund = g["baseline_redundant_to_inundated"].eq(1) | g["baseline_fragile_to_inundated"].eq(1)
            new_iso_plus = new_inund | g["baseline_redundant_to_isolated"].eq(1) | g["baseline_fragile_to_isolated"].eq(1)
            new_all = new_iso_plus | g["baseline_redundant_to_fragile"].eq(1)
            for name, mask in [("new_inundated", new_inund), ("new_isolated_or_inundated", new_iso_plus),
                               ("new_fragile_or_worse", new_all), ("added_by_fragile", new_all & ~new_iso_plus)]:
                rec[f"{name}_blocks"] = int(mask.sum())
                rec[f"{name}_pop20"] = int(g.loc[mask, "pop20"].sum())
            non_inund = new_all & ~new_inund
            rec["non_inundation_share_blocks"] = non_inund.sum() / new_all.sum() if new_all.sum() else np.nan
            rec["non_inundation_share_pop20"] = (g.loc[non_inund, "pop20"].sum() / g.loc[new_all, "pop20"].sum()
                                                 if g.loc[new_all, "pop20"].sum() else np.nan)
            trows.append(rec)
    trans = pd.DataFrame(trows)
    ref_t = trans.loc[trans["attachment_arm"].eq("reference")].set_index("slr_ft")
    for c in [c for c in trans.columns if c.endswith(("_blocks", "_pop20"))]:
        trans[f"{c}_diff"] = trans[c] - trans["slr_ft"].map(ref_t[c])
    for c in ["non_inundation_share_blocks", "non_inundation_share_pop20"]:
        trans[f"{c}_diff_pp"] = 100 * (trans[c] - trans["slr_ft"].map(ref_t[c]))
    al.atomic_to_csv(trans, out / "transition_population_comparison.csv")

    # Composition of changed / excluded origins (descriptive; one row per block, baseline).
    comp = []
    ref_base = long["reference"].loc[long["reference"]["slr_ft"].eq(0)].set_index("block_geoid")
    ref_all = ref_base.join(acs, on="block_group_geoid")
    def describe(frame, arm, group):
        rec = {"attachment_arm": arm, "group": group, "blocks": int(len(frame)), "pop20": int(frame["pop20"].sum())}
        for county, n in frame["county_name"].value_counts().items():
            rec[f"blocks_{county}"] = int(n)
        for c in COVARS:
            rec[f"mean_{c}"] = float(frame[c].mean()) if len(frame) else np.nan
        return rec
    comp.append(describe(ref_all, "reference", "all eligible blocks (reference)"))
    for a in arms:
        if a == "reference":
            continue
        arm_base = long[a].loc[long[a]["slr_ft"].eq(0)].set_index("block_geoid")
        excluded = ref_all.loc[~ref_all.index.isin(arm_base.index)]
        if len(excluded):
            comp.append(describe(excluded, a, "excluded vs reference"))
        common = ref_all.index.intersection(arm_base.index)
        changed = common[ref_all.loc[common, "baseline_status"].to_numpy() != arm_base.loc[common, "baseline_status"].to_numpy()]
        comp.append(describe(ref_all.loc[changed], a, "baseline state changed"))
        # any change at any scenario
        m = long["reference"][["block_geoid", "slr_ft", "scenario_status"]].merge(
            long[a][["block_geoid", "slr_ft", "scenario_status"]], on=["block_geoid", "slr_ft"], suffixes=("_r", "_a"))
        any_change = set(m.loc[m["scenario_status_r"] != m["scenario_status_a"], "block_geoid"])
        comp.append(describe(ref_all.loc[ref_all.index.isin(any_change)], a, "state changed at any scenario"))
    al.atomic_to_csv(pd.DataFrame(comp), out / "changed_unit_composition.csv")

    # Spatial review layers.
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    review = out / "spatial_review"
    review.mkdir(parents=True, exist_ok=True)
    gpkg = review / "attachment_review.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    f = fac.loc[fac["change_type"].ne("unchanged") | fac["snap_distance_m"].gt(250)].copy()
    geoms = []
    for r in f.itertuples():
        if np.isfinite(r.facility_x) and np.isfinite(r.x):
            geoms.append(LineString([(r.facility_x, r.facility_y), (r.x, r.y)]))
        elif np.isfinite(r.facility_x):
            geoms.append(Point(r.facility_x, r.facility_y))
        else:
            geoms.append(None)
    f["plausibility_review"] = "unverified: no imagery or road-entrance check performed"
    gpd.GeoDataFrame(f, geometry=geoms, crs="EPSG:32617").to_crs("EPSG:4326").to_file(gpkg, layer="facility_attachments", driver="GPKG")
    o = pd.read_csv(data_dir / "reference" / "origin_snap_audit.csv", dtype={"block_geoid": str})
    o = o.loc[o["snap_distance_m"].gt(500)].copy()
    o["plausibility_review"] = "unverified"
    lines = [LineString([(ox, oy), (nx_, ny_)]) for ox, oy, nx_, ny_ in zip(o["origin_x"], o["origin_y"], o["x"], o["y"])]
    gpd.GeoDataFrame(o, geometry=lines, crs="EPSG:32617").to_crs("EPSG:4326").to_file(
        gpkg, layer="origin_attachments_over_500m", driver="GPKG")
    print("wrote comparison tables to", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
