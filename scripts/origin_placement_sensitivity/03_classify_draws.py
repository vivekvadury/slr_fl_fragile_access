#!/usr/bin/env python
"""Classify randomized origins (and canonical points) with the node-state engine.

Requires 01_prepare_draws.py and 02_build_node_states.py for all seven scenarios.

For each draw and scenario: inundated if the point intersects the cached NOAA
scenario union (the production test), else unclassified if its LCC snap exceeds
2 km, else the snapped node's dry-graph network state. The SAME draw (same
coordinates, same snapped node) is used at 0 ft and every positive scenario;
transitions are always computed within a draw.

Validations written to <out>/validation/:
  V1a canonical_validation.csv / canonical_mismatches.csv
      engine on the canonical representative points vs the canonical della run
      (every block, every scenario)
  V1b production_pilot_comparison.csv
      engine vs UNMODIFIED production scenario_results_for_origins on pilot draws
  V1c (in work/node_states_slr_<s>ft_checks.json, summarized here)
  V2  inundation union vs raw-polygon cross-check; snap universe checks
  V3  transition-flag consistency and pairing checks

Example:
  python scripts/origin_placement_sensitivity/03_classify_draws.py --subset pilot --n-draws 10
  python scripts/origin_placement_sensitivity/03_classify_draws.py --subset all --n-draws 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ops_lib as ol  # noqa: E402

ol.ensure_env()
base, gpd, np, pd = ol.base, ol.gpd, ol.np, ol.pd
S = ol.STATE_CODES


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", choices=["pilot", "all"], required=True)
    p.add_argument("--n-draws", type=int, required=True)
    return p.parse_args()


def out_dir_for(subset: str) -> Path:
    return ol.OUT_DIR / ("pilot" if subset == "pilot" else "full")


def node_state_arrays():
    """state[s] indexed by node_id (node ids are contiguous 0..n-1; asserted)."""
    arrays, checks = {}, {}
    for s in ol.SCENARIOS:
        t = pd.read_parquet(ol.WORK_DIR / f"node_states_slr_{s}ft.parquet", columns=["node_id", "node_state", "raw_in_lcc"])
        ids = t["node_id"].to_numpy()
        assert np.array_equal(ids, np.arange(len(ids))), "node ids are not contiguous"
        arrays[s] = t["node_state"].to_numpy(dtype=np.int8)
        arrays["lcc"] = t["raw_in_lcc"].to_numpy(dtype=bool)
        checks[s] = ol.read_json(ol.WORK_DIR / f"node_states_slr_{s}ft_checks.json")
    return arrays, checks


def main():
    args = parse_args()
    out = out_dir_for(args.subset)
    (out / "validation").mkdir(parents=True, exist_ok=True)
    rss = ol.PeakRSS().start()
    t_all = time.time()
    timings = {}

    def stage(name, t0):
        timings[name] = round(time.time() - t0, 1)
        ol.log(f"{name}: {timings[name]} s (peak RSS {rss.peak_gib} GiB)")

    t0 = time.time()
    meta = pd.read_parquet(ol.WORK_DIR / "block_metadata.parquet")
    draws = pd.read_parquet(ol.WORK_DIR / "draws.parquet")
    draws = draws.loc[draws["rep"] < args.n_draws]
    if args.subset == "pilot":
        draws = draws.loc[draws["pilot_block"]]
    draws = draws.sort_values("draw_id").reset_index(drop=True)
    node_state, node_checks = node_state_arrays()
    stage("load", t0)

    # ---- V1c / bridge summaries from 02 --------------------------------------------------------
    v1c = pd.DataFrame([{
        "slr_ft": s,
        "bridge_structure_rows_mismatch": c["bridge_check"]["row_alignment_mismatch"],
        "bridge_retained_mismatch": c["bridge_check"]["retained_mismatch"],
        "bridge_dry_landing_count_mismatch": c["bridge_check"]["dry_landing_count_mismatch"],
        "n_total_edges_removed": c["bridge_check"]["n_total_edges_removed"],
        "lcc_nodes_checked": c.get("exhaustive_node_equivalence", {}).get("n_lcc_nodes_checked"),
        "lcc_node_state_mismatch": c.get("exhaustive_node_equivalence", {}).get("n_mismatch"),
        "production_other_state_nodes": c["node_table"]["production_other_state_nodes"],
        "node_table_peak_rss_gib": c["peak_rss_gib"], "node_table_seconds": c["total_s"],
    } for s, c in node_checks.items()])
    ol.atomic_to_csv(v1c, out / "validation" / "node_engine_equivalence.csv")

    # ---- V1a: canonical points through the engine -----------------------------------------------
    t0 = time.time()
    canon = ol.load_canonical_long(columns=["block_geoid", "slr_ft", "scenario_status", "baseline_status",
                                            "block_centroid_inundated", "origin_node_id", "analysis_eligible"])
    cll = ol.to_lonlat(meta["canonical_x"].to_numpy(), meta["canonical_y"].to_numpy())
    c_valid = meta["canonical_snap_valid"].to_numpy(dtype=bool)
    c_node = meta["canonical_node_id"].to_numpy(dtype=np.int64)
    engine_rows = []
    unions = {}
    for s in ol.SCENARIOS:
        unions[s] = ol.load_union(s)
        inund = ol.inundated_by_union(unions[s], cll)
        st = ol.classify(inund, c_valid, node_state[s][c_node])
        engine_rows.append(pd.DataFrame({"block_geoid": meta["block_geoid"].to_numpy(), "slr_ft": s,
                                         "engine_status": pd.Series(st).map(ol.CODE_STATES).to_numpy(),
                                         "engine_inundated": inund}))
    eng = pd.concat(engine_rows, ignore_index=True)
    cmp = canon.merge(eng, on=["block_geoid", "slr_ft"], how="outer", indicator=True, validate="one_to_one")
    assert (cmp["_merge"] == "both").all()
    cmp["match"] = cmp["scenario_status"].astype(str).eq(cmp["engine_status"])
    v1a = (cmp.groupby(["slr_ft", "analysis_eligible", "scenario_status"], observed=True)
           .agg(n_blocks=("match", "size"), n_match=("match", "sum")).reset_index())
    v1a["n_mismatch"] = v1a["n_blocks"] - v1a["n_match"]
    ol.atomic_to_csv(v1a, out / "validation" / "canonical_validation.csv")
    ol.atomic_to_csv(v1a, out / "canonical_validation.csv")
    ol.atomic_to_csv(cmp.loc[~cmp["match"]], out / "validation" / "canonical_mismatches.csv")
    v1a_summary = {"rows": int(len(cmp)), "mismatches": int((~cmp["match"]).sum()),
                   "eligible_mismatches": int((~cmp["match"] & cmp["analysis_eligible"]).sum()),
                   "inundation_flag_mismatches": int((cmp["engine_inundated"].astype(int)
                                                      != cmp["block_centroid_inundated"].astype(int)).sum())}
    ol.log(f"V1a canonical reproduction: {v1a_summary}")
    stage("v1a_canonical", t0)

    # ---- classify draws --------------------------------------------------------------------------
    t0 = time.time()
    dll = gpd.GeoSeries(gpd.points_from_xy(draws["lon"], draws["lat"]), crs="OGC:CRS84")
    valid = draws["snap_valid"].to_numpy(dtype=bool)
    node = draws["node_id"].to_numpy(dtype=np.int64)
    states = pd.DataFrame({"draw_id": draws["draw_id"].to_numpy(), "block_geoid": draws["block_geoid"].to_numpy(),
                           "rep": draws["rep"].to_numpy()})
    raw_cmp = []
    for s in ol.SCENARIOS:
        inund = ol.inundated_by_union(unions[s], dll)
        states[f"inund_{s}"] = inund
        states[f"state_{s}"] = ol.classify(inund, valid, node_state[s][node])
        raw = pd.read_parquet(ol.WORK_DIR / f"raw_polygon_inundation_slr_{s}ft.parquet")
        raw_ids = set(raw.loc[raw["kind"].eq("draw"), "key"].astype(np.int64))
        raw_flag = np.fromiter((d in raw_ids for d in states["draw_id"].to_numpy()), dtype=bool, count=len(states))
        raw_c = set(raw.loc[raw["kind"].eq("canonical"), "key"])
        canon_union = eng.loc[eng["slr_ft"].eq(s)].set_index("block_geoid")["engine_inundated"]
        raw_cmp.append({"slr_ft": s, "draws": int(len(states)), "union_inundated": int(inund.sum()),
                        "raw_polygon_inundated": int(raw_flag.sum()),
                        "draw_disagreements": int((raw_flag != inund).sum()),
                        "canonical_point_disagreements": int((canon_union.index.isin(raw_c) != canon_union.to_numpy()).sum())})
    del unions
    raw_cmp = pd.DataFrame(raw_cmp)
    ol.atomic_to_csv(raw_cmp, out / "validation" / "inundation_union_vs_raw_polygons.csv")
    ol.log(f"inundation cross-check:\n{raw_cmp.to_string(index=False)}")
    stage("classify_draws", t0)

    # ---- V1b: engine vs production function on pilot draws ----------------------------------------
    t0 = time.time()
    v1b = []
    for s in ol.SCENARIOS:
        prod = pd.read_parquet(ol.WORK_DIR / f"production_pilot_slr_{s}ft.parquet")
        m = states[["draw_id", f"state_{s}"]].merge(prod, on="draw_id", how="inner")
        m["engine_status"] = m[f"state_{s}"].map(ol.CODE_STATES)
        for st, g in m.groupby("production_status"):
            v1b.append({"slr_ft": s, "production_status": st, "n_draws": int(len(g)),
                        "n_mismatch": int((g["engine_status"] != g["production_status"]).sum())})
    v1b = pd.DataFrame(v1b)
    ol.atomic_to_csv(v1b, out / "validation" / "production_pilot_comparison.csv")
    ol.log(f"V1b engine vs production on pilot draws: {int(v1b['n_draws'].sum())} draw-scenarios, "
           f"{int(v1b['n_mismatch'].sum())} mismatches")
    stage("v1b_production_pilot", t0)

    # ---- V2: snap universe and geometry (re-asserted on the classified subset) --------------------
    lcc_ok = node_state["lcc"][node]
    v2 = {"draws": int(len(draws)), "blocks": int(draws["block_geoid"].nunique()),
          "snapped_node_not_in_raw_lcc": int((~lcc_ok).sum()),
          "snap_valid_vs_threshold_mismatch": int((valid != (draws["snap_distance_m"].to_numpy() <= base.MAX_ORIGIN_SNAP_M)).sum()),
          "snap_invalid_draws": int((~valid).sum()),
          "snap_invalid_and_inundated_at_0ft": int((~valid & states["inund_0"].to_numpy()).sum()),
          "duplicate_block_rep": int(draws.duplicated(["block_geoid", "rep"]).sum())}

    # ---- transitions (paired within draw) + V3 ------------------------------------------------------
    t0 = time.time()
    b = states["state_0"].to_numpy()
    trans_frames, v3 = [], []
    names = list(ol.TRANSITIONS)
    for s in ol.POSITIVE_SCENARIOS:
        sc = states[f"state_{s}"].to_numpy()
        flags = ol.transition_flags(b, sc)
        tf = pd.DataFrame({"draw_id": states["draw_id"].to_numpy(), "slr_ft": np.int8(s),
                           "baseline_state": b, "scenario_state": sc,
                           **{k: v for k, v in flags.items()}})
        # V3: independent recomputation from string labels, and invalid handling.
        bl = pd.Series(b).map(ol.CODE_STATES)
        sl = pd.Series(sc).map(ol.CODE_STATES)
        indep = {n: (bl.eq(o) & sl.isin(d)).to_numpy() for n, (o, d) in ol.TRANSITIONS.items()}
        v3.append({
            "slr_ft": s,
            "flag_mismatch_vs_label_recompute": int(sum((indep[n] != flags[n]).sum() for n in names)),
            "flags_set_with_unclassified_baseline": int(sum((flags[n] & (b == S["unclassified"])).sum() for n in names)),
            "flags_set_with_unclassified_scenario": int(sum((flags[n] & (sc == S["unclassified"])).sum() for n in names)),
            "five_transitions_overlap": int((np.vstack([flags[n] for n in ol.FIVE_TRANSITIONS]).sum(axis=0) > 1).sum()),
            "red_to_worse_ne_sum_of_three": int((flags["red_to_worse"] != (flags["red_to_fragile"] | flags["red_to_isolated"]
                                                                            | flags["red_to_inundated"])).sum()),
            "fragile_to_worse_ne_sum_of_two": int((flags["fragile_to_worse"] != (flags["fragile_to_isolated"]
                                                                                 | flags["fragile_to_inundated"])).sum()),
            "draws_with_unclassified_baseline": int((b == S["unclassified"]).sum()),
            "draws_with_unclassified_scenario": int((sc == S["unclassified"]).sum()),
            "unclassified_baseline_but_valid_scenario": int(((b == S["unclassified"]) & (sc != S["unclassified"]) & (sc != S["inundated"])).sum()),
        })
        trans_frames.append(tf)
    trans = pd.concat(trans_frames, ignore_index=True)
    # Pairing: each draw has exactly one baseline state; the transition table carries it unchanged.
    pair_chk = trans.merge(states[["draw_id", "state_0"]], on="draw_id")
    v3_df = pd.DataFrame(v3)
    v3_df["baseline_state_not_from_same_draw"] = int((pair_chk["baseline_state"] != pair_chk["state_0"]).sum())
    ol.atomic_to_csv(v3_df, out / "validation" / "transition_consistency.csv")
    ol.log(f"V3 transition consistency:\n{v3_df.to_string(index=False)}")
    stage("transitions", t0)

    # ---- write ------------------------------------------------------------------------------------------
    t0 = time.time()
    snap_cols = ["draw_id", "block_geoid", "rep", "seed", "x", "y", "lon", "lat", "land_area_geom_m2",
                 "land_area_m2_census", "node_id", "snap_distance_m", "snap_valid"]
    ol.atomic_to_parquet(draws[snap_cols], out / "random_origin_snaps.parquet")
    ol.atomic_to_parquet(states, out / "random_origin_states.parquet")
    ol.atomic_to_parquet(trans, out / "random_origin_transitions.parquet")
    ol.atomic_to_parquet(eng, out / "canonical_engine_states.parquet")
    stage("write", t0)
    rss.stop()
    manifest = {
        "script": "scripts/origin_placement_sensitivity/03_classify_draws.py", "subset": args.subset,
        "n_draws": args.n_draws, "seed": int(draws["seed"].iloc[0]),
        "state_codes": ol.STATE_CODES, "v1a": v1a_summary,
        "v1b_total_mismatch": int(v1b["n_mismatch"].sum()), "v1b_draw_scenarios": int(v1b["n_draws"].sum()),
        "v1c_total_node_mismatch": int(v1c["lcc_node_state_mismatch"].fillna(0).sum()),
        "v1c_bridge_mismatch": int(v1c[["bridge_structure_rows_mismatch", "bridge_retained_mismatch",
                                         "bridge_dry_landing_count_mismatch"]].sum().sum()),
        "v2": v2, "inundation_cross_check_disagreements": int(raw_cmp["draw_disagreements"].sum()),
        "v3_total_issues": int(v3_df[["flag_mismatch_vs_label_recompute", "flags_set_with_unclassified_baseline",
                                      "flags_set_with_unclassified_scenario", "five_transitions_overlap",
                                      "red_to_worse_ne_sum_of_three", "fragile_to_worse_ne_sum_of_two",
                                      "baseline_state_not_from_same_draw"]].sum().sum()),
        "timings_s": timings, "total_s": round(time.time() - t_all, 1), "peak_rss_gib": rss.peak_gib,
    }
    ol.atomic_write_json(out / "classify_manifest.json", manifest)
    ol.log(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
