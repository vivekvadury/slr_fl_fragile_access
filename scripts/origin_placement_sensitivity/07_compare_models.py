#!/usr/bin/env python
"""Summarize demographic-only transition models across origin replications.

Reads <run>/bg_model_estimates.csv (05_bg_models.R) and writes
  <run>/bg_model_reproduction_check.csv  canonical-aggregation fit vs production coefficients
  <run>/bg_model_sensitivity.csv         per transition x term: canonical vs MC distribution
  <run>/bg_outcome_sensitivity_summary.csv  block-group transition shares, canonical vs MC

No inferential claims: the clustered z is descriptive. Variation across
replications is origin-placement sensitivity of the point estimates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ops_lib as ol  # noqa: E402

np, pd = ol.np, ol.pd
SOCIAL = {"z_pct_black_nh": "Black share", "z_pct_hispanic": "Hispanic share", "z_renter_share": "Renter share",
          "z_log_median_income": "Log median income", "z_pct_age_65plus": "Age 65+ share",
          "z_no_vehicle_share": "No-vehicle household share"}


def validate_estimates(est, prod, n_draws):
    """Reject incomplete fits before publishing sensitivity summaries."""
    if est["fit_error"].isna().any() or est["fit_error"].astype(bool).any():
        raise ValueError("Model fits failed or have missing fit status; inspect bg_model_estimates.csv.")
    expected_reps = {"canonical", *(f"{r:03d}" for r in range(n_draws))}
    if set(est["rep"]) != expected_reps:
        raise ValueError("Model replications do not match classify_manifest.json.")
    keys = ["transition", "term"]
    expected_keys = set(prod[keys].itertuples(index=False, name=None))
    if not expected_keys or prod.duplicated(keys).any() or est.duplicated(["rep", *keys]).any():
        raise ValueError("Empty or duplicate model coefficient keys.")
    for rep, group in est.groupby("rep"):
        if set(group[keys].itertuples(index=False, name=None)) != expected_keys:
            raise ValueError(f"Incomplete model coefficients for replication {rep}.")
    if not np.isfinite(est[["coef", "coef_se_clustered"]].to_numpy()).all():
        raise ValueError("Non-finite model coefficients or standard errors.")
    can = est.loc[est["rep"].eq("canonical")].merge(prod, on=keys, validate="one_to_one")
    for actual, expected in [("coef", "estimate"), ("coef_se_clustered", "std.error")]:
        if not np.allclose(can[actual], can[expected], rtol=0, atol=1e-10):
            raise ValueError(f"Canonical {actual} does not reproduce production within 1e-10.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="full")
    run = ol.OUT_DIR / p.parse_args().run
    est = pd.read_csv(run / "bg_model_estimates.csv", dtype={"rep": str})
    prod = pd.read_csv(ol.PROJECT_ROOT / "outputs/tables/transition_model_coefficients_approach.csv")
    validate_estimates(est, prod, ol.read_json(run / "classify_manifest.json")["n_draws"])
    failed = est.loc[est["fit_error"].astype(bool), "rep"].unique().tolist()
    est = est.loc[~est["fit_error"].astype(bool)]
    est["z_clustered"] = est["coef"] / est["coef_se_clustered"]

    can = est.loc[est["rep"].eq("canonical")]
    chk = can.merge(prod[["transition", "term", "estimate", "std.error"]], on=["transition", "term"], how="outer", indicator=True)
    chk["abs_diff_coef"] = (chk["coef"] - chk["estimate"]).abs()
    chk["abs_diff_se"] = (chk["coef_se_clustered"] - chk["std.error"]).abs()
    ol.atomic_to_csv(chk, run / "bg_model_reproduction_check.csv")
    ol.log(f"reproduction: rows={len(chk)}, unmatched={int((chk['_merge'] != 'both').sum())}, "
           f"max |coef diff|={chk['abs_diff_coef'].max():.3g}, max |se diff|={chk['abs_diff_se'].max():.3g}")

    reps = est.loc[~est["rep"].eq("canonical")]
    rows = []
    for (tr, term), g in reps.groupby(["transition", "term"]):
        c = can.loc[(can["transition"] == tr) & (can["term"] == term)].iloc[0]
        rows.append({
            "transition": tr, "term": term, "term_label": SOCIAL.get(term, term), "n_replications": int(len(g)),
            "canonical_coef": c["coef"], "mc_coef_mean": g["coef"].mean(), "mc_coef_p2_5": g["coef"].quantile(0.025),
            "mc_coef_p97_5": g["coef"].quantile(0.975), "mc_coef_min": g["coef"].min(), "mc_coef_max": g["coef"].max(),
            "share_reps_same_sign_as_canonical": float((np.sign(g["coef"]) == np.sign(c["coef"])).mean()),
            "canonical_ame": c["ame"], "mc_ame_mean": g["ame"].mean(), "mc_ame_p2_5": g["ame"].quantile(0.025),
            "mc_ame_p97_5": g["ame"].quantile(0.975), "mc_ame_min": g["ame"].min(), "mc_ame_max": g["ame"].max(),
            "ame_rel_dev_mean_vs_canonical": (g["ame"].mean() - c["ame"]) / abs(c["ame"]) if c["ame"] else np.nan,
            "canonical_z_clustered": c["z_clustered"], "mc_z_clustered_min": g["z_clustered"].min(),
            "mc_z_clustered_max": g["z_clustered"].max(),
            "share_reps_abs_z_ge_1_96_descriptive": float((g["z_clustered"].abs() >= 1.96).mean()),
            "canonical_risk_set_blocks": c["risk_set_blocks"], "mc_risk_set_blocks_mean": g["risk_set_blocks"].mean(),
            "canonical_event_blocks": c["event_blocks"], "mc_event_blocks_mean": g["event_blocks"].mean(),
            "canonical_n_obs": c["n_obs"], "mc_n_obs_mean": g["n_obs"].mean(),
        })
    sens = pd.DataFrame(rows)
    # Relative ordering of the six social terms within each transition (by AME).
    order_rows = []
    for tr, g in reps.loc[reps["term"].isin(SOCIAL)].groupby("transition"):
        c_rank = can.loc[(can["transition"] == tr) & can["term"].isin(SOCIAL)].set_index("term")["ame"].rank()
        agree = []
        for r, gr in g.groupby("rep"):
            agree.append(gr.set_index("term")["ame"].rank().reindex(c_rank.index).corr(c_rank, method="spearman"))
        order_rows.append({"transition": tr, "mean_spearman_rank_corr_with_canonical": float(np.mean(agree)),
                           "min_spearman_rank_corr_with_canonical": float(np.min(agree))})
    sens = sens.merge(pd.DataFrame(order_rows), on="transition", how="left")
    ol.atomic_to_csv(sens, run / "bg_model_sensitivity.csv")

    bgs = pd.read_csv(run / "block_group_sensitivity.csv", dtype={"block_group_geoid": str})
    summ = (bgs.loc[bgs["slr_ft"].gt(0) | bgs["quantity"].isin(["baseline_redundant_n", "baseline_fragile_n"])]
            .groupby("quantity").apply(lambda g: pd.Series({
                "bg_slr_rows_defined": int(g["canonical"].notna().sum()),
                "mean_abs_dev_mc_mean_vs_canonical": float(g["abs_dev_mean_minus_canonical"].abs().mean()),
                "median_abs_dev_mc_mean_vs_canonical": float(g["abs_dev_mean_minus_canonical"].abs().median()),
                "share_rows_canonical_outside_mc_range": float(((g["canonical"] < g["mc_min"]) | (g["canonical"] > g["mc_max"]))
                                                             [g["canonical"].notna() & g["mc_min"].notna()].mean()),
                "corr_canonical_vs_mc_mean": float(g[["canonical", "mc_mean"]].dropna().corr().iloc[0, 1]),
            }), include_groups=False).reset_index())
    ol.atomic_to_csv(summ, run / "bg_outcome_sensitivity_summary.csv")
    ol.atomic_write_json(run / "bg_model_manifest.json", {"failed_reps": failed, "n_reps_fitted": int(reps["rep"].nunique()),
                                                          "max_abs_coef_diff_vs_production": float(chk["abs_diff_coef"].max()),
                                                          "max_abs_se_diff_vs_production": float(chk["abs_diff_se"].max())})
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(sens.loc[sens["term"].isin(SOCIAL), ["transition", "term_label", "canonical_ame", "mc_ame_mean", "mc_ame_p2_5",
                                                "mc_ame_p97_5", "share_reps_same_sign_as_canonical",
                                                "share_reps_abs_z_ge_1_96_descriptive", "canonical_z_clustered"]].round(4).to_string(index=False))
    print(summ.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
