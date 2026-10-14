#!/usr/bin/env python
"""Stability summaries, replication aggregates, correlates, figures, BG datasets.

Reads the outputs of 03_classify_draws.py for one run (``pilot`` or ``full``).
Population and eligibility follow the manuscript exactly: the universe is the
canonical analysis-eligible blocks, every draw of a block carries that block's
POP20, and a replication r is the set of every block's r-th draw (one
measurement per block). Population is never multiplied by the number of draws.

Denominators
  all-draw   : every draw of the block (N), failed snaps included
  snap-valid : draws whose LCC snap is within 2 km
Failed snaps keep the canonical rule (``unclassified`` unless the point is
inundated) and are reported explicitly, never dropped.

Example: python scripts/origin_placement_sensitivity/04_summarize.py --run full
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
S, CS = ol.STATE_CODES, ol.CODE_STATES
STATES5 = ["redundant", "fragile", "isolated", "inundated", "unclassified"]
ADVERSE5 = ol.FIVE_TRANSITIONS
ACS_COLUMNS = ["total_pop", "median_income", "median_age", "hh_no_vehicle", "no_vehicle_share",
               "pop_age_65plus", "pct_age_65plus", "pct_white_nh", "pct_black_nh", "pct_hispanic",
               "pct_nonwhite", "renter_share", "poverty_rate", "log_median_income"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", choices=["pilot", "full"], required=True)
    p.add_argument("--moran-permutations", type=int, default=199)
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--skip-bg", action="store_true", help="Do not (re)write block-group replication datasets.")
    return p.parse_args()


# ---------------------------------------------------------------------------------------------
# Loading: block x draw state matrices
# ---------------------------------------------------------------------------------------------

def load_matrices(run_dir):
    states = pd.read_parquet(run_dir / "random_origin_states.parquet")
    snaps = pd.read_parquet(run_dir / "random_origin_snaps.parquet",
                            columns=["draw_id", "snap_valid", "snap_distance_m"])
    states = states.merge(snaps, on="draw_id", validate="one_to_one").sort_values(["block_geoid", "rep"])
    n = int(states["rep"].max()) + 1
    counts = states.groupby("block_geoid").size()
    assert counts.eq(n).all(), "every block must have the same number of draws"
    assert (states["rep"].to_numpy().reshape(-1, n) == np.arange(n)).all()
    blocks = counts.index.to_numpy()
    mats = {s: states[f"state_{s}"].to_numpy(dtype=np.int8).reshape(-1, n) for s in ol.SCENARIOS}
    valid = states["snap_valid"].to_numpy(dtype=bool).reshape(-1, n)
    snapd = states["snap_distance_m"].to_numpy().reshape(-1, n)
    return blocks, n, mats, valid, snapd


def load_canonical(blocks):
    canon = ol.load_canonical_long(columns=["block_geoid", "block_group_geoid", "tract_geoid", "county_fips",
                                            "county_name", "slr_ft", "scenario_status", "pop20",
                                            "analysis_eligible", "bridge_rule_applied"])
    canon = canon.loc[canon["analysis_eligible"]]
    assert set(canon["bridge_rule_applied"]) == {"approach"}
    wide = canon.pivot(index="block_geoid", columns="slr_ft", values="scenario_status").loc[blocks]
    cmat = {s: wide[s].map(S).to_numpy(dtype=np.int8) for s in ol.SCENARIOS}
    info = canon.loc[canon["slr_ft"].eq(0)].set_index("block_geoid").loc[
        blocks, ["block_group_geoid", "tract_geoid", "county_fips", "county_name", "pop20"]]
    return cmat, info.reset_index()


# ---------------------------------------------------------------------------------------------
# A. Block-level state stability
# ---------------------------------------------------------------------------------------------

def state_counts(mat):
    return np.stack([(mat == S[k]).sum(axis=1) for k in STATES5], axis=1)


def modal(counts):
    mx = counts.max(axis=1)
    order = np.argmax(counts, axis=1)  # ties -> earlier in STATES5 (deterministic)
    tie = (counts == mx[:, None]).sum(axis=1) > 1
    return order, tie


def block_state_stability(blocks, info, n, mats, cmat, valid):
    rows = []
    for s in ol.SCENARIOS:
        m, c = mats[s], cmat[s]
        cnt = state_counts(m)
        match = m == c[:, None]
        nvalid = valid.sum(axis=1)
        mode_idx, tie = modal(cnt)
        canon_idx = np.array([STATES5.index(CS[v]) for v in c])
        df = pd.DataFrame({
            "block_geoid": blocks, "county_name": info["county_name"].to_numpy(), "pop20": info["pop20"].to_numpy(),
            "slr_ft": s, "canonical_state": [CS[v] for v in c], "n_draws": n,
            "n_snap_valid_draws": nvalid, "snap_valid_rate": nvalid / n,
            **{f"n_{k}": cnt[:, i] for i, k in enumerate(STATES5)},
            **{f"p_{k}_all": cnt[:, i] / n for i, k in enumerate(STATES5)},
            "share_match_canonical_all": match.mean(axis=1),
            "share_match_canonical_snap_valid": np.where(nvalid > 0, (match & valid).sum(axis=1) / np.maximum(nvalid, 1), np.nan),
            "modal_state": [STATES5[i] for i in mode_idx], "modal_tie": tie,
            "canonical_is_modal": cnt[np.arange(len(c)), canon_idx] == cnt.max(axis=1),
            "n_distinct_states": (cnt > 0).sum(axis=1),
        })
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------------------------
# B. Block-level transition stability (paired within draw)
# ---------------------------------------------------------------------------------------------

def adverse_any(b, sc):
    f = ol.transition_flags(b, sc)
    return np.logical_or.reduce([f[k] for k in ADVERSE5]), f


def block_transition_stability(blocks, info, n, mats, cmat):
    rows = []
    cb = cmat[0]
    b = mats[0]
    for s in ol.POSITIVE_SCENARIOS:
        sc, cs = mats[s], cmat[s]
        c_adv, _ = adverse_any(cb, cs)
        d_adv, dflags = adverse_any(b.ravel(), sc.ravel())
        d_adv = d_adv.reshape(b.shape)
        base_match = b == cb[:, None]
        exact = base_match & (sc == cs[:, None])
        code = b.astype(np.int16) * 10 + sc  # transition code per draw
        ccode = cb.astype(np.int16) * 10 + cs
        # most common alternative transition (excluding the canonical one)
        alt_code, alt_share = np.full(len(cb), -1), np.zeros(len(cb))
        uniq = np.unique(code)
        cnt = np.stack([(code == u).sum(axis=1) for u in uniq], axis=1)
        cnt_alt = np.where(uniq[None, :] == ccode[:, None], -1, cnt)
        j = cnt_alt.argmax(axis=1)
        has_alt = cnt_alt.max(axis=1) > 0
        alt_code[has_alt] = uniq[j[has_alt]]
        alt_share[has_alt] = cnt_alt.max(axis=1)[has_alt] / n
        nbm = base_match.sum(axis=1)
        df = pd.DataFrame({
            "block_geoid": blocks, "county_name": info["county_name"].to_numpy(), "pop20": info["pop20"].to_numpy(),
            "slr_ft": s, "canonical_baseline_state": [CS[v] for v in cb], "canonical_scenario_state": [CS[v] for v in cs],
            "canonical_transition": [f"{CS[x]}->{CS[y]}" for x, y in zip(cb, cs)], "canonical_adverse": c_adv,
            "n_draws": n,
            "share_baseline_match": base_match.mean(axis=1),
            "share_exact_transition": exact.mean(axis=1),
            "share_scenario_match_given_baseline_match": np.where(nbm > 0, exact.sum(axis=1) / np.maximum(nbm, 1), np.nan),
            "share_draws_adverse": d_adv.mean(axis=1),
            "share_adverse_outcome_match": (d_adv == c_adv[:, None]).mean(axis=1),
            "n_distinct_transitions": (cnt > 0).sum(axis=1),
            "top_alternative_transition": [f"{CS[c // 10]}->{CS[c % 10]}" if c >= 0 else "" for c in alt_code],
            "top_alternative_share": alt_share,
            **{f"p_{k}": dflags[k].reshape(b.shape).mean(axis=1) for k in ol.TRANSITIONS},
        })
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------------------------
# C. Aggregates for one complete replication (or the canonical assignment)
# ---------------------------------------------------------------------------------------------

def aggregate_one(states_by_s, pop):
    """Manuscript definitions (05_population_figures.py / notebook 03)."""
    out = {}
    nb = len(pop)
    for s in ol.SCENARIOS:
        st = states_by_s[s]
        for k in STATES5:
            m = st == S[k]
            out[("blocks", k, s)] = m.sum()
            out[("share_blocks", k, s)] = m.sum() / nb
            out[("pop", k, s)] = pop[m].sum()
            out[("share_pop", k, s)] = pop[m].sum() / pop.sum()
    b = states_by_s[0]
    for s in ol.POSITIVE_SCENARIOS:
        f = ol.transition_flags(b, states_by_s[s])
        for k, v in f.items():
            out[("blocks", k, s)] = v.sum()
            out[("pop", k, s)] = pop[v].sum()
        new_inund = f["red_to_inundated"] | f["fragile_to_inundated"]
        new_iso = new_inund | f["red_to_isolated"] | f["fragile_to_isolated"]
        new_all = new_iso | f["red_to_fragile"]
        added = new_all & ~new_iso
        for k, v in {"new_inundated": new_inund, "new_isolated_or_inundated": new_iso,
                     "new_fragile_or_worse": new_all, "added_by_fragile": added}.items():
            out[("blocks", k, s)] = v.sum()
            out[("pop", k, s)] = pop[v].sum()
        out[("share_pop", "added_by_fragile_of_new_fragile_or_worse", s)] = (
            pop[added].sum() / pop[new_all].sum() if pop[new_all].sum() else np.nan)
        out[("share_blocks", "added_by_fragile_of_new_fragile_or_worse", s)] = (
            added.sum() / new_all.sum() if new_all.sum() else np.nan)
    return out


def replication_table(n, mats, cmat, pop):
    recs = []
    for r in range(n):
        agg = aggregate_one({s: mats[s][:, r] for s in ol.SCENARIOS}, pop)
        recs += [{"rep": r, "measure": a, "quantity": q, "slr_ft": s, "value": float(v)} for (a, q, s), v in agg.items()]
    agg = aggregate_one(cmat, pop)
    canon = pd.DataFrame([{"measure": a, "quantity": q, "slr_ft": s, "canonical": float(v)} for (a, q, s), v in agg.items()])
    return pd.DataFrame(recs), canon


def summarize_reps(reps, canon):
    g = reps.groupby(["measure", "quantity", "slr_ft"])["value"]
    summ = pd.DataFrame({
        "mc_mean": g.mean(), "mc_sd": g.std(ddof=1), "mc_p2_5": g.quantile(0.025), "mc_p97_5": g.quantile(0.975),
        "mc_min": g.min(), "mc_max": g.max(), "n_replications": g.size()}).reset_index()
    out = canon.merge(summ, on=["measure", "quantity", "slr_ft"], how="outer", validate="one_to_one")
    out["abs_dev_mean_minus_canonical"] = out["mc_mean"] - out["canonical"]
    out["rel_dev_mean_vs_canonical"] = np.where(out["canonical"] != 0, out["abs_dev_mean_minus_canonical"] / out["canonical"], np.nan)
    out["canonical_inside_mc_interval"] = (out["canonical"] >= out["mc_p2_5"]) & (out["canonical"] <= out["mc_p97_5"])
    out["canonical_rank_pct"] = np.nan
    for i, row in out.iterrows():
        v = reps.loc[(reps["measure"] == row["measure"]) & (reps["quantity"] == row["quantity"])
                     & (reps["slr_ft"] == row["slr_ft"]), "value"]
        if len(v):
            out.at[i, "canonical_rank_pct"] = float((v < row["canonical"]).mean() + 0.5 * (v == row["canonical"]).mean())
    return out.sort_values(["measure", "quantity", "slr_ft"]).reset_index(drop=True)


def check_against_fig4(canon):
    """Canonical aggregates must reproduce the manuscript fig4 tables."""
    tabs = ol.PROJECT_ROOT / "outputs" / "tables"
    cum = pd.read_csv(tabs / "fig4_cumulative_population_by_slr_approach.csv")
    tr = pd.read_csv(tabs / "fig4_transition_population_by_slr_approach.csv")
    st = pd.read_csv(tabs / "fig4_status_population_by_slr_approach.csv")
    c = canon.set_index(["measure", "quantity", "slr_ft"])["canonical"]
    bad = []
    for _, r in cum.iterrows():
        for k in ["new_inundated", "new_isolated_or_inundated", "new_fragile_or_worse", "added_by_fragile"]:
            for meas, col in [("pop", f"{k}_pop20"), ("blocks", f"{k}_blocks")]:
                if c[(meas, k, int(r["slr_ft"]))] != r[col]:
                    bad.append((meas, k, int(r["slr_ft"]), c[(meas, k, int(r["slr_ft"]))], r[col]))
    lab = {"Redundant to fragile": "red_to_fragile", "Redundant to isolated": "red_to_isolated",
           "Redundant to inundated": "red_to_inundated", "Fragile to isolated": "fragile_to_isolated",
           "Fragile to inundated": "fragile_to_inundated"}
    for _, r in tr.iterrows():
        k = lab[r["transition"]]
        for meas, col in [("pop", "pop20"), ("blocks", "n_blocks")]:
            if c[(meas, k, int(r["slr_ft"]))] != r[col]:
                bad.append((meas, k, int(r["slr_ft"]), c[(meas, k, int(r["slr_ft"]))], r[col]))
    for _, r in st.iterrows():
        for meas, col in [("pop", "pop20"), ("blocks", "n_blocks")]:
            if c[(meas, r["scenario_status"], int(r["slr_ft"]))] != r[col]:
                bad.append((meas, r["scenario_status"], int(r["slr_ft"]), c[(meas, r["scenario_status"], int(r["slr_ft"]))], r[col]))
    return {"values_checked": int(len(cum) * 8 + len(tr) * 2 + len(st) * 2), "mismatches": bad}


# ---------------------------------------------------------------------------------------------
# D. Correlates and spatial concentration
# ---------------------------------------------------------------------------------------------

def binned(series, edges, labels):
    return pd.cut(series, bins=edges, labels=labels, right=False, include_lowest=True).astype(str)


def instability_correlates(bss, bts, meta, snapd):
    m = meta.set_index("block_geoid")
    blocks_order = bss.loc[bss["slr_ft"].eq(0), "block_geoid"].to_numpy()
    feat = pd.DataFrame({"block_geoid": blocks_order})
    mm = m.loc[blocks_order]
    feat["land_area_bin"] = pd.qcut(mm["land_area_m2"].to_numpy(), 10, labels=[f"D{i}" for i in range(1, 11)]).astype(str)
    feat["county"] = mm["county_name"].to_numpy()
    feat["coastal_1km"] = np.where(mm["coastal"].to_numpy(), "coastal (<=1 km)", "inland (>1 km)")
    wf = (mm["water_area_m2_census"] / (mm["water_area_m2_census"] + mm["land_area_m2"])).to_numpy()
    feat["water_fraction"] = binned(pd.Series(wf), [0, 1e-12, 0.1, 0.5, 1.0001], ["0", "(0,0.1)", "[0.1,0.5)", ">=0.5"]).to_numpy()
    edges_d = [0, 25, 50, 100, 200, 500, 2000.0001, np.inf]
    lab_d = ["<25", "25-50", "50-100", "100-200", "200-500", "500-2000", ">2000"]
    feat["canonical_snap_distance_m"] = binned(mm["canonical_snap_distance_m"].reset_index(drop=True), edges_d, lab_d).to_numpy()
    feat["median_draw_snap_distance_m"] = binned(pd.Series(np.median(snapd, axis=1)), edges_d, lab_d).to_numpy()
    feat["n_raw_2ecc_inside"] = binned(pd.Series(mm["n_raw_2ecc_inside"].to_numpy()), [0, 1, 2, 3, 6, 11, np.inf],
                                       ["0", "1", "2", "3-5", "6-10", ">10"]).to_numpy()
    rows = []

    def add(df, var, level_col, slr):
        g = df.groupby(level_col)
        for lvl, x in g:
            r = {"variable": var, "level": lvl, "slr_ft": slr, "n_blocks": int(len(x)), "pop20": int(x["pop20"].sum()),
                 "mean_state_agreement": x["share_match_canonical_all"].mean(),
                 "share_blocks_agreement_lt_1": (x["share_match_canonical_all"] < 1).mean(),
                 "share_blocks_agreement_lt_0_8": (x["share_match_canonical_all"] < 0.8).mean(),
                 "share_blocks_agreement_lt_0_5": (x["share_match_canonical_all"] < 0.5).mean(),
                 "share_blocks_canonical_not_modal": (~x["canonical_is_modal"]).mean()}
            if "share_exact_transition" in x:
                r.update({"mean_exact_transition_agreement": x["share_exact_transition"].mean(),
                          "share_blocks_transition_agreement_lt_0_8": (x["share_exact_transition"] < 0.8).mean(),
                          "mean_adverse_outcome_agreement": x["share_adverse_outcome_match"].mean()})
            rows.append(r)

    merged = bss.merge(feat, on="block_geoid")
    tm = bts.merge(bss[["block_geoid", "slr_ft", "share_match_canonical_all", "canonical_is_modal"]],
                   on=["block_geoid", "slr_ft"]).merge(feat, on="block_geoid")
    base0 = merged.loc[merged["slr_ft"].eq(0)]
    for var in ["land_area_bin", "county", "coastal_1km", "water_fraction", "canonical_snap_distance_m",
                "median_draw_snap_distance_m", "n_raw_2ecc_inside"]:
        add(base0, var, var, 0)
        add(tm, var, var, "1-6 pooled")
    add(base0, "canonical_baseline_state", "canonical_state", 0)
    tm["canonical_baseline_state_"] = tm["canonical_baseline_state"]
    add(tm, "canonical_baseline_state", "canonical_baseline_state_", "1-6 pooled")
    for s in ol.SCENARIOS:
        x = merged.loc[merged["slr_ft"].eq(s)].copy()
        x["all"] = "all blocks"
        if s > 0:
            x = x.merge(bts.loc[bts["slr_ft"].eq(s), ["block_geoid", "share_exact_transition", "share_adverse_outcome_match"]],
                        on="block_geoid")
        add(x, "scenario", "all", s)
        x["dist_bin"] = binned(pd.Series(mm[f"dist_to_extent_boundary_{s}ft_m"].to_numpy()), [0, 50, 100, 250, 500, 1000, 2000, np.inf],
                               ["<50", "50-100", "100-250", "250-500", "500-1000", "1000-2000", ">2000"]).to_numpy()
        x["straddle"] = np.where(mm[f"land_straddles_extent_{s}ft"].to_numpy(), "land straddles extent boundary",
                                 "land does not straddle")
        add(x, "dist_canonical_point_to_extent_boundary_m", "dist_bin", s)
        add(x, "land_polygon_straddles_extent", "straddle", s)
    return pd.DataFrame(rows), feat


def instability_logit(bss, meta, snapd):
    import statsmodels.formula.api as smf

    x = bss.loc[bss["slr_ft"].eq(0)].merge(meta, on="block_geoid", suffixes=("", "_m"))
    x["unstable"] = (x["share_match_canonical_all"] < 0.8).astype(int)
    x["log10_land_area"] = np.log10(x["land_area_m2"].clip(lower=1))
    x["log10_canon_snap"] = np.log10(x["canonical_snap_distance_m"].clip(lower=1))
    x["log10_dist_0ft_boundary"] = np.log10(x["dist_to_extent_boundary_0ft_m"].clip(lower=1))
    x["water_fraction"] = x["water_area_m2_census"] / (x["water_area_m2_census"] + x["land_area_m2"])
    x["log1p_2ecc"] = np.log1p(x["n_raw_2ecc_inside"])
    x["coastal_i"] = x["coastal"].astype(int)
    x = x.loc[x["canonical_state"].isin(["redundant", "fragile", "isolated"])]  # inundated baseline: separate
    fit = smf.logit("unstable ~ log10_land_area + log10_canon_snap + log10_dist_0ft_boundary + water_fraction"
                    " + log1p_2ecc + coastal_i + C(county_name) + C(canonical_state, Treatment('redundant'))",
                    data=x).fit(disp=False, maxiter=200)
    ci = fit.conf_int()
    return pd.DataFrame({"term": fit.params.index, "coef": fit.params.values, "odds_ratio": np.exp(fit.params.values),
                         "or_ci_low": np.exp(ci[0].values), "or_ci_high": np.exp(ci[1].values), "p_value": fit.pvalues.values,
                         "n": int(fit.nobs), "outcome": "baseline canonical-state agreement < 0.8 (descriptive model)"})


def morans_i(xy, values, k=8, permutations=199, seed=1):
    tree = base.cKDTree(xy)
    _, idx = tree.query(xy, k=k + 1)
    nbr = idx[:, 1:]
    z = values - values.mean()
    denom = (z ** 2).sum()
    if denom == 0:
        return np.nan, np.nan
    obs = (z * z[nbr].mean(axis=1)).sum() / denom
    rng = np.random.default_rng(seed)
    sims = np.empty(permutations)
    for i in range(permutations):
        zp = rng.permutation(z)
        sims[i] = (zp * zp[nbr].mean(axis=1)).sum() / denom
    p = (1 + (np.abs(sims) >= abs(obs)).sum()) / (permutations + 1)
    return float(obs), float(p)


def spatial_concentration(bss, bts, meta, info, perms):
    m = meta.set_index("block_geoid")
    rows = []
    for s in ol.SCENARIOS:
        x = bss.loc[bss["slr_ft"].eq(s)]
        xy = m.loc[x["block_geoid"], ["canonical_x", "canonical_y"]].to_numpy()
        dis = 1 - x["share_match_canonical_all"].to_numpy()
        I, p = morans_i(xy, dis, permutations=perms)
        bg = x.assign(bg=info.set_index("block_geoid").loc[x["block_geoid"], "block_group_geoid"].to_numpy(),
                      unstable=(x["share_match_canonical_all"] < 0.8))
        bgc = bg.groupby("bg")["unstable"].agg(["sum", "size"]).sort_values("sum", ascending=False)
        tot = bgc["sum"].sum()
        top = int(np.ceil(0.1 * len(bgc)))
        rows.append({"slr_ft": s, "quantity": "state disagreement (1 - share_match_canonical_all)",
                     "morans_i_knn8": I, "permutation_p": p, "n_blocks": int(len(x)),
                     "unstable_blocks_lt_0_8": int(tot),
                     "share_of_unstable_in_top10pct_block_groups": float(bgc["sum"].head(top).sum() / tot) if tot else np.nan,
                     "block_groups_with_any_unstable": int((bgc["sum"] > 0).sum()), "block_groups": int(len(bgc))})
        if s > 0:
            y = bts.loc[bts["slr_ft"].eq(s)]
            I, p = morans_i(xy, 1 - y["share_exact_transition"].to_numpy(), permutations=perms)
            rows.append({"slr_ft": s, "quantity": "transition disagreement (1 - share_exact_transition)",
                         "morans_i_knn8": I, "permutation_p": p, "n_blocks": int(len(y))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# F. Block-group datasets per replication (inputs to 05_bg_models.R)
# ---------------------------------------------------------------------------------------------

def bg_dataset(states_by_s, info, acs):
    frames = []
    b = states_by_s[0]
    for s in ol.SCENARIOS:
        sc = states_by_s[s]
        f = ol.transition_flags(b, sc)
        d = pd.DataFrame({"block_group_geoid": info["block_group_geoid"].to_numpy(),
                          "county_fips": info["county_fips"].to_numpy(), "county_name": info["county_name"].to_numpy(),
                          "tract_geoid": info["tract_geoid"].to_numpy(), "slr_ft": s, "pop20": info["pop20"].to_numpy()})
        for k in STATES5:
            d[f"block_centroid_{k}"] = (sc == S[k]).astype(int)
            d[f"baseline_block_centroid_{k}"] = (b == S[k]).astype(int)
        worse = np.isin(sc, [S["fragile"], S["isolated"], S["inundated"]])
        d["fragile_or_worse"] = worse.astype(int)
        d["any_loss_of_redundancy"] = ((b == S["redundant"]) & worse).astype(int)
        d["baseline_redundant_to_fragile"] = f["red_to_fragile"].astype(int)
        d["baseline_redundant_to_isolated"] = f["red_to_isolated"].astype(int)
        d["baseline_redundant_to_inundated"] = f["red_to_inundated"].astype(int)
        d["baseline_fragile_to_isolated"] = f["fragile_to_isolated"].astype(int)
        d["baseline_fragile_to_inundated"] = f["fragile_to_inundated"].astype(int)
        d["baseline_isolated_to_inundated"] = ((b == S["isolated"]) & (sc == S["inundated"])).astype(int)
        pos = s != 0
        d["new_fragile_due_to_slr"] = (pos & (b == S["redundant"]) & (sc == S["fragile"])).astype(int)
        d["new_isolated_due_to_slr"] = (pos & (b != S["isolated"]) & (sc == S["isolated"])).astype(int)
        d["new_inundated_due_to_slr"] = (pos & (b != S["inundated"]) & (sc == S["inundated"])).astype(int)
        frames.append(d)
    long = pd.concat(frames, ignore_index=True)
    keys = ["block_group_geoid", "county_fips", "county_name", "tract_geoid", "slr_ft"]
    count_cols = [c for c in long.columns if c not in keys + ["pop20"]]
    bg = long.groupby(keys).agg(**{c: (c, "sum") for c in count_cols}, total_blocks=("pop20", "size"),
                                eligible_pop20=("pop20", "sum")).reset_index()
    bg["bridge_rule_applied"] = "approach"
    return bg.merge(acs, on="block_group_geoid", how="left", validate="many_to_one")


def load_acs():
    prod = pd.read_csv(ol.CANONICAL_BG, dtype={"block_group_geoid": str, "tract_geoid": str, "county_fips": str})
    prod["block_group_geoid"] = prod["block_group_geoid"].str.zfill(12)
    acs = prod[["block_group_geoid"] + ACS_COLUMNS]
    assert acs.groupby("block_group_geoid").nunique(dropna=False).le(1).all(axis=None)
    return acs.drop_duplicates("block_group_geoid").reset_index(drop=True), prod


def bg_sensitivity(bgs, canon_bg):
    """Per BG x SLR: canonical vs MC distribution of risk sets and transition shares."""
    def quantities(bg):
        q = bg[["block_group_geoid", "slr_ft"]].copy()
        base0 = bg.loc[bg["slr_ft"].eq(0)].set_index("block_group_geoid")
        red = base0.loc[bg["block_group_geoid"], "block_centroid_redundant"].to_numpy()
        fra = base0.loc[bg["block_group_geoid"], "block_centroid_fragile"].to_numpy()
        q["baseline_redundant_n"] = red
        q["baseline_fragile_n"] = fra
        with np.errstate(invalid="ignore", divide="ignore"):
            q["prop_red_to_worse"] = np.where(red > 0, bg["any_loss_of_redundancy"] / red, np.nan)
            q["prop_red_to_fragile"] = np.where(red > 0, bg["baseline_redundant_to_fragile"] / red, np.nan)
            q["prop_red_to_isolated"] = np.where(red > 0, bg["baseline_redundant_to_isolated"] / red, np.nan)
            q["prop_red_to_inundated"] = np.where(red > 0, bg["baseline_redundant_to_inundated"] / red, np.nan)
            fw = bg["baseline_fragile_to_isolated"] + bg["baseline_fragile_to_inundated"]
            q["prop_fragile_to_worse"] = np.where(fra > 0, fw / fra, np.nan)
            q["prop_fragile_to_isolated"] = np.where(fra > 0, bg["baseline_fragile_to_isolated"] / fra, np.nan)
            q["prop_fragile_to_inundated"] = np.where(fra > 0, bg["baseline_fragile_to_inundated"] / fra, np.nan)
            q["share_fragile_or_worse"] = bg["fragile_or_worse"] / bg["total_blocks"]
        return q

    cq = quantities(canon_bg).melt(id_vars=["block_group_geoid", "slr_ft"], var_name="quantity", value_name="canonical")
    reps = pd.concat([quantities(b).assign(rep=r) for r, b in bgs.items()], ignore_index=True)
    rl = reps.melt(id_vars=["block_group_geoid", "slr_ft", "rep"], var_name="quantity", value_name="value")
    g = rl.groupby(["block_group_geoid", "slr_ft", "quantity"])["value"]
    summ = pd.DataFrame({"mc_mean": g.mean(), "mc_p2_5": g.quantile(0.025), "mc_p97_5": g.quantile(0.975),
                         "mc_min": g.min(), "mc_max": g.max(), "n_reps_defined": g.count()}).reset_index()
    out = cq.merge(summ, on=["block_group_geoid", "slr_ft", "quantity"], how="left")
    out["abs_dev_mean_minus_canonical"] = out["mc_mean"] - out["canonical"]
    return out


# ---------------------------------------------------------------------------------------------
# E. Figures
# ---------------------------------------------------------------------------------------------

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQ = ["#1f4e79", "#2a78d6", "#8fb8ea", "#d9e6f6"]  # one hue, dark (worst) -> light
# Left-closed bins so labels match the "< x" thresholds used in the tables.
AGREE_BINS = [(0.0, 0.5, "< 50%"), (0.5, 0.8, "50% to < 80%"), (0.8, 1.0, "80% to < 100%"), (1.0, 1.01, "100%")]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def agreement_bars(ax, frame, col, xcol, xlabels, title):
    bottom = np.zeros(len(xlabels))
    for (lo, hi, lab), c in zip(AGREE_BINS, SEQ):
        vals = np.array([((frame.loc[frame[xcol].eq(x), col] >= lo) & (frame.loc[frame[xcol].eq(x), col] < hi)).mean()
                         for x in xlabels])
        ax.bar(range(len(xlabels)), vals, bottom=bottom, color=c, width=0.7, edgecolor="white", linewidth=1.5, label=lab)
        bottom += vals
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels([f"{x} ft" for x in xlabels])
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(__import__("matplotlib.ticker", fromlist=["PercentFormatter"]).PercentFormatter(1.0))
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    style(ax)


def zoom_bars(ax, frame, col, xcol, xlabels, title):
    """Share of blocks with agreement < 100% / < 80% / < 50%, as a zoomed companion."""
    for (thr, lab), c in zip([(1.0, "< 100%"), (0.8, "< 80%"), (0.5, "< 50%")], [SEQ[2], SEQ[1], SEQ[0]]):
        vals = [(frame.loc[frame[xcol].eq(x), col] < thr).mean() for x in xlabels]
        ax.plot(range(len(xlabels)), vals, marker="o", markersize=6, linewidth=2, color=c, label=f"agreement {lab}")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels([f"{x} ft" for x in xlabels])
    ax.yaxis.set_major_formatter(__import__("matplotlib.ticker", fromlist=["PercentFormatter"]).PercentFormatter(1.0, decimals=1))
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    style(ax)


def make_figures(run_dir, bss, bts, summ, meta, n):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    agreement_bars(axes[0], bss, "share_match_canonical_all", "slr_ft", ol.SCENARIOS,
                   f"Share of draws reproducing the canonical state ({n} draws/block)")
    axes[0].set_ylabel("Share of eligible blocks", color=INK2)
    axes[0].legend(title="Draws matching canonical", fontsize=8, title_fontsize=8, frameon=False,
                   loc="upper left", bbox_to_anchor=(1.0, 1.0))
    zoom_bars(axes[1], bss, "share_match_canonical_all", "slr_ft", ol.SCENARIOS, "Blocks below each agreement level")
    axes[1].legend(fontsize=8, frameon=False)
    fig.savefig(run_dir / "state_agreement_distribution.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    agreement_bars(axes[0], bts, "share_exact_transition", "slr_ft", ol.POSITIVE_SCENARIOS,
                   "Exact baseline->scenario transition reproduced (all blocks)")
    axes[0].set_ylabel("Share of eligible blocks", color=INK2)
    adv = bts.loc[bts["canonical_adverse"]]
    agreement_bars(axes[1], adv, "share_exact_transition", "slr_ft", ol.POSITIVE_SCENARIOS,
                   "Exact transition reproduced (canonical adverse blocks)")
    agreement_bars(axes[2], adv, "share_adverse_outcome_match", "slr_ft", ol.POSITIVE_SCENARIOS,
                   "Any adverse transition reproduced (canonical adverse blocks)")
    axes[2].legend(title="Draws matching canonical", fontsize=8, title_fontsize=8, frameon=False,
                   loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(run_dir / "transition_agreement_distribution.png", dpi=200)
    plt.close(fig)

    # Population: canonical (markers) vs Monte Carlo mean and 2.5-97.5% interval.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    x = np.array(ol.POSITIVE_SCENARIOS)
    for k, lab, c in zip(["new_inundated", "new_isolated_or_inundated", "new_fragile_or_worse"],
                         ["New inundated", "+ new isolated", "+ new fragile"], SERIES):
        d = summ.loc[summ["measure"].eq("pop") & summ["quantity"].eq(k)].set_index("slr_ft").loc[x]
        axes[0].fill_between(x, d["mc_p2_5"] / 1e6, d["mc_p97_5"] / 1e6, color=c, alpha=0.18, linewidth=0)
        axes[0].plot(x, d["mc_mean"] / 1e6, color=c, linewidth=2, label=f"{lab}: MC mean (band 2.5-97.5%)")
        axes[0].plot(x, d["canonical"] / 1e6, linestyle="none", marker="D", markersize=6, markerfacecolor="white",
                     markeredgecolor=INK, markeredgewidth=1.2)
    axes[0].plot([], [], linestyle="none", marker="D", markerfacecolor="white", markeredgecolor=INK, label="Canonical (single origin)")
    axes[0].set_ylabel("2020 population (millions)", color=INK2)
    axes[0].set_xlabel("Sea-level rise (ft)", color=INK2)
    axes[0].set_title("Cumulative population with new adverse transitions", fontsize=10.5, loc="left", color=INK)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    style(axes[0])
    d = summ.loc[summ["measure"].eq("pop") & summ["quantity"].eq("added_by_fragile")].set_index("slr_ft").loc[x]
    lo, hi = d["mc_p2_5"] / 1e3, d["mc_p97_5"] / 1e3
    axes[1].vlines(x, lo, hi, color=SERIES[1], linewidth=3, label="MC 2.5-97.5% interval")
    axes[1].plot(x, d["mc_mean"] / 1e3, linestyle="none", marker="o", markersize=7, color=SERIES[1], label="MC mean")
    axes[1].plot(x, d["canonical"] / 1e3, linestyle="none", marker="D", markersize=7, markerfacecolor="white",
                 markeredgecolor=INK, label="Canonical")
    axes[1].set_ylabel("2020 population (thousands)", color=INK2)
    axes[1].set_xlabel("Sea-level rise (ft)", color=INK2)
    axes[1].set_title("Population added by fragility beyond isolation + inundation", fontsize=10.5, loc="left", color=INK)
    axes[1].legend(fontsize=8, frameon=False, loc="upper left")
    style(axes[1])
    fig.savefig(run_dir / "population_sensitivity_by_slr.png", dpi=200)
    plt.close(fig)

    # Instability vs block area (baseline).
    b0 = bss.loc[bss["slr_ft"].eq(0)].merge(meta[["block_geoid", "land_area_m2"]], on="block_geoid")
    b0["area_bin"] = pd.qcut(b0["land_area_m2"], 20)
    g = b0.groupby("area_bin", observed=True).agg(med_area=("land_area_m2", "median"),
                                                  lt1=("share_match_canonical_all", lambda v: (v < 1).mean()),
                                                  lt08=("share_match_canonical_all", lambda v: (v < 0.8).mean()),
                                                  mean_dis=("share_match_canonical_all", lambda v: 1 - v.mean()))
    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    for col, lab, c in [("lt1", "Blocks with any draw disagreeing", SEQ[2]), ("lt08", "Blocks with agreement < 80%", SEQ[1]),
                        ("mean_dis", "Mean share of draws disagreeing", SEQ[0])]:
        ax.plot(g["med_area"], g[col], marker="o", markersize=5, linewidth=2, color=c, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("Census land area (m², median of ventile bin, log scale)", color=INK2)
    ax.set_ylabel("Share", color=INK2)
    ax.yaxis.set_major_formatter(__import__("matplotlib.ticker", fromlist=["PercentFormatter"]).PercentFormatter(1.0))
    ax.set_title("Baseline (0 ft) origin-placement instability by block land area", fontsize=10.5, loc="left", color=INK)
    ax.legend(fontsize=8, frameon=False)
    style(ax)
    fig.savefig(run_dir / "instability_vs_block_area.png", dpi=200)
    plt.close(fig)

    # Map. Block polygons span 1 m^2 to 700 km^2, so a block-level regional map is
    # dominated by a few huge rural blocks. Panel A aggregates to block groups
    # (share of eligible blocks with baseline agreement < 80%); panels B-C show
    # blocks at full resolution in an urban window.
    from matplotlib.patches import Patch

    blocks = base.maybe_to_projected(base.prepare_blocks_layer(base.read_vector(base.BLOCKS_PATH)))[
        ["block_geoid", "block_group_geoid", "geometry"]]
    b0 = bss.loc[bss["slr_ft"].eq(0), ["block_geoid", "share_match_canonical_all"]]
    gdf = blocks.merge(b0, on="block_geoid", how="inner")
    gdf["unstable"] = gdf["share_match_canonical_all"] < 0.8
    bg = gdf.dissolve(by="block_group_geoid", aggfunc={"unstable": "mean"}).reset_index()
    t4 = bts.loc[bts["slr_ft"].eq(4), ["block_geoid", "share_exact_transition"]]
    g4 = blocks.merge(t4, on="block_geoid", how="inner")
    fig, axes = plt.subplots(1, 3, figsize=(16, 8.5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.15, 1.15]})
    bg_bins = [(0.0, 0.1, "#d9e6f6", "< 10%"), (0.1, 0.25, "#8fb8ea", "10% to < 25%"),
               (0.25, 0.5, "#2a78d6", "25% to < 50%"), (0.5, 1.01, "#1f4e79", ">= 50%")]
    handles = []
    for lo, hi, c, lab in bg_bins:
        sub = bg.loc[(bg["unstable"] >= lo) & (bg["unstable"] < hi)]
        if len(sub):
            sub.plot(ax=axes[0], color=c, linewidth=0)
        handles.append(Patch(color=c, label=f"{lab} ({len(sub):,} block groups)"))
    axes[0].legend(handles=handles, title="Share of blocks with baseline\nagreement < 80%", fontsize=8,
                   title_fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=2)
    axes[0].set_title("A. Block groups: share of blocks with\nlow baseline agreement (0 ft)", fontsize=10.5, loc="left")
    # Urban window (EPSG:32617 metres), central Broward / north Miami-Dade.
    win = (575_000, 2_860_000, 590_000, 2_880_000)
    panels = [(axes[1], gdf, "share_match_canonical_all", "B. Blocks, 0 ft: share of draws\nreproducing canonical state"),
              (axes[2], g4, "share_exact_transition", "C. Blocks, 4 ft: share of draws\nreproducing canonical transition")]
    for ax, frame, col, title in panels:
        sub_all = frame.cx[win[0]:win[2], win[1]:win[3]]
        sub_all.plot(ax=ax, color="#ecebe7", linewidth=0.1, edgecolor="white")
        handles = []
        for (lo, hi, lab), c in zip(AGREE_BINS[:3], SEQ[:3]):
            sub = sub_all.loc[(sub_all[col] >= lo) & (sub_all[col] < hi)]
            if len(sub):
                sub.plot(ax=ax, color=c, linewidth=0.1, edgecolor="white")
            handles.append(Patch(color=c, label=lab))
        handles.append(Patch(color="#ecebe7", label="100%"))
        ax.set_xlim(win[0], win[2])
        ax.set_ylim(win[1], win[3])
        ax.legend(handles=handles, title="Draws matching canonical", fontsize=8, title_fontsize=8,
                  frameon=True, framealpha=0.9, loc="lower left")
        ax.set_title(title, fontsize=10.5, loc="left")
    for ax in axes:
        ax.set_axis_off()
    axes[0].indicate_inset(bounds=(win[0], win[1], win[2] - win[0], win[3] - win[1]), edgecolor="#eb6834", linewidth=1.5)
    fig.savefig(run_dir / "spatial_instability_map.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------

def main():
    args = parse_args()
    run_dir = ol.OUT_DIR / args.run
    t_all = time.time()
    rss = ol.PeakRSS().start()
    meta = pd.read_parquet(ol.WORK_DIR / "block_metadata.parquet")
    blocks, n, mats, valid, snapd = load_matrices(run_dir)
    cmat, info = load_canonical(blocks)
    pop = info["pop20"].to_numpy(dtype=np.int64)
    ol.log(f"{len(blocks):,} blocks x {n} draws")

    bss = block_state_stability(blocks, info, n, mats, cmat, valid)
    ol.atomic_to_csv(bss, run_dir / "block_state_stability.csv")
    bts = block_transition_stability(blocks, info, n, mats, cmat)
    ol.atomic_to_csv(bts, run_dir / "block_transition_stability.csv")
    ol.log("block stability tables written")

    reps, canon = replication_table(n, mats, cmat, pop)
    ol.atomic_to_csv(reps, run_dir / "replication_aggregates_long.csv")
    summ = summarize_reps(reps, canon)
    fig4_check = check_against_fig4(canon) if args.run == "full" else {"skipped": "pilot subset"}
    ol.log(f"fig4 reproduction check: {fig4_check}")
    ol.atomic_to_csv(summ.loc[summ["measure"].isin(["blocks", "share_blocks"])], run_dir / "aggregate_sensitivity_by_slr.csv")
    ol.atomic_to_csv(summ.loc[summ["measure"].isin(["pop", "share_pop"])], run_dir / "population_sensitivity_by_slr.csv")

    corr, _ = instability_correlates(bss, bts, meta, snapd)
    ol.atomic_to_csv(corr, run_dir / "instability_correlates.csv")
    try:
        logit = instability_logit(bss, meta, snapd)
        ol.atomic_to_csv(logit, run_dir / "instability_logit_baseline.csv")
    except Exception as exc:  # report, do not hide
        ol.log(f"instability logit failed: {exc}")
    sp = spatial_concentration(bss, bts, meta, info, args.moran_permutations)
    ol.atomic_to_csv(sp, run_dir / "spatial_concentration.csv")
    ol.log("correlates and spatial concentration written")

    # Block-group datasets: canonical (validation) + one per replication.
    acs, prod_bg = load_acs()
    bg_dir = run_dir / "work_bg"
    bg_dir.mkdir(parents=True, exist_ok=True)
    canon_bg = bg_dataset(cmat, info, acs)
    bg_check = {"skipped": True}
    if args.run == "full" and not args.skip_bg:
        cols = ["block_centroid_" + k for k in STATES5] + ["any_loss_of_redundancy", "baseline_redundant_to_fragile",
                                                           "baseline_redundant_to_isolated", "baseline_redundant_to_inundated",
                                                           "baseline_fragile_to_isolated", "baseline_fragile_to_inundated",
                                                           "total_blocks", "eligible_pop20"]
        pb = prod_bg.copy()
        m = canon_bg.merge(pb[["block_group_geoid", "slr_ft"] + cols], on=["block_group_geoid", "slr_ft"],
                           suffixes=("", "_prod"), how="outer", indicator=True)
        bg_check = {"rows": int(len(m)), "row_alignment_mismatch": int((m["_merge"] != "both").sum()),
                    **{f"{c}_mismatch": int((m[c] != m[f"{c}_prod"]).sum()) for c in cols}}
        ol.log(f"canonical BG aggregation vs production dataset: {bg_check}")
    if not args.skip_bg:
        ol.atomic_to_csv(canon_bg, bg_dir / "bg_rep_canonical.csv")
        bgs = {}
        for r in range(n):
            bgs[r] = bg_dataset({s: mats[s][:, r] for s in ol.SCENARIOS}, info, acs)
            ol.atomic_to_csv(bgs[r], bg_dir / f"bg_rep_{r:03d}.csv")
        bgsens = bg_sensitivity(bgs, canon_bg)
        ol.atomic_to_csv(bgsens, run_dir / "block_group_sensitivity.csv")
        ol.log("block-group datasets written")

    if not args.skip_figures:
        make_figures(run_dir, bss, bts, summ, meta, n)
        ol.log("figures written")
    rss.stop()
    ol.atomic_write_json(run_dir / "summarize_manifest.json", {
        "run": args.run, "n_blocks": int(len(blocks)), "n_draws": n, "fig4_check": fig4_check,
        "bg_aggregation_check": bg_check, "total_s": round(time.time() - t_all, 1), "peak_rss_gib": rss.peak_gib})
    return 0


if __name__ == "__main__":
    sys.exit(main())
