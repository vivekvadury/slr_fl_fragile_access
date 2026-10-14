#!/usr/bin/env python
"""Probe: do alternative ORIGIN attachment rules reduce origin-placement sensitivity?

Exploratory follow-up to the Monte Carlo experiment. Reuses the validated node-state
tables, the 50 land-polygon draws per block, and their per-scenario inundation flags;
only the node an origin attaches to changes.

Rules compared (all within the raw-graph LCC, 2 km validity as in production):
  canonical     nearest LCC node (production rule)
  public_street nearest LCC node incident to at least one non-`service` edge
                (driveways, parking aisles, and other service stubs cannot be the
                origin's attachment point; they remain in the graph as connectors)
  loop          nearest LCC node in a non-singleton raw 2-edge-connected component,
                falling back to the nearest LCC node: the facility rule of section 4.2
                applied to origins

Also reports, for the canonical rule, how far fragile origins sit from the redundant
network: network distance from the snapped node to the nearest node that is
redundant at 0 ft (computed on the raw graph, which differs from the 0 ft dry graph by
1,097 of 1.86M edges).

Outputs: outputs/origin_placement_sensitivity/probe/
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ops_lib as ol  # noqa: E402

ol.ensure_env()
base, np, pd = ol.base, ol.np, ol.pd
S, CS = ol.STATE_CODES, ol.CODE_STATES
OUT = ol.OUT_DIR / "probe"


def node_sets():
    edges = pd.read_parquet(ol.SOURCE_CACHE_DIR / f"segmentized_edges_{ol.V2_STEM}.parquet",
                            columns=["u", "v", "highway", "length_m"])
    nodes = pd.read_parquet(ol.SOURCE_CACHE_DIR / f"segmentized_nodes_{ol.V2_STEM}.parquet", columns=["node_id", "x", "y"])
    n = len(nodes)
    assert (nodes["node_id"].to_numpy() == np.arange(n)).all()
    public = np.zeros(n, dtype=bool)
    nonservice = edges["highway"].ne("service").to_numpy()
    public[edges["u"].to_numpy()[nonservice]] = True
    public[edges["v"].to_numpy()[nonservice]] = True
    mem = pd.read_parquet(ol.SOURCE_CACHE_DIR / f"raw_graph_membership_{ol.V2_STEM}.parquet")
    lcc = np.zeros(n, dtype=bool)
    lcc[mem.loc[mem["raw_in_lcc"], "node_id"].to_numpy()] = True
    loop = np.zeros(n, dtype=bool)
    loop[mem.loc[mem["raw_2ecc_size"] >= 2, "node_id"].to_numpy()] = True
    return nodes[["x", "y"]].to_numpy(), edges, lcc, public, loop


def snap(xy_nodes, mask, pts, max_m=base.MAX_ORIGIN_SNAP_M):
    ids = np.flatnonzero(mask)
    d, i = base.cKDTree(xy_nodes[ids]).query(pts, k=1)
    return ids[i], d, d <= max_m


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    xy, edges, lcc, public, loop = node_sets()
    states = {s: pd.read_parquet(ol.WORK_DIR / f"node_states_slr_{s}ft.parquet", columns=["node_state"])["node_state"].to_numpy()
              for s in ol.SCENARIOS}

    meta = pd.read_parquet(ol.WORK_DIR / "block_metadata.parquet")
    meta = meta.loc[meta["analysis_eligible"].eq(True)].sort_values("block_geoid").reset_index(drop=True)
    canon_inund = pd.read_parquet(ol.OUT_DIR / "full" / "canonical_engine_states.parquet")
    canon_inund = canon_inund.pivot(index="block_geoid", columns="slr_ft", values="engine_inundated").loc[meta["block_geoid"]]
    st = pd.read_parquet(ol.OUT_DIR / "full" / "random_origin_states.parquet").sort_values(["block_geoid", "rep"])
    snaps = pd.read_parquet(ol.OUT_DIR / "full" / "random_origin_snaps.parquet", columns=["draw_id", "x", "y"])
    st = st.merge(snaps, on="draw_id").sort_values(["block_geoid", "rep"])
    assert (st["block_geoid"].to_numpy()[::50] == meta["block_geoid"].to_numpy()).all()
    nb, k = len(meta), 50
    pop = meta["pop20"].to_numpy()
    cpts = meta[["canonical_x", "canonical_y"]].to_numpy()
    dpts = st[["x", "y"]].to_numpy()

    rules = {"canonical": lcc, "public_street": lcc & public, "loop": lcc & loop}
    rows, per_block = [], {}
    for rule, mask in rules.items():
        c_node, c_d, c_ok = snap(xy, mask, cpts)
        d_node, d_d, d_ok = snap(xy, mask, dpts)
        if rule == "loop":  # facility-style fallback to the nearest LCC node
            fc_node, fc_d, fc_ok = snap(xy, lcc, cpts)
            fd_node, fd_d, fd_ok = snap(xy, lcc, dpts)
            c_node, c_d, c_ok = np.where(c_ok, c_node, fc_node), np.where(c_ok, c_d, fc_d), c_ok | fc_ok
            d_node, d_d, d_ok = np.where(d_ok, d_node, fd_node), np.where(d_ok, d_d, fd_d), d_ok | fd_ok
        cs = {s: ol.classify(canon_inund[s].to_numpy(dtype=bool), c_ok, states[s][c_node]) for s in ol.SCENARIOS}
        ds = {s: ol.classify(st[f"inund_{s}"].to_numpy(dtype=bool), d_ok, states[s][d_node]).reshape(nb, k) for s in ol.SCENARIOS}
        agree0 = (ds[0] == cs[0][:, None]).mean(axis=1)
        per_block[rule] = pd.DataFrame({"block_geoid": meta["block_geoid"], "pop20": pop, "canonical_state_0ft": [CS[v] for v in cs[0]],
                                        "share_draws_fragile_0ft": (ds[0] == S["fragile"]).mean(axis=1), "agreement_0ft": agree0})
        r = {"rule": rule,
             "median_snap_m_canonical": float(np.median(c_d)), "median_snap_m_draws": float(np.median(d_d)),
             "canonical_snap_failures": int((~c_ok).sum()), "draw_snap_failures": int((~d_ok).sum()),
             "fragile_share_canonical": float((cs[0] == S["fragile"]).mean()),
             "fragile_share_mc_mean": float((ds[0] == S["fragile"]).mean()),
             "fragile_pop_canonical": int(pop[cs[0] == S["fragile"]].sum()),
             "fragile_pop_mc_mean": float(np.mean([pop[ds[0][:, j] == S["fragile"]].sum() for j in range(k)])),
             "mean_agreement_0ft": float(agree0.mean()),
             "pop_weighted_agreement_0ft": float((agree0 * pop).sum() / pop.sum()),
             "blocks_any_disagreement_0ft": float((agree0 < 1).mean()),
             "blocks_agreement_lt_0_8_0ft": float((agree0 < 0.8).mean())}
        for s in (2, 4, 6):
            def newfw(b, x):
                f = ol.transition_flags(b, x)
                return f["red_to_worse"] | f["fragile_to_worse"]
            r[f"new_fragile_or_worse_pop_{s}ft_canonical"] = int(pop[newfw(cs[0], cs[s])].sum())
            r[f"new_fragile_or_worse_pop_{s}ft_mc_mean"] = float(np.mean([pop[newfw(ds[0][:, j], ds[s][:, j])].sum() for j in range(k)]))
            r[f"exact_transition_agreement_{s}ft"] = float(((ds[0] == cs[0][:, None]) & (ds[s] == cs[s][:, None])).mean())
        rows.append(r)
        ol.log(f"{rule}: {r}")
    summary = pd.DataFrame(rows)
    ol.atomic_to_csv(summary, OUT / "attachment_rule_comparison.csv")

    # How far are canonical-rule fragile origins from the redundant network?
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    n = len(xy)
    g = coo_matrix((edges["length_m"].to_numpy(), (edges["u"].to_numpy(), edges["v"].to_numpy())), shape=(n, n)).tocsr()
    red = np.flatnonzero(states[0] == S["redundant"])
    dist = dijkstra(g, directed=False, indices=red, min_only=True)
    c_node, _, c_ok = snap(xy, lcc, cpts)
    hw = edges.assign(service=edges["highway"].eq("service"))
    incident_public = public  # node touches a non-service edge
    frag = (states[0][c_node] == S["fragile"]) & c_ok & ~canon_inund[0].to_numpy(dtype=bool)
    dfr = pd.DataFrame({"block_geoid": meta["block_geoid"][frag], "pop20": pop[frag],
                        "dist_to_redundant_m": dist[c_node[frag]],
                        "snapped_node_service_only": ~incident_public[c_node[frag]]})
    bins = [0, 25, 50, 100, 250, 500, 1000, np.inf]
    dfr["bin"] = pd.cut(dfr["dist_to_redundant_m"], bins, right=False).astype(str)
    tab = dfr.groupby("bin", sort=False).agg(blocks=("block_geoid", "size"), pop20=("pop20", "sum"),
                                             share_service_only=("snapped_node_service_only", "mean")).reset_index()
    tab["share_blocks"] = tab["blocks"] / tab["blocks"].sum()
    ol.atomic_to_csv(tab, OUT / "canonical_fragile_distance_to_redundant.csv")
    ol.atomic_to_csv(pd.concat([v.assign(rule=k) for k, v in per_block.items()]), OUT / "per_block_by_rule.csv")
    ol.log(f"canonical fragile blocks: {int(frag.sum())}; service-only snapped node share {dfr['snapped_node_service_only'].mean():.3f}; "
           f"median dist to redundant {np.median(dfr['dist_to_redundant_m']):.1f} m")
    print(tab.to_string(index=False))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(summary.T.to_string())
    ol.log(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
