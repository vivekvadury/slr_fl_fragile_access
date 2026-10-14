#!/usr/bin/env python
"""Mechanism check: does origin depth inside the block explain the fragile excess?

Hypothesis (to test, not assume): Census blocks are usually bounded by streets,
the canonical representative point lies deep in the block interior, and nodes
nearest an interior point are more often on dead-end (fragile) parts of the
network, while uniformly drawn points frequently fall near the boundary streets.

For every draw and every canonical point this computes the distance to the
block's own polygon boundary, a scale-free depth (distance / sqrt(land area)),
and the 0 ft node state of the snapped node, then tabulates the fragile share by
depth bin separately for draws and canonical points.

Outputs (outputs/origin_placement_sensitivity/full/):
  mechanism_depth_vs_state.csv, mechanism_summary.json, mechanism_depth_vs_state.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ops_lib as ol  # noqa: E402

ol.ensure_env()
base, gpd, np, pd = ol.base, ol.gpd, ol.np, ol.pd
import shapely  # noqa: E402

S = ol.STATE_CODES
DEPTH_EDGES = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, np.inf]
DEPTH_LABELS = ["<0.05", "0.05-0.1", "0.1-0.15", "0.15-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.6", ">=0.6"]


def main():
    run = ol.OUT_DIR / "full"
    meta = pd.read_parquet(ol.WORK_DIR / "block_metadata.parquet")
    meta = meta.loc[meta["analysis_eligible"].eq(True)].reset_index(drop=True)
    blocks = base.maybe_to_projected(base.prepare_blocks_layer(base.read_vector(base.BLOCKS_PATH)))
    geom = blocks.set_index("block_geoid").loc[meta["block_geoid"], "geometry"].values
    bnd = shapely.boundary(geom)
    shapely.prepare(bnd)
    scale = np.sqrt(meta["land_area_m2"].clip(lower=1).to_numpy())
    node0 = pd.read_parquet(ol.WORK_DIR / "node_states_slr_0ft.parquet", columns=["node_state", "dry_degree"])
    state0 = node0["node_state"].to_numpy()
    deg0 = node0["dry_degree"].to_numpy()

    draws = pd.read_parquet(run / "random_origin_snaps.parquet", columns=["block_geoid", "x", "y", "node_id", "snap_distance_m"])
    pos = pd.Series(np.arange(len(meta)), index=meta["block_geoid"]).loc[draws["block_geoid"]].to_numpy()
    d_draw = shapely.distance(bnd[pos], gpd.points_from_xy(draws["x"], draws["y"]))
    c_pts = gpd.points_from_xy(meta["canonical_x"], meta["canonical_y"])
    d_can = shapely.distance(bnd, c_pts)

    frames = []
    for kind, dist, node, sc, snapd in [
        ("random draw", d_draw, draws["node_id"].to_numpy(), scale[pos], draws["snap_distance_m"].to_numpy()),
        ("canonical point", d_can, meta["canonical_node_id"].to_numpy(), scale, meta["canonical_snap_distance_m"].to_numpy()),
    ]:
        f = pd.DataFrame({"kind": kind, "depth": dist / sc, "dist_to_block_boundary_m": dist,
                          "node_fragile": state0[node] == S["fragile"], "node_degree1": deg0[node] == 1,
                          "snap_distance_m": snapd})
        f["depth_bin"] = pd.cut(f["depth"], DEPTH_EDGES, labels=DEPTH_LABELS, right=False).astype(str)
        frames.append(f)
    allf = pd.concat(frames, ignore_index=True)
    tab = (allf.groupby(["kind", "depth_bin"]).agg(n=("depth", "size"), share_node_fragile=("node_fragile", "mean"),
                                                   share_node_degree1=("node_degree1", "mean"),
                                                   median_snap_distance_m=("snap_distance_m", "median"),
                                                   median_dist_to_boundary_m=("dist_to_block_boundary_m", "median"))
           .reset_index())
    tab["share_of_kind"] = tab["n"] / tab.groupby("kind")["n"].transform("sum")
    tab["depth_bin"] = pd.Categorical(tab["depth_bin"], DEPTH_LABELS, ordered=True)
    tab = tab.sort_values(["kind", "depth_bin"])
    ol.atomic_to_csv(tab, run / "mechanism_depth_vs_state.csv")

    # Reweighting: draws' depth-specific fragile rates applied to the canonical depth distribution.
    dr = tab.loc[tab["kind"].eq("random draw")].set_index("depth_bin")
    cn = tab.loc[tab["kind"].eq("canonical point")].set_index("depth_bin")
    reweighted = float((dr["share_node_fragile"] * cn["share_of_kind"]).sum())
    summary = {
        "draw_share_node_fragile": float(frames[0]["node_fragile"].mean()),
        "canonical_share_node_fragile": float(frames[1]["node_fragile"].mean()),
        "draw_rates_reweighted_to_canonical_depth_distribution": reweighted,
        "median_depth_draw": float(np.median(frames[0]["depth"])),
        "median_depth_canonical": float(np.median(frames[1]["depth"])),
        "median_dist_to_boundary_m_draw": float(np.median(d_draw)),
        "median_dist_to_boundary_m_canonical": float(np.median(d_can)),
        "median_snap_distance_m_draw": float(np.median(frames[0]["snap_distance_m"])),
        "median_snap_distance_m_canonical": float(np.median(frames[1]["snap_distance_m"])),
        "note": "node state = 0 ft network state of the snapped node (ignores point inundation); eligible blocks only",
    }
    ol.atomic_write_json(run / "mechanism_summary.json", summary)
    ol.log(summary)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    x = np.arange(len(DEPTH_LABELS))
    for (kind, c, mk) in [("random draw", "#2a78d6", "o"), ("canonical point", "#eb6834", "D")]:
        t = tab.loc[tab["kind"].eq(kind)].set_index("depth_bin").reindex(DEPTH_LABELS)
        axes[0].plot(x, t["share_node_fragile"], marker=mk, markersize=7, linewidth=2, color=c, label=kind)
        axes[1].bar(x + (-0.2 if kind == "random draw" else 0.2), t["share_of_kind"], width=0.38, color=c, label=kind)
    for ax, title, yl in [(axes[0], "Share snapping to a fragile node (0 ft), by origin depth", "Share fragile node"),
                          (axes[1], "Where origins fall: depth distribution", "Share of origins")]:
        ax.set_xticks(x)
        ax.set_xticklabels(DEPTH_LABELS, fontsize=8)
        ax.set_xlabel("Depth = distance to block boundary / sqrt(land area)", color="#52514e")
        ax.set_ylabel(yl, color="#52514e")
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color="#e4e3df")
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, frameon=False)
    fig.savefig(run / "mechanism_depth_vs_state.png", dpi=200)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
