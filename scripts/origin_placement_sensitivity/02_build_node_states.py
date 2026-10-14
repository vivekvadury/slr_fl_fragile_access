#!/usr/bin/env python
"""Per-scenario node-state tables and production-equivalence checks.

For one SLR scenario this script rebuilds the canonical ``approach`` dry graph
with production functions (``build_bridge_structures``, ``load_slr_layer``,
``apply_bridge_rule_to_edges``, ``build_graph``), then writes:

  work/node_states_slr_<s>ft.parquet
      network state of every raw-graph node on the dry graph (``ops_lib.node_state_table``)
  work/raw_polygon_inundation_slr_<s>ft.parquet
      draw_ids / block_geoids whose point intersects any raw NOAA polygon
      (cross-check of the union-based inundation test used in classification)
  work/production_pilot_slr_<s>ft.parquet
      UNMODIFIED production ``scenario_results_for_origins`` applied to every
      pilot draw (Validation 1b)
  work/node_states_slr_<s>ft_checks.json
      bridge-audit comparison with the canonical run, exhaustive node-level
      equivalence (Validation 1c), timings, memory

Validation 1c runs production ``scenario_results_for_origins`` with every raw-LCC
node as an origin located on that node (``slr_layer=None`` so only network
state is tested) and compares the result with the node table, node by node.

Performance patches used ONLY inside the production-function checks:
  * ``build_nearest_service_lookup`` returns an empty table. Its output feeds
    distance/detour columns and the legacy collocated rule, which is disabled
    in the canonical run, so classification cannot depend on it; the
    canonical-point comparison in 03 (against the della output, produced with
    the real lookup) tests this empirically.
  * ``attachment_lib.GraphMemo`` caches component and 2-edge-component
    computations per graph object (results identical; verified in the
    attachment experiment).
  * ``attachment_lib.union_passthrough`` hands the production function the
    cached scenario union instead of re-unioning it.

Example: python scripts/origin_placement_sensitivity/02_build_node_states.py --scenario 3
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ops_lib as ol  # noqa: E402

ol.ensure_env()
base, al, gpd, np, pd = ol.base, ol.al, ol.gpd, ol.np, ol.pd
import shapely  # noqa: E402

CHUNK = 250_000


@contextlib.contextmanager
def no_nearest_lookup():
    original = base.build_nearest_service_lookup

    def empty(graph, services):
        return pd.DataFrame(columns=["node_id", "nearest_service_node_id", "nearest_service_id",
                                     "nearest_service_type", "nearest_service_snap_rule", "distance_m"])

    base.build_nearest_service_lookup = empty
    try:
        yield
    finally:
        base.build_nearest_service_lookup = original


def production_classify(*, slr_ft, slr_layer, graph, services, origins, geometry, boundary_node_ids, retention):
    """Call production scenario_results_for_origins exactly as main() does."""
    boundary = pd.DataFrame({"block_geoid": origins["block_geoid"].to_numpy(),
                             "boundary_distance_m": 0.0, "boundary_flag": False})
    return base.scenario_results_for_origins(
        slr_ft=slr_ft, slr_layer_name=ol.LAYERS[slr_ft], slr_layer=slr_layer, graph=graph,
        services=services, origins=origins, centroid_boundary=boundary,
        centroid_geometry_source=geometry, baseline_nearest=pd.DataFrame(),
        dry_boundary_node_ids=boundary_node_ids, unclassify_failed_origins=True,
        legacy_collocated_rule=False, legacy_centroid_inundation_join=False,
        bridge_rule_applied=ol.BRIDGE_RULE, bridge_structure_retained_lookup=retention,
    )


def synthetic_origins(keys, node_ids, snap_valid, snap_distance, land_area):
    n = len(keys)
    return pd.DataFrame({
        "block_geoid": keys, "block_group_geoid": "", "tract_geoid": "", "block": "",
        "county_fips": "", "county_name": "", "pop20": 0, "land_area_m2": land_area,
        "origin_geometry_method": "random_draw", "node_id": node_ids, "snap_distance_m": snap_distance,
        "snap_valid": snap_valid, "origin_in_lcc": True,
        "nearby_bridge_structure_id": pd.array([pd.NA] * n, dtype="Int64"),
        "nearby_bridge_structure_distance_m": np.nan,
    })


def state_from_production(frame):
    return base.classify_status_columns(
        frame, inundated_col="block_centroid_inundated", isolated_col="block_centroid_isolated",
        redundant_col="block_centroid_redundant", fragile_col="block_centroid_fragile",
        unclassified_col="block_centroid_unclassified")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", type=int, required=True, choices=ol.SCENARIOS)
    p.add_argument("--skip-exhaustive", action="store_true", help="Skip Validation 1c (all LCC nodes).")
    args = p.parse_args()
    s = args.scenario
    rss = ol.PeakRSS().start()
    t_all = time.time()
    timings = {}

    def stage(name, t0):
        timings[name] = round(time.time() - t0, 1)
        ol.log(f"[{s} ft] {name}: {timings[name]} s (peak RSS {rss.peak_gib} GiB)")

    prep = ol.read_json(ol.WORK_DIR / "prepare_manifest.json")
    road_bounds = tuple(prep["road_bounds_lonlat"])
    boundary_polygon = base.build_study_area_boundary(road_bounds, "OGC:CRS84")
    source_clip_polygon = base.box(*road_bounds)

    t0 = time.time()
    nodes, edges = ol.load_network()
    edges, structures, landing = base.build_bridge_structures(edges)
    raw_membership = ol.load_raw_membership()
    services = pd.read_parquet(ol.WORK_DIR / "services_attached.parquet")
    services = services.loc[services["snap_valid"].astype(bool)].copy()
    boundary_node_ids = base.build_boundary_node_set(nodes, boundary_polygon)
    stage("load_network", t0)

    t0 = time.time()
    slr_layer = base.load_slr_layer(ol.LAYERS[s], source_clip_polygon)
    dry_edges, audit, summary, retention = base.apply_bridge_rule_to_edges(
        edges=edges, nodes=nodes, slr_layer=slr_layer, structures=structures, landing_nodes=landing,
        bridge_rule=ol.BRIDGE_RULE, slr_ft=s)
    canon_audit = pd.read_csv(ol.CANONICAL_RUN / f"bridge_structures_slr_{s}ft.csv")
    a = audit.merge(canon_audit, on="structure_id", suffixes=("", "_canon"), how="outer", indicator=True)
    bridge_check = {
        "n_structures": int(len(audit)), "n_canonical": int(len(canon_audit)),
        "row_alignment_mismatch": int((a["_merge"] != "both").sum()),
        **{f"{c}_mismatch": int((a[c].fillna("").astype(str) != a[f"{c}_canon"].fillna("").astype(str)).sum())
           for c in ["landing_node_ids", "dry_landing_count", "retained", "removed_edge_count"]},
        **{k: int(v) for k, v in summary.iloc[0].items() if k.startswith("n_")},
    }
    ol.log(f"[{s} ft] bridge audit vs canonical: {bridge_check}")
    stage("dry_edges", t0)

    # Raw-polygon inundation cross-check (union-free) for all draws and canonical points.
    t0 = time.time()
    draws = pd.read_parquet(ol.WORK_DIR / "draws.parquet",
                            columns=["draw_id", "lon", "lat", "node_id", "snap_valid", "snap_distance_m",
                                     "land_area_m2_census", "pilot_block"])
    meta = pd.read_parquet(ol.WORK_DIR / "block_metadata.parquet",
                           columns=["block_geoid", "canonical_x", "canonical_y"])
    raw_flags = {}
    if slr_layer is not None:
        # Tree on the points, prepared polygons as queries: intersects(polygon, point)
        # with a prepared polygon is fast; the reverse orientation is ~100x slower.
        polys = slr_layer.geometry.values.copy()
        shapely.prepare(polys)
        tree = shapely.STRtree(gpd.points_from_xy(draws["lon"], draws["lat"]))
        raw_flags["draw"] = np.unique(tree.query(polys, predicate="intersects")[1])
        cpts = ol.to_lonlat(meta["canonical_x"].to_numpy(), meta["canonical_y"].to_numpy())
        ctree = shapely.STRtree(cpts.values)
        raw_flags["canonical"] = np.unique(ctree.query(polys, predicate="intersects")[1])
        del tree, ctree, polys
    else:
        raw_flags = {"draw": np.array([], dtype=int), "canonical": np.array([], dtype=int)}
    ol.atomic_to_parquet(pd.concat([
        pd.DataFrame({"kind": "draw", "key": draws["draw_id"].to_numpy()[raw_flags["draw"]].astype(str)}),
        pd.DataFrame({"kind": "canonical", "key": meta["block_geoid"].to_numpy()[raw_flags["canonical"]]}),
    ], ignore_index=True), ol.WORK_DIR / f"raw_polygon_inundation_slr_{s}ft.parquet")
    del slr_layer, edges, audit
    stage("raw_polygon_inundation", t0)

    t0 = time.time()
    dry_graph = base.build_graph(dry_edges)
    del dry_edges
    graph_info = {"dry_nodes": int(dry_graph.number_of_nodes()), "dry_edges": int(dry_graph.number_of_edges())}
    stage("build_dry_graph", t0)

    checks = {"slr_ft": s, "layer": ol.LAYERS[s], "bridge_check": bridge_check, "graph": graph_info}
    with no_nearest_lookup(), al.GraphMemo(base) as memo:
        t0 = time.time()
        all_nodes = nodes["node_id"].to_numpy(dtype=np.int64)
        table = ol.node_state_table(dry_graph, services, boundary_node_ids, all_nodes)
        table["raw_in_lcc"] = table["node_id"].isin(raw_membership["largest_component_nodes"])
        ol.atomic_to_parquet(table, ol.WORK_DIR / f"node_states_slr_{s}ft.parquet")
        lcc_table = table.loc[table["raw_in_lcc"]]
        checks["node_table"] = {
            "n_nodes": int(len(table)), "n_lcc_nodes": int(len(lcc_table)),
            "lcc_state_counts": {ol.CODE_STATES[k]: int(v) for k, v in lcc_table["node_state"].value_counts().items()},
            "lcc_nodes_absent_from_dry_graph": int((~lcc_table["in_dry_graph"]).sum()),
            "production_other_state_nodes": int(table["node_state_other"].sum()),
        }
        ol.log(f"[{s} ft] node table: {checks['node_table']}")
        stage("node_table", t0)

        # Validation 1c: exhaustive node-level equivalence with the production function.
        if not args.skip_exhaustive:
            t0 = time.time()
            lcc_nodes = lcc_table["node_id"].to_numpy()
            expected = lcc_table["node_state"].map(ol.CODE_STATES).to_numpy()
            mismatches, n_checked = [], 0
            node_xy = nodes.set_index("node_id").loc[lcc_nodes, ["x", "y"]].to_numpy()
            for start in range(0, len(lcc_nodes), CHUNK):
                sl = slice(start, start + CHUNK)
                keys = lcc_nodes[sl].astype(str)
                origins = synthetic_origins(keys, lcc_nodes[sl], True, 0.0, 1)
                geom = gpd.GeoDataFrame({"block_geoid": keys},
                                        geometry=gpd.points_from_xy(node_xy[sl, 0], node_xy[sl, 1]),
                                        crs=base.PROJECTED_CRS)
                res = production_classify(slr_ft=s, slr_layer=None, graph=dry_graph, services=services,
                                          origins=origins, geometry=geom, boundary_node_ids=boundary_node_ids,
                                          retention=retention)
                got = state_from_production(res).to_numpy()
                bad = np.flatnonzero(got != expected[sl])
                mismatches.extend(zip(lcc_nodes[sl][bad].tolist(), expected[sl][bad].tolist(), got[bad].tolist()))
                n_checked += len(keys)
            checks["exhaustive_node_equivalence"] = {
                "n_lcc_nodes_checked": int(n_checked), "n_mismatch": int(len(mismatches)),
                "examples": mismatches[:20], "memo_hits": int(memo.hits), "memo_misses": int(memo.misses),
            }
            ol.log(f"[{s} ft] exhaustive node equivalence: {checks['exhaustive_node_equivalence']}")
            stage("validation_1c_exhaustive_nodes", t0)

        # Validation 1b: production function on every pilot draw (with inundation).
        t0 = time.time()
        pilot = draws.loc[draws["pilot_block"]].reset_index(drop=True)
        keys = pilot["draw_id"].astype(str).to_numpy()
        origins = synthetic_origins(keys, pilot["node_id"].to_numpy(), pilot["snap_valid"].to_numpy(),
                                    pilot["snap_distance_m"].to_numpy(), pilot["land_area_m2_census"].to_numpy())
        union_geom = ol.load_union(s)
        dissolved = gpd.GeoDataFrame({"Id": [0]}, geometry=[union_geom], crs="OGC:CRS84")
        geom = gpd.GeoDataFrame({"block_geoid": keys}, geometry=gpd.points_from_xy(pilot["lon"], pilot["lat"]),
                                crs="OGC:CRS84")
        with al.union_passthrough(base, union_geom):
            res = production_classify(slr_ft=s, slr_layer=dissolved, graph=dry_graph, services=services,
                                      origins=origins, geometry=geom, boundary_node_ids=boundary_node_ids,
                                      retention=retention)
        out = pd.DataFrame({"draw_id": pilot["draw_id"].to_numpy(),
                            "production_status": state_from_production(res).to_numpy(),
                            "production_inundated": res["block_centroid_inundated"].to_numpy(),
                            "production_analysis_eligible": res["analysis_eligible"].to_numpy(),
                            "production_max_edge_disjoint_paths": res["max_edge_disjoint_paths_any_service"].to_numpy(),
                            "production_n_reachable_services": res["n_reachable_services"].to_numpy()})
        ol.atomic_to_parquet(out, ol.WORK_DIR / f"production_pilot_slr_{s}ft.parquet")
        checks["production_pilot"] = {"n_pilot_draws": int(len(out)),
                                      "status_counts": out["production_status"].value_counts().to_dict()}
        stage("validation_1b_production_pilot", t0)

    rss.stop()
    checks.update({"timings_s": timings, "total_s": round(time.time() - t_all, 1), "peak_rss_gib": rss.peak_gib})
    ol.atomic_write_json(ol.WORK_DIR / f"node_states_slr_{s}ft_checks.json", checks)
    ol.log(f"[{s} ft] done in {checks['total_s']} s; peak RSS {rss.peak_gib} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
