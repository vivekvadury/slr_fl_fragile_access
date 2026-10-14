#!/usr/bin/env python
"""Stage 1-2: block geometry, randomized within-block origins, and LCC snaps.

For every canonical analysis-eligible block (68,521 in the approach run) this
script draws ``--n-draws`` points uniformly inside the block's *land* polygon
(2020 TIGER block polygon minus intersecting 2020 TIGER AREAWATER polygons) in
the canonical projected CRS (EPSG:32617), and snaps every point to the raw
graph's largest connected component with the production snap function and the
canonical 2 km validity threshold. The canonical representative point is kept
separately (``block_metadata.parquet``) and is never counted as a draw.

Reproducibility: block b uses its own generator
``np.random.default_rng([seed, int(GEOID)])``. Candidates are drawn in stream
order from the land polygon's bounding box and accepted when
``shapely.contains_xy(land_polygon, x, y)`` holds (rejection sampling with an
explicit polygon-membership test). The r-th accepted candidate is draw r, so a
10-draw run reproduces the first 10 draws of a 50-draw run exactly.

Outputs (outputs/origin_placement_sensitivity/work/):
  draws.parquet            one row per (block, draw): coordinates, land area,
                           seed, snapped node, snap distance, snap validity
  block_metadata.parquet   one row per block (all 70,695): canonical point and
                           snap, land/water geometry checks, coastal distance,
                           network-complexity counts, pilot strata
  services_attached.parquet  production service attachment (validated)
  prepare_manifest.json
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
import shapely  # noqa: E402

PILOT_STRATA_N = {"small_urban": 300, "large_suburban": 300, "coastal": 300,
                  "network_complex": 300, "near_inundation_boundary": 300, "random": 500}
BOUNDARY_DENSIFY_M = 25.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-draws", type=int, default=50)
    p.add_argument("--seed", type=int, default=ol.DEFAULT_SEED)
    p.add_argument("--max-candidates", type=int, default=50_000_000,
                   help="Hard cap on rejection-sampling candidates per block.")
    return p.parse_args()


def load_blocks():
    blocks = base.maybe_to_projected(base.prepare_blocks_layer(base.read_vector(base.BLOCKS_PATH)))
    blocks = blocks.merge(base.load_block_attributes(), on="block_geoid", how="left", validate="one_to_one")
    aw = base.pyogrio.read_dataframe(base.CENSUS_BLOCK_ATTRIBUTES_PATH, columns=["GEOID20", "AWATER20"],
                                     read_geometry=False)
    aw["block_geoid"] = aw["GEOID20"].astype(str).str.zfill(15)
    blocks = blocks.merge(aw[["block_geoid", "AWATER20"]], on="block_geoid", how="left", validate="one_to_one")
    blocks = blocks.rename(columns={"AWATER20": "water_area_m2_census"})
    assert blocks[["pop20", "land_area_m2", "water_area_m2_census"]].notna().all(axis=None)
    return blocks.reset_index(drop=True)


def load_water():
    frames = [base.pyogrio.read_dataframe(f"zip://{z.as_posix()}") for z in ol.AREAWATER_ZIPS]
    water = pd.concat(frames, ignore_index=True)
    water = gpd.GeoDataFrame(water, geometry="geometry", crs=frames[0].crs)
    water = base.maybe_to_projected(water)  # same EPSG:4269 handling as the blocks
    water["geometry"] = shapely.make_valid(water.geometry.values)
    return water.reset_index(drop=True)


def build_land_polygons(blocks, water):
    """Block polygon minus intersecting AREAWATER polygons (projected CRS)."""
    block_geoms = shapely.make_valid(blocks.geometry.values)
    pairs = water.sindex.query(block_geoms, predicate="intersects")
    land = block_geoms.copy()
    hits = pd.DataFrame({"b": pairs[0], "w": pairs[1]}).groupby("b")["w"].apply(list)
    wg = water.geometry.values
    for b, ws in hits.items():
        land[b] = shapely.make_valid(shapely.difference(block_geoms[b], shapely.union_all(wg[ws])))
    # Keep only polygonal parts (differences can leave line/point slivers).
    land = np.array([shapely.union_all([g for g in shapely.get_parts(geom)
                                        if g.geom_type in ("Polygon", "MultiPolygon")])
                     if geom.geom_type == "GeometryCollection" else geom for geom in land], dtype=object)
    return block_geoms, land, hits.index.to_numpy()


def sample_block(poly, n, seed, geoid_int, max_candidates):
    rng = np.random.default_rng([seed, geoid_int])
    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny
    ratio = max(poly.area / (w * h), 1e-9)
    shapely.prepare(poly)
    xs, ys, used = [], [], 0
    got = 0
    while got < n:
        k = int(min(max(np.ceil((n - got) / ratio * 1.3) + 16, 64), 5_000_000))
        u = rng.random((k, 2))  # stream order: (x0, y0, x1, y1, ...) regardless of k
        used += k
        x = minx + u[:, 0] * w
        y = miny + u[:, 1] * h
        ok = shapely.contains_xy(poly, x, y)
        xs.append(x[ok])
        ys.append(y[ok])
        got += int(ok.sum())
        if used > max_candidates and got < n:
            raise RuntimeError(f"Rejection sampling exceeded {max_candidates} candidates for {geoid_int}.")
    return np.concatenate(xs)[:n], np.concatenate(ys)[:n], used


def boundary_vertex_tree(geom_lonlat, densify_m=BOUNDARY_DENSIFY_M):
    """KD-tree on densified boundary vertices of a lon/lat polygon (projected)."""
    from pyproj import Transformer

    tr = Transformer.from_crs("OGC:CRS84", base.PROJECTED_CRS, always_xy=True)
    bnd = shapely.boundary(geom_lonlat)
    coords = shapely.get_coordinates(bnd)
    x, y = tr.transform(coords[:, 0], coords[:, 1])
    # Densify in projected space by re-segmentizing the projected boundary.
    line = shapely.transform(bnd, lambda c: np.column_stack(tr.transform(c[:, 0], c[:, 1])))
    dense = shapely.get_coordinates(shapely.segmentize(line, densify_m))
    return base.cKDTree(dense), len(coords), len(dense)


def main():
    args = parse_args()
    ol.WORK_DIR.mkdir(parents=True, exist_ok=True)
    rss = ol.PeakRSS().start()
    t_all = time.time()
    timings = {}

    def stage(name, t0):
        timings[name] = round(time.time() - t0, 1)
        ol.log(f"{name}: {timings[name]} s (peak RSS {rss.peak_gib} GiB)")

    # ---- canonical universe ---------------------------------------------------------------
    t0 = time.time()
    canon = ol.load_canonical_long(columns=[
        "block_geoid", "slr_ft", "analysis_eligible", "exclusion_reason", "origin_node_id",
        "origin_snap_distance_m", "scenario_status", "pop20", "land_area_m2"])
    canon0 = canon.loc[canon["slr_ft"].eq(0)].set_index("block_geoid")
    stage("load_canonical", t0)

    # ---- blocks, canonical points, services (production functions) ----------------------
    t0 = time.time()
    blocks = load_blocks()
    centroids = base.make_centroids(blocks, use_representative_point=True)
    roads = base.load_roads(smoke=False).set_crs("OGC:CRS84", allow_override=True)
    road_bounds = tuple(float(v) for v in roads.total_bounds)
    del roads
    boundary_polygon = base.build_study_area_boundary(road_bounds, "OGC:CRS84")
    stage("load_blocks_roads", t0)

    t0 = time.time()
    nodes = gpd.read_parquet(ol.SOURCE_CACHE_DIR / f"segmentized_nodes_{ol.V2_STEM}.parquet")
    raw_membership = ol.load_raw_membership()
    tree, _, node_ids = base.build_node_kdtree(nodes)
    services = base.attach_services_to_raw_graph(
        base.filter_services_by_buffer(base.load_services(), boundary_polygon),
        nodes=nodes, unconstrained_tree=tree, unconstrained_node_ids=node_ids,
        raw_membership=raw_membership, use_eligible_service_nodes=True)
    del tree
    canon_svc = pd.read_csv(ol.CANONICAL_RUN / "service_snapping_audit.csv")
    svc_cmp = services[["service_record_id", "service_id", "node_id", "snap_valid", "service_snap_rule"]].merge(
        canon_svc[["service_record_id", "service_id", "node_id", "snap_valid", "service_snap_rule"]],
        on="service_record_id", suffixes=("", "_canon"), how="outer", indicator=True)
    service_check = {
        "n_services": int(len(services)), "n_canonical": int(len(canon_svc)),
        "row_alignment_mismatch": int((svc_cmp["_merge"] != "both").sum()),
        **{f"{c}_mismatch": int((svc_cmp[c].astype(str) != svc_cmp[f"{c}_canon"].astype(str)).sum())
           for c in ["service_id", "node_id", "snap_valid", "service_snap_rule"]},
        "n_valid": int(services["snap_valid"].sum()),
    }
    ol.log(f"service attachment vs canonical audit: {service_check}")
    svc_out = pd.DataFrame(services.drop(columns="geometry"))
    ol.atomic_to_parquet(svc_out, ol.WORK_DIR / "services_attached.parquet")
    stage("services", t0)

    # ---- canonical point snap (V1 precursor) -------------------------------------------------
    t0 = time.time()
    csnap = ol.lcc_snap(centroids, id_col="block_geoid", nodes=nodes, raw_membership=raw_membership)
    cmp = csnap.set_index("block_geoid").join(canon0, how="outer")
    canon_snap_check = {
        "n_blocks": int(len(cmp)),
        "node_id_mismatch": int((cmp["node_id"] != cmp["origin_node_id"]).sum()),
        "snap_distance_max_abs_diff": float((cmp["snap_distance_m"] - cmp["origin_snap_distance_m"]).abs().max()),
        "snap_valid_vs_exclusion_mismatch": int((~cmp["snap_valid"]).ne(cmp["exclusion_reason"].eq("origin_snap_failed")).sum()),
    }
    ol.log(f"canonical point snap vs canonical run: {canon_snap_check}")
    stage("canonical_snap", t0)

    # ---- land polygons --------------------------------------------------------------------------
    t0 = time.time()
    water = load_water()
    block_geoms, land, water_blocks = build_land_polygons(blocks, water)
    land_area_geom = shapely.area(land)
    blocks["land_area_geom_m2"] = land_area_geom
    blocks["block_area_geom_m2"] = shapely.area(block_geoms)
    blocks["has_areawater_overlap"] = False
    blocks.loc[water_blocks, "has_areawater_overlap"] = True
    blocks["land_geom_fallback"] = (land_area_geom < 1.0) & blocks["land_area_m2"].gt(0)
    land = np.where(blocks["land_geom_fallback"].to_numpy(), block_geoms, land)
    blocks["sampling_area_m2"] = shapely.area(land)
    rp = centroids.geometry.values
    blocks["canonical_point_on_land"] = shapely.contains(land, rp) | shapely.touches(land, rp)
    blocks["canonical_point_in_block"] = shapely.covers(block_geoms, rp)
    stage("land_polygons", t0)

    # ---- coastal distance, network complexity -----------------------------------------------------
    t0 = time.time()
    coastal = water.loc[water["MTFCC"].isin(ol.COASTAL_MTFCC)]
    coast_union = shapely.union_all(coastal.geometry.values)
    shapely.prepare(coast_union)
    coast_tree = base.cKDTree(shapely.get_coordinates(shapely.segmentize(shapely.boundary(coast_union),
                                                                         BOUNDARY_DENSIFY_M)))
    d, _ = coast_tree.query(np.column_stack([centroids.geometry.x, centroids.geometry.y]), k=1)
    in_coastal_water = shapely.contains(coast_union, rp)
    blocks["coastal_distance_m"] = np.where(in_coastal_water, 0.0, d)
    blocks["coastal"] = blocks["coastal_distance_m"].le(ol.COASTAL_DISTANCE_M)
    lcc = np.array(sorted(raw_membership["largest_component_nodes"]))
    lcc_nodes = nodes.set_index("node_id").loc[lcc]
    lcc_pts = gpd.GeoSeries(lcc_nodes.geometry.values, crs=base.PROJECTED_CRS)
    pairs = lcc_pts.sindex.query(block_geoms, predicate="contains")
    pair_df = pd.DataFrame({"b": pairs[0], "node_id": lcc[pairs[1]]})
    pair_df["raw_2ecc_id"] = pair_df["node_id"].map(raw_membership["node_to_2ecc_id"])
    pair_df["raw_2ecc_size"] = pair_df["node_id"].map(raw_membership["node_to_2ecc_size"])
    agg = pair_df.groupby("b").agg(n_lcc_nodes_inside=("node_id", "size"),
                                   n_raw_2ecc_inside=("raw_2ecc_id", "nunique"),
                                   n_raw_bridge_nodes_inside=("raw_2ecc_size", lambda s: int((s == 1).sum())))
    for c in agg.columns:
        blocks[c] = agg[c].reindex(range(len(blocks))).fillna(0).astype(int).to_numpy()
    stage("coastal_and_network_counts", t0)

    # ---- inundation-boundary geometry per scenario ------------------------------------------------
    t0 = time.time()
    # Point-based (polygon-vs-union predicates degrade badly against the
    # multi-million-vertex NOAA unions): a land polygon straddles the extent
    # boundary when any boundary vertex (densified to 25 m) lies inside it.
    rp_xy = np.column_stack([centroids.geometry.x, centroids.geometry.y])
    rp_ll = ol.to_lonlat(rp_xy[:, 0], rp_xy[:, 1])
    bounds = shapely.bounds(land)
    centers = np.column_stack([(bounds[:, 0] + bounds[:, 2]) / 2, (bounds[:, 1] + bounds[:, 3]) / 2])
    radii = np.hypot(bounds[:, 2] - bounds[:, 0], bounds[:, 3] - bounds[:, 1]) / 2 + 1.0
    bb = shapely.bounds(block_geoms)
    block_diag = np.hypot(bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1]) + 1.0
    boundary_meta = {}
    for s in ol.SCENARIOS:
        ts = time.time()
        u = ol.load_union(s)
        rp_in = ol.inundated_by_union(u, rp_ll)
        bt, nv, nd = boundary_vertex_tree(u)
        dist, _ = bt.query(rp_xy, k=1)
        cand = np.flatnonzero((dist <= block_diag) & np.isfinite(centers).all(axis=1) & np.isfinite(radii))
        straddle = np.zeros(len(land), dtype=bool)
        hits = bt.query_ball_point(centers[cand], radii[cand])
        for i, idx in zip(cand, hits):
            if idx:
                v = bt.data[idx]
                straddle[i] = bool(shapely.contains_xy(land[i], v[:, 0], v[:, 1]).any())
        blocks[f"canonical_point_inundated_{s}ft"] = rp_in
        blocks[f"land_straddles_extent_{s}ft"] = straddle
        blocks[f"dist_to_extent_boundary_{s}ft_m"] = dist
        boundary_meta[s] = {"union_vertices": int(nv), "densified_vertices": int(nd),
                            "blocks_straddling": int(straddle.sum()),
                            "seconds": round(time.time() - ts, 1)}
        ol.log(f"  {s} ft boundary metrics: {boundary_meta[s]}")
        del u, bt
    stage("inundation_boundary_metrics", t0)

    # ---- metadata table + pilot strata ------------------------------------------------------------
    meta = pd.DataFrame(blocks.drop(columns="geometry"))
    meta["canonical_x"] = centroids.geometry.x.to_numpy()
    meta["canonical_y"] = centroids.geometry.y.to_numpy()
    meta = meta.merge(csnap.rename(columns={"node_id": "canonical_node_id", "snap_distance_m": "canonical_snap_distance_m",
                                            "snap_valid": "canonical_snap_valid", "x": "canonical_node_x",
                                            "y": "canonical_node_y", "origin_in_lcc": "canonical_origin_in_lcc"}),
                      on="block_geoid", how="left", validate="one_to_one")
    meta = meta.merge(canon0[["analysis_eligible", "exclusion_reason", "scenario_status"]]
                      .rename(columns={"scenario_status": "canonical_baseline_status"}),
                      left_on="block_geoid", right_index=True, how="left", validate="one_to_one")
    elig = meta["analysis_eligible"].to_numpy(dtype=bool)
    rng = np.random.default_rng(args.seed + 1)
    p25, p90, p99 = np.quantile(meta.loc[elig, "land_area_m2"], [0.25, 0.90, 0.99])
    straddle_any = meta[[f"land_straddles_extent_{s}ft" for s in ol.SCENARIOS]].any(axis=1)
    strata = {
        "small_urban": elig & meta["land_area_m2"].le(p25) & meta["county_name"].isin(["Miami-Dade", "Broward"]),
        "large_suburban": elig & meta["land_area_m2"].between(p90, p99),
        "coastal": elig & meta["coastal"],
        "network_complex": elig & meta["n_raw_2ecc_inside"].ge(np.quantile(meta.loc[elig, "n_raw_2ecc_inside"], 0.9)),
        "near_inundation_boundary": elig & straddle_any,
        "random": pd.Series(elig, index=meta.index),
    }
    pilot = np.zeros(len(meta), dtype=bool)
    strata_meta = {}
    for name, mask in strata.items():
        idx = np.flatnonzero(np.asarray(mask))
        take = rng.choice(idx, size=min(PILOT_STRATA_N[name], len(idx)), replace=False)
        meta[f"pilot_{name}"] = False
        meta.loc[take, f"pilot_{name}"] = True
        meta[f"stratum_{name}"] = np.asarray(mask)
        pilot[take] = True
        strata_meta[name] = {"eligible_in_stratum": int(len(idx)), "sampled": int(len(take))}
    meta["pilot_block"] = pilot
    strata_meta["thresholds"] = {"land_area_p25_m2": float(p25), "land_area_p90_m2": float(p90),
                                 "land_area_p99_m2": float(p99)}
    ol.log(f"pilot blocks: {int(pilot.sum())}; strata {strata_meta}")

    # ---- random draws -------------------------------------------------------------------------------
    t0 = time.time()
    rows_geoid, rows_rep, xs, ys, cand = [], [], [], [], {}
    elig_idx = np.flatnonzero(elig)
    for i in elig_idx:
        geoid = meta.at[i, "block_geoid"]
        x, y, used = sample_block(land[i], args.n_draws, args.seed, int(geoid), args.max_candidates)
        xs.append(x)
        ys.append(y)
        rows_geoid.append(np.repeat(geoid, args.n_draws))
        rows_rep.append(np.arange(args.n_draws, dtype=np.int16))
        cand[geoid] = used
    draws = pd.DataFrame({"block_geoid": np.concatenate(rows_geoid), "rep": np.concatenate(rows_rep),
                          "x": np.concatenate(xs), "y": np.concatenate(ys)})
    draws["seed"] = np.int64(args.seed)
    stage("sampling", t0)

    # Validation 2 geometry checks on every draw.
    t0 = time.time()
    pos = meta.reset_index().set_index("block_geoid").loc[draws["block_geoid"], "index"].to_numpy()
    in_land = shapely.contains_xy(land[pos], draws["x"].to_numpy(), draws["y"].to_numpy())
    in_block = shapely.covers(block_geoms[pos], gpd.points_from_xy(draws["x"], draws["y"]))
    # Must not fall in any other block (TIGER blocks tile the plane).
    owner = gpd.GeoSeries(block_geoms, crs=base.PROJECTED_CRS).sindex.query(
        gpd.points_from_xy(draws["x"], draws["y"]), predicate="within")
    owner_df = pd.DataFrame({"d": owner[0], "b": owner[1]})
    n_owner = owner_df.groupby("d").size().reindex(range(len(draws)), fill_value=0)
    wrong_owner = owner_df.loc[owner_df["b"].to_numpy() != pos[owner_df["d"].to_numpy()]]
    draws["land_area_geom_m2"] = meta["sampling_area_m2"].to_numpy()[pos]
    draws["land_area_m2_census"] = meta["land_area_m2"].to_numpy()[pos]
    ll = ol.to_lonlat(draws["x"].to_numpy(), draws["y"].to_numpy())
    draws["lon"] = ll.x.to_numpy()
    draws["lat"] = ll.y.to_numpy()
    geom_check = {
        "n_draws": int(len(draws)), "n_blocks": int(draws["block_geoid"].nunique()),
        "draws_per_block_min": int(draws.groupby("block_geoid").size().min()),
        "draws_per_block_max": int(draws.groupby("block_geoid").size().max()),
        "draws_outside_land_polygon": int((~in_land).sum()),
        "draws_outside_block_polygon": int((~in_block).sum()),
        "draws_strictly_within_n_blocks_ne_1": int((n_owner != 1).sum()),
        "draws_within_a_different_block": int(len(wrong_owner)),
        "duplicate_coordinates": int(draws.duplicated(["x", "y"]).sum()),
        "candidates_total": int(sum(cand.values())),
        "acceptance_rate": float(len(draws) / sum(cand.values())),
    }
    # Nesting property: a fresh 10-draw run reproduces the first 10 of the full run.
    chk_blocks = meta.loc[meta["pilot_block"]].head(200)
    nest_bad = 0
    for i, geoid in zip(chk_blocks.index, chk_blocks["block_geoid"]):
        x10, y10, _ = sample_block(land[i], 10, args.seed, int(geoid), args.max_candidates)
        full = draws.loc[draws["block_geoid"].eq(geoid)].sort_values("rep").head(10)
        nest_bad += int(not (np.array_equal(x10, full["x"].to_numpy()) and np.array_equal(y10, full["y"].to_numpy())))
    geom_check["nesting_check_blocks"] = int(len(chk_blocks))
    geom_check["nesting_check_failures"] = int(nest_bad)
    ol.log(f"draw geometry checks: {geom_check}")
    stage("geometry_checks", t0)

    # ---- snap draws with the production function ---------------------------------------------------
    t0 = time.time()
    draws["draw_id"] = np.arange(len(draws), dtype=np.int64)
    dgeo = gpd.GeoDataFrame({"draw_key": draws["draw_id"].astype(str)},
                            geometry=gpd.points_from_xy(draws["x"], draws["y"]), crs=base.PROJECTED_CRS)
    dsnap = ol.lcc_snap(dgeo, id_col="draw_key", nodes=nodes, raw_membership=raw_membership)
    assert (dsnap["draw_key"].to_numpy() == dgeo["draw_key"].to_numpy()).all()
    draws["node_id"] = dsnap["node_id"].to_numpy(dtype=np.int64)
    draws["snap_distance_m"] = dsnap["snap_distance_m"].to_numpy(dtype=float)
    draws["snap_valid"] = dsnap["snap_valid"].to_numpy(dtype=bool)
    draws["node_in_lcc"] = dsnap["origin_in_lcc"].to_numpy(dtype=bool)
    snap_check = {
        "snap_invalid_draws": int((~draws["snap_valid"]).sum()),
        "snap_invalid_share": float((~draws["snap_valid"]).mean()),
        "blocks_with_any_invalid_draw": int(draws.loc[~draws["snap_valid"], "block_geoid"].nunique()),
        "all_snapped_nodes_in_lcc": bool(draws["node_in_lcc"].all()),
        "snap_distance_p50": float(draws["snap_distance_m"].median()),
        "snap_distance_p99": float(draws["snap_distance_m"].quantile(0.99)),
        "snap_distance_max": float(draws["snap_distance_m"].max()),
    }
    ol.log(f"draw snap checks: {snap_check}")
    stage("snap_draws", t0)

    draws["pilot_block"] = meta.set_index("block_geoid").loc[draws["block_geoid"], "pilot_block"].to_numpy()
    cols = ["draw_id", "block_geoid", "rep", "seed", "x", "y", "lon", "lat", "land_area_geom_m2",
            "land_area_m2_census", "node_id", "snap_distance_m", "snap_valid", "node_in_lcc", "pilot_block"]
    ol.atomic_to_parquet(draws[cols], ol.WORK_DIR / "draws.parquet")
    ol.atomic_to_parquet(meta, ol.WORK_DIR / "block_metadata.parquet")

    land_ratio = (meta.loc[elig, "sampling_area_m2"] / meta.loc[elig, "land_area_m2"]).describe(
        percentiles=[0.001, 0.01, 0.5, 0.99, 0.999]).to_dict()
    rss.stop()
    manifest = {
        "script": "scripts/origin_placement_sensitivity/01_prepare_draws.py",
        "seed": args.seed, "n_draws": args.n_draws,
        "rng": "numpy default_rng(SeedSequence([seed, int(block_geoid)])); rejection sampling in land-polygon bbox",
        "projected_crs": base.PROJECTED_CRS, "max_origin_snap_m": base.MAX_ORIGIN_SNAP_M,
        "road_bounds_lonlat": road_bounds,
        "eligible_blocks": int(elig.sum()), "all_blocks": int(len(meta)),
        "service_check": service_check, "canonical_snap_check": canon_snap_check,
        "land_geometry": {
            "blocks_with_areawater_overlap": int(meta["has_areawater_overlap"].sum()),
            "land_geom_fallback_blocks": int(meta["land_geom_fallback"].sum()),
            "eligible_land_area_ratio_geom_over_ALAND20": {k: float(v) for k, v in land_ratio.items()},
            "eligible_canonical_points_not_on_land": int((~meta.loc[elig, "canonical_point_on_land"]).sum()),
            "eligible_canonical_points_not_in_block": int((~meta.loc[elig, "canonical_point_in_block"]).sum()),
        },
        "coastal_definition": f"canonical representative point within {ol.COASTAL_DISTANCE_M} m of TIGER 2020 "
                              f"AREAWATER MTFCC {list(ol.COASTAL_MTFCC)} (bay/estuary/sound, ocean); distance to "
                              f"boundary vertices densified at {BOUNDARY_DENSIFY_M} m",
        "boundary_metrics": boundary_meta,
        "pilot_strata": strata_meta,
        "geometry_check": geom_check, "snap_check": snap_check,
        "timings_s": timings, "total_s": round(time.time() - t_all, 1), "peak_rss_gib": rss.peak_gib,
        "inputs": {str(p.relative_to(ol.PROJECT_ROOT)): ol.sha256_file(p) for p in ol.AREAWATER_ZIPS},
    }
    ol.atomic_write_json(ol.WORK_DIR / "prepare_manifest.json", manifest)
    ol.log(f"done in {manifest['total_s']} s; peak RSS {rss.peak_gib} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
