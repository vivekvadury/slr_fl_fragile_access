#!/usr/bin/env python
"""Run block-level access classification for the attachment-sensitivity arms.

All arms share one road graph, one set of NOAA extents, and the ``approach``
bridge rule. For each scenario the dry graph is built once; every arm is then
classified by the *production* ``scenario_results_for_origins`` with that arm's
origin validity and facility map. Facility-independent graph computations are
memoized per graph object (see ``attachment_lib.GraphMemo``).

Outputs (never inside the production directories):
  data/processed/access/attachment_sensitivity/<experiment>/<arm>/block_access_flags_long.parquet
  .../<arm>/service_attachment_audit.csv, origin_snap_audit.csv, arm_manifest.json, _COMPLETE.json
  .../shared/bridge_structures_slr_<s>ft.csv, experiment_manifest.json, stage timings

Example:
  python scripts/attachment_sensitivity/run_access_arms.py --experiment-id attach_20260925
  python scripts/attachment_sensitivity/run_access_arms.py --experiment-id bench --scenarios 0 --arms reference
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attachment_lib as al  # noqa: E402

base = al.load_base_module()
gpd, nx, np, pd = base.gpd, base.nx, base.np, base.pd

PROJECT_ROOT = al.PROJECT_ROOT
REFERENCE_RUN = PROJECT_ROOT / "data/processed/access/edited/della_runs/positive_layer_20260814_approach"
DEFAULT_SOURCE_CACHE = PROJECT_ROOT / "data/processed/access/cache"
V2_STEM = "fe639d75a84cb809_full_v2"


def log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class PeakRSS:
    """Sample this process's resident memory in a background thread."""

    def __init__(self, interval: float = 2.0):
        import psutil

        self.proc = psutil.Process()
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        self.peak = max(self.peak, self.proc.memory_info().rss)

    @property
    def peak_gib(self) -> float:
        return round(self.peak / 2**30, 2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--arms", default=",".join(al.ARM_ORDER), help="Comma-separated attachment arms.")
    p.add_argument("--scenarios", default="0,1,2,3,4,5,6")
    p.add_argument("--source-cache-dir", type=Path, default=DEFAULT_SOURCE_CACHE,
                   help="Directory holding the verified v2 segmentized cache (read-only).")
    p.add_argument("--out-root", type=Path,
                   default=PROJECT_ROOT / "data/processed/access/attachment_sensitivity")
    p.add_argument("--union-method", choices=["union_all", "coverage_union_all"], default="union_all",
                   help="How to build an UNCACHED scenario union. coverage_union_all is valid only for "
                        "non-overlapping polygons; it was checked against union_all at 0 and 5 ft "
                        "(identical area, 0 mismatches at every origin and graph node).")
    p.add_argument("--skip-input-hashes", action="store_true",
                   help="Skip SHA-256 of large inputs (benchmarks only).")
    return p.parse_args()


def git_state() -> dict:
    def run(*cmd):
        try:
            return subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            return f"unavailable: {exc}"

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "status_porcelain": run("git", "status", "--porcelain"),
    }


def source_hashes() -> dict:
    here = Path(__file__).resolve().parent
    files = [base.BASE_SCRIPT if hasattr(base, "BASE_SCRIPT") else al.BASE_SCRIPT,
             here / "attachment_lib.py", Path(__file__).resolve()]
    return {str(Path(f).relative_to(PROJECT_ROOT)): al.sha256_file(f) for f in files}


def input_hashes(skip: bool) -> dict:
    ref_manifest = json.loads((REFERENCE_RUN / "run_manifest.json").read_text(encoding="utf-8"))
    ref_by_name = {Path(rec["path"]).name: rec for rec in ref_manifest["input_files"]}
    paths = [base.BLOCKS_PATH, base.CENSUS_BLOCK_ATTRIBUTES_PATH, base.NOAA_GPKG_PATH,
             base.PRIVATE_SCHOOLS_PATH, base.PUBLIC_SCHOOLS_PATH, base.FIRE_STATIONS_PATH, base.ROAD_PBF_PATH]
    out = {}
    for path in paths:
        rec = {"path": str(path.relative_to(PROJECT_ROOT)), "size_bytes": path.stat().st_size}
        ref = ref_by_name.get(path.name)
        rec["reference_manifest_sha256"] = ref["sha256"] if ref else None
        if not skip:
            rec["sha256"] = al.sha256_file(path)
            rec["matches_reference_manifest"] = (ref is not None and rec["sha256"] == ref["sha256"])
        out[path.name] = rec
    return out


def versions() -> dict:
    import importlib.metadata as md

    pkgs = ["geopandas", "shapely", "pyogrio", "pandas", "numpy", "networkx", "scipy", "pyproj", "pyarrow"]
    return {"python": sys.version, "platform": platform.platform(), **{p: md.version(p) for p in pkgs}}


def load_cached_network(source_cache_dir: Path):
    """Load the v2 segmentized cache and apply the corrected positive layer gate.

    The v2 cache stores raw ``bridge_tag_present`` and ``layer_value`` per edge;
    ``segmentize_roads`` defines bridge_like = bridge tag OR positive layer, so
    recomputing that column reproduces the corrected cache exactly. Node/edge
    topology does not depend on the layer gate.
    """
    nodes = gpd.read_parquet(source_cache_dir / f"segmentized_nodes_{V2_STEM}.parquet")
    edges = gpd.read_parquet(source_cache_dir / f"segmentized_edges_{V2_STEM}.parquet")
    legacy_bridge_like = edges["bridge_like"].astype(bool)
    positive = edges["layer_value"].map(base.is_positive_layer_value).astype(bool)
    edges["bridge_like"] = edges["bridge_tag_present"].astype(bool) | positive
    changed = int((legacy_bridge_like != edges["bridge_like"]).sum())
    return nodes, edges, {"v2_cache_stem": V2_STEM, "bridge_like_reclassified_edges": changed,
                          "n_nodes": int(len(nodes)), "n_edges": int(len(edges))}


def main() -> int:
    args = parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = sorted(set(arms) - set(al.ARM_SPECS))
    if unknown:
        raise SystemExit(f"Unknown arms: {unknown}")
    scenarios = sorted({int(s) for s in args.scenarios.split(",")})
    if 0 not in scenarios:
        raise SystemExit("Scenario 0 is required: transitions are relative to the 0 ft dry graph.")
    layers = {0: base.BASELINE_SLR_LAYER, **base.SLR_LAYER_MAP}

    exp_dir = args.out_root / args.experiment_id
    shared_dir = exp_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    rss = PeakRSS().start()
    t_all = time.time()

    def stage(name, t0):
        timings[name] = round(time.time() - t0, 1)
        log(f"stage {name}: {timings[name]} s; peak RSS so far {round(max(rss.peak, rss.proc.memory_info().rss) / 2**30, 2)} GiB")

    # ---- provenance ---------------------------------------------------------
    t0 = time.time()
    provenance = {
        "experiment_id": args.experiment_id,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": git_state(),
        "source_sha256": source_hashes(),
        "versions": versions(),
        "arms": {a: al.ARM_SPECS[a] for a in arms},
        "bridge_rule": "approach",
        "scenarios": scenarios,
        "production_constants": {
            "MAX_ORIGIN_SNAP_M": base.MAX_ORIGIN_SNAP_M,
            "MAX_SERVICE_SNAP_M": base.MAX_SERVICE_SNAP_M,
            "SERVICE_BUFFER_M": base.SERVICE_BUFFER_M,
            "DROP_PRIVATE_ACCESS_EDGES": base.DROP_PRIVATE_ACCESS_EDGES,
            "MAX_EDGE_DISJOINT_PATHS_CAP": base.MAX_EDGE_DISJOINT_PATHS_CAP,
        },
        "input_files": input_hashes(args.skip_input_hashes),
    }
    al.atomic_write_json(shared_dir / "experiment_manifest.json", provenance)
    stage("provenance", t0)

    # ---- blocks, facilities, roads (as production main()) --------------------
    t0 = time.time()
    blocks = base.maybe_to_projected(base.prepare_blocks_layer(base.read_vector(base.BLOCKS_PATH)))
    blocks = blocks.merge(base.load_block_attributes(), on="block_geoid", how="left", validate="one_to_one")
    if blocks[["pop20", "land_area_m2"]].isna().any(axis=None):
        raise ValueError("Blocks missing POP20/ALAND20.")
    centroids = base.make_centroids(blocks, use_representative_point=True)
    candidates_all = base.load_services()
    roads = base.load_roads(smoke=False).set_crs("OGC:CRS84", allow_override=True)
    road_bounds = tuple(float(v) for v in roads.total_bounds)
    boundary_polygon = base.build_study_area_boundary(road_bounds, roads.crs)
    source_clip_polygon = base.box(*road_bounds)
    del roads
    centroid_boundary = base.compute_origin_boundary_fields(centroids, boundary_polygon)
    candidates = base.filter_services_by_buffer(candidates_all, boundary_polygon)
    centroids_source = centroids.to_crs("OGC:CRS84")
    stage("load_blocks_facilities_roads", t0)
    log(f"blocks={len(blocks):,} candidates_in_buffer={len(candidates):,} road_bounds={road_bounds}")

    # ---- network ------------------------------------------------------------
    t0 = time.time()
    nodes, edges, cache_info = load_cached_network(args.source_cache_dir)
    edges, bridge_structures, bridge_landing_nodes = base.build_bridge_structures(edges)
    graph_raw = base.build_graph(edges)
    tree, _, node_ids = base.build_node_kdtree(nodes)
    stage("load_network", t0)
    log(f"nodes={len(nodes):,} edges={len(edges):,} bridge_structures={len(bridge_structures):,} "
        f"reclassified_bridge_like={cache_info['bridge_like_reclassified_edges']:,}")

    t0 = time.time()
    raw_membership = base.compute_raw_graph_membership(graph_raw)
    cached_membership = pd.read_parquet(args.source_cache_dir / f"raw_graph_membership_{V2_STEM}.parquet")
    fresh = base.raw_membership_to_frame(raw_membership)
    cmp = fresh.merge(cached_membership, on="node_id", suffixes=("", "_cached"), validate="one_to_one")
    membership_check = {
        "rows_fresh": int(len(fresh)), "rows_cached": int(len(cached_membership)),
        "lcc_size": int(raw_membership["largest_component_size"]),
        "in_lcc_identical": bool((cmp["raw_in_lcc"] == cmp["raw_in_lcc_cached"]).all()),
        "2ecc_size_identical": bool((cmp["raw_2ecc_size"] == cmp["raw_2ecc_size_cached"]).all()),
    }
    stage("raw_membership", t0)
    log(f"raw membership check: {membership_check}")

    # ---- origins per distinct origin limit ------------------------------------
    t0 = time.time()
    origin_cols = ["block_geoid", "block_group_geoid", "tract_geoid", "block", "county_fips", "county_name",
                   "pop20", "land_area_m2", "origin_geometry_method", "geometry"]
    origins_base = centroids[origin_cols].copy()
    origins_by_limit = {}
    for limit in sorted({float(al.ARM_SPECS[a]["origin_limit_m"]) for a in arms}):
        snap = al.snap_origins(base, origins_base, nodes=nodes, raw_membership=raw_membership, limit_m=limit)
        merged = origins_base.merge(snap, on="block_geoid", how="left")
        merged = base.attach_nearest_bridge_structure(merged, nodes, bridge_landing_nodes)
        origins_by_limit[limit] = merged
    stage("origin_snaps", t0)

    # ---- facility maps per distinct rule ---------------------------------------
    t0 = time.time()
    facility_maps = {}
    for arm in arms:
        spec = al.ARM_SPECS[arm]
        rule = spec["facility_rule"]
        if rule == "reference_map":
            key = ("preferred_then_fallback", 1000.0)
        else:
            key = (rule, None if spec["facility_limit_m"] is None else float(spec["facility_limit_m"]))
        if key not in facility_maps:
            facility_maps[key] = al.attach_facilities(
                base, candidates, rule=key[0], limit_m=key[1], nodes=nodes,
                unconstrained_tree=tree, unconstrained_node_ids=node_ids, raw_membership=raw_membership,
            )
    arm_facility_key = {}
    for arm in arms:
        spec = al.ARM_SPECS[arm]
        arm_facility_key[arm] = (("preferred_then_fallback", 1000.0) if spec["facility_rule"] == "reference_map"
                                 else (spec["facility_rule"], None if spec["facility_limit_m"] is None
                                       else float(spec["facility_limit_m"])))
    stage("facility_maps", t0)

    # ---- per-arm audits and baseline (raw-graph) nearest lookups ----------------
    t0 = time.time()
    audit_cols = ["service_record_id", "service_id", "service_type", "service_source", "service_name",
                  "unconstrained_node_id", "unconstrained_snap_distance_m", "node_id", "snap_distance_m",
                  "snap_valid", "service_snap_rule", "service_raw_2ecc_size",
                  "service_snap_distance_penalty_m", "service_node_moved", "x", "y"]
    services_by_key, baseline_nearest_by_key = {}, {}
    for key, fmap in facility_maps.items():
        services_by_key[key] = fmap.loc[fmap["snap_valid"].astype(bool)].copy()
        baseline_nearest_by_key[key] = base.build_nearest_service_lookup(graph_raw, services_by_key[key])
    for arm in arms:
        arm_dir = exp_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        fmap = facility_maps[arm_facility_key[arm]].copy()
        fmap["facility_x"] = fmap.geometry.x
        fmap["facility_y"] = fmap.geometry.y
        al.atomic_to_csv(pd.DataFrame(fmap[audit_cols + ["facility_x", "facility_y"]]).assign(attachment_arm=arm),
                         arm_dir / "service_attachment_audit.csv")
        limit = float(al.ARM_SPECS[arm]["origin_limit_m"])
        o = origins_by_limit[limit]
        al.atomic_to_csv(pd.DataFrame(o.drop(columns="geometry")).assign(
                             attachment_arm=arm, origin_x=o.geometry.x.to_numpy(), origin_y=o.geometry.y.to_numpy()),
                         arm_dir / "origin_snap_audit.csv")
    boundary_node_ids = base.build_boundary_node_set(nodes, boundary_polygon)
    del graph_raw
    stage("audits_and_baseline_nearest", t0)

    # ---- scenario loop -----------------------------------------------------------
    scenario_meta = []
    with al.GraphMemo(base) as memo:
        for slr_ft in scenarios:
            layer_name = layers[slr_ft]
            parts_done = all((exp_dir / a / "parts" / f"slr_{slr_ft}ft.parquet").exists() for a in arms)
            if parts_done and (shared_dir / f"bridge_structures_slr_{slr_ft}ft.csv").exists():
                log(f"SLR {slr_ft} ft: all arm parts present; skipping (resume).")
                continue
            t0 = time.time()
            slr_layer = base.load_slr_layer(layer_name, source_clip_polygon)
            dry_edges, bridge_audit, bridge_summary, retention = base.apply_bridge_rule_to_edges(
                edges=edges, nodes=nodes, slr_layer=slr_layer, structures=bridge_structures,
                landing_nodes=bridge_landing_nodes, bridge_rule="approach", slr_ft=slr_ft,
            )
            al.atomic_to_csv(bridge_audit, shared_dir / f"bridge_structures_slr_{slr_ft}ft.csv")
            dry_graph = base.build_graph(dry_edges)
            del dry_edges
            # The scenario union is computed once (as production does once per
            # scenario) and cached for resume; arms receive it through
            # al.union_passthrough so it is not recomputed six times.
            union_path = shared_dir / f"slr_union_{slr_ft}ft.parquet"
            dissolved_geom = None
            if slr_layer is not None:
                t_union = time.time()
                method_path = shared_dir / f"slr_union_{slr_ft}ft_method.json"
                if union_path.exists():
                    dissolved_geom = gpd.read_parquet(union_path).geometry.iloc[0]
                    union_source = "cache"
                else:
                    if args.union_method == "coverage_union_all":
                        import shapely

                        dissolved_geom = shapely.coverage_union_all(slr_layer.geometry.values)
                    else:
                        dissolved_geom = slr_layer.geometry.union_all()
                    union_source = args.union_method
                    al.atomic_to_parquet(gpd.GeoDataFrame({"slr_ft": [slr_ft]}, geometry=[dissolved_geom],
                                                          crs=slr_layer.crs), union_path)
                    al.atomic_write_json(method_path, {"slr_ft": slr_ft, "method": union_source,
                                                       "seconds": round(time.time() - t_union, 1)})
                log(f"SLR {slr_ft} ft union ready in {time.time() - t_union:.1f} s ({union_source})")
                dissolved = gpd.GeoDataFrame({"Id": [0]}, geometry=[dissolved_geom], crs=slr_layer.crs)
            else:
                dissolved = None
            del slr_layer
            t_graph = time.time() - t0
            union_ctx = al.union_passthrough(base, dissolved_geom) if dissolved_geom is not None else al.contextlib.nullcontext()
            with union_ctx:
              for arm in arms:
                  t_arm = time.time()
                  limit = float(al.ARM_SPECS[arm]["origin_limit_m"])
                  key = arm_facility_key[arm]
                  result = base.scenario_results_for_origins(
                      slr_ft=slr_ft, slr_layer_name=layer_name, slr_layer=dissolved, graph=dry_graph,
                      services=services_by_key[key], origins=origins_by_limit[limit].drop(columns="geometry"),
                      centroid_boundary=centroid_boundary,
                      centroid_geometry_source=centroids_source[["block_geoid", "geometry"]],
                      baseline_nearest=baseline_nearest_by_key[key], dry_boundary_node_ids=boundary_node_ids,
                      unclassify_failed_origins=True, legacy_collocated_rule=False,
                      legacy_centroid_inundation_join=False, bridge_rule_applied="approach",
                      bridge_structure_retained_lookup=retention,
                  )
                  result["attachment_arm"] = arm
                  al.atomic_to_parquet(result, exp_dir / arm / "parts" / f"slr_{slr_ft}ft.parquet")
                  log(f"SLR {slr_ft} ft arm {arm}: {time.time() - t_arm:.1f} s")
            memo.forget_graph(dry_graph)
            del dry_graph
            meta = {"slr_ft": slr_ft, "layer": layer_name, "graph_build_s": round(t_graph, 1),
                    "total_s": round(time.time() - t0, 1),
                    **{k: (int(v) if isinstance(v, (np.integer, int)) else v)
                       for k, v in bridge_summary.iloc[0].to_dict().items()}}
            scenario_meta.append(meta)
            al.atomic_write_json(shared_dir / f"scenario_{slr_ft}ft_meta.json", meta)
            log(f"SLR {slr_ft} ft done in {meta['total_s']} s (memo hits={memo.hits}, misses={memo.misses})")

    # ---- assemble per-arm outputs --------------------------------------------------
    t0 = time.time()
    for arm in arms:
        parts = [pd.read_parquet(exp_dir / arm / "parts" / f"slr_{s}ft.parquet") for s in scenarios]
        raw = pd.concat(parts, ignore_index=True)
        attachment = raw.pop("attachment_arm")
        results = base.add_baseline_comparison_fields(raw)
        results["attachment_arm"] = attachment.to_numpy()
        al.atomic_to_parquet(results, exp_dir / arm / "block_access_flags_long.parquet")
        base_rows = results.loc[results["slr_ft"].eq(0)]
        fmap = facility_maps[arm_facility_key[arm]]
        arm_manifest = {
            "attachment_arm": arm, **al.ARM_SPECS[arm], "bridge_rule": "approach",
            "scenarios": scenarios,
            "n_rows": int(len(results)),
            "eligible_blocks": int(base_rows["analysis_eligible"].sum()),
            "eligible_pop20": int(base_rows.loc[base_rows["analysis_eligible"], "pop20"].sum()),
            "exclusions": base_rows.loc[~base_rows["analysis_eligible"], "exclusion_reason"].value_counts().to_dict(),
            "facility_candidates": int(len(fmap)),
            "facility_valid": int(fmap["snap_valid"].astype(bool).sum()),
            "facility_rule_counts": fmap.groupby(["service_snap_rule", "snap_valid"]).size().rename("n")
                                     .reset_index().astype({"snap_valid": str}).to_dict("records"),
        }
        al.atomic_write_json(exp_dir / arm / "arm_manifest.json", arm_manifest)
        al.atomic_write_json(exp_dir / arm / "_COMPLETE.json", {
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "rows": int(len(results)), "scenarios": scenarios,
            "unique_keys": bool(not results.duplicated(["block_geoid", "slr_ft"]).any()),
        })
        log(f"arm {arm}: {len(results):,} rows, eligible={arm_manifest['eligible_blocks']:,}, "
            f"facilities valid={arm_manifest['facility_valid']:,}")
    stage("assemble", t0)
    rss.stop()
    provenance.update({
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "network_cache": cache_info, "raw_membership_check": membership_check,
        "road_bounds_lonlat": road_bounds, "candidate_facilities_in_buffer": int(len(candidates)),
        "stage_seconds": timings, "total_seconds": round(time.time() - t_all, 1),
        "peak_rss_gib": rss.peak_gib, "scenario_meta": scenario_meta,
    })
    al.atomic_write_json(shared_dir / "experiment_manifest.json", provenance)
    log(f"finished in {provenance['total_seconds']} s; peak RSS {rss.peak_gib} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
