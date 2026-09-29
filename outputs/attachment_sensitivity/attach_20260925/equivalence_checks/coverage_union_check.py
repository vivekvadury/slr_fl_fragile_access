"""Is shapely.coverage_union_all equivalent to union_all for the NOAA layers?

For each layer: time coverage_union_all, then compare its point-in-polygon results
against the cached production-style union at every block representative point and
every road-graph node (the only points the analysis tests), plus area difference.
"""
import sys, time
sys.path.insert(0, 'scripts/attachment_sensitivity')
import attachment_lib as al
import shapely

b = al.load_base_module()
gpd, np = b.gpd, b.np
clip = b.box(-80.9689146, 25.0967434, -80.0323306, 27.037165)
exp = al.PROJECT_ROOT / "data/processed/access/attachment_sensitivity/attach_20260925/shared"

blocks = b.maybe_to_projected(b.prepare_blocks_layer(b.read_vector(b.BLOCKS_PATH)))
origins = b.make_centroids(blocks, use_representative_point=True).to_crs("OGC:CRS84").geometry.values
nodes = gpd.read_parquet(al.PROJECT_ROOT / "data/processed/access/cache/segmentized_nodes_fe639d75a84cb809_full_v2.parquet")
node_pts = nodes.to_crs("OGC:CRS84").geometry.values
print(f"points: origins={len(origins):,} nodes={len(node_pts):,}", flush=True)

for s in [int(x) for x in sys.argv[1:]]:
    layer = b.load_slr_layer(b.BASELINE_SLR_LAYER if s == 0 else b.SLR_LAYER_MAP[s], clip)
    t = time.time()
    cov = shapely.coverage_union_all(layer.geometry.values)
    t_cov = time.time() - t
    ref = gpd.read_parquet(exp / f"slr_union_{s}ft.parquet").geometry.iloc[0]
    shapely.prepare(cov); shapely.prepare(ref)
    o_ref, o_cov = shapely.intersects(ref, origins), shapely.intersects(cov, origins)
    n_ref, n_cov = shapely.intersects(ref, node_pts), shapely.intersects(cov, node_pts)
    print(f"{s} ft: coverage_union {t_cov:.1f}s | valid={shapely.is_valid(cov)} | "
          f"area ref={ref.area:.10f} cov={cov.area:.10f} | "
          f"origin mismatches={int((o_ref != o_cov).sum())} (inundated {int(o_ref.sum()):,}) | "
          f"node mismatches={int((n_ref != n_cov).sum())} (in polygon {int(n_ref.sum()):,})", flush=True)
