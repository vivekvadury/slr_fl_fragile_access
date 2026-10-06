#!/usr/bin/env python
"""Synthetic-network tests for the attachment-sensitivity experiment.

Run:  python scripts/attachment_sensitivity/test_attachment_synthetic.py
Each test builds a tiny road network in projected metres and pushes it through
the *production* functions (via attachment_lib), so what is tested is the code
the experiment actually runs. pytest is not installed in this environment, so a
minimal runner is included; the exit code is non-zero if any test fails.
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attachment_lib as al  # noqa: E402

base = al.load_base_module()
gpd, nx, np, pd = base.gpd, base.nx, base.np, base.pd
Point = base.Point
CRS = base.PROJECTED_CRS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_network(coords: dict[int, tuple[float, float]], edges: list[tuple[int, int]]):
    nodes = gpd.GeoDataFrame(
        {
            "node_id": list(coords),
            "x": [xy[0] for xy in coords.values()],
            "y": [xy[1] for xy in coords.values()],
        },
        geometry=[Point(xy) for xy in coords.values()],
        crs=CRS,
    )
    graph = nx.Graph()
    for u, v in edges:
        (x1, y1), (x2, y2) = coords[u], coords[v]
        graph.add_edge(u, v, weight=math.hypot(x2 - x1, y2 - y1))
    membership = base.compute_raw_graph_membership(graph)
    tree, _, node_ids = base.build_node_kdtree(nodes)
    return nodes, graph, membership, tree, node_ids


def make_candidates(points: list[tuple[str, float, float]]):
    return gpd.GeoDataFrame(
        {
            "service_id": [p[0] for p in points],
            "service_type": ["school"] * len(points),
            "service_source": ["synthetic"] * len(points),
            "service_name": [p[0] for p in points],
        },
        geometry=[Point(p[1], p[2]) for p in points],
        crs=CRS,
    )


def attach(rule, limit, candidates, net):
    nodes, _, membership, tree, node_ids = net
    return al.attach_facilities(
        base,
        candidates,
        rule=rule,
        limit_m=limit,
        nodes=nodes,
        unconstrained_tree=tree,
        unconstrained_node_ids=node_ids,
        raw_membership=membership,
    )


def classify(net, facilities, origin_nodes: dict[str, int]):
    """Run the production scenario classifier on the (dry = raw) test graph."""
    nodes, graph, _, _, _ = net
    services = facilities.loc[facilities["snap_valid"]].copy()
    origins = pd.DataFrame(
        {
            "block_geoid": list(origin_nodes),
            "block_group_geoid": "g",
            "tract_geoid": "t",
            "block": "b",
            "county_fips": "086",
            "county_name": "Test",
            "pop20": 1,
            "land_area_m2": 1,
            "origin_geometry_method": "representative_point",
            "node_id": list(origin_nodes.values()),
            "snap_distance_m": 0.0,
            "snap_valid": True,
            "origin_in_lcc": True,
        }
    )
    boundary = pd.DataFrame(
        {"block_geoid": list(origin_nodes), "boundary_distance_m": 1e6, "boundary_flag": False}
    )
    xy = nodes.set_index("node_id")
    centroids = gpd.GeoDataFrame(
        {"block_geoid": list(origin_nodes)},
        geometry=[Point(xy.loc[n, "x"], xy.loc[n, "y"]) for n in origin_nodes.values()],
        crs=CRS,
    )
    out = base.scenario_results_for_origins(
        slr_ft=0,
        slr_layer_name="none",
        slr_layer=None,
        graph=graph,
        services=services,
        origins=origins,
        centroid_boundary=boundary,
        centroid_geometry_source=centroids,
        baseline_nearest=base.build_nearest_service_lookup(graph, services),
        dry_boundary_node_ids=set(),
        unclassify_failed_origins=True,
        legacy_collocated_rule=False,
        legacy_centroid_inundation_join=False,
        bridge_rule_applied="approach",
    )
    status = base.classify_status_columns(
        out,
        inundated_col="block_centroid_inundated",
        isolated_col="block_centroid_isolated",
        redundant_col="block_centroid_redundant",
        fragile_col="block_centroid_fragile",
        unclassified_col="block_centroid_unclassified",
    )
    return dict(zip(out["block_geoid"], status))


def grid(n: int = 3, spacing: float = 100.0, offset: tuple[float, float] = (0.0, 0.0), start: int = 0):
    coords, edges = {}, []
    for i in range(n):
        for j in range(n):
            coords[start + i * n + j] = (offset[0] + j * spacing, offset[1] + i * spacing)
    for i in range(n):
        for j in range(n):
            node = start + i * n + j
            if j + 1 < n:
                edges.append((node, node + 1))
            if i + 1 < n:
                edges.append((node, node + n))
    return coords, edges


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dead_end_spur_preferred_vs_nearest():
    """Facility at the end of a driveway spur: nearest attaches to the spur tip,
    preferred attaches to the loop, and classification differs for the intended
    reason (the spur edge is a cut edge shared by every route)."""
    coords, edges = grid()
    coords[100] = (250.0, 0.0)      # spur tip 50 m east of node 2 (200, 0)
    edges.append((2, 100))
    net = make_network(coords, edges)
    cand = make_candidates([("fac", 252.0, 0.0)])
    near = attach("unconstrained_nearest", 1000.0, cand, net)
    pref = attach("preferred_then_fallback", 1000.0, cand, net)
    assert int(near["node_id"].iloc[0]) == 100
    assert int(pref["node_id"].iloc[0]) == 2
    assert pref["service_snap_rule"].iloc[0] == "raw_lcc_2ecc"
    assert classify(net, near, {"o": 4})["o"] == "fragile"
    assert classify(net, pref, {"o": 4})["o"] == "redundant"


def test_real_single_access_site_is_bypassed_by_preferred_rule():
    """Documents a real limitation: a facility inside a development reachable by
    one access road. The preferred rule attaches it to the loop, so origins are
    classified redundant even though real access to the building depends on the
    single access road. This is expected behavior of the rule, not a mapping
    error; the test pins the behavior so the report can describe it."""
    coords, edges = grid()
    coords[200] = (200.0, -150.0)   # single access road south from node 2
    coords[201] = (200.0, -300.0)   # development interior
    coords[202] = (280.0, -300.0)
    coords[203] = (280.0, -220.0)
    edges += [(2, 200), (200, 201), (201, 202), (202, 203), (203, 200)]  # small loop behind one link
    net = make_network(coords, edges)
    # Facility sits beside the development's internal loop.
    cand = make_candidates([("campus", 250.0, -330.0)])
    near = attach("unconstrained_nearest", 1000.0, cand, net)
    pref = attach("preferred_then_fallback", 1000.0, cand, net)
    # The internal loop is itself 2-edge-connected (size 4), so the preferred rule
    # keeps the facility inside the development here; the single access road
    # (2, 200) remains a shared dependency and origins in the grid are fragile.
    assert int(near["node_id"].iloc[0]) == int(pref["node_id"].iloc[0])
    assert classify(net, pref, {"o": 4})["o"] == "fragile"
    # If the development interior is a tree (no internal loop), the preferred rule
    # instead jumps to the nearest non-singleton component and bypasses the link.
    coords2, edges2 = grid()
    coords2[200] = (200.0, -150.0)
    coords2[201] = (200.0, -300.0)
    edges2 += [(2, 200), (200, 201)]
    net2 = make_network(coords2, edges2)
    cand2 = make_candidates([("campus", 200.0, -310.0)])
    near2 = attach("unconstrained_nearest", 1000.0, cand2, net2)
    pref2 = attach("preferred_then_fallback", 1000.0, cand2, net2)
    assert int(near2["node_id"].iloc[0]) == 201
    assert int(pref2["node_id"].iloc[0]) == 2          # bypasses the single access road
    assert classify(net2, near2, {"o": 4})["o"] == "fragile"
    assert classify(net2, pref2, {"o": 4})["o"] == "redundant"


def test_facility_500_falls_back_to_ordinary_node():
    """Preferred (2ECC) node at 600 m but an ordinary dead-end node at 300 m."""
    coords, edges = grid()
    coords[300] = (800.0, 0.0)      # dead-end chain node
    coords[301] = (1100.0, 0.0)
    edges += [(2, 300), (300, 301)]
    net = make_network(coords, edges)
    cand = make_candidates([("fac", 800.0, 300.0)])  # 300 m from node 300; 671 m from node 2... check
    ref = attach("preferred_then_fallback", 1000.0, cand, net)
    f500 = attach("preferred_then_fallback", 500.0, cand, net)
    d_pref = math.hypot(800 - 200, 300 - 200)        # nearest 2ECC node is (200, 200) -> 608 m
    assert 500 < d_pref < 1000
    assert int(ref["node_id"].iloc[0]) == 8 and ref["service_snap_rule"].iloc[0] == "raw_lcc_2ecc"
    assert int(f500["node_id"].iloc[0]) == 300
    assert f500["service_snap_rule"].iloc[0] == "fallback_unconstrained"
    assert bool(f500["snap_valid"].iloc[0])


def test_add_uncapped_admits_only_excluded_and_preserves_reference():
    coords, edges = grid()
    net = make_network(coords, edges)
    cand = make_candidates([
        ("near", 105.0, 5.0),        # valid in reference
        ("far", 1700.0, 100.0),      # 1,500 m from nearest node: excluded by reference
    ])
    ref = attach("preferred_then_fallback", 1000.0, cand, net)
    add = attach("preserve_valid_add_excluded_uncapped_nearest", None, cand, net)
    near1000 = attach("unconstrained_nearest", 1000.0, cand, net)
    assert ref["snap_valid"].tolist() == [True, False]
    assert add["snap_valid"].tolist() == [True, True]
    assert near1000["snap_valid"].tolist() == [True, False]
    assert int(add["node_id"].iloc[0]) == int(ref["node_id"].iloc[0])
    assert add["service_snap_rule"].iloc[0] == ref["service_snap_rule"].iloc[0]
    assert add["service_snap_rule"].iloc[1] == "added_uncapped_nearest"
    assert int(add["node_id"].iloc[1]) == 5          # (200, 100)
    try:
        attach("preserve_valid_add_excluded_uncapped_nearest", 1000.0, cand, net)
    except ValueError:
        pass
    else:
        raise AssertionError("add-uncapped rule must reject a finite limit")


def test_invalid_coordinates_stay_excluded_when_uncapped():
    coords, edges = grid()
    net = make_network(coords, edges)
    cand = make_candidates([("ok", 5.0, 5.0), ("nan", float("nan"), float("nan"))])
    add = attach("preserve_valid_add_excluded_uncapped_nearest", None, cand, net)
    assert add["snap_valid"].tolist() == [True, False]


def test_origin_threshold_inclusive_and_eligibility():
    coords, edges = grid()
    nodes, graph, membership, tree, node_ids = make_network(coords, edges)
    # Distances from node 0 at (0, 0) straight south; all nearest to node 0.
    dists = [499.9, 500.0, 500.1, 999.9, 1000.0, 1000.1, 2000.0, 2000.1]
    origins = gpd.GeoDataFrame(
        {"block_geoid": [f"b{i}" for i in range(len(dists))]},
        geometry=[Point(0.0, -d) for d in dists],
        crs=CRS,
    )
    for limit in (500.0, 1000.0, 2000.0):
        snapped = al.snap_origins(base, origins, nodes=nodes, raw_membership=membership, limit_m=limit)
        expected = [d <= limit for d in dists]
        assert snapped["snap_valid"].tolist() == expected, (limit, snapped["snap_valid"].tolist())
    # Production constant restored after the patch.
    assert base.MAX_ORIGIN_SNAP_M == 2000
    assert base.MAX_SERVICE_SNAP_M == 1000


def test_eligibility_land_area_and_zero_population():
    """analysis_eligible = valid origin snap AND positive land area; population is
    never used. Exercise through the production classifier."""
    coords, edges = grid()
    net = make_network(coords, edges)
    fac = attach("preferred_then_fallback", 1000.0, make_candidates([("f", 0.0, 0.0)]), net)
    nodes, graph, _, _, _ = net
    services = fac.loc[fac["snap_valid"]].copy()
    origins = pd.DataFrame(
        {
            "block_geoid": ["pop0", "land0", "badsnap"],
            "block_group_geoid": "g", "tract_geoid": "t", "block": "b",
            "county_fips": "086", "county_name": "Test",
            "pop20": [0, 10, 10],
            "land_area_m2": [100, 0, 100],
            "origin_geometry_method": "representative_point",
            "node_id": [4, 4, 4],
            "snap_distance_m": [10.0, 10.0, 2500.0],
            "snap_valid": [True, True, False],
            "origin_in_lcc": True,
        }
    )
    boundary = pd.DataFrame({"block_geoid": origins["block_geoid"], "boundary_distance_m": 1e6, "boundary_flag": False})
    centroids = gpd.GeoDataFrame({"block_geoid": origins["block_geoid"]}, geometry=[Point(100, 100)] * 3, crs=CRS)
    out = base.scenario_results_for_origins(
        slr_ft=0, slr_layer_name="none", slr_layer=None, graph=graph, services=services,
        origins=origins, centroid_boundary=boundary, centroid_geometry_source=centroids,
        baseline_nearest=base.build_nearest_service_lookup(graph, services),
        dry_boundary_node_ids=set(), unclassify_failed_origins=True,
        legacy_collocated_rule=False, legacy_centroid_inundation_join=False,
        bridge_rule_applied="approach",
    ).set_index("block_geoid")
    assert bool(out.loc["pop0", "analysis_eligible"]) is True
    assert bool(out.loc["land0", "analysis_eligible"]) is False
    assert out.loc["land0", "exclusion_reason"] == "zero_land_area"
    assert bool(out.loc["badsnap", "analysis_eligible"]) is False
    assert out.loc["badsnap", "exclusion_reason"] == "origin_snap_failed"
    assert int(out.loc["badsnap", "block_centroid_unclassified"]) == 1


def test_euclidean_attachment_crosses_barrier():
    """Two street networks separated by a canal (joined far to the north). A
    facility on the east bank is closer to a west-bank node, so Euclidean
    attachment puts it on the wrong side. Documented limitation; not corrected."""
    # Canal centre line at x = 220. West streets end at x = 200; the nearest east
    # street is at x = 400, so an east-bank site at x = 240 is 40 m from a west
    # node and 160 m from any east node.
    west, west_edges = grid(3, 100.0, (0.0, 0.0), 0)
    east, east_edges = grid(3, 100.0, (400.0, 0.0), 10)
    coords = {**west, **east}
    edges = west_edges + east_edges
    coords[50] = (300.0, 2000.0)                 # distant bridge joining the banks
    edges += [(8, 50), (50, 16)]
    net = make_network(coords, edges)
    cand = make_candidates([("east_school", 240.0, 0.0)])
    pref = attach("preferred_then_fallback", 1000.0, cand, net)
    near = attach("unconstrained_nearest", 1000.0, cand, net)
    assert int(pref["node_id"].iloc[0]) == 2      # (200, 0): WEST bank, across the canal
    assert int(near["node_id"].iloc[0]) == 2
    assert abs(float(pref["snap_distance_m"].iloc[0]) - 40.0) < 1e-6


def test_graph_memo_equivalence():
    coords, edges = grid(4)
    coords[99] = (500.0, 0.0)
    edges.append((3, 99))
    _, graph, _, _, _ = make_network(coords, edges)
    plain = sorted(sorted(c) for c in base.k_edge_components(graph, 2))
    with al.GraphMemo(base) as memo:
        first = sorted(sorted(c) for c in base.k_edge_components(graph, 2))
        second = sorted(sorted(c) for c in base.k_edge_components(graph, 2))
        assert memo.misses == 1 and memo.hits == 1
    assert plain == first == second
    assert base.k_edge_components is al.load_base_module().k_edge_components


def test_union_passthrough_is_identity_scoped():
    from shapely.geometry import box as sbox

    layer = gpd.GeoDataFrame({"Id": [1, 2]}, geometry=[sbox(0, 0, 10, 10), sbox(20, 0, 30, 10)], crs=CRS)
    union = layer.geometry.union_all()
    one = gpd.GeoDataFrame({"Id": [0]}, geometry=[union], crs=CRS)
    other = gpd.GeoDataFrame({"Id": [0]}, geometry=[sbox(0, 0, 1, 1)], crs=CRS)
    with al.union_passthrough(base, union):
        assert one.geometry.union_all() is union
        assert other.geometry.union_all().equals(sbox(0, 0, 1, 1))
        assert layer.geometry.union_all().equals(union)
    assert gpd.GeoSeries.union_all.__name__ == "union_all"
    pts = gpd.GeoSeries([Point(5, 5), Point(15, 5), Point(25, 5)], crs=CRS)
    assert pts.intersects(union).tolist() == [True, False, True]


def test_memo_nearest_and_components_are_equivalent():
    coords, edges = grid(3)
    net = make_network(coords, edges)
    nodes, graph, *_ = net
    fac = attach("preferred_then_fallback", 1000.0, make_candidates([("a", 0.0, 0.0), ("b", 200.0, 200.0)]), net)
    services = fac.loc[fac["snap_valid"]].copy()
    plain_near = base.build_nearest_service_lookup(graph, services)
    plain_comp = base.build_component_maps(graph, services, set())
    with al.GraphMemo(base) as memo:
        b1 = set()
        n1 = base.build_nearest_service_lookup(graph, services)
        n2 = base.build_nearest_service_lookup(graph, services)
        c1 = base.build_component_maps(graph, services, b1)
        c2 = base.build_component_maps(graph, services, b1)
        assert n1 is n2 and c1 is c2
        fewer = services.iloc[:1].copy()
        n3 = base.build_nearest_service_lookup(graph, fewer)
        assert n3 is not n1                      # different facility map -> recomputed
    assert plain_near.sort_values("node_id").reset_index(drop=True).equals(n1.sort_values("node_id").reset_index(drop=True))
    assert plain_comp == c1


TESTS = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]


def main() -> int:
    failures = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
