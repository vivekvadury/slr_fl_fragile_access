"""Shared helpers for the origin-placement Monte Carlo sensitivity analysis.

Exploratory measurement-validity analysis. Nothing here modifies the canonical
pipeline: ``scripts/02_access_flags.py`` is imported as a module (via the
attachment-sensitivity loader) and its functions are called unchanged.

Classification shortcut (validated, not assumed)
------------------------------------------------
For a non-legacy run (``legacy_collocated_rule=False``), production
``scenario_results_for_origins`` classifies an origin from exactly three inputs:

1. whether the origin point intersects the scenario's NOAA inundation union
   (``inundated`` takes precedence over everything);
2. whether the origin snapped to a raw-graph LCC node within 2 km
   (failure -> ``unclassified`` unless flooded);
3. per-node quantities on that scenario's dry graph: membership, the service
   count of the node's connected component, the node's degree, and the service
   count of its 2-edge-connected component (``capped_local_edge_connectivity``).

The nearest-service Dijkstra result only feeds distance columns. Item 3 is a
function of (node, scenario) alone, so it is computed once per node and scenario
with the production helper functions (``node_state_table``) and reused for every
origin. ``02_build_node_states.py`` checks this exhaustively against the
production function for every LCC node, and against the canonical run output for
every canonical representative point.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "attachment_sensitivity"))
import attachment_lib as al  # noqa: E402

base = al.load_base_module()
gpd, nx, np, pd = base.gpd, base.nx, base.np, base.pd

PROJECT_ROOT = al.PROJECT_ROOT
CANONICAL_RUN = PROJECT_ROOT / "data/processed/access/edited/della_runs/positive_layer_20260814_approach"
CANONICAL_BLOCK_LEVEL = PROJECT_ROOT / "data/processed/analysis/block_level_long_dataset_approach.csv"
CANONICAL_BG = PROJECT_ROOT / "data/processed/analysis/block_group_analysis_dataset_approach.csv"
SOURCE_CACHE_DIR = PROJECT_ROOT / "data/processed/access/cache"
V2_STEM = "fe639d75a84cb809_full_v2"
UNION_CACHE_DIR = PROJECT_ROOT / "data/processed/access/attachment_sensitivity/attach_20260925/shared"
AREAWATER_DIR = PROJECT_ROOT / "data/raw/census/areawater/2020"
AREAWATER_ZIPS = [AREAWATER_DIR / f"tl_2020_12{c}_areawater.zip" for c in ("011", "086", "099")]

OUT_DIR = PROJECT_ROOT / "outputs/origin_placement_sensitivity"
WORK_DIR = OUT_DIR / "work"

SCENARIOS = [0, 1, 2, 3, 4, 5, 6]
POSITIVE_SCENARIOS = [1, 2, 3, 4, 5, 6]
LAYERS = {0: base.BASELINE_SLR_LAYER, **base.SLR_LAYER_MAP}
BRIDGE_RULE = "approach"
DEFAULT_SEED = 20261009

# Integer state codes used in all per-draw tables.
STATE_CODES = {"inundated": 0, "isolated": 1, "fragile": 2, "redundant": 3, "unclassified": 4}
CODE_STATES = {v: k for k, v in STATE_CODES.items()}
STATE_ORDER = ["redundant", "fragile", "isolated", "inundated", "unclassified"]

# Coastal measure: TIGER AREAWATER bay/estuary/sound (H2051) and ocean/sea (H2053).
COASTAL_MTFCC = ("H2051", "H2053")
COASTAL_DISTANCE_M = 1_000

# Five manuscript transitions (notebook 03 cell 12) and the two composites
# (04_transition_models.R: Redundant -> Worse = any_loss_of_redundancy;
# Fragile -> Worse = fragile -> isolated + fragile -> inundated).
TRANSITIONS = {
    "red_to_fragile": ("redundant", ("fragile",)),
    "red_to_isolated": ("redundant", ("isolated",)),
    "red_to_inundated": ("redundant", ("inundated",)),
    "fragile_to_isolated": ("fragile", ("isolated",)),
    "fragile_to_inundated": ("fragile", ("inundated",)),
    "red_to_worse": ("redundant", ("fragile", "isolated", "inundated")),
    "fragile_to_worse": ("fragile", ("isolated", "inundated")),
}
FIVE_TRANSITIONS = ["red_to_fragile", "red_to_isolated", "red_to_inundated",
                    "fragile_to_isolated", "fragile_to_inundated"]


def log(msg: str) -> None:
    import datetime as dt

    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_env() -> None:
    """Environment quirks on this machine (see project memory)."""
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    gdal = Path(sys.executable).parent / "Library" / "share" / "gdal"
    if gdal.exists():
        os.environ.setdefault("GDAL_DATA", str(gdal))


atomic_write_json = al.atomic_write_json
atomic_to_parquet = al.atomic_to_parquet
atomic_to_csv = al.atomic_to_csv
sha256_file = al.sha256_file


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class PeakRSS:
    """Sample this process's resident memory in a background thread."""

    def __init__(self, interval: float = 1.0):
        import threading

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
        return round(max(self.peak, self.proc.memory_info().rss) / 2**30, 2)


# ---------------------------------------------------------------------------
# Network and services (production functions; v2 cache as in attachment runner)
# ---------------------------------------------------------------------------

def load_network():
    """Segmentized nodes/edges with the corrected positive-layer bridge gate.

    Identical to ``attachment_sensitivity/run_access_arms.load_cached_network``:
    the v2 cache stores raw bridge and layer tags, and recomputing
    ``bridge_like`` reproduces the production v3 cache. That reproduction was
    exact in the attachment experiment (all columns, every block).
    """
    nodes = gpd.read_parquet(SOURCE_CACHE_DIR / f"segmentized_nodes_{V2_STEM}.parquet")
    edges = gpd.read_parquet(SOURCE_CACHE_DIR / f"segmentized_edges_{V2_STEM}.parquet")
    positive = edges["layer_value"].map(base.is_positive_layer_value).astype(bool)
    edges["bridge_like"] = edges["bridge_tag_present"].astype(bool) | positive
    return nodes, edges


def load_raw_membership() -> dict:
    frame = pd.read_parquet(SOURCE_CACHE_DIR / f"raw_graph_membership_{V2_STEM}.parquet")
    return base.raw_membership_from_frame(frame)


def lcc_snap(points: "gpd.GeoDataFrame", *, id_col: str, nodes, raw_membership) -> "pd.DataFrame":
    """Production origin snap (raw-graph LCC nodes, 2 km validity)."""
    frame = points[[id_col, "geometry"]].rename(columns={id_col: "block_geoid"})
    snapped = base.snap_origins_to_raw_graph(
        frame, nodes=nodes, raw_membership=raw_membership, restrict_to_lcc=True
    )
    return snapped.rename(columns={"block_geoid": id_col})


# ---------------------------------------------------------------------------
# Inundation
# ---------------------------------------------------------------------------

def load_union(slr_ft: int):
    """Cached scenario union (attachment experiment) as a prepared shapely geometry."""
    import shapely

    geom = gpd.read_parquet(UNION_CACHE_DIR / f"slr_union_{slr_ft}ft.parquet").geometry.iloc[0]
    shapely.prepare(geom)
    return geom


def to_lonlat(x: "np.ndarray", y: "np.ndarray") -> "gpd.GeoSeries":
    """Projected points -> OGC:CRS84 exactly as production (GeoDataFrame.to_crs)."""
    pts = gpd.GeoSeries(gpd.points_from_xy(x, y), crs=base.PROJECTED_CRS)
    return pts.to_crs("OGC:CRS84")


def inundated_by_union(union_geom, lonlat_points: "gpd.GeoSeries") -> "np.ndarray":
    import shapely

    return shapely.intersects(union_geom, lonlat_points.values).astype(bool)


# ---------------------------------------------------------------------------
# Node-state table (the validated shortcut)
# ---------------------------------------------------------------------------

def node_state_table(dry_graph, services, boundary_node_ids, node_ids: "np.ndarray") -> "pd.DataFrame":
    """Per-node network state on one scenario's dry graph via production helpers.

    Mirrors ``scenario_results_for_origins`` for an origin that is validly
    snapped to ``node`` and not flooded (non-legacy collocated rule).
    """
    comp_lookup, comp_services, _, _ = base.build_component_maps(dry_graph, services, boundary_node_ids)
    two_lookup, two_services = base.build_two_edge_component_maps(dry_graph, services)
    n = len(node_ids)
    in_dry = np.zeros(n, dtype=bool)
    reach = np.zeros(n, dtype=np.int32)
    degree = np.zeros(n, dtype=np.int32)
    paths = np.zeros(n, dtype=np.int8)
    two_svc = np.zeros(n, dtype=np.int32)
    for i, node in enumerate(node_ids.tolist()):
        comp = comp_lookup.get(node)
        if comp is None:
            continue
        in_dry[i] = True
        reach[i] = comp_services.get(comp, 0)
        degree[i] = dry_graph.degree(node)
        tid = two_lookup.get(node)
        two_svc[i] = two_services.get(tid, 0) if tid is not None else 0
        if reach[i] > 0:
            paths[i] = base.capped_local_edge_connectivity(
                dry_graph, node, nearest_service_node=None,
                two_edge_component_lookup=two_lookup,
                two_edge_component_service_counts=two_services,
                legacy_collocated_rule=False,
            )
    state = np.full(n, STATE_CODES["isolated"], dtype=np.int8)
    ok = in_dry & (reach > 0)
    state[ok & (paths >= 2)] = STATE_CODES["redundant"]
    state[ok & (paths == 1)] = STATE_CODES["fragile"]
    # paths == 0 with reach > 0 would be production "other"; flag it loudly.
    other = ok & (paths == 0)
    return pd.DataFrame({
        "node_id": node_ids.astype(np.int64), "in_dry_graph": in_dry, "component_services": reach,
        "dry_degree": degree, "two_edge_component_services": two_svc, "capped_paths": paths,
        "node_state": state, "node_state_other": other,
    })


def classify(inundated: "np.ndarray", snap_valid: "np.ndarray", node_state: "np.ndarray") -> "np.ndarray":
    """Production precedence: inundated > unclassified > node network state."""
    out = node_state.astype(np.int8).copy()
    out[~snap_valid] = STATE_CODES["unclassified"]
    out[inundated] = STATE_CODES["inundated"]
    return out


def transition_flags(base_state: "np.ndarray", scen_state: "np.ndarray") -> dict:
    """Seven manuscript transition indicators from paired state codes."""
    out = {}
    for name, (origin, dests) in TRANSITIONS.items():
        dest_mask = np.isin(scen_state, [STATE_CODES[d] for d in dests])
        out[name] = (base_state == STATE_CODES[origin]) & dest_mask
    return out


def load_canonical_long(columns=None) -> "pd.DataFrame":
    return pd.read_parquet(CANONICAL_RUN / "block_access_flags_long.parquet", columns=columns)
