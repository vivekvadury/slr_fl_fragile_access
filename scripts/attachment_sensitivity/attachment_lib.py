"""Shared helpers for the attachment-sensitivity experiment.

Everything here reuses the production functions in ``scripts/02_access_flags.py``
unchanged. Arm-specific distance limits are applied by temporarily overriding the
production module constants (``MAX_SERVICE_SNAP_M`` / ``MAX_ORIGIN_SNAP_M``) inside
a ``try/finally`` block, so the production algorithm itself is what runs; the
production file is never edited.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = PROJECT_ROOT / "scripts" / "02_access_flags.py"

ARM_SPECS: dict[str, dict[str, object]] = {
    "reference": {
        "origin_limit_m": 2000.0,
        "facility_rule": "preferred_then_fallback",
        "facility_limit_m": 1000.0,
        "purpose": "Reproduce current behavior",
    },
    "origin_1000": {
        "origin_limit_m": 1000.0,
        "facility_rule": "reference_map",
        "facility_limit_m": 1000.0,
        "purpose": "Tighter origin tolerance",
    },
    "origin_500": {
        "origin_limit_m": 500.0,
        "facility_rule": "reference_map",
        "facility_limit_m": 1000.0,
        "purpose": "Stronger origin restriction",
    },
    "facility_500": {
        "origin_limit_m": 2000.0,
        "facility_rule": "preferred_then_fallback",
        "facility_limit_m": 500.0,
        "purpose": "Tighter facility attachment policy",
    },
    "facility_add_uncapped": {
        "origin_limit_m": 2000.0,
        "facility_rule": "preserve_valid_add_excluded_uncapped_nearest",
        # JSON null: no finite cap for the added candidates only.
        "facility_limit_m": None,
        "purpose": "Isolate exclusion of distant facilities",
    },
    "facility_nearest_1000": {
        "origin_limit_m": 2000.0,
        "facility_rule": "unconstrained_nearest",
        "facility_limit_m": 1000.0,
        "purpose": "Test preferential attachment separately",
    },
}
ARM_ORDER = list(ARM_SPECS)


def load_base_module():
    """Import scripts/02_access_flags.py as a module without running main()."""
    name = "access_flags_base"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def patched_constant(module, name: str, value) -> Iterator[None]:
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    # allow_nan=False guarantees standard JSON (no Infinity/NaN tokens).
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str))


def atomic_to_parquet(frame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_to_csv(frame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Facility attachment rules
# ---------------------------------------------------------------------------

def _recompute_derived_facility_fields(base, attached, nodes, raw_membership):
    """Recompute the fields the production function derives from node_id."""
    import numpy as np  # noqa: F401  (kept local; base already imported numpy)

    node_to_2ecc_size = raw_membership["node_to_2ecc_size"]
    attached = attached.copy()
    attached["service_raw_2ecc_size"] = (
        attached["node_id"].map(node_to_2ecc_size).fillna(0).astype(int)
    )
    attached["service_snap_distance_penalty_m"] = (
        attached["snap_distance_m"] - attached["unconstrained_snap_distance_m"]
    )
    attached["service_node_moved"] = attached["node_id"].ne(attached["unconstrained_node_id"])
    chosen_xy = nodes.set_index("node_id")[["x", "y"]]
    attached = attached.drop(columns=["x", "y"], errors="ignore").join(chosen_xy, on="node_id")
    return attached


def attach_facilities(
    base,
    candidates,
    *,
    rule: str,
    limit_m: float | None,
    nodes,
    unconstrained_tree,
    unconstrained_node_ids,
    raw_membership,
):
    """Return the full candidate audit frame (valid and invalid) for one rule.

    Rules
    -----
    preferred_then_fallback
        The production algorithm (``attach_services_to_raw_graph`` with
        ``use_eligible_service_nodes=True``), with both the preferred and the
        fallback search capped at ``limit_m``.
    unconstrained_nearest
        The production algorithm with ``use_eligible_service_nodes=False``
        (the ``--legacy-service-snap`` behavior): nearest full-raw-graph node
        within ``limit_m``, no component preference.
    preserve_valid_add_excluded_uncapped_nearest
        Reference attachments (preferred_then_fallback, 1,000 m) are kept
        exactly; candidates the reference rejects are added at their
        unconstrained nearest full-raw-graph node with no distance cap.
        Candidates with non-finite coordinates stay invalid.
    """
    import numpy as np

    if rule == "preferred_then_fallback":
        with patched_constant(base, "MAX_SERVICE_SNAP_M", float(limit_m)):
            return base.attach_services_to_raw_graph(
                candidates,
                nodes=nodes,
                unconstrained_tree=unconstrained_tree,
                unconstrained_node_ids=unconstrained_node_ids,
                raw_membership=raw_membership,
                use_eligible_service_nodes=True,
            )
    if rule == "unconstrained_nearest":
        with patched_constant(base, "MAX_SERVICE_SNAP_M", float(limit_m)):
            return base.attach_services_to_raw_graph(
                candidates,
                nodes=nodes,
                unconstrained_tree=unconstrained_tree,
                unconstrained_node_ids=unconstrained_node_ids,
                raw_membership=raw_membership,
                use_eligible_service_nodes=False,
            )
    if rule == "preserve_valid_add_excluded_uncapped_nearest":
        if limit_m is not None:
            raise ValueError("The add-uncapped rule takes no finite limit (limit_m must be None).")
        reference = attach_facilities(
            base,
            candidates,
            rule="preferred_then_fallback",
            limit_m=1000.0,
            nodes=nodes,
            unconstrained_tree=unconstrained_tree,
            unconstrained_node_ids=unconstrained_node_ids,
            raw_membership=raw_membership,
        )
        out = reference.copy()
        finite = np.isfinite(out["unconstrained_snap_distance_m"].to_numpy(dtype=float)) & out[
            "unconstrained_node_id"
        ].ge(0).to_numpy()
        add_mask = (~out["snap_valid"].astype(bool).to_numpy()) & finite
        out.loc[add_mask, "node_id"] = out.loc[add_mask, "unconstrained_node_id"].astype(int)
        out.loc[add_mask, "snap_distance_m"] = out.loc[add_mask, "unconstrained_snap_distance_m"].astype(float)
        out.loc[add_mask, "snap_valid"] = True
        out.loc[add_mask, "service_snap_rule"] = "added_uncapped_nearest"
        return _recompute_derived_facility_fields(base, out, nodes, raw_membership)
    raise ValueError(f"Unknown facility rule: {rule}")


def snap_origins(base, origins, *, nodes, raw_membership, limit_m: float):
    """Production origin snap (raw LCC nodes) with the arm's validity limit."""
    with patched_constant(base, "MAX_ORIGIN_SNAP_M", float(limit_m)):
        return base.snap_origins_to_raw_graph(
            origins,
            nodes=nodes,
            raw_membership=raw_membership,
            restrict_to_lcc=True,
        )


@contextlib.contextmanager
def union_passthrough(base, dissolved_geometry) -> Iterator[None]:
    """Return ``dissolved_geometry`` when a one-row series holding exactly that
    object is unioned.

    ``scenario_results_for_origins`` calls ``slr_layer.geometry.union_all()`` on
    every call. The runner passes a one-row layer whose geometry is the scenario's
    union (computed once, as production does once per scenario). Re-unioning an
    already-unioned geometry returns the same point set but costs minutes for the
    2-million-vertex NOAA polygons, so this identity-checked pass-through skips
    only that redundant recomputation. Any other union call runs unchanged.
    """
    series_cls = base.gpd.GeoSeries
    original = series_cls.union_all

    def patched(self, *args, **kwargs):
        if len(self) == 1 and self.iloc[0] is dissolved_geometry:
            return dissolved_geometry
        return original(self, *args, **kwargs)

    series_cls.union_all = patched
    try:
        yield
    finally:
        series_cls.union_all = original


# ---------------------------------------------------------------------------
# Memoization of facility-independent graph computations
# ---------------------------------------------------------------------------

class GraphMemo:
    """Cache results that depend only on (graph object, inputs).

    ``k_edge_components(G, 2)`` depends only on the graph, so it is cached per
    graph object and reused across attachment arms. The production module looks
    ``k_edge_components`` up as a module global at call time, so replacing the
    global routes every production call through this cache. Component order is
    fixed by the first call and identical for later calls.
    """

    def __init__(self, base):
        self.base = base
        self._original = base.k_edge_components
        self._original_nearest = base.build_nearest_service_lookup
        self._original_components = base.build_component_maps
        self._cache: dict[tuple, object] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def services_signature(services) -> str:
        """Hash of every service column the memoized functions read."""
        import pandas as pd

        cols = [c for c in ("node_id", "service_id", "service_type", "service_snap_rule", "snap_distance_m") if c in services.columns]
        frame = services[cols].reset_index(drop=True)
        return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()

    def _lookup(self, key, compute):
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        value = compute()
        self._cache[key] = value
        return value

    def __enter__(self):
        original = self._original
        original_nearest = self._original_nearest
        original_components = self._original_components

        def cached_k_edge_components(graph, k):
            key = ("k_edge", id(graph), graph.number_of_edges(), int(k))
            value = self._lookup(key, lambda: [set(c) for c in original(graph, k)])
            return iter(value)

        def cached_nearest(graph, services):
            key = ("nearest", id(graph), graph.number_of_edges(), self.services_signature(services))
            return self._lookup(key, lambda: original_nearest(graph, services))

        def cached_components(graph, services, boundary_node_ids):
            key = (
                "components", id(graph), graph.number_of_edges(),
                self.services_signature(services), id(boundary_node_ids), len(boundary_node_ids),
            )
            return self._lookup(key, lambda: original_components(graph, services, boundary_node_ids))

        self.base.k_edge_components = cached_k_edge_components
        self.base.build_nearest_service_lookup = cached_nearest
        self.base.build_component_maps = cached_components
        return self

    def forget_graph(self, graph) -> None:
        for key in [key for key in self._cache if key[1] == id(graph)]:
            del self._cache[key]

    def __exit__(self, *exc):
        self.base.k_edge_components = self._original
        self.base.build_nearest_service_lookup = self._original_nearest
        self.base.build_component_maps = self._original_components
        self._cache.clear()
        return False
