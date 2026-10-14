#!/usr/bin/env python
"""Write outputs/origin_placement_sensitivity/run_manifest.json from stage manifests."""

from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ops_lib as ol  # noqa: E402


def git(*args):
    return subprocess.run(["git", *args], cwd=ol.PROJECT_ROOT, capture_output=True, text=True).stdout.strip()


def main():
    here = Path(__file__).resolve().parent
    full, pilot = ol.OUT_DIR / "full", ol.OUT_DIR / "pilot"
    node_checks = {s: ol.read_json(ol.WORK_DIR / f"node_states_slr_{s}ft_checks.json") for s in ol.SCENARIOS}
    manifest = {
        "analysis": "Monte Carlo origin-placement sensitivity (exploratory measurement-validity analysis)",
        "git_commit": git("rev-parse", "HEAD"), "git_status_porcelain": git("status", "--porcelain"),
        "python": sys.version, "platform": platform.platform(),
        "versions": {p: md.version(p) for p in ["geopandas", "shapely", "pyogrio", "pandas", "numpy", "networkx",
                                                 "scipy", "pyproj", "pyarrow", "statsmodels", "matplotlib"]},
        "canonical_reference": {
            "run_dir": str(ol.CANONICAL_RUN.relative_to(ol.PROJECT_ROOT)),
            "run_manifest_sha256": ol.sha256_file(ol.CANONICAL_RUN / "run_manifest.json"),
            "block_access_flags_long_sha256": ol.sha256_file(ol.CANONICAL_RUN / "block_access_flags_long.parquet"),
            "bridge_rule": ol.BRIDGE_RULE, "scenarios": ol.SCENARIOS,
        },
        "reused_artifacts": {
            "segmentized_graph_cache": f"data/processed/access/cache/*_{ol.V2_STEM}.parquet (bridge_like recomputed with positive-layer gate)",
            "raw_graph_membership_cache": f"data/processed/access/cache/raw_graph_membership_{ol.V2_STEM}.parquet",
            "scenario_unions": str(ol.UNION_CACHE_DIR.relative_to(ol.PROJECT_ROOT)) + "/slr_union_<s>ft.parquet",
        },
        "new_inputs": ol.read_json(ol.WORK_DIR / "prepare_manifest.json")["inputs"],
        "design": {"seed": ol.read_json(ol.WORK_DIR / "prepare_manifest.json")["seed"],
                   "draws_per_block_full": 50, "draws_per_block_pilot": 10,
                   "sampling_geometry": "2020 TIGER block polygon minus 2020 TIGER AREAWATER (land polygon), EPSG:32617",
                   "rng": ol.read_json(ol.WORK_DIR / "prepare_manifest.json")["rng"],
                   "origin_snap": "production snap_origins_to_raw_graph (raw-graph LCC nodes, 2,000 m validity)",
                   "failed_snap_rule": "canonical: unclassified unless the point is inundated; kept explicit",
                   "universe": "canonical analysis-eligible blocks (68,521); POP20 fixed per block"},
        "stage_manifests": {
            "prepare": "work/prepare_manifest.json",
            "node_states": {s: f"work/node_states_slr_{s}ft_checks.json" for s in ol.SCENARIOS},
            "classify_pilot": "pilot/classify_manifest.json", "classify_full": "full/classify_manifest.json",
            "summarize_pilot": "pilot/summarize_manifest.json", "summarize_full": "full/summarize_manifest.json",
            "models": "full/bg_model_manifest.json",
        },
        "validation_summary": {
            "pilot": {k: v for k, v in ol.read_json(pilot / "classify_manifest.json").items()
                      if k.startswith("v") or k.startswith("inundation")},
            "full": {k: v for k, v in ol.read_json(full / "classify_manifest.json").items()
                     if k.startswith("v") or k.startswith("inundation")},
            "fig4_reproduction": ol.read_json(full / "summarize_manifest.json")["fig4_check"],
            "bg_aggregation_reproduction": ol.read_json(full / "summarize_manifest.json")["bg_aggregation_check"],
            "model_reproduction": ol.read_json(full / "bg_model_manifest.json"),
        },
        "runtime": {
            "prepare_s": ol.read_json(ol.WORK_DIR / "prepare_manifest.json")["total_s"],
            "node_states_s": {s: c["total_s"] for s, c in node_checks.items()},
            "node_states_peak_rss_gib": max(c["peak_rss_gib"] for c in node_checks.values()),
            "classify_full_s": ol.read_json(full / "classify_manifest.json")["total_s"],
            "summarize_full_s": ol.read_json(full / "summarize_manifest.json")["total_s"],
        },
        "script_sha256": {p.name: ol.sha256_file(p) for p in sorted(here.glob("*")) if p.suffix in {".py", ".R", ".sh"}},
    }
    ol.atomic_write_json(ol.OUT_DIR / "run_manifest.json", manifest)
    print("wrote", ol.OUT_DIR / "run_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
