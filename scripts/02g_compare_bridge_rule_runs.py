#!/usr/bin/env python
"""Compare completed access runs generated under the three bridge rules.

The script is read-only with respect to completed run directories.  It writes
comparison tables and a Markdown report to outputs/run_comparison/.  Group A
is a hard comparability gate: if it fails, Groups B--F are not run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
DIAGNOSE_SCRIPT = SCRIPT_DIR / "02b_diagnose_access_run.py"
VALIDITY_SCRIPT = SCRIPT_DIR / "02d_measurement_validity_diagnostics.py"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "run_comparison"
DEFAULT_RUN_PARENT = (
    PROJECT_ROOT / "data" / "processed" / "access" / "edited" / "della_runs"
)

RUN_ORDER = ("intersect", "approach", "retain")
STATUS_FLAGS = (
    "block_centroid_inundated",
    "block_centroid_unclassified",
    "block_centroid_isolated",
    "block_centroid_redundant",
    "block_centroid_fragile",
)
STATUS_PRECEDENCE = ("inundated", "unclassified", "isolated", "redundant", "fragile")
STATUS_ORDER = ("redundant", "fragile", "isolated", "inundated", "unclassified")
SCENARIO_STATUS_COLUMNS = {
    "inundated": "block_centroid_inundated",
    "unclassified": "block_centroid_unclassified",
    "isolated": "block_centroid_isolated",
    "redundant": "block_centroid_redundant",
    "fragile": "block_centroid_fragile",
}
BASELINE_STATUS_COLUMNS = {
    state: f"baseline_{column}"
    for state, column in SCENARIO_STATUS_COLUMNS.items()
}
NEW_STATUS_FLAGS = {
    "fragile": "new_fragile_due_to_slr",
    "isolated": "new_isolated_due_to_slr",
    "inundated": "new_inundated_due_to_slr",
}

PRE_FIX_BRIDGE_STRUCTURE_COUNT = 660
PRE_FIX_BRIDGE_LENGTH_KM = 57.6
PRE_FIX_BRIDGE_REMOVAL_SHARE = 0.71
PRE_FIX_SERVICE_SINGLETON_SHARE = 0.263
PRE_FIX_FIRE_SINGLETON_SHARE = 0.451
PRE_FIX_SERVICE_MOVE_MEDIAN_M = 13.5
PRE_FIX_NEGATIVE_LAYER_SEGMENTS = 18
PRE_FIX_MOVABLE_SEGMENTS = 186
PRE_FIX_BASELINE_FRAGILE_SHARE = 0.247
PRE_FIX_CUMULATIVE_AFFECTED_POPULATION = 660_000


def log(message: str) -> None:
    print(f"[compare_bridge_rules] {message}", flush=True)


def load_module(name: str, path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"Required helper script is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare intersect, approach, and retain bridge-rule runs without "
            "modifying or rerunning them."
        )
    )
    parser.add_argument(
        "--intersect-run-dir",
        default=str(DEFAULT_RUN_PARENT / "corrected_intersect"),
    )
    parser.add_argument(
        "--approach-run-dir",
        default=str(DEFAULT_RUN_PARENT / "corrected_canonical"),
    )
    parser.add_argument(
        "--retain-run-dir",
        default=str(DEFAULT_RUN_PARENT / "corrected_retain"),
    )
    parser.add_argument(
        "--pre-fix-run-dir",
        default=None,
        help="Optional completed pre-fix run directory for direct comparisons.",
    )
    parser.add_argument(
        "--rebuild-graph",
        action="store_true",
        help=(
            "Explicitly rebuild the raw graph and service snapping diagnostics in "
            "memory. Without this flag, existing service audit artifacts are required."
        ),
    )
    return parser.parse_args()


def require_columns(frame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def require_path(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {description}: {resolved}")
    return resolved


def read_manifest(run_dir: Path) -> dict[str, object]:
    path = require_path(run_dir / "run_manifest.json", "run manifest")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Run manifest must contain a JSON object: {path}")
    return manifest


def detect_bridge_rule(manifest: dict[str, object], run_dir: Path) -> str:
    candidates: list[object] = [manifest.get("bridge_rule")]
    flags = manifest.get("cli_flags")
    if isinstance(flags, dict):
        candidates.append(flags.get("bridge_rule"))
    parameters = manifest.get("resolved_parameters")
    if isinstance(parameters, dict):
        candidates.append(parameters.get("bridge_rule"))
    for candidate in candidates:
        if candidate in RUN_ORDER:
            return str(candidate)
    raise ValueError(f"Could not detect a valid bridge rule in {run_dir / 'run_manifest.json'}")


def write_csv(frame, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    frame.to_csv(path, index=False)
    log(f"saved={path}")
    return path


def markdown_table(frame, *, max_rows: int = 30) -> str:
    if frame is None or frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    columns = [str(column) for column in view.columns]

    def clean(value: object) -> str:
        if value is None:
            return ""
        try:
            if bool(value != value):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(value, float):
            text = f"{value:.6g}"
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in view.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(clean(value) for value in values) + " |")
    if len(frame) > max_rows:
        rows.append(f"\n_First {max_rows} of {len(frame):,} rows shown._")
    return "\n".join(rows)


def truth_mask(series, context: str):
    if series.isna().any():
        raise ValueError(f"{context} contains missing Boolean values")
    allowed = series.isin([True, False, 1, 0])
    if not bool(allowed.all()):
        sample = series.loc[~allowed].astype(str).drop_duplicates().head(10).tolist()
        raise ValueError(f"{context} contains non-Boolean values: {sample}")
    return series.eq(True) | series.eq(1)


def status_from_precedence(frame, columns: dict[str, str], context: str):
    require_columns(frame, columns.values(), context)
    normalized = frame[columns["fragile"]].astype("object").copy()
    normalized.loc[:] = "other"
    # Assign lowest priority first so each later assignment enforces precedence.
    for state in reversed(STATUS_PRECEDENCE):
        mask = truth_mask(frame[columns[state]], f"{context} {columns[state]}")
        normalized.loc[mask] = state
    return normalized


def recompute_status_labels(diag02b: ModuleType, frame, rule: str):
    pd = diag02b.pd
    require_columns(frame, ["slr_ft", "scenario_status", "baseline_status"], f"{rule} statuses")
    output = frame.copy()
    output["scenario_status_input"] = output["scenario_status"].astype("string")
    output["baseline_status_input"] = output["baseline_status"].astype("string")
    output["scenario_status"] = status_from_precedence(
        output, SCENARIO_STATUS_COLUMNS, f"{rule} scenario flags"
    )
    output["baseline_status"] = status_from_precedence(
        output, BASELINE_STATUS_COLUMNS, f"{rule} baseline flags"
    )

    reassignment_records: list[dict[str, object]] = []
    for status_field in ("scenario_status", "baseline_status"):
        original = output[f"{status_field}_input"]
        reassigned = original.eq("unclassified") & output[status_field].eq("inundated")
        for slr_ft, group_index in output.groupby("slr_ft", sort=True).groups.items():
            count = int(reassigned.loc[group_index].sum())
            log(
                f"status_reassignment rule={rule} status_field={status_field} "
                f"slr_ft={int(slr_ft)} unclassified_to_inundated={count}"
            )
            reassignment_records.append(
                {
                    "record_type": "status_reassignment",
                    "bridge_rule": rule,
                    "slr_ft": int(slr_ft),
                    "check": f"{status_field}_unclassified_to_inundated",
                    "value": count,
                    "detail": "input label overridden from flag precedence",
                    "verdict": "REVIEW" if count else "PASS",
                }
            )
    return output, pd.DataFrame.from_records(reassignment_records)


def load_run(diag02b: ModuleType, rule: str, run_dir: Path) -> dict[str, object]:
    files = diag02b.find_result_files(run_dir)
    inventory = diag02b.load_file_inventory(files)
    frame, duplicates = diag02b.load_and_combine_results(files)
    frame, status_reassignments = recompute_status_labels(diag02b, frame, rule)
    return {
        "rule": rule,
        "run_dir": run_dir,
        "manifest": read_manifest(run_dir),
        "files": files,
        "inventory": inventory,
        "frame": frame,
        "duplicates": duplicates,
        "status_reassignments": status_reassignments,
    }


def run_group_a(diag02b: ModuleType, runs: dict[str, dict[str, object]]):
    pd = diag02b.pd
    records: list[dict[str, object]] = []
    differences: list[dict[str, object]] = []
    failed = False

    required = {
        "block_geoid",
        "slr_ft",
        "pop20",
        "land_area_m2",
        "analysis_eligible",
        "scenario_status",
        "baseline_status",
        *STATUS_FLAGS,
    }
    for rule, run in runs.items():
        frame = run["frame"]
        require_columns(frame, required, f"{rule} long output")
        levels = sorted(frame["slr_ft"].unique().tolist())
        records.append(
            {
                "record_type": "run_summary",
                "bridge_rule": rule,
                "check": "unique_blocks",
                "value": int(frame["block_geoid"].nunique()),
                "detail": "",
                "verdict": "PASS",
            }
        )
        records.append(
            {
                "record_type": "run_summary",
                "bridge_rule": rule,
                "check": "slr_levels",
                "value": len(levels),
                "detail": ",".join(str(value) for value in levels),
                "verdict": "PASS",
            }
        )
        duplicate_count = len(run["duplicates"])
        if duplicate_count:
            failed = True
        records.append(
            {
                "record_type": "run_summary",
                "bridge_rule": rule,
                "check": "duplicate_block_slr_rows",
                "value": int(duplicate_count),
                "detail": "",
                "verdict": "FAIL" if duplicate_count else "PASS",
            }
        )
        for file_row in run["inventory"].to_dict("records"):
            records.append(
                {
                    "record_type": "file_inventory",
                    "bridge_rule": rule,
                    "check": "result_file",
                    "value": file_row.get("rows", file_row.get("n_rows", "")),
                    "detail": json.dumps(file_row, default=str, sort_keys=True),
                    "verdict": "PASS",
                }
            )
        records.extend(run["status_reassignments"].to_dict("records"))

        eligible, pop_positive, land_positive = analysis_masks(frame)
        analysis_universe = eligible & pop_positive & land_positive
        effective_state_masks = pd.DataFrame(
            {
                state: frame["scenario_status"].eq(state)
                for state in STATUS_PRECEDENCE
            },
            index=frame.index,
        )
        status_sum = effective_state_masks.sum(axis=1)
        violation = status_sum.ne(1)
        analysis_violation = violation & analysis_universe
        outside_violation = violation & ~analysis_universe
        n_analysis_violations = int(analysis_violation.sum())
        n_outside_violations = int(outside_violation.sum())
        if n_analysis_violations:
            failed = True
            bad = frame.loc[
                analysis_violation,
                [
                    "block_geoid", "slr_ft", *STATUS_FLAGS,
                    "scenario_status_input", "scenario_status",
                    "analysis_eligible", "pop20", "land_area_m2",
                ],
            ].copy()
            bad["effective_status_count"] = status_sum.loc[analysis_violation].to_numpy()
            for row in bad.to_dict("records"):
                differences.append(
                    {
                        "record_type": "analysis_universe_status_partition_violation",
                        "bridge_rule": rule,
                        "comparison": "",
                        "verdict": "FAIL",
                        **row,
                    }
                )
        records.extend(
            [
                {
                    "record_type": "assertion",
                    "bridge_rule": rule,
                    "check": "five_state_partition_within_analysis_universe",
                    "value": n_analysis_violations,
                    "detail": "effective inundated > unclassified > isolated > redundant > fragile states must sum to 1",
                    "verdict": "FAIL" if n_analysis_violations else "PASS",
                },
                {
                    "record_type": "descriptive",
                    "bridge_rule": rule,
                    "check": "five_state_partition_violations_outside_analysis_universe",
                    "value": n_outside_violations,
                    "detail": "outside analysis_eligible & pop20 > 0 & land_area_m2 > 0",
                    "verdict": "REVIEW" if n_outside_violations else "PASS",
                },
            ]
        )

        effective_baseline_masks = pd.DataFrame(
            {
                state: frame["baseline_status"].eq(state)
                for state in STATUS_PRECEDENCE
            },
            index=frame.index,
        )
        baseline_status_sum = effective_baseline_masks.sum(axis=1)
        baseline_violation = baseline_status_sum.ne(1)
        baseline_analysis_violation = baseline_violation & analysis_universe
        baseline_outside_violation = baseline_violation & ~analysis_universe
        n_baseline_analysis_violations = int(baseline_analysis_violation.sum())
        n_baseline_outside_violations = int(baseline_outside_violation.sum())
        if n_baseline_analysis_violations:
            failed = True
            bad = frame.loc[
                baseline_analysis_violation,
                [
                    "block_geoid", "slr_ft", *BASELINE_STATUS_COLUMNS.values(),
                    "baseline_status_input", "baseline_status",
                    "analysis_eligible", "pop20", "land_area_m2",
                ],
            ].copy()
            bad["effective_status_count"] = baseline_status_sum.loc[
                baseline_analysis_violation
            ].to_numpy()
            for row in bad.to_dict("records"):
                differences.append(
                    {
                        "record_type": "analysis_universe_baseline_status_partition_violation",
                        "bridge_rule": rule,
                        "comparison": "",
                        "verdict": "FAIL",
                        **row,
                    }
                )
        records.extend(
            [
                {
                    "record_type": "assertion",
                    "bridge_rule": rule,
                    "check": "five_state_baseline_partition_within_analysis_universe",
                    "value": n_baseline_analysis_violations,
                    "detail": "recomputed baseline state must be exactly one of five states",
                    "verdict": "FAIL" if n_baseline_analysis_violations else "PASS",
                },
                {
                    "record_type": "descriptive",
                    "bridge_rule": rule,
                    "check": "five_state_baseline_partition_violations_outside_analysis_universe",
                    "value": n_baseline_outside_violations,
                    "detail": "outside analysis_eligible & pop20 > 0 & land_area_m2 > 0",
                    "verdict": "REVIEW" if n_baseline_outside_violations else "PASS",
                },
            ]
        )

        raw_unclassified = truth_mask(
            frame["block_centroid_unclassified"],
            f"{rule} block_centroid_unclassified",
        )
        raw_inundated = truth_mask(
            frame["block_centroid_inundated"],
            f"{rule} block_centroid_inundated",
        )
        for slr_ft, group_index in frame.groupby("slr_ft", sort=True).groups.items():
            group_unclassified = raw_unclassified.loc[group_index]
            unclassified_index = group_unclassified.index[group_unclassified]
            unclassified_analysis_eligible = int(eligible.loc[unclassified_index].sum())
            if unclassified_analysis_eligible:
                failed = True
            records.append(
                {
                    "record_type": "unclassified_descriptive",
                    "bridge_rule": rule,
                    "slr_ft": int(slr_ft),
                    "check": "raw_unclassified_blocks",
                    "value": int(group_unclassified.sum()),
                    "unclassified_count": int(group_unclassified.sum()),
                    "also_flagged_inundated_count": int(
                        raw_inundated.loc[unclassified_index].sum()
                    ),
                    "unclassified_pop20": int(frame.loc[unclassified_index, "pop20"].sum()),
                    "unclassified_analysis_eligible_count": unclassified_analysis_eligible,
                    "detail": "counts use block_centroid_unclassified before precedence",
                    "verdict": "FAIL" if unclassified_analysis_eligible else "PASS",
                }
            )

        baseline = frame.loc[frame["slr_ft"].eq(0)].copy()
        baseline_eligible, baseline_pop_positive, baseline_land_positive = analysis_masks(baseline)
        baseline_metrics = {
            "baseline_blocks_total": pd.Series(True, index=baseline.index),
            "baseline_zero_land_area_blocks": ~baseline_land_positive,
            "baseline_zero_population_blocks": ~baseline_pop_positive,
            "baseline_zero_land_and_zero_population_blocks": (
                ~baseline_land_positive & ~baseline_pop_positive
            ),
            "baseline_analysis_eligible_before_population_filter": baseline_eligible,
            "baseline_analysis_universe_after_population_filter": (
                baseline_eligible & baseline_pop_positive & baseline_land_positive
            ),
        }
        for check, mask in baseline_metrics.items():
            records.append(
                {
                    "record_type": "analysis_universe_descriptive",
                    "bridge_rule": rule,
                    "slr_ft": 0,
                    "check": check,
                    "value": int(mask.sum()),
                    "detail": "zero-land and zero-population exclusions reported separately",
                    "verdict": "PASS",
                }
            )

    reference_rule = RUN_ORDER[0]
    reference = runs[reference_rule]["frame"]
    reference_blocks = set(reference["block_geoid"])
    reference_levels = set(reference["slr_ft"])
    reference_keys = set(
        reference[["block_geoid", "slr_ft"]].itertuples(index=False, name=None)
    )
    for rule in RUN_ORDER[1:]:
        frame = runs[rule]["frame"]
        other_blocks = set(frame["block_geoid"])
        symmetric = sorted(reference_blocks.symmetric_difference(other_blocks))
        if symmetric:
            failed = True
            for geoid in symmetric[:20]:
                differences.append(
                    {
                        "record_type": "block_universe_difference",
                        "bridge_rule": rule,
                        "comparison": f"{reference_rule}_vs_{rule}",
                        "block_geoid": geoid,
                        "slr_ft": "",
                    }
                )
        records.append(
            {
                "record_type": "assertion",
                "bridge_rule": rule,
                "check": "identical_block_universe",
                "value": len(symmetric),
                "detail": f"symmetric_difference_vs_{reference_rule}; sample_saved={min(20, len(symmetric))}",
                "verdict": "FAIL" if symmetric else "PASS",
            }
        )
        levels = set(frame["slr_ft"])
        if levels != reference_levels:
            failed = True
        records.append(
            {
                "record_type": "assertion",
                "bridge_rule": rule,
                "check": "identical_slr_levels",
                "value": len(reference_levels.symmetric_difference(levels)),
                "detail": f"reference={sorted(reference_levels)}; other={sorted(levels)}",
                "verdict": "PASS" if levels == reference_levels else "FAIL",
            }
        )
        other_keys = set(
            frame[["block_geoid", "slr_ft"]].itertuples(index=False, name=None)
        )
        key_difference = sorted(reference_keys.symmetric_difference(other_keys))
        if key_difference:
            failed = True
            for geoid, slr_ft in key_difference[:20]:
                differences.append(
                    {
                        "record_type": "block_scenario_key_difference",
                        "bridge_rule": rule,
                        "comparison": f"{reference_rule}_vs_{rule}",
                        "block_geoid": geoid,
                        "slr_ft": int(slr_ft),
                    }
                )
        records.append(
            {
                "record_type": "assertion",
                "bridge_rule": rule,
                "check": "identical_block_scenario_keys",
                "value": len(key_difference),
                "detail": f"symmetric_difference_vs_{reference_rule}; sample_saved={min(20, len(key_difference))}",
                "verdict": "FAIL" if key_difference else "PASS",
            }
        )

    if all(
        set(
            runs[rule]["frame"][["block_geoid", "slr_ft"]].itertuples(
                index=False, name=None
            )
        ) == reference_keys
        for rule in RUN_ORDER
    ):
        index_columns = ["block_geoid", "slr_ft"]
        reference_values = reference.set_index(index_columns)["block_centroid_inundated"]
        for rule in RUN_ORDER[1:]:
            other = runs[rule]["frame"].set_index(index_columns)["block_centroid_inundated"]
            other = other.reindex(reference_values.index)
            mismatch = reference_values.ne(other) | reference_values.isna() | other.isna()
            n_mismatch = int(mismatch.sum())
            if n_mismatch:
                failed = True
                for (geoid, slr_ft), reference_value in reference_values.loc[mismatch].items():
                    differences.append(
                        {
                            "record_type": "centroid_inundation_difference",
                            "bridge_rule": rule,
                            "comparison": f"{reference_rule}_vs_{rule}",
                            "block_geoid": geoid,
                            "slr_ft": int(slr_ft),
                            "reference_value": reference_value,
                            "other_value": other.loc[(geoid, slr_ft)],
                        }
                    )
            records.append(
                {
                    "record_type": "assertion",
                    "bridge_rule": rule,
                    "check": "identical_block_centroid_inundated",
                    "value": n_mismatch,
                    "detail": f"differences_vs_{reference_rule}",
                    "verdict": "FAIL" if n_mismatch else "PASS",
                }
            )

    summary = pd.DataFrame.from_records(records)
    detail = pd.DataFrame.from_records(differences)
    if not detail.empty:
        detail_rows = detail.copy()
        detail_rows["check"] = detail_rows["record_type"]
        detail_rows["value"] = 1
        detail_rows["detail"] = detail_rows.apply(
            lambda row: json.dumps(row.dropna().to_dict(), default=str, sort_keys=True), axis=1
        )
        if "verdict" not in detail_rows.columns:
            detail_rows["verdict"] = "FAIL"
        else:
            detail_rows["verdict"] = detail_rows["verdict"].fillna("FAIL")
        summary = pd.concat(
            [summary, detail_rows[["record_type", "bridge_rule", "check", "value", "detail", "verdict"]]],
            ignore_index=True,
        )
    return summary, detail, ("FAIL" if failed else "PASS")


def locate_cache_paths(base: ModuleType) -> dict[str, Path]:
    cache_dir = PROJECT_ROOT / "data" / "processed" / "access" / "cache"
    paths = base.graph_cache_paths(cache_dir, base.DRIVABLE_HIGHWAYS, smoke=False)
    missing = [path for key, path in paths.items() if key in {"nodes", "edges"} and not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required local full-extent graph cache is missing: "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def numeric_layer(series, pd, context: str):
    text = series.astype("string").str.strip()
    present = text.notna() & text.ne("")
    parsed = pd.to_numeric(text.where(present), errors="coerce")
    invalid = present & parsed.isna()
    if invalid.any():
        sample = text.loc[invalid].drop_duplicates().head(10).tolist()
        raise ValueError(f"{context} contains nonnumeric layer tags: {sample}")
    return parsed


def prepare_bridge_inputs(validity: ModuleType):
    base = validity.base
    pd = validity.pd
    gpd = validity.gpd
    cache_paths = locate_cache_paths(base)
    log(f"Loading cached segmentized edges (no graph rebuild): {cache_paths['edges']}")
    edges = gpd.read_parquet(cache_paths["edges"])
    require_columns(
        edges,
        [
            "edge_id", "u", "v", "osm_id", "highway", "bridge_tag_present",
            "layer_value", "bridge_like", "length_m", "geometry",
        ],
        "segmentized edge cache",
    )
    if edges["edge_id"].duplicated().any():
        raise ValueError("Segmentized edge cache contains duplicate edge_id values")

    log("Loading parent OSM ways to recover exact bridge and layer tags.")
    roads = base.load_roads()
    require_columns(roads, ["osm_id", "other_tags"], "road ways")
    roads = roads.copy()
    roads["osm_id"] = roads["osm_id"].astype(str)
    roads["bridge_tag_value"] = roads["other_tags"].map(
        lambda value: base.parse_other_tag(value, "bridge")
    )
    roads["parent_layer_value"] = roads["other_tags"].map(
        lambda value: base.parse_other_tag(value, "layer")
    )
    parent = roads[["osm_id", "bridge_tag_value", "parent_layer_value"]].drop_duplicates()
    conflicts = parent.groupby("osm_id", dropna=False).size()
    if bool(conflicts.gt(1).any()):
        bad = conflicts.loc[conflicts.gt(1)].head(10).index.tolist()
        raise ValueError(f"Parent OSM ways have conflicting tag records: {bad}")
    parent = parent.drop_duplicates("osm_id")

    output = edges.copy()
    output["osm_id"] = output["osm_id"].astype(str)
    output = output.merge(parent, on="osm_id", how="left", validate="many_to_one")
    missing_parent = output["bridge_tag_value"].isna() & output["parent_layer_value"].isna()
    # A parent can legitimately have neither tag.  Verify existence separately.
    parent_ids = set(parent["osm_id"])
    unknown = ~output["osm_id"].isin(parent_ids)
    if unknown.any():
        sample = output.loc[unknown, "osm_id"].drop_duplicates().head(10).tolist()
        raise ValueError(f"Segmentized edges could not be joined to parent ways: {sample}")
    del missing_parent

    layer = numeric_layer(output["parent_layer_value"], pd, "parent OSM layer tags")
    output["layer_numeric"] = layer
    output["nonzero_layer"] = layer.notna() & layer.ne(0)
    output["positive_layer"] = layer.notna() & layer.gt(0)
    output["negative_layer"] = layer.notna() & layer.lt(0)
    output["bridge_tag_present_exact"] = output["bridge_tag_value"].notna()
    output["movable_tag"] = output["bridge_tag_value"].astype("string").str.lower().eq("movable")
    output["fixed_bridge_tag"] = output["bridge_tag_present_exact"] & ~output["movable_tag"]
    output["bridge_like_expected_positive_gate"] = output["fixed_bridge_tag"] | output["positive_layer"]
    output["bridge_like_cached"] = truth_mask(output["bridge_like"], "cached bridge_like")
    output["bridge_classification_mismatch"] = output["bridge_like_cached"].ne(
        output["bridge_like_expected_positive_gate"]
    )

    log("Reconstructing bridge structure IDs from cached bridge-like edges (still no raw graph rebuild).")
    output, structures, landing_nodes = base.build_bridge_structures(output)
    return roads, output, structures, landing_nodes, cache_paths


def bridge_audit_path(run_dir: Path, slr_ft: int) -> Path:
    return require_path(
        run_dir / f"bridge_structures_slr_{slr_ft}ft.csv",
        f"{slr_ft} ft bridge-structure audit",
    )


def manifest_removed_counts(manifest: dict[str, object], slr_ft: int):
    summary = manifest.get("bridge_structure_counts")
    if not isinstance(summary, list):
        raise ValueError("Manifest is missing bridge_structure_counts")
    matches = [row for row in summary if isinstance(row, dict) and int(row.get("slr_ft", -1)) == slr_ft]
    if len(matches) != 1:
        raise ValueError(
            f"Manifest must contain exactly one bridge_structure_counts row for {slr_ft} ft; "
            f"found {len(matches)}"
        )
    row = matches[0]
    source_keys = {
        "non_bridge_edges_removed": "n_non_bridge_edges_removed",
        "bridge_edges_removed": "n_bridge_edges_removed",
        "total_edges_removed": "n_total_edges_removed",
    }
    missing = [source for source in source_keys.values() if source not in row]
    if missing:
        raise ValueError(f"Manifest bridge summary at {slr_ft} ft is missing: {missing}")
    return {target: int(row[source]) for target, source in source_keys.items()}


def removed_positions_for_rule(validity: ModuleType, edges, slr_layer, audit, rule: str):
    np = validity.np
    intersecting = validity.query_removed_edge_positions(edges, slr_layer)
    intersecting_set = set(int(value) for value in intersecting.tolist())
    bridge_mask = edges["bridge_like_cached"].to_numpy(dtype=bool)
    nonbridge_intersecting = {position for position in intersecting_set if not bridge_mask[position]}
    if rule == "intersect":
        removed = intersecting_set
    elif rule == "retain":
        removed = nonbridge_intersecting
    elif rule == "approach":
        require_columns(audit, ["structure_id", "removed"], "approach bridge audit")
        removed_structures = set(
            audit.loc[truth_mask(audit["removed"], "bridge audit removed"), "structure_id"].astype(int)
        )
        structure_values = edges["structure_id"]
        bridge_removed = set(
            np.flatnonzero(structure_values.isin(removed_structures).to_numpy()).astype(int).tolist()
        )
        removed = nonbridge_intersecting | bridge_removed
    else:
        raise ValueError(f"Unsupported bridge rule: {rule}")
    return np.asarray(sorted(removed), dtype=int), intersecting


def run_group_b(validity: ModuleType, runs: dict[str, dict[str, object]]):
    pd = validity.pd
    roads, edges, structures, landing_nodes, cache_paths = prepare_bridge_inputs(validity)
    del structures, landing_nodes, cache_paths
    slr_layers = validity.load_zero_and_one_ft_layers(roads)
    records: list[dict[str, object]] = []
    removed_lookup: dict[tuple[str, int], set[int]] = {}

    inventory_masks = {
        "all_segments": pd.Series(True, index=edges.index),
        "bridge_tag_present": edges["bridge_tag_present_exact"],
        "nonzero_layer": edges["nonzero_layer"],
        "nonzero_layer_no_bridge_tag": edges["nonzero_layer"] & ~edges["bridge_tag_present_exact"],
        "neither_bridge_nor_nonzero_layer": ~edges["bridge_tag_present_exact"] & ~edges["nonzero_layer"],
        "movable_tag": edges["movable_tag"],
        "negative_layer": edges["negative_layer"],
        "positive_gate_classification_mismatch": edges["bridge_classification_mismatch"],
    }

    for rule in RUN_ORDER:
        run = runs[rule]
        for slr_ft, slr_layer in zip((0, 1), slr_layers):
            audit = pd.read_csv(bridge_audit_path(run["run_dir"], slr_ft), low_memory=False)
            removed_positions, intersecting = removed_positions_for_rule(
                validity, edges, slr_layer, audit, rule
            )
            removed_lookup[(rule, slr_ft)] = set(removed_positions.tolist())
            removed_mask = pd.Series(False, index=edges.index)
            removed_mask.iloc[removed_positions] = True
            manifest_counts = manifest_removed_counts(run["manifest"], slr_ft)
            actual_nonbridge = int((removed_mask & ~edges["bridge_like_cached"]).sum())
            actual_bridge = int((removed_mask & edges["bridge_like_cached"]).sum())
            actual_total = int(removed_mask.sum())
            actual = {
                "non_bridge_edges_removed": actual_nonbridge,
                "bridge_edges_removed": actual_bridge,
                "total_edges_removed": actual_total,
            }
            if actual != manifest_counts:
                raise ValueError(
                    f"Reconstructed {rule} removals at {slr_ft} ft do not match manifest: "
                    f"reconstructed={actual}, manifest={manifest_counts}"
                )

            for category, category_mask in inventory_masks.items():
                selected = removed_mask & category_mask
                records.append(
                    {
                        "record_type": "removed_segments",
                        "bridge_rule": rule,
                        "slr_ft": slr_ft,
                        "category": category,
                        "segment_count": int(selected.sum()),
                        "length_m": float(edges.loc[selected, "length_m"].sum()),
                        "length_km": float(edges.loc[selected, "length_m"].sum() / 1000),
                        "parent_way_count": int(edges.loc[selected, "osm_id"].nunique()),
                        "structure_count": int(edges.loc[selected, "structure_id"].dropna().nunique()),
                        "inventory_segment_count": int(category_mask.sum()),
                        "inventory_length_m": float(edges.loc[category_mask, "length_m"].sum()),
                        "inventory_length_km": float(edges.loc[category_mask, "length_m"].sum() / 1000),
                        "intersecting_segment_count": int(len(intersecting)),
                        "reference_value": "",
                        "delta": "",
                    }
                )

    intersect0 = pd.DataFrame.from_records(records)
    intersect0 = intersect0.loc[
        (intersect0["bridge_rule"] == "intersect") & (intersect0["slr_ft"] == 0)
    ]
    bridge_row = intersect0.loc[intersect0["category"] == "bridge_tag_present"].iloc[0]
    all_row = intersect0.loc[intersect0["category"] == "all_segments"].iloc[0]
    count_share = float(bridge_row["segment_count"] / all_row["segment_count"]) if all_row["segment_count"] else math.nan
    length_share = float(bridge_row["length_m"] / all_row["length_m"]) if all_row["length_m"] else math.nan
    control_close = (
        abs(float(bridge_row["parent_way_count"]) - PRE_FIX_BRIDGE_STRUCTURE_COUNT)
        <= PRE_FIX_BRIDGE_STRUCTURE_COUNT * 0.10
        and abs(float(bridge_row["length_km"]) - PRE_FIX_BRIDGE_LENGTH_KM)
        <= PRE_FIX_BRIDGE_LENGTH_KM * 0.10
        and min(abs(count_share - PRE_FIX_BRIDGE_REMOVAL_SHARE), abs(length_share - PRE_FIX_BRIDGE_REMOVAL_SHARE))
        <= 0.05
    )
    for metric, current, reference in (
        ("bridge_parent_way_count", float(bridge_row["parent_way_count"]), PRE_FIX_BRIDGE_STRUCTURE_COUNT),
        ("bridge_length_km", float(bridge_row["length_km"]), PRE_FIX_BRIDGE_LENGTH_KM),
        ("bridge_segment_count_share", count_share, PRE_FIX_BRIDGE_REMOVAL_SHARE),
        ("bridge_length_share", length_share, PRE_FIX_BRIDGE_REMOVAL_SHARE),
    ):
        records.append(
            {
                "record_type": "pre_fix_comparison",
                "bridge_rule": "intersect",
                "slr_ft": 0,
                "category": metric,
                "segment_count": "",
                "length_m": "",
                "length_km": "",
                "parent_way_count": "",
                "structure_count": "",
                "inventory_segment_count": "",
                "intersecting_segment_count": "",
                "reference_value": reference,
                "current_value": current,
                "delta": current - reference,
            }
        )

    bridge_counts = {
        rule: next(
            row["segment_count"]
            for row in records
            if row["record_type"] == "removed_segments"
            and row["bridge_rule"] == rule
            and row["slr_ft"] == 0
            and row["category"] == "bridge_tag_present"
        )
        for rule in RUN_ORDER
    }
    pattern_holds = (
        bool(control_close)
        and bridge_counts["approach"] < bridge_counts["intersect"]
        and bridge_counts["retain"] == 0
    )
    mismatch_count = int(edges["bridge_classification_mismatch"].sum())
    movable_inventory = int(edges["movable_tag"].sum())
    negative_inventory = int(edges["negative_layer"].sum())
    negative_gate_eligible = edges["negative_layer"] & ~edges["fixed_bridge_tag"]
    negative_wrongly_protected = negative_gate_eligible & edges["bridge_like_cached"]
    movable_gate_eligible = edges["movable_tag"] & ~edges["positive_layer"]
    movable_wrongly_protected = movable_gate_eligible & edges["bridge_like_cached"]
    control_movable_scenario_segments = sum(
        int(
            row["segment_count"]
        )
        for row in records
        if row["record_type"] == "removed_segments"
        and row["bridge_rule"] == "intersect"
        and row["slr_ft"] in {0, 1}
        and row["category"] == "movable_tag"
    )
    control_negative_scenario_segments = sum(
        int(row["segment_count"])
        for row in records
        if row["record_type"] == "removed_segments"
        and row["bridge_rule"] == "intersect"
        and row["slr_ft"] in {0, 1}
        and row["category"] == "negative_layer"
    )
    records.extend(
        [
            {
                "record_type": "assertion",
                "bridge_rule": "all",
                "slr_ft": 0,
                "category": "expected_control_and_collapse_pattern",
                "current_value": str(pattern_holds),
                "reference_value": "intersect approximately pre-fix; approach lower; retain zero",
                "delta": "",
            },
            {
                "record_type": "tag_inventory",
                "bridge_rule": "all",
                "slr_ft": "",
                "category": "movable_tag_segments",
                "segment_count": movable_inventory,
                "reference_value": "",
                "delta": "",
            },
            {
                "record_type": "tag_inventory",
                "bridge_rule": "all",
                "slr_ft": "",
                "category": "negative_layer_segments",
                "segment_count": negative_inventory,
                "reference_value": "",
                "delta": "",
            },
            {
                "record_type": "pre_fix_comparison",
                "bridge_rule": "intersect",
                "slr_ft": "0_and_1",
                "category": "movable_removed_person_scenarios",
                "current_value": control_movable_scenario_segments,
                "reference_value": PRE_FIX_MOVABLE_SEGMENTS,
                "delta": control_movable_scenario_segments - PRE_FIX_MOVABLE_SEGMENTS,
            },
            {
                "record_type": "pre_fix_comparison",
                "bridge_rule": "intersect",
                "slr_ft": "0_and_1",
                "category": "negative_layer_removed_person_scenarios",
                "current_value": control_negative_scenario_segments,
                "reference_value": PRE_FIX_NEGATIVE_LAYER_SEGMENTS,
                "delta": control_negative_scenario_segments - PRE_FIX_NEGATIVE_LAYER_SEGMENTS,
            },
            {
                "record_type": "assertion",
                "bridge_rule": "all",
                "slr_ft": "",
                "category": "positive_layer_and_fixed_span_gate_matches_cache",
                "segment_count": mismatch_count,
                "current_value": str(mismatch_count == 0),
                "reference_value": 0,
                "delta": mismatch_count,
            },
            {
                "record_type": "assertion",
                "bridge_rule": "all",
                "slr_ft": "",
                "category": "negative_layer_without_fixed_bridge_is_removal_eligible",
                "inventory_segment_count": int(negative_gate_eligible.sum()),
                "segment_count": int(negative_wrongly_protected.sum()),
                "current_value": str(not negative_wrongly_protected.any()),
                "reference_value": 0,
                "delta": int(negative_wrongly_protected.sum()),
            },
            {
                "record_type": "assertion",
                "bridge_rule": "all",
                "slr_ft": "",
                "category": "movable_without_positive_layer_is_removal_eligible",
                "inventory_segment_count": int(movable_gate_eligible.sum()),
                "segment_count": int(movable_wrongly_protected.sum()),
                "current_value": str(not movable_wrongly_protected.any()),
                "reference_value": 0,
                "delta": int(movable_wrongly_protected.sum()),
            },
        ]
    )
    frame = pd.DataFrame.from_records(records)
    verdict = "PASS" if pattern_holds and mismatch_count == 0 else (
        "FAIL" if mismatch_count else "REVIEW"
    )
    return frame, verdict


def locate_service_audit(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "service_snapping_audit.csv",
        run_dir / "diagnostics" / "validity" / "service_2ecc_assignments.csv",
    ]
    return next((path for path in candidates if path.exists()), None)


def rebuild_service_audit(validity: ModuleType):
    base = validity.base
    paths = locate_cache_paths(base)
    log("--rebuild-graph supplied: rebuilding the full raw graph in memory.")
    nodes = validity.gpd.read_parquet(paths["nodes"])
    edges = validity.gpd.read_parquet(paths["edges"])
    graph = base.build_graph(edges)
    membership = base.compute_raw_graph_membership(graph)
    roads = base.load_roads()
    services_source = base.load_services()
    boundary = base.build_study_area_boundary(tuple(roads.total_bounds), roads.crs)
    services = base.filter_services_by_buffer(services_source, boundary)
    tree, _, node_ids = base.build_node_kdtree(nodes)
    attached = base.attach_services_to_raw_graph(
        services,
        nodes=nodes,
        unconstrained_tree=tree,
        unconstrained_node_ids=node_ids,
        raw_membership=membership,
        use_eligible_service_nodes=True,
    )
    return attached


def normalize_service_audit(frame, context: str):
    frame = frame.copy()
    if "service_raw_2ecc_size" not in frame.columns:
        if "two_edge_component_size" not in frame.columns:
            raise ValueError(
                f"{context} lacks service_raw_2ecc_size or two_edge_component_size"
            )
        frame["service_raw_2ecc_size"] = frame["two_edge_component_size"]
    require_columns(frame, ["service_type", "service_raw_2ecc_size"], context)
    return frame


def run_group_c(validity: ModuleType, runs: dict[str, dict[str, object]], rebuild_graph: bool):
    pd = validity.pd
    rebuilt = rebuild_service_audit(validity) if rebuild_graph else None
    records: list[dict[str, object]] = []
    for rule in RUN_ORDER:
        if rebuilt is not None:
            audit = rebuilt.copy()
            source = "rebuilt_in_memory"
        else:
            path = locate_service_audit(runs[rule]["run_dir"])
            if path is None:
                raise FileNotFoundError(
                    f"No service validity artifact found in {runs[rule]['run_dir']}. "
                    "Pass --rebuild-graph to rebuild it explicitly."
                )
            log(f"Reading existing service diagnostic for {rule}: {path}")
            audit = pd.read_csv(path, low_memory=False)
            source = str(path)
            if path.name == "service_2ecc_assignments.csv":
                recovery_path = path.parent / "service_resnapping.csv"
                if recovery_path.exists():
                    recovery = pd.read_csv(recovery_path, low_memory=False)
                    require_columns(
                        recovery,
                        [
                            "service_record_id", "old_snap_distance_m",
                            "new_snap_distance_m", "new_snap_valid", "moved",
                            "added_snap_distance_m",
                        ],
                        f"{rule} service resnapping artifact",
                    )
                    if recovery["service_record_id"].duplicated().any():
                        raise ValueError(
                            f"{rule} service resnapping artifact has duplicate service_record_id values"
                        )
                    recovery_columns = [
                        "service_record_id", "old_snap_distance_m",
                        "new_snap_distance_m", "new_snap_valid", "moved",
                        "added_snap_distance_m",
                    ]
                    audit = audit.merge(
                        recovery[recovery_columns],
                        on="service_record_id",
                        how="left",
                        validate="one_to_one",
                    )
                    use_new = truth_mask(audit["new_snap_valid"], f"{rule} new_snap_valid")
                    audit["snap_distance_m"] = audit["old_snap_distance_m"]
                    audit.loc[use_new, "snap_distance_m"] = audit.loc[
                        use_new, "new_snap_distance_m"
                    ]
                    audit["service_node_moved"] = audit["moved"]
                    audit["service_snap_distance_penalty_m"] = audit[
                        "added_snap_distance_m"
                    ]
                    source += f" + {recovery_path}"
        audit = normalize_service_audit(audit, f"{rule} service audit")
        groups = [("all", audit)] + [
            (str(service_type), group)
            for service_type, group in audit.groupby("service_type", dropna=False, sort=True)
        ]
        for service_type, group in groups:
            sizes = pd.to_numeric(group["service_raw_2ecc_size"], errors="raise")
            singleton = sizes.eq(1)
            pre_fix = (
                PRE_FIX_SERVICE_SINGLETON_SHARE if service_type == "all"
                else PRE_FIX_FIRE_SINGLETON_SHARE if service_type == "fire_station"
                else math.nan
            )
            records.append(
                {
                    "record_type": "singleton_2ecc",
                    "bridge_rule": rule,
                    "service_type": service_type,
                    "service_count": int(len(group)),
                    "singleton_count": int(singleton.sum()),
                    "singleton_share": float(singleton.mean()) if len(group) else math.nan,
                    "pre_fix_share": pre_fix,
                    "share_delta": float(singleton.mean() - pre_fix) if len(group) and not math.isnan(pre_fix) else math.nan,
                    "source": source,
                }
            )

        if "service_node_moved" in audit.columns:
            moved = truth_mask(audit["service_node_moved"], f"{rule} service_node_moved")
        elif {"node_id", "unconstrained_node_id"}.issubset(audit.columns):
            moved = audit["node_id"].ne(audit["unconstrained_node_id"])
        else:
            moved = pd.Series(False, index=audit.index)
        penalty_column = "service_snap_distance_penalty_m"
        if penalty_column not in audit.columns and {
            "snap_distance_m", "unconstrained_snap_distance_m"
        }.issubset(audit.columns):
            penalty = audit["snap_distance_m"] - audit["unconstrained_snap_distance_m"]
        elif penalty_column in audit.columns:
            penalty = pd.to_numeric(audit[penalty_column], errors="raise")
        else:
            penalty = pd.Series(math.nan, index=audit.index)
        moved_penalty = penalty.loc[moved].dropna()
        require_columns(audit, ["snap_distance_m"], f"{rule} service audit")
        chosen_distance = pd.to_numeric(audit["snap_distance_m"], errors="raise")
        records.append(
            {
                "record_type": "resnapping",
                "bridge_rule": rule,
                "service_type": "all",
                "service_count": int(len(audit)),
                "moved_count": int(moved.sum()),
                "moved_share": float(moved.mean()) if len(audit) else math.nan,
                "added_distance_median_m": float(moved_penalty.median()) if len(moved_penalty) else math.nan,
                "added_distance_p90_m": float(moved_penalty.quantile(0.90)) if len(moved_penalty) else math.nan,
                "added_distance_max_m": float(moved_penalty.max()) if len(moved_penalty) else math.nan,
                "exceeds_max_service_snap_count": int(chosen_distance.gt(validity.base.MAX_SERVICE_SNAP_M).sum()),
                "pre_fix_added_distance_median_m": PRE_FIX_SERVICE_MOVE_MEDIAN_M,
                "source": source,
            }
        )
    frame = pd.DataFrame.from_records(records)
    all_rows = frame.loc[(frame["record_type"] == "singleton_2ecc") & (frame["service_type"] == "all")]
    fire_rows = frame.loc[(frame["record_type"] == "singleton_2ecc") & (frame["service_type"] == "fire_station")]
    changed = bool(
        all_rows["singleton_share"].lt(PRE_FIX_SERVICE_SINGLETON_SHARE).all()
        and fire_rows["singleton_share"].lt(PRE_FIX_FIRE_SINGLETON_SHARE).all()
    )
    return frame, ("PASS" if changed else "REVIEW")


def analysis_masks(frame):
    require_columns(
        frame,
        ["pop20", "land_area_m2", "analysis_eligible", "baseline_status"],
        "block-universe output",
    )
    pop = frame["pop20"]
    land = frame["land_area_m2"]
    if pop.isna().any() or land.isna().any():
        raise ValueError("pop20 or land_area_m2 contains missing values")
    eligible = truth_mask(frame["analysis_eligible"], "analysis_eligible")
    pop_positive = pop.gt(0)
    land_positive = land.gt(0)
    return eligible, pop_positive, land_positive


def run_group_d(diag02b: ModuleType, runs: dict[str, dict[str, object]]):
    pd = diag02b.pd
    records: list[dict[str, object]] = []
    comparison_universe_violations = 0
    for rule in RUN_ORDER:
        frame = runs[rule]["frame"]
        baseline = frame.loc[frame["slr_ft"].eq(0)].copy()
        eligible, pop_positive, land_positive = analysis_masks(baseline)
        comparison_universe = eligible & pop_positive & land_positive
        masks = {
            "blocks_before_filtering": pd.Series(True, index=baseline.index),
            "blocks_after_analysis_eligible": eligible,
            "blocks_after_population_criterion": pop_positive,
            "blocks_after_land_area_criterion": land_positive,
            "blocks_after_population_and_land_criteria": pop_positive & land_positive,
            "blocks_after_comparison_analysis_universe": comparison_universe,
            "dropped_by_population_from_pipeline_eligible": eligible & ~pop_positive,
            "dropped_by_zero_land_from_full_universe": ~land_positive,
            "dropped_population_only": ~pop_positive & land_positive,
            "dropped_land_area_only": pop_positive & ~land_positive,
            "dropped_both_population_and_land": ~pop_positive & ~land_positive,
        }
        for metric, mask in masks.items():
            records.append(
                {
                    "record_type": "filter_count",
                    "bridge_rule": rule,
                    "universe": "baseline",
                    "baseline_status": "all",
                    "criterion": metric,
                    "flag_value": "",
                    "block_count": int(mask.sum()),
                }
            )
        for universe, universe_mask in (
            ("all_blocks", pd.Series(True, index=baseline.index)),
            ("analysis_eligible", eligible),
            ("comparison_analysis_universe", comparison_universe),
        ):
            for criterion, zero_mask in (
                ("pop20_eq_0", ~pop_positive),
                ("land_area_m2_eq_0", ~land_positive),
            ):
                table = pd.crosstab(
                    baseline.loc[universe_mask, "baseline_status"],
                    zero_mask.loc[universe_mask],
                    dropna=False,
                )
                for status in table.index:
                    for flag in table.columns:
                        records.append(
                            {
                                "record_type": "status_crosstab",
                                "bridge_rule": rule,
                                "universe": universe,
                                "baseline_status": status,
                                "criterion": criterion,
                                "flag_value": bool(flag),
                                "block_count": int(table.loc[status, flag]),
                            }
                        )
        comparison_universe_violations += int(
            (comparison_universe & (~pop_positive | ~land_positive)).sum()
        )
    frame = pd.DataFrame.from_records(records)
    # Zero-population blocks may be pipeline-eligible, but are excluded here.
    verdict = "PASS" if comparison_universe_violations == 0 else "FAIL"
    return frame, verdict


def add_population_metric(records, *, rule, slr_ft, metric, value, universe, reference_value=math.nan):
    for count_type in ("person_scenario", "unique_person"):
        records.append(
            {
                "bridge_rule": rule,
                "slr_ft": slr_ft,
                "metric": metric,
                "value": float(value),
                "unit": "persons",
                "universe": universe,
                "population_count_type": count_type,
                "reference_value": reference_value,
                "delta": float(value - reference_value) if not math.isnan(reference_value) else math.nan,
            }
        )


def headline_for_run(pd, rule: str, frame, *, universe_label: str):
    require_columns(
        frame,
        [
            "block_geoid", "slr_ft", "baseline_status", "scenario_status", "pop20",
            *NEW_STATUS_FLAGS.values(),
        ],
        f"{rule} headline output",
    )
    eligible, pop_positive, land_positive = analysis_masks(frame)
    if universe_label == "all_blocks":
        universe = pd.Series(True, index=frame.index)
    elif universe_label == "analysis_universe":
        universe = eligible & pop_positive & land_positive
    else:
        raise ValueError(universe_label)
    work = frame.loc[universe].copy()
    records: list[dict[str, object]] = []
    baseline = work.loc[work["slr_ft"].eq(0)].copy()
    for status in STATUS_ORDER:
        status_rows = baseline["baseline_status"].eq(status)
        records.append(
            {
                "bridge_rule": rule,
                "slr_ft": 0,
                "metric": f"baseline_status_{status}_blocks",
                "value": float(status_rows.sum()),
                "unit": "blocks",
                "universe": universe_label,
                "population_count_type": "not_applicable",
                "reference_value": math.nan,
                "delta": math.nan,
            }
        )
        add_population_metric(
            records,
            rule=rule,
            slr_ft=0,
            metric=f"baseline_status_{status}_population",
            value=float(baseline.loc[status_rows, "pop20"].sum()),
            universe=universe_label,
        )
    fragile_share = float(baseline["baseline_status"].eq("fragile").mean()) if len(baseline) else math.nan
    records.append(
        {
            "bridge_rule": rule,
            "slr_ft": 0,
            "metric": "baseline_fragile_share",
            "value": fragile_share,
            "unit": "share",
            "universe": universe_label,
            "population_count_type": "not_applicable",
            "reference_value": PRE_FIX_BASELINE_FRAGILE_SHARE,
            "delta": fragile_share - PRE_FIX_BASELINE_FRAGILE_SHARE,
        }
    )

    for slr_ft, scenario in work.groupby("slr_ft", sort=True):
        for status, column in NEW_STATUS_FLAGS.items():
            mask = truth_mask(scenario[column], f"{rule} {column} at {slr_ft} ft")
            records.append(
                {
                    "bridge_rule": rule,
                    "slr_ft": int(slr_ft),
                    "metric": f"new_{status}_blocks",
                    "value": float(mask.sum()),
                    "unit": "blocks",
                    "universe": universe_label,
                    "population_count_type": "not_applicable",
                    "reference_value": math.nan,
                    "delta": math.nan,
                }
            )
            add_population_metric(
                records,
                rule=rule,
                slr_ft=int(slr_ft),
                metric=f"new_{status}_population",
                value=float(scenario.loc[mask, "pop20"].sum()),
                universe=universe_label,
            )

    positive_levels = sorted(int(value) for value in work["slr_ft"].unique() if int(value) > 0)
    for threshold in positive_levels:
        subset = work.loc[work["slr_ft"].between(1, threshold)].copy()
        affected = pd.Series(False, index=subset.index)
        for column in NEW_STATUS_FLAGS.values():
            affected |= truth_mask(subset[column], f"{rule} cumulative {column}")
        affected_rows = subset.loc[affected]
        person_scenario = float(affected_rows["pop20"].sum())
        unique = float(
            affected_rows[["block_geoid", "pop20"]]
            .drop_duplicates("block_geoid")["pop20"]
            .sum()
        )
        reference = PRE_FIX_CUMULATIVE_AFFECTED_POPULATION if threshold == 6 else math.nan
        for count_type, value in (
            ("person_scenario", person_scenario),
            ("unique_person", unique),
        ):
            records.append(
                {
                    "bridge_rule": rule,
                    "slr_ft": threshold,
                    "metric": "cumulative_newly_affected_population",
                    "value": value,
                    "unit": "persons",
                    "universe": universe_label,
                    "population_count_type": count_type,
                    "reference_value": reference,
                    "delta": value - reference if not math.isnan(reference) else math.nan,
                }
            )
    return records


def transition_rows(pd, rule: str, frame, universe_label: str):
    eligible, pop_positive, land_positive = analysis_masks(frame)
    universe = (
        pd.Series(True, index=frame.index)
        if universe_label == "all_blocks"
        else eligible & pop_positive & land_positive
    )
    work = frame.loc[universe]
    records: list[dict[str, object]] = []
    for slr_ft, scenario in work.groupby("slr_ft", sort=True):
        grouped = (
            scenario.groupby(["baseline_status", "scenario_status"], dropna=False)
            .agg(block_count=("block_geoid", "size"), population=("pop20", "sum"))
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            base_record = {
                "record_type": "transition",
                "bridge_rule": rule,
                "slr_ft": int(slr_ft),
                "universe": universe_label,
                "baseline_status": row.baseline_status,
                "scenario_status": row.scenario_status,
                "block_count": int(row.block_count),
            }
            for count_type in ("person_scenario", "unique_person"):
                records.append(
                    {
                        **base_record,
                        "population": float(row.population),
                        "population_count_type": count_type,
                    }
                )
    return records


def load_optional_pre_fix(diag02b: ModuleType, base: ModuleType, path: Path | None):
    if path is None:
        return None
    run_dir = require_path(path, "pre-fix run directory")
    files = diag02b.find_result_files(run_dir)
    frame, duplicates = diag02b.load_and_combine_results(files)
    if len(duplicates):
        raise ValueError(f"Pre-fix run contains {len(duplicates)} duplicate block/scenario rows")
    if "pop20" not in frame.columns or "land_area_m2" not in frame.columns:
        attributes = base.load_block_attributes()
        frame = frame.merge(attributes, on="block_geoid", how="left", validate="many_to_one")
    if "analysis_eligible" not in frame.columns:
        frame["analysis_eligible"] = frame["land_area_m2"].gt(0)
    log(f"Loaded optional pre-fix comparison run: {run_dir}")
    return {"rule": "pre_fix", "run_dir": run_dir, "frame": frame}


def run_group_e(diag02b: ModuleType, runs: dict[str, dict[str, object]], pre_fix):
    pd = diag02b.pd
    headline_records: list[dict[str, object]] = []
    transition_records: list[dict[str, object]] = []
    comparison_runs = {rule: runs[rule] for rule in RUN_ORDER}
    if pre_fix is not None:
        comparison_runs["pre_fix"] = pre_fix
    for rule, run in comparison_runs.items():
        for universe_label in ("all_blocks", "analysis_universe"):
            headline_records.extend(
                headline_for_run(pd, rule, run["frame"], universe_label=universe_label)
            )
            transition_records.extend(
                transition_rows(pd, rule, run["frame"], universe_label)
            )
    headline = pd.DataFrame.from_records(headline_records)
    transitions = pd.DataFrame.from_records(transition_records)
    check = pd.concat(
        [
            headline.assign(record_type="headline"),
            transitions,
        ],
        ignore_index=True,
        sort=False,
    )
    verdict = "PASS" if pre_fix is not None else "REVIEW"
    return check, headline, verdict


def run_group_f(diag02b: ModuleType, runs: dict[str, dict[str, object]]):
    pd = diag02b.pd
    records: list[dict[str, object]] = []
    failed = False
    counts_by_rule: dict[str, dict[int, int]] = {}
    for rule in RUN_ORDER:
        frame = runs[rule]["frame"]
        inundated = (
            frame.assign(_flag=truth_mask(frame["block_centroid_inundated"], f"{rule} inundated"))
            .groupby("slr_ft", sort=True)["_flag"]
            .sum()
        )
        previous = None
        for slr_ft, count in inundated.items():
            violation = previous is not None and int(count) < int(previous)
            failed |= violation
            records.append(
                {
                    "check": "inundated_non_decreasing",
                    "bridge_rule": rule,
                    "slr_ft": int(slr_ft),
                    "value": int(count),
                    "previous_value": "" if previous is None else int(previous),
                    "violation_magnitude": 0 if not violation else int(previous - count),
                    "verdict": "FAIL" if violation else "PASS",
                }
            )
            previous = int(count)
        worse = frame["scenario_status"].isin(["fragile", "isolated", "inundated"])
        counts_by_rule[rule] = (
            frame.assign(_worse=worse).groupby("slr_ft", sort=True)["_worse"].sum().astype(int).to_dict()
        )

    levels = sorted(set.intersection(*(set(values) for values in counts_by_rule.values())))
    for slr_ft in levels:
        retain = counts_by_rule["retain"][slr_ft]
        approach = counts_by_rule["approach"][slr_ft]
        intersect = counts_by_rule["intersect"][slr_ft]
        violation = not (retain <= approach <= intersect)
        magnitude = max(retain - approach, approach - intersect, 0)
        failed |= violation
        records.append(
            {
                "check": "fragile_or_worse_order_retain_le_approach_le_intersect",
                "bridge_rule": "all",
                "slr_ft": int(slr_ft),
                "value": f"retain={retain};approach={approach};intersect={intersect}",
                "previous_value": "",
                "violation_magnitude": int(magnitude),
                "verdict": "FAIL" if violation else "PASS",
            }
        )
    return pd.DataFrame.from_records(records), ("FAIL" if failed else "PASS")


def what_moved(pd, headline):
    keys = ["slr_ft", "metric", "unit", "universe", "population_count_type"]
    available_rules = [
        rule for rule in (*RUN_ORDER, "pre_fix")
        if rule in set(headline["bridge_rule"])
    ]
    selected = headline.loc[headline["bridge_rule"].isin(available_rules)].copy()
    wide = selected.pivot(index=keys, columns="bridge_rule", values="value").reset_index()
    pairs = [
        ("intersect", "approach"),
        ("approach", "retain"),
        ("intersect", "retain"),
    ]
    if "pre_fix" in available_rules:
        pairs.extend(("pre_fix", rule) for rule in RUN_ORDER)
    records: list[dict[str, object]] = []
    for row in wide.itertuples(index=False):
        values = {rule: getattr(row, rule, math.nan) for rule in available_rules}
        for reference_rule, comparison_rule in pairs:
            reference = values.get(reference_rule, math.nan)
            current = values.get(comparison_rule, math.nan)
            if pd.isna(reference) or pd.isna(current):
                continue
            if row.unit == "share":
                moved = abs(current - reference) > 0.01
                change = (current - reference) * 100
                change_label = f"{change:+.3f} percentage points"
            else:
                relative = math.inf if reference == 0 and current != 0 else (abs(current - reference) / abs(reference) if reference else 0)
                moved = relative > 0.01
                change_label = f"{current - reference:+,.0f} ({relative:.2%})"
            if moved:
                records.append(
                    {
                        "comparison": f"{comparison_rule}_vs_{reference_rule}",
                        "slr_ft": row.slr_ft,
                        "metric": row.metric,
                        "universe": row.universe,
                        "population_count_type": row.population_count_type,
                        "reference_value": reference,
                        "comparison_value": current,
                        "change": change_label,
                    }
                )
    return pd.DataFrame.from_records(records)


def build_summary(
    diag02b: ModuleType,
    runs: dict[str, dict[str, object]],
    verdicts: dict[str, str],
    tables: dict[str, object],
    moved=None,
) -> str:
    pd = diag02b.pd
    lines = [
        "# Bridge-rule run comparison",
        "",
        "## Run identification",
        "",
    ]
    identification = pd.DataFrame(
        [
            {
                "bridge_rule": rule,
                "detected_rule": run["detected_rule"],
                "run_directory": str(run["run_dir"]),
                "result_files": len(run["files"]),
                "unique_blocks": run["frame"]["block_geoid"].nunique(),
                "slr_levels": ",".join(map(str, sorted(run["frame"]["slr_ft"].unique()))),
            }
            for rule, run in runs.items()
        ]
    )
    lines.extend([markdown_table(identification), "", "## Verdicts", ""])
    verdict_table = pd.DataFrame(
        [{"check_group": group, "verdict": verdicts.get(group, "NOT RUN")} for group in "ABCDEF"]
    )
    lines.extend([markdown_table(verdict_table), ""])

    if verdicts.get("A") == "FAIL":
        failed = tables["A"].loc[tables["A"]["verdict"].eq("FAIL")]
        lines.extend(
            [
                "## Group A — comparability (FAIL)",
                "",
                "The hard comparability gate failed. Groups B–F were not run.",
                "",
                markdown_table(failed[["bridge_rule", "check", "value", "detail"]], max_rows=30),
                "",
            ]
        )
        return "\n".join(lines)

    for group, title in (
        ("A", "Comparability"),
        ("B", "Bridge fix"),
        ("C", "Service snapping"),
        ("D", "Block universe filter"),
        ("E", "Headline numbers"),
        ("F", "Ordering"),
    ):
        lines.extend([f"## Group {group} — {title} ({verdicts[group]})", ""])
        table = tables[group]
        if group == "A":
            display = table.loc[
                table["record_type"].isin(
                    [
                        "run_summary",
                        "assertion",
                        "descriptive",
                        "status_reassignment",
                        "unclassified_descriptive",
                        "analysis_universe_descriptive",
                    ]
                )
            ]
        elif group == "B":
            display = table.loc[
                (table["record_type"].isin(["pre_fix_comparison", "assertion", "tag_inventory"]))
                | ((table["record_type"] == "removed_segments") & table["category"].isin(["all_segments", "bridge_tag_present", "movable_tag", "negative_layer"]))
            ]
        elif group == "C":
            display = table
        elif group == "D":
            display = table.loc[table["record_type"] == "filter_count"]
        elif group == "E":
            display = table.loc[
                (table["record_type"] == "headline")
                & table["metric"].isin(
                    ["baseline_fragile_share", "cumulative_newly_affected_population"]
                )
                & table["slr_ft"].isin([0, 1, 6])
            ]
        else:
            display = table.loc[table["check"].str.contains("fragile_or_worse", na=False)]
        if group == "B":
            pattern_rows = table.loc[
                (table["record_type"] == "assertion")
                & table["category"].eq("expected_control_and_collapse_pattern")
            ]
            pattern_holds = (
                not pattern_rows.empty
                and str(pattern_rows.iloc[0].get("current_value", "")).lower() == "true"
            )
            lines.append(
                "The expected control pattern "
                + ("holds" if pattern_holds else "does not hold")
                + ": intersect should approximate the pre-fix bridge removals, "
                "while approach and retain should sharply reduce them."
            )
            lines.append("")
        elif group == "C":
            lines.extend(
                [
                    "Singleton 2-edge-connected-component shares are compared directly with the pre-fix 26.3% overall and 45.1% fire-station rates.",
                    "",
                ]
            )
        elif group == "D":
            lines.extend(
                [
                    "The filter counts and status cross-tabs distinguish the stored analysis_eligible flag from the population and land-area criteria.",
                    "",
                ]
            )
        elif group == "E":
            lines.extend(
                [
                    "Population metrics are emitted as both person-scenario totals and block-deduplicated unique-person totals.",
                    "",
                ]
            )
        elif group == "F":
            lines.extend(
                [
                    "Ordering is checked as retain ≤ approach ≤ intersect at every SLR level.",
                    "",
                ]
            )
        lines.extend(
            [markdown_table(display, max_rows=(len(display) if group == "A" else 80)), ""]
        )

    lines.extend(["## What moved", ""])
    if moved is None or moved.empty:
        lines.append("No corrected-run headline difference exceeded 1 percentage point or 1% relative.")
    else:
        lines.append(
            "The following pairwise headline values differ by more than 1 percentage point or 1% relative."
        )
        lines.extend(["", markdown_table(moved, max_rows=len(moved)), ""])
    return "\n".join(lines)


def print_condensed(verdicts: dict[str, str], tables: dict[str, object]) -> None:
    print("\nBRIDGE-RULE RUN COMPARISON", flush=True)
    print(" ".join(f"{group}={verdicts.get(group, 'NOT RUN')}" for group in "ABCDEF"), flush=True)
    if verdicts.get("A") == "FAIL":
        failures = tables["A"].loc[
            tables["A"]["verdict"].eq("FAIL")
            & tables["A"]["record_type"].isin(["run_summary", "assertion"])
        ]
        for row in failures.itertuples(index=False):
            print(f"A_FAIL rule={row.bridge_rule} check={row.check} count={row.value}", flush=True)
        print("Groups B-F not run because Group A is a hard gate.", flush=True)
        return
    bridge = tables["B"]
    selected = bridge.loc[
        (bridge["record_type"] == "removed_segments")
        & bridge["slr_ft"].isin([0, 1])
        & bridge["category"].eq("bridge_tag_present")
    ]
    for row in selected.itertuples(index=False):
        print(
            f"bridge_removed rule={row.bridge_rule} slr_ft={row.slr_ft} "
            f"segments={row.segment_count} length_km={row.length_km:.3f}",
            flush=True,
        )
    services = tables["C"]
    services = services.loc[
        (services["record_type"] == "singleton_2ecc") & services["service_type"].eq("all")
    ]
    for row in services.itertuples(index=False):
        print(
            f"service_singleton rule={row.bridge_rule} count={row.singleton_count} "
            f"share={row.singleton_share:.3%}",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    validity = load_module("measurement_validity_helpers_02d", VALIDITY_SCRIPT)
    versions = validity.configure_dependencies(sensitivity=True)
    if versions is None:
        return 2
    validity.base = validity.load_base_module()
    diag02b = load_module("diagnose_access_helpers_02b", DIAGNOSE_SCRIPT)
    if dict(diag02b.RESULT_DTYPE) != dict(validity.RESULT_DTYPE):
        raise ValueError(
            "02b and 02d RESULT_DTYPE maps differ; refusing to choose one silently. "
            f"02b={diag02b.RESULT_DTYPE}; 02d={validity.RESULT_DTYPE}"
        )
    log(f"RESULT_DTYPE verified identical across 02b and 02d: {diag02b.RESULT_DTYPE}")

    requested_dirs = {
        "intersect": Path(args.intersect_run_dir),
        "approach": Path(args.approach_run_dir),
        "retain": Path(args.retain_run_dir),
    }
    runs: dict[str, dict[str, object]] = {}
    for expected_rule in RUN_ORDER:
        run_dir = require_path(requested_dirs[expected_rule], f"{expected_rule} run directory")
        manifest = read_manifest(run_dir)
        detected = detect_bridge_rule(manifest, run_dir)
        print(
            f"RUN expected_rule={expected_rule} detected_rule={detected} path={run_dir}",
            flush=True,
        )
        if detected != expected_rule:
            raise ValueError(
                f"Run labeled {expected_rule} reports bridge_rule={detected}: {run_dir}"
            )
        run = load_run(diag02b, expected_rule, run_dir)
        run["detected_rule"] = detected
        runs[expected_rule] = run
    if args.pre_fix_run_dir:
        print(f"PRE_FIX_RUN path={Path(args.pre_fix_run_dir).expanduser().resolve()}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    verdicts: dict[str, str] = {}
    tables: dict[str, object] = {}

    group_a, _group_a_details, verdicts["A"] = run_group_a(diag02b, runs)
    tables["A"] = group_a
    write_csv(group_a, "check_a_comparability.csv")
    stale_group_a_detail = OUTPUT_DIR / "check_a_comparability_failures.csv"
    if stale_group_a_detail.exists():
        log(
            "ignored_legacy_output="
            f"{stale_group_a_detail}; current Group A details are in check_a_comparability.csv"
        )
    if verdicts["A"] == "FAIL":
        summary = build_summary(diag02b, runs, verdicts, tables)
        summary_path = OUTPUT_DIR / "run_comparison_summary.md"
        summary_path.write_text(summary, encoding="utf-8")
        log(f"saved={summary_path}")
        print_condensed(verdicts, tables)
        print(
            "ERROR: Group A comparability failed; Groups B-F were not run. "
            "See check_a_comparability.csv.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    tables["B"], verdicts["B"] = run_group_b(validity, runs)
    write_csv(tables["B"], "check_b_bridge_fix.csv")
    tables["C"], verdicts["C"] = run_group_c(validity, runs, args.rebuild_graph)
    write_csv(tables["C"], "check_c_service_snapping.csv")
    tables["D"], verdicts["D"] = run_group_d(diag02b, runs)
    write_csv(tables["D"], "check_d_block_universe.csv")

    pre_fix = load_optional_pre_fix(
        diag02b,
        validity.base,
        Path(args.pre_fix_run_dir) if args.pre_fix_run_dir else None,
    )
    tables["E"], headline, verdicts["E"] = run_group_e(diag02b, runs, pre_fix)
    write_csv(tables["E"], "check_e_headline_numbers.csv")
    write_csv(headline, "headline_numbers_by_rule.csv")
    tables["F"], verdicts["F"] = run_group_f(diag02b, runs)
    write_csv(tables["F"], "check_f_ordering.csv")

    moved = what_moved(diag02b.pd, headline)
    summary = build_summary(diag02b, runs, verdicts, tables, moved=moved)
    summary_path = OUTPUT_DIR / "run_comparison_summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    log(f"saved={summary_path}")
    print_condensed(verdicts, tables)
    return 1 if any(value == "FAIL" for value in verdicts.values()) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ImportError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc
