# Attachment sensitivity experiment `attach_20260925`

Specification: `docs/attachment_sensitivity/agent_prompt.md`. Prespecified assessment
(written before any arm ran): `00_prespecified_assessment.md`. Results report:
`REPORT.md`. Workbook: `scripts/07_attachment_sensitivity_workbook.ipynb`.

## Status

| Stage | Status | Evidence |
|---|---|---|
| Implementation | done | `scripts/attachment_sensitivity/` |
| Synthetic tests | 11/11 pass | `scripts/attachment_sensitivity/test_attachment_synthetic.py` |
| Access runs, 6 arms x 0-6 ft | completed (local) | `data/processed/access/attachment_sensitivity/attach_20260925/<arm>/_COMPLETE.json` |
| Reference reproduction + invariants | 30/30 checks pass | `validation/validation_summary.csv`, `validation_log.txt` |
| Analysis datasets | rebuilt for all arms | `.../<arm>/_DATASETS_COMPLETE.json` |
| Access comparisons | done | tables below |
| Models, both specs, 199-success bootstrap | see REPORT.md | `models/<arm>/` |

## Execution record

- Local machine, Windows 11, 31.5 GB RAM, research-geo env (Python 3.11). Peak RSS 7.35 GiB
  (0 ft benchmark `attach_bench_0ft`), 6.19 GiB (resume). No Della run was needed.
- 2026-09-25 19:05 - 2026-09-26 ~02:00: scenarios 0-5 ft completed for all arms
  (`runner_log_20260925_interrupted.txt`); the machine shut down during the 6 ft union.
  All part files are written atomically; no partial part existed.
- 2026-09-29 16:08-17:43: resumed (`runner_log_resume_20260929.txt`); 0-5 ft skipped from
  completed parts, 6 ft computed, all arms assembled.
- Scenario unions: 0-5 ft `GeoSeries.union_all` (0 ft seeded from the benchmark run,
  `union_cache_seed.txt`); 6 ft `shapely.coverage_union_all`, which was first checked
  equal to `union_all` at 0 and 5 ft (identical area; 0 mismatches at 70,695 origins and
  1,659,411 graph nodes; `equivalence_checks/`). The 6 ft reference inundation flags then
  reproduced production exactly.
- Inputs: SHA-256 of all seven access inputs equal the production Della manifest
  (`data/.../shared/experiment_manifest.json`). Source: commit 8c091bf plus uncommitted
  experiment files; file hashes in the manifest. The one production edit is
  `scripts/04_transition_models.R` (`TRANSITION_TABLE_DIR`, default-equivalent).

## Commands (repository root, Git Bash)

```bash
export GDAL_DATA="C:/Users/Vivek/miniforge3/envs/research-geo/Library/share/gdal"
PY=/c/Users/Vivek/miniforge3/envs/research-geo/python.exe
$PY scripts/attachment_sensitivity/test_attachment_synthetic.py
$PY -u scripts/attachment_sensitivity/run_access_arms.py --experiment-id attach_20260925 --union-method coverage_union_all   # resumable
$PY scripts/attachment_sensitivity/verify_arms.py --experiment-id attach_20260925
$PY scripts/attachment_sensitivity/build_analysis_datasets.py --experiment-id attach_20260925
$PY scripts/attachment_sensitivity/compare_arms.py --experiment-id attach_20260925
bash scripts/attachment_sensitivity/run_models.sh attach_20260925 10 reference origin_1000 origin_500 facility_500 facility_nearest_1000
$PY scripts/attachment_sensitivity/compare_models.py --experiment-id attach_20260925
```

`facility_add_uncapped` is not refit: its block-group model input is identical to the
reference in every model column (checked), so its models equal the reference models.

## Output index (this directory)

| File | Content |
|---|---|
| `attachment_audit.csv` | one row per (arm, facility candidate): rule, validity, node, distances, change type, added distance and node displacement |
| `attachment_audit_summary.csv` | counts of added/excluded/reattached/fallback and distance distributions |
| `origin_attachment_summary.csv` | eligible-origin attachment distance distributions, all and populated blocks |
| `eligibility_comparison.csv` | eligible blocks/population/block groups and exclusions by reason |
| `state_comparison.csv` | state counts, population, shares by arm x SLR, with deltas (count and pp) |
| `state_crosstab_vs_reference.csv` | reference-vs-arm state cross-tab by GEOID on common eligible origins |
| `transition_population_comparison.csv` | five transitions and 05-style cumulative population totals, non-inundation share |
| `changed_unit_composition.csv` | descriptive composition of excluded / changed origins |
| `spatial_review/attachment_review.gpkg` | changed or >250 m facility attachments; origin attachments >500 m (plausibility unverified) |
| `validation/` | reproduction and invariant checks |
| `models/<arm>/` | R outputs per arm and spec, logs, completion markers |
| `ame_comparison_*.csv`, `model_sample_comparison.csv`, `reference_vs_production_models.csv` | model comparisons |
