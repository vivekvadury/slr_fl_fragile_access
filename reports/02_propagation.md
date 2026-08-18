# Phase 2 - Downstream propagation

## Batch stop: Phase 2 is not clean

**STOP BEFORE PHASE 3.** The canonical data pass every notebook-core and equivalent
risk-set invariant, but this host has no `CENSUS_API_KEY` and no R/Rscript runtime.
Consequently, notebook 03 could not complete its ACS-dependent half and script 04 could
not fit models. The ACS-independent notebook core and script 05 completed on the approach
arm, but the Phase 3 prerequisite (all arms built through 03 and 04) is not met. No model
coefficient or AME movement is inferred or fabricated. Source:
`reports/02_runtime_and_validation.csv`.

The notebook now fails clearly when `CENSUS_API_KEY` is unset. The exposed historical key
has been removed from the notebook, but it must be rotated because exposure in repository
history is not repaired by deleting the current literal. Source:
`reports/02_runtime_and_validation.csv`; `scripts/03_build_extension_dataset_and_memo.ipynb`
cell 4.

## Changes made

### Notebook 03

- `ARM` replaces the date/run-directory switch, defaults to `approach`, and resolves to
  `positive_layer_20260814_{ARM}`. Every generated CSV, PNG, PDF, and GeoPackage is tagged
  by arm. Source: `reports/02_change_log.csv`.
- The notebook reads exactly one `block_access_flags_long.parquet`. Six-file discovery,
  cross-file baseline comparison, stacking, and duplicate dropping are gone. Assertions
  require one logical Parquet, unique `(block_geoid, slr_ft)` keys, levels 0-6, 70,695
  source blocks, and a complete seven-level panel. Source: `reports/02_change_log.csv`;
  `scripts/03_build_extension_dataset_and_memo.ipynb` cells 6-7.
- `analysis_eligible == True` is applied once near the top; there is no `pop20 > 0`
  filter. The producer-supplied `pop20` supplies population weights. Source:
  `reports/02_change_log.csv`; `scripts/03_build_extension_dataset_and_memo.ipynb`
  cells 9-10.
- Status is now an ordered five-level categorical in the order `unclassified`,
  `inundated`, `isolated`, `fragile`, `redundant`; unknown or missing eligible statuses
  fail an assertion. `unclassified` is included in the count/share aggregation and Figure
  1 palette, and five status counts must sum to each block-group/SLR eligible denominator.
  Source: `reports/02_change_log.csv`; `scripts/03_build_extension_dataset_and_memo.ipynb`
  cells 4, 12, 18, and 25.
- Block-group output now carries `bridge_rule_applied`, the retained-nearby-structure
  share, the share with `origin_in_lcc == False`, and shares by observed
  `origin_geometry_method`. Both scenario and baseline five-state counts are retained for
  the model risk-set checks. Source: `reports/02_change_log.csv`.
- The existing `baseline_baseline_shortest_path_distance_m` is read as-is and no second
  variant is created. Source: `reports/02_change_log.csv`;
  `scripts/03_build_extension_dataset_and_memo.ipynb` cells 7 and 12.

The notebook was written only through `nbformat` 5.10.4. All 49 cells and all embedded
figure outputs were preserved. Twenty-two stale text/table outputs attached to the
rewritten setup/data cells were cleared after confirming that none contained image MIME
data; this removes the retired filenames from the rendered notebook without deleting a
cell or figure. An immediate reparse, `nbformat.validate()`, and compilation of every code
cell passed. Its final SHA-256 is
`ac988996a26f12fff8b01e4ab1fe693a83f4c5185ba7ea170e64064fa58f5e61`.
Source: `reports/02_runtime_and_validation.csv`.

### Transition models 04

The script accepts `--arm`/`BRIDGE_ARM` and a positional, `--data`, or
`TRANSITION_DATA_PATH` input override; approach is the default. Model workbooks, LaTeX,
and sample diagnostics are arm-tagged. Source: `scripts/04_transition_models.R:24-130`.

`prepare_transition_data` now:

1. asserts the five eligible states sum to `total_blocks` for every block-group/SLR row;
2. uses `baseline_total_blocks` to verify the baseline partition and stable universe;
3. defines redundant and fragile grouped-binomial weights from the corresponding baseline
   state counts inside that verified eligible universe; and
4. asserts all transition numerators are nonnegative integers within their state-specific
   risk sets.

Sources: `scripts/04_transition_models.R:151-254` and
`scripts/04_transition_models.R:275-397`.

The six covariates, county fixed effect, SLR fixed effect, binomial family, clustered
standard errors, and transition specifications are unchanged. Source:
`scripts/04_transition_models.R:431`; `scripts/04_transition_models.R:501`;
`reports/02_runtime_and_validation.csv`.

The script now writes counts of block groups and rows removed by covariate completeness and
the two positive-risk filters. Because R could not run, the report-ready preflight using
the canonical eligible approach groups and the repository's existing ACS demographic
columns finds: 338 block groups removed by covariate completeness, 162 then removed from
the redundant risk set, and 426 then removed from the fragile risk set. These are a
preflight audit, not R output. Source: `reports/02_sample_diagnostics_preflight.csv`.

### Population figures 05

The script defaults to approach, reads only
`block_level_long_dataset_{arm}.csv`, requires every row to be eligible, validates unique
keys and all seven levels, and represents all five statuses explicitly. The input `pop20`
is authoritative. The separate raw Census block join remains in place but is marked
redundant and now acts only as an audit; row-level values and totals at every SLR level
must agree before outputs are written. Sources: `scripts/05_population_figures.py:35-194`.

All three population tables and both PNG/PDF figure pairs are arm-tagged. Source:
`scripts/05_population_figures.py:218-355`; the generated approach files under
`outputs/tables/fig4_*_approach.csv` and `outputs/figures/fig4*_approach.*`.

## Universe and reconciliation

All three arms have the same eligible universe at all seven levels: **68,521 blocks and
6,135,688 people**, retaining zero-population eligible blocks. Source:
`reports/02_arm_eligible_universe.csv`.

At baseline, the one-time filter removes:

| Exclusion reason | Blocks | Population |
|---|---:|---:|
| `origin_snap_failed` | 262 | 1,848 |
| `zero_land_area` | 1,912 | 797 |
| **Total** | **2,174** | **2,645** |

Source: `outputs/tables/eligibility_exclusions_approach.csv`.

The approach eligible-universe baseline fragile share is **17,335 / 68,521 = 0.252988**.
It is its own estimate, between but not reproducing the legacy all-block value 0.2469
(70,695 blocks) and legacy `pop20 > 0` value 0.2585 (55,411 blocks). The difference is the
2,174 exclusions above, not a populated-block restriction. Source:
`reports/02_universe_reconciliation.csv`;
`outputs/tables/eligibility_exclusions_approach.csv`.

## Risk-set weight change

On approach, filtering to the decided universe changes the aggregate baseline redundant
weight from 51,452 to 50,835, a reduction of 617 (1.20%) across 435 block groups. The
fragile weight changes from 17,729 to 17,335, a reduction of 394 (2.22%) across 269 block
groups. The corresponding reductions are 1.21%/2.20% under intersect and 1.20%/2.23%
under retain. Source: `reports/02_risk_set_weight_changes.csv`.

Whether any coefficient or AME moves cannot be measured on this host because R is absent.
No substitute-language fit was attempted. Source: `reports/02_runtime_and_validation.csv`.

## Approach smoke results

The ACS-independent notebook core ran through block-group and tract aggregation and wrote
the arm-tagged block-level input. Script 05 then completed in the exact pinned
`research-geo` conda environment; the redundant POP20 audit agreed block-by-block and by
SLR level. The model step did not run. Source: `reports/02_runtime_and_validation.csv`.

Five-state eligible-block distribution:

| Status | 0 ft count | 0 ft share | 6 ft count | 6 ft share |
|---|---:|---:|---:|---:|
| unclassified | 0 | 0.0000% | 0 | 0.0000% |
| inundated | 188 | 0.2744% | 13,202 | 19.2671% |
| isolated | 163 | 0.2379% | 6,489 | 9.4701% |
| fragile | 17,335 | 25.2988% | 13,742 | 20.0552% |
| redundant | 50,835 | 74.1889% | 35,088 | 51.2077% |

Source: `outputs/tables/fig4_status_population_by_slr_approach.csv`.

The non-inundation pathway share is **69.7816% at 2 ft** and **37.5678% at 6 ft** on
approach. Source: `outputs/tables/transition_summary_by_slr_approach.csv`.

No notebook-core assertion fired for status vocabulary, the eligible universe, five-state
closure, or transition denominators in any arm. The equivalent canonical audit also found
zero failures for the new model risk-set invariants. The R assertions themselves did not
execute. Source: `reports/02_runtime_and_validation.csv`.

## Required resume conditions

To complete Phase 2 and unlock Phases 3-5, rerun from Phase 2 smoke after both:

1. `CENSUS_API_KEY` is supplied in the execution environment; and
2. an existing compatible R runtime with the script's declared packages is made available
   without changing the pinned Python stack.

Until then, Phases 3-5 are intentionally not started. Source:
`reports/02_runtime_and_validation.csv`.
