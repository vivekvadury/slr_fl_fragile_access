# Phase 2 - Downstream propagation

## Outcome

**PASS.** The complete-ACS approach run finished through
`03_build_extension_dataset_and_memo.ipynb` -> `04_transition_models.R` ->
`05_population_figures.py`. All notebook universe/status assertions, R risk-set
assertions, the five-rep AME smoke, and the population audits passed. The run produced
42 AME rows, 42 clustered coefficient rows, and arm-tagged analysis, table, and figure
outputs. Sources: `reports/02_runtime_and_validation.csv` and
`reports/02_smoke_completion.csv`.

## ACS retry and completeness repair

The first full attempt encountered a Broward read timeout. The ACS fetch now allows three
120-second attempts per variable chunk, retries transient failures with backoff, and raises
instead of returning a partial county after the final attempt. It also asserts the expected
county counts, unique block-group GEOIDs, and the tri-county total before demographic merge
or output. Sources: `scripts/03_build_extension_dataset_and_memo.ipynb:2748-2829` and
`reports/02_runtime_and_validation.csv`.

The completed pull contains exactly **3,946 source ACS block groups**: 1,121 in Broward,
1,843 in Miami-Dade, and 982 in Palm Beach. The analysis has **3,942 eligible block
groups** because four block groups contain no eligible block after the decided universe
filter. Sources: `reports/02_acs_counts.csv`, `reports/02_smoke_completion.csv`, and
`reports/02_arm_eligible_universe.csv`.

The Census credential is read only from `os.environ["CENSUS_API_KEY"]`; the execution used
a process-local credential and the serialized notebook contains no literal key. The key
previously exposed in repository history still must be rotated. Sources:
`scripts/03_build_extension_dataset_and_memo.ipynb:198-205` and
`reports/02_runtime_and_validation.csv`.

## Changes made

### Notebook 03

- `ARM` replaces the date/run-directory switch, defaults to `approach`, and resolves to
  `positive_layer_20260814_{ARM}`. Every generated CSV, PNG, PDF, and GeoPackage is tagged
  by arm. Sources: `scripts/03_build_extension_dataset_and_memo.ipynb:179-182` and
  `reports/02_change_log.csv`.
- The notebook reads exactly one `block_access_flags_long.parquet`. Six-file discovery,
  cross-file baseline comparison, stacking, and duplicate dropping are gone. Assertions
  require one logical Parquet, unique `(block_geoid, slr_ft)` keys, levels 0-6, 70,695
  source blocks, and a complete seven-level panel. Sources:
  `scripts/03_build_extension_dataset_and_memo.ipynb:419-453` and
  `reports/02_change_log.csv`.
- `analysis_eligible == True` is applied once near the top; there is no `pop20 > 0`
  filter. Producer-supplied `pop20` supplies population weights. The existing
  `baseline_baseline_shortest_path_distance_m` is read as-is, with no second variant.
  Sources: `scripts/03_build_extension_dataset_and_memo.ipynb:561-619` and
  `reports/02_change_log.csv`.
- Status is an ordered five-level categorical in the order `unclassified`, `inundated`,
  `isolated`, `fragile`, `redundant`. Unknown or missing eligible statuses fail an
  assertion; `unclassified` is included in count/share aggregation and the Figure 1
  palette; five status counts must sum to every block-group/SLR eligible denominator.
  Sources: `scripts/03_build_extension_dataset_and_memo.ipynb:220`,
  `scripts/03_build_extension_dataset_and_memo.ipynb:994-997`, and
  `reports/02_change_log.csv`.
- Block-group output carries `bridge_rule_applied`, the retained-nearby-structure share,
  the share with `origin_in_lcc == False`, and shares by observed
  `origin_geometry_method`. Both scenario and baseline five-state counts are retained for
  model risk-set checks. Source: `reports/02_change_log.csv`.

The notebook was written only through `nbformat` 5.10.4. All 49 cells and all embedded
figure outputs were preserved. Twenty-two stale text/table outputs attached to rewritten
setup/data cells were cleared only after confirming that none contained image MIME data.
The final complete-ACS notebook reparsed, passed `nbformat.validate()`, and every code cell
compiled; its SHA-256 is
`83aaff011bef56899e16e0555f1cbc5bde3d5accbb0598b5dcc92ad8b249cb2f`.
Source: `reports/02_runtime_and_validation.csv`.

The final spatial export contains exactly `slr_0ft` through `slr_6ft` plus
`all_scenarios`; there is no missing-SLR `slr_nanft` layer. Each scenario layer contains
the same 3,942 eligible block groups. The seven per-SLR layers retain spatial indexes;
the duplicated `all_scenarios` layer omits only its optional index so the GeoPackage
remains below the Git hosting limit without changing any feature or value. Sources:
`outputs/spatial/slr_block_group_analysis_approach.gpkg` and
`reports/02_runtime_and_validation.csv`.

### Transition models 04

The script accepts `--arm`/`BRIDGE_ARM` and a positional, `--data`, or
`TRANSITION_DATA_PATH` input override; approach is the default. AME workbooks, LaTeX,
sample diagnostics, and coefficient diagnostics are arm-tagged. Sources:
`scripts/04_transition_models.R:24-133`, `scripts/04_transition_models.R:645-661`, and
`scripts/04_transition_models.R:943-947`.

`prepare_transition_data` now:

1. asserts the five eligible states sum to `total_blocks` for every block-group/SLR row;
2. uses `baseline_total_blocks` to verify the baseline partition and stable universe;
3. defines redundant and fragile grouped-binomial weights from their corresponding
   baseline state counts inside the verified eligible universe; and
4. asserts all transition numerators are nonnegative integers within those state-specific
   risk sets.

Sources: `scripts/04_transition_models.R:193-266` and
`scripts/04_transition_models.R:279-398`.

The six covariates, binomial family, county fixed effect, SLR-scenario fixed effect,
transition outcomes, and block-group clustered coefficient standard errors are unchanged.
Sources: `scripts/04_transition_models.R:135-153` and
`scripts/04_transition_models.R:433-444`.

### Population figures 05

The script defaults to approach and reads only
`block_level_long_dataset_{arm}.csv`. It requires every row to be eligible, validates
unique keys and all seven levels, and represents all five statuses explicitly. The input
`pop20` is authoritative. The separate raw Census block join remains in place but is
marked redundant and acts only as an audit; row-level values and totals at every SLR level
must agree before outputs are written. Sources: `scripts/05_population_figures.py:35-194`
and `reports/02_smoke_completion.csv`.

All three population tables and both PNG/PDF figure pairs are arm-tagged. Sources:
`scripts/05_population_figures.py:218-355` and generated approach files under
`outputs/tables/fig4_*_approach.csv` and `outputs/figures/fig4*_approach.*`.

## Universe and smoke results

All three arms have the same eligible universe at every SLR level: **68,521 blocks and
6,135,688 people**, retaining eligible zero-population blocks. The approach output contains
479,647 block/SLR rows and 27,594 block-group/SLR rows. Sources:
`reports/02_arm_eligible_universe.csv` and `reports/02_smoke_completion.csv`.

At baseline, the one-time filter removes:

| Exclusion reason | Blocks | Population |
|---|---:|---:|
| `origin_snap_failed` | 262 | 1,848 |
| `zero_land_area` | 1,912 | 797 |
| **Total** | **2,174** | **2,645** |

Source: `outputs/tables/eligibility_exclusions_approach.csv`.

The approach eligible-universe baseline fragile share remains
**17,335 / 68,521 = 0.252988 (25.2988%)**. This is its own estimate, between but not
reproducing the legacy all-block value 0.2469 (70,695 blocks) and legacy `pop20 > 0`
value 0.2585 (55,411 blocks). Sources: `reports/02_universe_reconciliation.csv` and
`outputs/tables/fig4_status_population_by_slr_approach.csv`.

Five-state eligible-block distribution:

| Status | 0 ft count | 0 ft share | 6 ft count | 6 ft share |
|---|---:|---:|---:|---:|
| unclassified | 0 | 0.0000% | 0 | 0.0000% |
| inundated | 188 | 0.2744% | 13,202 | 19.2671% |
| isolated | 163 | 0.2379% | 6,489 | 9.4701% |
| fragile | 17,335 | 25.2988% | 13,742 | 20.0552% |
| redundant | 50,835 | 74.1889% | 35,088 | 51.2077% |

Source: `outputs/tables/fig4_status_population_by_slr_approach.csv`.

The approach non-inundation pathway share is **69.7816% at 2 ft** and **37.5678% at
6 ft**. Source: `outputs/tables/transition_summary_by_slr_approach.csv`.

## Risk sets and estimation samples

Before demographic completeness filtering, the raw canonical approach baseline weights
change as follows when the decided universe is applied:

| Risk set | All blocks | Eligible | Eligible - all | Reduction | Block groups changed |
|---|---:|---:|---:|---:|---:|
| baseline redundant | 51,452 | 50,835 | -617 | 1.1992% | 435 |
| baseline fragile | 17,729 | 17,335 | -394 | 2.2223% | 269 |

Source: `reports/02_risk_set_weight_changes.csv`.

Actual approach model attrition is:

| Filter | Input rows / groups | Dropped rows / groups | Retained rows / groups |
|---|---:|---:|---:|
| Complete six-covariate cases | 27,594 / 3,942 | 2,366 / 338 | 25,228 / 3,604 |
| `baseline_redundant_n > 0` after removing 0 ft | 21,624 / 3,604 | 972 / 162 | 20,652 / 3,442 |
| `baseline_fragile_n > 0` after removing 0 ft | 21,624 / 3,604 | 2,556 / 426 | 19,068 / 3,178 |

Source: `outputs/tables/transition_sample_diagnostics_approach.csv`.

## Controlled all-block versus eligible coefficient diagnostic

The controlled diagnostic holds the canonical approach arm, demographics, six
covariates, model formulas, fixed effects, and estimation code fixed, changing only the
all-block versus eligible aggregation/risk sets. After the common complete-covariate
filter, the redundant weight is 47,512 all-block versus 46,954 eligible (-558), and the
fragile weight is 16,467 versus 16,114 eligible (-353). Source:
`outputs/tables/transition_weight_comparison_approach_eligible_vs_all_blocks_diagnostic.csv`.

Both controlled fits used `AME_BOOT_REPS=5` for the plumbing diagnostic. The coefficient
comparison below uses the models' block-group-clustered coefficient standard errors and
p-values, not the five-draw AME bootstrap p-values. Sources:
`outputs/tables/ame_bootstrap_results_approach_all_blocks_diagnostic.xlsx`,
`outputs/tables/ame_bootstrap_results_approach_eligible_diagnostic.xlsx`, and
`outputs/tables/transition_model_coefficient_comparison_approach_eligible_vs_all_blocks_diagnostic.csv`.

Across 42 coefficients, there is one sign flip: the `no_vehicle_share` coefficient for
Redundant -> Worse moves from -0.002376 to +0.001813, but is null in both fits
(p=0.9652 and p=0.9735). There is one p=0.05 threshold loss: log median income for
Redundant -> Isolated moves from +0.132922 (p=0.04961) to +0.132224 (p=0.05056).
No other sign or p=0.05 classification changes. Source:
`outputs/tables/transition_model_coefficient_comparison_approach_eligible_vs_all_blocks_diagnostic.csv`.

The largest Black-coefficient movement is for Fragile -> Inundated: -0.656766 all-block
to -0.698637 eligible, a -0.041872 change (6.3754% in magnitude relative to the all-block
coefficient). All 14 Black/Hispanic coefficients remain negative and significant at
p<0.05 in both controlled fits. Thus the universe/risk-weight change does not alter the
racial-composition sign or significance pattern in this diagnostic. Source:
`outputs/tables/transition_model_coefficient_comparison_approach_eligible_vs_all_blocks_diagnostic.csv`.

## Combined legacy-to-current AME smoke comparison

This comparison combines multiple changes: the untagged legacy all-block analysis versus
the canonical approach arm on the eligible universe. It does **not** isolate the universe
or bridge rule. The source workbooks have the same 42 unique `(transition, term)` keys.
The old workbook uses 199 successful bootstrap replications; the current smoke workbook
uses five. AME point-estimate differences are exact for these fits, but p-value and
significance comparisons are **non-inferential smoke diagnostics** because the bootstrap
replication counts differ and five replications are inadequate for inference. Source:
`reports/02_ame_combined_legacy_to_approach_smoke_comparison.csv`.

The largest absolute AME movements are:

| Transition / term | Legacy AME | Current smoke AME | Change | Relative magnitude |
|---|---:|---:|---:|---:|
| Fragile -> Worse / age 65+ | 0.035274 | 0.022183 | -0.013091 | 37.11% |
| Fragile -> Worse / renter share | 0.061287 | 0.050975 | -0.010312 | 16.83% |
| Fragile -> Isolated / age 65+ | 0.017634 | 0.010926 | -0.006708 | 38.04% |
| Fragile -> Inundated / age 65+ | 0.018513 | 0.011925 | -0.006588 | 35.58% |
| Fragile -> Isolated / renter share | 0.023549 | 0.017370 | -0.006179 | 26.24% |

Source: `reports/02_ame_combined_legacy_to_approach_smoke_comparison.csv`.

Two combined-comparison AMEs flip sign, both while nonsignificant in both artifacts:
Redundant -> Inundated / log income (-0.000915 to +0.002317) and Redundant -> Worse /
no-vehicle share (-0.002137 to +0.000126). The smoke-only p<0.05 classification gains are
Redundant -> Isolated / log income and Redundant -> Worse / log income; Fragile ->
Inundated / log income has a smoke-only loss. All 14 Black/Hispanic AMEs remain negative
and smoke-significant, with no racial sign flip. These p-value statements are descriptive
of the five-rep smoke only. Source:
`reports/02_ame_combined_legacy_to_approach_smoke_comparison.csv`.

## Runtime and final validation

The model run used R 4.6.1 with tidyverse 2.0.0, fixest 0.14.2,
marginaleffects 0.32.0, and openxlsx 4.2.8.1. The R script parsed, the model specification
remained unchanged, all 27,594 eligible block-group/SLR risk-set rows passed, all 42 AME
rows obtained five successful and zero failed replications, and all 42 clustered
coefficient rows were written. Sources: `reports/02_runtime_and_validation.csv` and
`reports/02_smoke_completion.csv`.

The complete approach pipeline is clean: every notebook code cell completed, no status,
universe, transition, or risk-set assertion fired, and the population join agreed both
row-by-row and by SLR level. Source: `reports/02_runtime_and_validation.csv`.
