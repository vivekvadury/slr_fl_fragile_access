# Analysis manifest — final draft freeze (1 October 2026)

Branch `attachment-sensitivity` (7c77caa) plus the uncommitted final-draft scripts listed below.
Arm: **approach** (bridge rule). Eligible universe: 68,521 blocks, 6,135,688 residents, 3,942 block groups.

## How to regenerate

```bash
# From the repository root. About 9 minutes; no bootstrap is run.
"C:/Program Files/R/R-4.6.1/bin/Rscript.exe" scripts/08_final_draft_inference.R
# Bridge-arm population comparison. About 2 minutes.
MPLBACKEND=Agg C:/Users/Vivek/miniforge3/envs/research-geo/python.exe scripts/08b_final_draft_bridge_arms.py
```

`08_final_draft_inference.R` does not re-implement anything. It evaluates
`scripts/04_transition_models.R` verbatim up to its bootstrap call, so data
preparation, QA assertions, risk sets and model fits are the production code.
It stops with an error if that script changes shape.

## Canonical pipeline

| Step | Script | Status |
|---|---|---|
| Access states | `02_access_flags.py` | production, unchanged |
| Block → block-group dataset | `03_build_extension_dataset_and_memo.ipynb` | production, unchanged |
| Physical covariates | `03b_join_elevation_drainage.py` | production, unchanged |
| Model specification | `04_shared_model_spec.R` | production, unchanged |
| Transition models + bootstrap | `04_transition_models.R` | production; only change is the `TRANSITION_TABLE_DIR` override (default unchanged) |
| Moran's I diagnostics | `04b_spatial_residual_diagnostics.R` | production, unchanged |
| Population tables | `05_population_figures.py` | production, unchanged |
| **Final-draft inference** | `08_final_draft_inference.R` | **new** |
| **Bridge-arm population comparison** | `08b_final_draft_bridge_arms.py` | **new** |

The handoff named `04_regressions.R`, `05_regressions_4_poster.R` and
`05_export_model_tables.R`. Those were deleted on 10 August 2026 (commit
01c7d04) and replaced by `04_transition_models.R`. The deleted
`04_regressions.R` used **median age**. The poster script and the current
pipeline use **age 65+ share and no-vehicle share**. The current pipeline has
the six intended covariates: Black share, Hispanic share, renter share, log
median income, age 65+ share and no-vehicle share.

## QA (fails loudly; all passed)

These checks are in `04_transition_models.R`:
- one row per (block group, SLR);
- the five state counts sum to `total_blocks` in every row;
- the eligible universe is constant across SLR;
- each block group has a unique 0 ft baseline;
- transition counts are nonnegative integers;
- redundant-origin events sum to `any_loss_of_redundancy` and do not exceed the redundant baseline;
- fragile-origin events do not exceed the fragile baseline;
- the physical-covariate join is one-to-one;
- every model converges and keeps every covariate.

These checks are in `08_final_draft_inference.R`:
- the 5 km Conley matrix equals the canonical script's;
- point estimates are identical across all covariance types;
- point estimates equal the 199-replication bootstrap tables to ≤1e-10;
- every bootstrap table has n_boot = 199;
- state totals are constant at 68,521 blocks and 6,135,688 residents at every SLR level, with no unclassified blocks;
- population thresholds are monotone and nested.

`08b` checks that the approach arm reproduces the production fig4 tables exactly.

## Conley implementation (verified, see `conley_verification.csv`)

- **Function and arguments.** `fixest::vcov_conley`. The cutoff is in km, the kernel is uniform, distance is fixest's default "triangular" approximation, the small-sample factor is (N−1)/(N−K), and `vcov_fix = TRUE`. Coordinates are block-group centroids, computed in EPSG:5070 and returned in WGS84.
- **Same-location rows are inside the kernel.** With a cutoff below every between-block-group distance, the Conley matrix equals the unadjusted block-group-clustered matrix (relative difference about 6e-15). So within-block-group, cross-scenario dependence is covered.
- **The meat is reproduced by hand.** A hand-built version, the sum of S_g S_h′ over block-group pairs within the cutoff, matches fixest exactly once fixest's Earth radius is used.
- **fixest's spherical distance uses R = 6,376 km.** This was found by probing the compiled routine with synthetic pairs. The default triangular distance places the "5 km" boundary at about 5.009 km on the 6,371 km sphere, a difference of under 10 m.
- **AMEs.** `marginaleffects::avg_slopes(model, vcov = V)`, with delta-method SEs and normal-theory 95% CIs and p-values.
- **Positive-definiteness** (raw matrices; see `conley_positive_definiteness.csv`):
  - 5 km: 7/7 in both specifications.
  - 10 km: 7/7 demographic-only, 5/7 physical.
  - 15 km: 2/7 demographic-only, 1/7 physical.
  - Non-PD matrices are eigenvalue-repaired by fixest. **This should be disclosed** wherever 10/15 km results are reported.

## Output files

| File | Contents |
|---|---|
| `main_transition_ames_conley5.{csv,xlsx,tex}` | **Primary table.** Demographic-only, 7 transitions × 6 covariates, Conley 5 km. The xlsx also has a with_physical sheet. |
| `transition_ames_conley5_with_physical.csv` | Physical-adjusted specification, Conley 5 km |
| `conley_cutoff_sensitivity.csv` | Both specifications × 5/10/15 km, with PD flags |
| `cluster_vs_conley.csv` | Clustered (delta), Conley 5 km and bootstrap SE/p side by side, with SE ratios and agreement |
| `physical_controls_comparison.csv` | Demographic-only vs physical-adjusted AMEs (Conley 5 km) |
| `model_sample_sizes.csv`, `model_covariate_filter_diagnostics.csv` | Observations, block groups, risk sets, events |
| `diagnostics_summary.csv` | Convergence, Pearson dispersion (with top-row concentration), Moran summary, Conley PD |
| `conley_verification.csv`, `conley_positive_definiteness.csv` | Implementation checks |
| `descriptive_access_by_slr.csv` | Blocks and population by access state and SLR |
| `population_thresholds_by_slr.csv` | Nested population thresholds; non-inundation share (manuscript §4.5 definition) |
| `bridge_rule_comparison.csv`, `bridge_arms/` | Bridge rules on the manuscript's population definitions |
| `all_ames_all_vcovs.csv` | Every AME under every covariance type |
| `model_run/` | Coefficient and sample diagnostics written by the canonical prefix |
| `r_session_info.txt`, `run_log.txt` | Provenance |

## Model classification

| Class | Models / outputs |
|---|---|
| **Primary** | Seven grouped-binomial transition models, demographic-only, approach arm; AMEs with Conley 5 km SEs |
| **Robustness (report in §4.9 / §5.6)** | Conley 10/15 km; block-group-clustered and 199-rep cluster-bootstrap SEs; physical-adjusted specification; bridge rules (intersect/retain); attachment arms (`outputs/attachment_sensitivity/attach_20260925/`); residual Moran's I |
| **Supplementary only** | Population-weighted AMEs (see the caution in the results memo); all-blocks vs eligible-universe diagnostic (`*_all_blocks_diagnostic`, `*_eligible_diagnostic`); Pearson dispersion |
| **Drop (superseded; do not cite)** | `all_regression_models_prelim.xlsx`, `regression_summary.csv` (April, median-age era); `ame_bootstrap_results.xlsx` and `ame_bootstrap_transition_table.tex` (untagged, July); `ame_bootstrap_transition_table_beamerposter.tex` (May); `fig4_cumulative_population_by_slr.csv` (untagged, April); `conley_sweep/` and `conley_cutoff_comparison_*.csv` (pre-freeze; superseded by this folder); `bridge_rule_sensitivity.tex` (definitions differ from the manuscript; superseded by `bridge_rule_comparison.csv`) |

## Manuscript (v42) claims vs implementation

| Manuscript claim | Status |
|---|---|
| §4.5: affected populations combine transitions from baseline-redundant and baseline-fragile blocks | **Implemented** (fig4 tables; `population_thresholds_by_slr.csv`). INCONSISTENCY: `bridge_rule_sensitivity.tex` also counts baseline-isolated→inundated blocks (2,058,681 vs 2,048,545 at 6 ft). Superseded. |
| §4.5: non-inundation share over population, five transitions | **Implemented** here. INCONSISTENCY: `bridge_rule_sensitivity.tex` and `transition_summary_by_slr_*.csv` use a block-based share (69.78% vs 75.00% at 2 ft). |
| §4.5: "sensitivity to population weighting … when more populous blocks receive greater weight" | **Implemented differently.** Code weights *block groups* by eligible population in the AME average only; the fit is unchanged. In a logit without interactions this rescales every AME within a model by the same factor, so it cannot change sign, relative size or the pattern. **Decision needed** (see memo). |
| §4.6: standardized indicators | Implemented. Detail for Methods: means/SDs come from all 3,942 block groups with a nonmissing value (the stacked panel), *before* the complete-case and risk-set filters, not from each model's estimation sample. |
| §4.6: complete cases | Implemented: 338 block groups dropped, 3,604 retained. |
| §4.7: AMEs averaged equally across block-group-by-scenario rows | Implemented (`avg_slopes`, no weights). |
| §4.7: bootstrap with 199 replications, percentile CI, normal-approximation p | Implemented. Seed is 20260411 + transition index. |
| §4.7: Conley 5 km reference, 10/15 km sensitivity | Implemented. Not disclosed: non-PD repair at 10/15 km. |
| §4.7: "robust only where also significant under 5 km Conley" | Implemented as a reporting rule. Moot if Conley 5 km becomes primary. |
| §3.2 / §4.8: elevation and drainage sensitivity specification | Implemented. Note the extrapolation outlier in the memo. |
| §4.9: residual spatial autocorrelation tests | Implemented in `04b` for all 7 models × 6 scenarios (42 tests per specification). |
| §4.2: attachment sensitivity | Completed 29 September (`outputs/attachment_sensitivity/attach_20260925/REPORT.md`). |
| §4.3.1: bridge-rule sensitivity (intersect/retain) | Implemented (models and population). |
