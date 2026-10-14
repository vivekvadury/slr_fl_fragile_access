# Elevation and drainage adjustment: Section 5.5 audit

Audit date: 6 October 2026. Primary arm: `approach`. Manuscript reviewed:
`SLR_Access_Degredation_IDJRR_Manuscript (32).pdf`, Sections 4.7-4.8 (PDF pages 18-20),
Table 5 (page 26), and empty Section 5.5 (page 29). The manuscript was not modified.

## Verdict

**Current and directly comparable: yes. Both final-draft specifications use 5-km
Conley spatial HAC covariance and delta-method AME uncertainty. No estimation
rerun is required to update Section 5.5.** This audit reproduced all 14 production
fits from the current local inputs, without a bootstrap, in an isolated folder.
All 98 AMEs (42 social-only and 56 including physical terms), standard errors,
p-values, z statistics and confidence limits matched the final-draft export
within the recorded numerical tolerance. See `saved_vs_current_checks.csv`.
The 42-row comparison includes only the six requested social indicators.

No model, input, manuscript, or production output was changed. Existing unrelated
social-vulnerability audit work was left alone. Results here apply to the current
local finalized analysis inputs, not a newly rebuilt road network.

## Exact pipeline and files

All paths below are relative to the repository root.

1. `scripts/02_access_flags.py` produces the designated approach access run:
   `data/processed/access/edited/della_runs/positive_layer_20260814_approach/block_access_flags_long.parquet`
   (CSV sibling and `run_manifest.json` also present). Manifest: `approach`, 0-6 ft,
   all legacy flags false. This audit did not rebuild the graph.
2. `scripts/03_build_extension_dataset_and_memo.ipynb` retains eligible blocks,
   constructs transition indicators, aggregates them, joins ACS covariates and writes
   `data/processed/analysis/block_level_long_dataset_approach.csv` and
   **`data/processed/analysis/block_group_analysis_dataset_approach.csv`**.
   The latter is the direct regression input for both specifications.
3. `scripts/03b_join_elevation_drainage.py` writes the second adjusted-model input:
   **`data/processed/analysis/block_group_physical_covariates.csv`**. The fitted
   additions are `elevation_m_mean` and `drainage_distance_km`; `elevation_m_median`
   is validated/read but is not a regression covariate.
   - Geometry: `outputs/spatial/slr_block_group_analysis_approach.gpkg`, layer `slr_0ft`.
   - Elevation: valid-cell-center zonal mean from pinned USGS 3DEP 1/3-arc-second
     tiles `USGS_13_n26w081_20260225.tif` and `USGS_13_n27w081_20251216.tif`, cached
     under `data/raw/physical_covariates/usgs_3dep_13_arc_second/`.
   - Drainage: geometric-centroid distance to the nearest SFWMD AHED feature with
     `HYDRO_ORDER = 'PRIMARY'`, measured in EPSG:32617 and converted to km.
     Cache: `data/raw/physical_covariates/sfwmd/sfwmd_ahed_primary_drainage.geojson`
     and `.metadata.json` (7,803 features; retrieved 2 September 2026).
4. `scripts/04_shared_model_spec.R` defines the seven formulas, the covariates,
   grouped-binomial fits, risk-set weights and Conley covariance helper.
5. `scripts/04_transition_models.R` reads the direct inputs, standardizes variables,
   constructs estimation samples and fits the models; `MODEL_SPEC=with_physical`
   adds the two physical controls to the same primary specification.
6. **`scripts/08_final_draft_inference.R` is the final-draft AME exporter.** It fixes
   `ARM='approach'`, evaluates script 04 up to its bootstrap call for both specs,
   then computes Conley/delta-method AMEs. Its relevant outputs are:
   - `outputs/final_draft/main_transition_ames_conley5.csv` (42 primary AMEs);
   - `outputs/final_draft/transition_ames_conley5_with_physical.csv` (56 adjusted AMEs);
   - `outputs/final_draft/main_transition_ames_conley5.xlsx` (both specifications);
   - `outputs/final_draft/all_ames_all_vcovs.csv` (all covariance variants);
   - `outputs/final_draft/physical_controls_comparison.csv` (older comparison layout);
   - `outputs/final_draft/model_run/transition_model_coefficients_approach{,_with_physical}.csv`;
   - `outputs/final_draft/model_sample_sizes.csv`, `model_covariate_filter_diagnostics.csv`,
     `conley_positive_definiteness.csv`, `conley_verification.csv`, `diagnostics_summary.csv`.

Hashes of current pipeline scripts, direct inputs, access states, geometry and final
AME exports are saved in `input_provenance_sha256.csv`. The CSV comparison is generated
from the freshly reproduced AMEs, after comparison with both final-draft tables.

## Comparability checks

| Component | Primary vs adjusted |
| --- | --- |
| Seven outcomes | Identical: redundant to fragile, isolated, inundated, worse; fragile to isolated, inundated, worse. |
| Worse | Redundant: fragile + isolated + inundated; fragile: isolated + inundated. |
| Estimation rows | Identical row by row within every corresponding model, not merely equal sample sizes. |
| Baseline denominators | Identical counts of eligible blocks in the appropriate 0-ft baseline state; also binomial trial weights. |
| Social indicators | Identical standardized Black share, Hispanic share, renter share, log median household income, age-65+ share and no-vehicle household share. |
| Standardization | `scale()` before complete-case and risk-set filters on the stacked 0-6-ft panel; shared social z-scores are exactly equal. |
| Fixed effects | Identical county and SLR-scenario factors; estimation uses the six positive 1-6-ft scenarios. |
| Bridge treatment | Identical `approach`. |
| Access/transition data | Identical current finalized block-group input; independently traced to the designated source run. |
| AME averaging | Equal weight for each estimation-sample block-group/scenario row; model trial weights remain baseline block counts. |
| Only added regressors | Standardized mean elevation and standardized nearest-primary-drainage distance. |

The physical join is complete for all 3,942 block groups. Both models exclude the
same 338 groups for missing social covariates, leaving 3,604 complete-case groups
before baseline-state risk-set filtering. Final samples:

| Baseline family | Models | Block groups | BG-scenario observations | Baseline blocks | Block-scenario trials |
| --- | ---: | ---: | ---: | ---: | ---: |
| Redundant | 4 | 3,442 | 20,652 | 46,954 | 281,724 |
| Fragile | 3 | 3,178 | 19,068 | 16,114 | 96,684 |

`sample_identity_checks.csv` verifies exact row keys, outcomes, denominators, social
z-scores, county/scenario values and coordinates. `data_lineage_checks.csv` verifies
479,647 eligible block-scenario rows (68,521 blocks x 7 scenarios), plus exact
reaggregation of all relevant state/transition counts in the 27,594 block-group rows.

## Variance-estimator audit

The final-draft exports use `fixest::vcov_conley()` with a **5-km cutoff, uniform
kernel**, block-group centroid coordinates, the default triangular distance
approximation (this is a distance algorithm, not a triangular kernel), and fixest's
small-sample adjustment. Centroids come from the same approach GeoPackage.
Repeated scenario observations at the same centroid are included in the spatial
covariance calculation. The final-draft implementation verification documents a
small-cutoff equivalence to unadjusted block-group clustering and a hand-built
Conley reproduction; that independent kernel reconstruction was not rerun here.

AMEs use `marginaleffects::avg_slopes(model, vcov=V)` with the supplied Conley matrix,
delta-method SEs, normal-theory 95% intervals and two-sided normal p-values. All 14
raw 5-km covariance matrices are positive definite in the fresh audit; no eigenvalue
repair is needed at 5 km. The audit also checked p-values and confidence bounds
algebraically. Significance below means nominal p<0.05, without multiplicity adjustment.

**Do not confuse other files with these outputs.** The `std.error`, p-value and CI
columns in `outputs/tables/ame_bootstrap_results_approach_with_physical.xlsx` are from
the 199-successful-replication block-group cluster bootstrap: SE is the SD of bootstrap
AME draws, CI is percentile-based, and p is a normal approximation using the bootstrap
SE. When Conley is requested, script 04 adds separate `conley_se`/`conley_cutoff_km`
columns; it does not turn those bootstrap CI/p columns into Conley inference.
Coefficient diagnostic CSVs likewise have clustered `std.error` and separate
coefficient-level `conley_se`; those are not AME SEs. Earlier `conley_sweep/` files
are superseded for the current manuscript. No such older uncertainty columns were
used in this comparison.

The current final-draft files agree with manuscript Sections 4.7-4.8, which explicitly
specify 5-km Conley inference and delta-method AME SEs for these adjusted models.
The old workflow README and script-04 log wording still emphasize bootstrap or
clustered inference; this documentation is not a reliable guide to the final-draft table.

## Results to prioritize for Section 5.5

- **Black share:** negative and significant in all seven adjusted models, with
  magnitude attenuation of 40.1-65.6%. The least decisive is redundant-to-isolated
  (p=0.0412); do not describe every association as strongly significant.
- **Hispanic share:** negative and significant in all seven, attenuating 30.0-64.0%.
- **Renter share:** positive in all seven, attenuating 20.5-61.5%; significant in
  six, not seven. Redundant-to-isolated changes from +1.045 to +0.403 pp (95% CI
  -0.018 to +0.823; p=0.0606).
- **Age 65+:** five significant primary associations become zero significant
  adjusted associations. The two primary inundation associations were already
  nonsignificant. Adjusted inundation effects and fragile-to-worse change sign,
  but no adjusted age association is distinguishable from zero at 5%.
- **Income:** no adjusted association is significant. The sole significant primary
  association, fragile-to-isolated, falls from +0.846 to +0.171 pp (p=0.470).
  Five estimates reverse sign; all five are nonsignificant before and after.
- **No-vehicle share:** none is significant in the primary models; after adjustment,
  redundant-to-inundated is positive and significant (+0.602 pp; 95% CI +0.071 to
  +1.133; p=0.0264). The other six remain nonsignificant. Four signs change.

Useful overall-worsening anchors (all in percentage points): Black share changes
from -4.904 to -2.382 for redundant-to-worse and -5.860 to -2.667 for fragile-to-worse;
Hispanic share from -6.280 to -3.837 and -7.289 to -3.656; renter share from +4.290
to +2.250 and +5.098 to +2.272. All six adjusted worsening associations remain significant.

### Prose-ready summary

Adding block-group mean elevation and distance to the nearest primary drainage
feature attenuated, but did not eliminate, several social associations with access
degradation. Black and Hispanic population shares remained negatively associated
with all seven transitions, with AME magnitudes declining by approximately 40-66%
and 30-64%, respectively. Renter share remained positively associated with every
transition, although redundant-to-isolated no longer met the 5% significance
threshold; the other six associations remained significant. In contrast, none of
the age-65+ associations remained statistically significant, and the sole significant
primary income association also became nonsignificant. No-vehicle share was
positively associated with redundant-to-inundated transitions after adjustment,
while its other associations remained nonsignificant. Thus, several social
associations persist after conditioning on these two physical characteristics,
while the older-adult pattern is substantially attenuated.

Interpret these as conditional block-group associations. Attenuation is descriptive,
not the percentage causally explained by elevation/drainage. A change in whether
p crosses 0.05 is not a formal test that two AMEs differ. This audit does not estimate
a paired SE/CI for their difference, and does not interpret the physical coefficients.

## Caveats and actions before writing

1. **No freshness or variance-estimator rerun is needed.** Use the verified final-draft
   Conley tables or this audit CSV, not older bootstrap or preliminary outputs.
2. **Model diagnostics deserve review.** A separate current-input refit reproduced the
   saved final-draft diagnostics (`verified_adjusted_diagnostics.csv`), including
   adjusted redundant-to-worse Pearson dispersion around 4,792, with about 99.4% of
   the Pearson statistic from BG `120860100261` at 6 ft (fitted p about 1.22e-7).
   Dispersion remains about 28.95 after removing that row's contribution. This is
   a warning about fit/extreme predictions, not evidence that its removal would
   change the social AMEs by a known amount. All current fits converge and retain
   all coefficients; convergence alone does not resolve this diagnostic. No row
   was dropped or specification changed. A formal influence/refit sensitivity is
   a separate research decision, not required to make the existing tables current.
3. **Upstream source provenance is incomplete.** The access manifest records commit
   `27ac074` with a dirty working tree. `reports/01_provenance_and_layer_gate.md`
   already documents that no runtime source hash/patch identifies that historical
   checkout exactly. Current transition data match that designated run exactly,
   but this audit cannot certify a fresh road-network rebuild would be byte-identical.
4. **Existing comparison percent field is misleading if called attenuation.**
   `outputs/final_draft/physical_controls_comparison.csv` defines `pct_change` as
   100*(adjusted-primary)/abs(primary). That is a signed change normalized by primary
   magnitude, not generally percent attenuation. Use the explicitly defined fields below.

## CSV definitions and complete social-indicator comparisons

File: `primary_vs_elevation_drainage_ames.csv`. All AMEs, SEs, CIs and changes use
**percentage points per one-SD increase** in the indicator (income uses standardized
log median income). Confidence intervals are for individual AMEs, not their differences.

- `signed_change_pp` = adjusted-primary; `absolute_change_pp` = abs(adjusted-primary).
- `magnitude_reduction_pp` = abs(primary)-abs(adjusted).
- `raw_percent_reduction_absolute_magnitude` = 100*(1-abs(adjusted)/abs(primary)).
- `percent_attenuation_absolute_magnitude` reports that ratio only for same-sign
  estimates with abs(primary)>=0.1 pp; otherwise blank, with an explanation. The
  0.1-pp threshold is solely a transparent reporting guard against near-zero ratios.
  Negative values mean amplification. Ratios from nonsignificant primary estimates
  remain descriptive and should not be emphasized as evidence of an underlying effect.
- `significance_change` and `significance_transition` compare nominal p<0.05.

In the tables below, stars are * p<0.05, ** p<0.01, *** p<0.001. Delta is the signed
change in pp; the CSV additionally supplies its absolute value, both SEs and both CIs.
"NA" attenuation denotes a sign change or a near-zero primary estimate as defined above.
"Lost/gained" refers only to crossing p=0.05, not a test of the between-model difference.

### Black share

| Transition | Primary AME | Adjusted AME | Adjusted SE | Adjusted 95% CI | Adjusted p | Delta pp | Attenuation % | Sign flips? | Significance |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| Redundant -> Fragile | -0.688*** | -0.412** | 0.149 | [-0.704, -0.120] | 0.005712 | +0.276 | 40.1 | no | unchanged |
| Redundant -> Isolated | -1.376** | -0.653* | 0.320 | [-1.279, -0.026] | 0.04118 | +0.723 | 52.5 | no | unchanged |
| Redundant -> Inundated | -2.885*** | -1.210*** | 0.365 | [-1.927, -0.494] | 0.0009237 | +1.675 | 58.0 | no | unchanged |
| Redundant -> Worse | -4.904*** | -2.382*** | 0.530 | [-3.421, -1.344] | 6.926e-06 | +2.521 | 51.4 | no | unchanged |
| Fragile -> Isolated | -2.397** | -1.307** | 0.460 | [-2.209, -0.405] | 0.004507 | +1.090 | 45.5 | no | unchanged |
| Fragile -> Inundated | -3.517*** | -1.211*** | 0.353 | [-1.903, -0.519] | 0.0006003 | +2.306 | 65.6 | no | unchanged |
| Fragile -> Worse | -5.860*** | -2.667*** | 0.618 | [-3.878, -1.456] | 1.59e-05 | +3.193 | 54.5 | no | unchanged |

### Hispanic share

| Transition | Primary AME | Adjusted AME | Adjusted SE | Adjusted 95% CI | Adjusted p | Delta pp | Attenuation % | Sign flips? | Significance |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| Redundant -> Fragile | -1.082*** | -0.758*** | 0.188 | [-1.126, -0.390] | 5.387e-05 | +0.324 | 30.0 | no | unchanged |
| Redundant -> Isolated | -1.728*** | -0.990** | 0.318 | [-1.614, -0.366] | 0.001877 | +0.738 | 42.7 | no | unchanged |
| Redundant -> Inundated | -3.493*** | -1.791*** | 0.485 | [-2.741, -0.841] | 0.0002202 | +1.702 | 48.7 | no | unchanged |
| Redundant -> Worse | -6.280*** | -3.837*** | 0.650 | [-5.112, -2.563] | 3.59e-09 | +2.443 | 38.9 | no | unchanged |
| Fragile -> Isolated | -2.869*** | -1.628*** | 0.375 | [-2.363, -0.893] | 1.424e-05 | +1.241 | 43.2 | no | unchanged |
| Fragile -> Inundated | -4.313*** | -1.554*** | 0.393 | [-2.324, -0.785] | 7.511e-05 | +2.759 | 64.0 | no | unchanged |
| Fragile -> Worse | -7.289*** | -3.656*** | 0.564 | [-4.762, -2.550] | 9.13e-11 | +3.633 | 49.8 | no | unchanged |

### Renter share

| Transition | Primary AME | Adjusted AME | Adjusted SE | Adjusted 95% CI | Adjusted p | Delta pp | Attenuation % | Sign flips? | Significance |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| Redundant -> Fragile | +0.852*** | +0.677*** | 0.138 | [+0.407, +0.947] | 9.176e-07 | -0.175 | 20.5 | no | unchanged |
| Redundant -> Isolated | +1.045*** | +0.403 | 0.215 | [-0.018, +0.823] | 0.06059 | -0.642 | 61.5 | no | lost |
| Redundant -> Inundated | +2.420*** | +1.121*** | 0.314 | [+0.506, +1.737] | 0.0003536 | -1.299 | 53.7 | no | unchanged |
| Redundant -> Worse | +4.290*** | +2.249*** | 0.437 | [+1.394, +3.105] | 2.594e-07 | -2.040 | 47.6 | no | unchanged |
| Fragile -> Isolated | +1.737*** | +0.779*** | 0.217 | [+0.354, +1.204] | 0.000328 | -0.958 | 55.1 | no | unchanged |
| Fragile -> Inundated | +3.405*** | +1.427*** | 0.366 | [+0.709, +2.144] | 9.813e-05 | -1.978 | 58.1 | no | unchanged |
| Fragile -> Worse | +5.098*** | +2.272*** | 0.440 | [+1.409, +3.134] | 2.418e-07 | -2.826 | 55.4 | no | unchanged |

### Log median income

| Transition | Primary AME | Adjusted AME | Adjusted SE | Adjusted 95% CI | Adjusted p | Delta pp | Attenuation % | Sign flips? | Significance |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| Redundant -> Fragile | +0.251 | +0.129 | 0.091 | [-0.050, +0.308] | 0.1583 | -0.122 | 48.7 | no | unchanged |
| Redundant -> Isolated | +0.387 | -0.005 | 0.218 | [-0.433, +0.424] | 0.9836 | -0.392 | NA | yes | unchanged |
| Redundant -> Inundated | +0.232 | -0.422 | 0.330 | [-1.069, +0.225] | 0.2009 | -0.654 | NA | yes | unchanged |
| Redundant -> Worse | +0.906 | -0.145 | 0.490 | [-1.106, +0.816] | 0.7682 | -1.051 | NA | yes | unchanged |
| Fragile -> Isolated | +0.846* | +0.171 | 0.236 | [-0.292, +0.633] | 0.4697 | -0.675 | 79.8 | no | lost |
| Fragile -> Inundated | +0.736 | -0.449 | 0.301 | [-1.040, +0.142] | 0.1362 | -1.185 | NA | yes | unchanged |
| Fragile -> Worse | +1.615 | -0.245 | 0.439 | [-1.106, +0.616] | 0.5769 | -1.860 | NA | yes | unchanged |

### Age 65+ share

| Transition | Primary AME | Adjusted AME | Adjusted SE | Adjusted 95% CI | Adjusted p | Delta pp | Attenuation % | Sign flips? | Significance |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| Redundant -> Fragile | +0.436*** | +0.111 | 0.091 | [-0.067, +0.289] | 0.2202 | -0.324 | 74.5 | no | lost |
| Redundant -> Isolated | +0.798** | +0.031 | 0.204 | [-0.368, +0.430] | 0.8778 | -0.767 | 96.1 | no | lost |
| Redundant -> Inundated | +0.986 | -0.372 | 0.375 | [-1.107, +0.362] | 0.3206 | -1.359 | NA | yes | unchanged |
| Redundant -> Worse | +2.231** | +0.152 | 0.441 | [-0.713, +1.016] | 0.7306 | -2.080 | 93.2 | no | lost |
| Fragile -> Isolated | +1.093*** | +0.164 | 0.267 | [-0.358, +0.687] | 0.5382 | -0.928 | 85.0 | no | lost |
| Fragile -> Inundated | +1.193 | -0.594 | 0.535 | [-1.642, +0.454] | 0.2666 | -1.787 | NA | yes | unchanged |
| Fragile -> Worse | +2.218* | -0.209 | 0.632 | [-1.448, +1.029] | 0.7405 | -2.428 | NA | yes | lost |

### No-vehicle household share

| Transition | Primary AME | Adjusted AME | Adjusted SE | Adjusted 95% CI | Adjusted p | Delta pp | Attenuation % | Sign flips? | Significance |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| Redundant -> Fragile | -0.069 | +0.034 | 0.084 | [-0.132, +0.199] | 0.6891 | +0.103 | NA | yes | unchanged |
| Redundant -> Isolated | -0.238 | -0.063 | 0.175 | [-0.406, +0.280] | 0.7179 | +0.175 | 73.5 | no | unchanged |
| Redundant -> Inundated | +0.210 | +0.602* | 0.271 | [+0.071, +1.133] | 0.02638 | +0.392 | -186.1 | no | gained |
| Redundant -> Worse | +0.013 | +0.566 | 0.365 | [-0.150, +1.282] | 0.1213 | +0.553 | NA | no | unchanged |
| Fragile -> Isolated | -0.173 | +0.100 | 0.214 | [-0.319, +0.519] | 0.6391 | +0.273 | NA | yes | unchanged |
| Fragile -> Inundated | -0.097 | +0.502 | 0.332 | [-0.148, +1.152] | 0.13 | +0.599 | NA | yes | unchanged |
| Fragile -> Worse | -0.213 | +0.525 | 0.436 | [-0.330, +1.380] | 0.2287 | +0.738 | NA | yes | unchanged |

## Verification artifacts and rerun commands

Run from the repository root with the existing R/Python research environments:

```text
Rscript outputs/elevation_drainage_audit_20261006/verify_current_models.R
Rscript outputs/elevation_drainage_audit_20261006/verify_adjusted_diagnostics.R
python outputs/elevation_drainage_audit_20261006/verify_data_lineage.py
python outputs/elevation_drainage_audit_20261006/build_comparison.py
```

These audit scripts write only in this audit folder. They do not regenerate production outputs. Verification tables, source hashes, run logs and R session details are saved beside this report.
