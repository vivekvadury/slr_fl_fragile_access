# Physical-covariate specification and Conley spatial SEs

This documents the September 2026 extension to the transition-model pipeline:
an alternate, clearly labeled `with_physical` specification (elevation and
drainage) and an optional Conley/HAC spatial standard-error mode. Neither
changes the default demographic-only pipeline that
`05_population_figures.py` and `06_placeholders_in_draft.ipynb` depend on.

New / edited files:

| File | Change |
|---|---|
| `scripts/03b_join_elevation_drainage.py` | NEW — builds `data/processed/analysis/block_group_physical_covariates.csv` |
| `scripts/04_shared_model_spec.R` | NEW — shared covariate lists, `fit_transition_model()`, `compute_conley_vcov()` |
| `scripts/04_transition_models.R` | sources the shared file; adds `MODEL_SPEC`, `CONLEY_CUTOFF_KM`, `compare_specifications()` |
| `scripts/04b_spatial_residual_diagnostics.R` | sources the shared file; adds `--spec {demographic_only,with_physical,both}`; live Pearson dispersion |
| `scripts/00_README_workflow.md` | documents `03b`, the two new `04` options, and `04b --spec` |

---

## 1. Elevation source and resolution

**USGS 3DEP seamless 1/3-arc-second (~10 m) bare-earth DEM**, NAD83 (EPSG:4269)
horizontally, meters above NAVD88 vertically. Two immutable, dated
distribution tiles cover the tri-county extent and are pinned in `03b` by
URL, byte length, and USGS MD5:

| Tile | File | Bytes | MD5 |
|---|---|---|---|
| n26w081 | `USGS_13_n26w081_20260225.tif` | 364,091,865 | `65dbd55b1cfab770111ea513f327ffbd` |
| n27w081 | `USGS_13_n27w081_20251216.tif` | 474,839,904 | `81f0d97961dd16c175ba17fa9b76920c` |

Both pins were verified against the live USGS S3 bucket (`Content-Length` and
the base64 `x-amz-meta-md5chksum` header decode to the pinned values exactly).
Pinning dated `historical/` tiles rather than `current/` keeps the covariates
from silently changing when USGS republishes.

**Why this product.** It is in the same USGS/TNM lidar-derived elevation
family that NOAA OCM used as source data when building the Southeast Florida
3-m conditioned DEM behind the Sea Level Rise Viewer, so it is broadly
consistent with the inundation polygons already in this pipeline
(`data/raw/noaa/FL_SE_slr_final_dist.gpkg`). It is **not** NOAA's specially
conditioned DEM and the script does not claim the two surfaces are identical.
The 1/3-arc-second product is fine enough for block-group zonal summaries
while remaining practical to download (~0.8 GB) and rerun; the 1-m product
would require many very large project tiles.

`03b` computes **zonal mean and median** elevation per `slr_0ft` block-group
polygon over valid DEM cell centers, removing the 6-pixel tile overlap at the
nominal 26 degN seam so no cell is counted twice.

Result over 3,942 block groups: `elevation_m_mean` mean 2.66 m (range
-1.15 to 15.4 m; 15 block groups below the NAVD88 datum), `elevation_m_median`
tracks it closely. These are plausible for the Miami-Dade / Broward / Palm
Beach coastal plain and the Atlantic Coastal Ridge.

## 2. Drainage representation

**Implemented: a continuous proxy** — `drainage_distance_km`, the distance
(km, computed in EPSG:32617) from each block-group centroid to the nearest
**SFWMD Arc Hydro Enhanced Database (AHED) feature with
`HYDRO_ORDER = 'PRIMARY'`** (7,803 primary canal / primary-flowline
features). Result: median 1.19 km, range 0.001-13.7 km, no missing values.

**Rejected: a categorical basin fixed effect.** Investigated before
implementing:

* The SFWMD AHED Basin (HUC6) layer supplies **one** basin over the study
  extent — no identifying variation.
* An audit of the full USGS WBD **HUC12** layer assigned the 3,942 analysis
  block-group centroids to **43 HUC12s**, of which **3 are singletons** and
  **11 have fewer than 10 block groups**. Adding that many fixed-effect
  levels to rare transition-outcome models (Redundant/Fragile -> Inundated)
  creates avoidable separation / incidental-parameter risk, and block groups
  that straddle a watershed boundary would need an arbitrary assignment.

The continuous distance is complete, parsimonious, and preserves
within-county variation.

**Bug fixed while implementing.** SFWMD's `geoweb.sfwmd.gov` ArcGIS host sits
behind a Web Application Firewall that returns HTTP 403 to non-browser
`User-Agent` strings. `03b` now sends a browser-style UA (overridable with
the `SLR_FL_HTTP_USER_AGENT` environment variable); the USGS S3 bucket is
indifferent to the header.

## 3. Output contract

`data/processed/analysis/block_group_physical_covariates.csv` — exactly
`block_group_geoid, elevation_m_mean, elevation_m_median,
drainage_distance_km`; **one row per block group** (3,942), not per SLR
scenario. `data/` is git-ignored, matching every other file under
`data/processed/analysis/`; rerun `03b` to regenerate.

**Modeling warning (carried in the `03b` docstring).** Elevation is
mechanically close to the classifier's own inundation rule, so downstream
models on this alternate specification must be checked for separation and
convergence, especially any Inundated-outcome model. This warning is borne
out — see Section 6.

---

## 4. Step 2 regression test (pure-extraction check): PASS

The pre-refactor `04_transition_models.R` (`HEAD`) and the refactored version
(which `source()`s `04_shared_model_spec.R`) were each run with **identical
settings** — `AME_BOOT_REPS=25`, everything else default (`MODEL_SPEC` unset,
`CONLEY_CUTOFF_KM` unset):

| Output | Result |
|---|---|
| `transition_model_coefficients_approach.csv` | byte-identical (and identical to the committed copy) |
| `transition_sample_diagnostics_approach.csv` | byte-identical (and identical to the committed copy) |
| `ame_bootstrap_transition_table_approach.tex` | byte-identical |
| `ame_bootstrap_results_approach.xlsx` | content-identical (`identical()` in R on the read-back data frame; only the xlsx container timestamp differs) |

The bootstrap is seeded and the refactor adds no RNG consumer (the only
per-replication random draw is `sample()` of clusters; formula construction
moved to `make_transition_formula()` but touches no RNG), so a 25-replication
match implies the 199-replication default matches — it is the same seeded
stream, just longer. A subsequent full default run (`AME_BOOT_REPS` unset =
199) reconfirmed `transition_model_coefficients_approach.csv` and
`transition_sample_diagnostics_approach.csv` byte-identical to the committed
copies.

**Pre-existing discrepancy, not caused by the refactor.** The *committed*
`ame_bootstrap_results_approach.xlsx` / `.tex` were generated at
`AME_BOOT_REPS=49` (the `.tex` literally embeds "49-replication cluster
bootstrap"), which is not the script default of 199 (unchanged since commit
`01c7d04`, 2026-08-10). A default-settings run regenerates them at 199 reps:
AME **point estimates** and every deterministic output are unchanged; only
bootstrap SEs move slightly and two significance stars on `Log median income
(z)` shift. Neither `05_population_figures.py` nor
`06_placeholders_in_draft.ipynb` reads `transition_model_coefficients_*`,
`ame_bootstrap_*`, or `transition_sample_diagnostics_*` (they consume only
`fig4_*` and `block_level_long_dataset*`), so nothing downstream breaks.

---

## 5. New `04_transition_models.R` options

* `MODEL_SPEC` — `demographic_only` (default; behavior and filenames
  unchanged) or `with_physical`. `with_physical` left-joins
  `block_group_physical_covariates.csv` by `block_group_geoid` and adds
  standardized `z_elevation_m_mean` and `z_drainage_distance_km` to the RHS.
  Its arm-tagged outputs gain a `_with_physical` suffix
  (`transition_model_coefficients_approach_with_physical.csv`, etc.) so the
  demographic-only files are never overwritten.
* `CONLEY_CUTOFF_KM` — unset by default (no Conley SEs, unchanged behavior).
  When set, after each model is fit `04` also computes
  `fixest::vcov_conley()` at block-group centroids (derived from
  `slr_block_group_analysis_<arm>.gpkg`) at the given cutoff, and adds
  `conley_se` + `conley_cutoff_km` columns to the coefficient CSV **and** the
  AME (`avg_slopes`) output — always *alongside*, never replacing, the
  block-group-clustered SE and the cluster bootstrap. Smoke-tested at
  `CONLEY_CUTOFF_KM=50` on both specs (exit 0, columns populated; e.g.
  demographic-only `z_pct_black_nh` in Redundant -> Fragile: clustered SE
  0.068, Conley SE 0.030).
* `AME_POP_WEIGHT` — unset by default (unchanged behavior). When set to
  `1`/`true` (or an explicit column name; the shorthand resolves to
  `eligible_pop20`), `04` additionally writes
  `ame_bootstrap_results_approach_popweighted.xlsx` /
  `ame_bootstrap_transition_table_approach_popweighted.tex`. The grouped-
  binomial model fit is identical; only `avg_slopes()` aggregation is
  weighted, so each block group counts in proportion to its eligible
  population rather than once. It runs its own cluster bootstrap with the same
  seeds (hence the same resampled block groups) as Table 2. See Section 9.
* `compare_specifications()` writes
  `transition_model_spec_comparison_<arm>.csv` whenever both spec coefficient
  files exist for an arm — see Section 6.

---

## 6. Coefficient comparison — does the racial-composition finding survive `with_physical`?

**Yes in sign and significance; no in magnitude — it attenuates by roughly a
quarter to a half.** From `transition_model_spec_comparison_approach.csv`
(estimates are bootstrap-independent):

| Transition | z_pct_black_nh: demog. -> physical | % change | z_pct_hispanic: demog. -> physical | % change |
|---|---|---|---|---|
| Redundant -> Fragile | -0.498 -> -0.290 | **-41.8%** | -0.783 -> -0.533 | **-32.0%** |
| Redundant -> Isolated | -0.470 -> -0.228 | **-51.5%** | -0.590 -> -0.346 | **-41.4%** |
| Redundant -> Inundated | -0.663 -> -0.354 | **-46.6%** | -0.803 -> -0.524 | **-34.8%** |
| Redundant -> Worse | -0.703 -> -0.516 | **-26.5%** | -0.900 -> -0.832 | **-7.6%** |
| Fragile -> Isolated | -0.658 -> -0.392 | **-40.5%** | -0.787 -> -0.488 | **-38.0%** |
| Fragile -> Inundated | -0.699 -> -0.335 | **-52.1%** | -0.857 -> -0.429 | **-49.9%** |
| Fragile -> Worse | -0.821 -> -0.578 | **-29.7%** | -1.022 -> -0.792 | **-22.5%** |

Every `with_physical` Black-share and Hispanic-share coefficient stays
negative and highly significant (p from ~1e-5 to ~1e-27). Mean attenuation
across the 14 cells is about -37%.

**Why `with_physical` is a sensitivity spec, not a replacement.** The
`z_elevation_m_mean` coefficients are implausibly large on the standardized
scale — -1.42, -1.95, **-3.69**, **-4.06**, -1.67, **-3.38**, **-3.70**
across the seven transitions (odds ratios down to ~0.017 per SD of
elevation), largest for the Inundated and "Worse" outcomes. `04b` finds one
`with_physical` model's live Pearson dispersion at about **4,800** (vs
1.83-6.98 for demographic-only). Elevation is partly re-encoding the
classifier's inundation rule, so its coefficient is not cleanly
interpretable and the attenuation of the racial coefficients is a mix of
plausible confounder adjustment (redlining sorted communities by elevation)
and over-control / near-collider adjustment on the outcome definition
itself. Report the core association from the demographic-only specification;
cite `with_physical` as the robustness check that shows the sign and
significance hold while the magnitude is elevation-sensitive.

---

## 7. Paired Moran's I comparison (`04b --spec both`)

`moran_residual_diagnostics_spec_comparison.csv`, 42 tests:

| | demographic_only | with_physical |
|---|---|---|
| Significant at 0.05 | **40 / 42** | **35 / 42** |
| redundant-risk | 23 / 24 | 20 / 24 |
| fragile-risk | 17 / 18 | 15 / 18 |

**5 tests lose 0.05 significance under `with_physical`; 0 gain it.** The five
are Redundant -> Inundated (1 ft), Redundant -> Worse (1 ft and 6 ft),
Fragile -> Inundated (1 ft), and Fragile -> Worse (1 ft) — all adverse
outcomes, four of them at the smallest SLR level. Residual Moran's I falls
under `with_physical` for almost every test (`moran_i_change_with_physical`
is negative in 36 of 42 rows), consistent with elevation and drainage
absorbing part of a spatially clustered omitted component — but 35 of 42
tests still reject spatial randomness, so spatial dependence in the
residuals is **reduced, not removed**. The block-group-clustered plus
cluster-bootstrap uncertainty still needs a spatially robust supplement
(the `CONLEY_CUTOFF_KM` option) under either specification.

The Pearson dispersion range quoted in Sections 4-5 of
`final_methods_verification.md` is now computed live from `04b`'s own fitted
models: demographic-only reproduces the previously hardcoded "1.832-6.984",
and `with_physical` would report its true, much wider range automatically.

---

## 8. Conley/HAC cutoff sensitivity sweep (10 / 25 / 50 km)

`outputs/tables/conley_cutoff_comparison_approach.csv` (294 rows = 2 specs x 3
cutoffs x 49 transition/term pairs — 7 transitions x 7 terms) was assembled
from six `CONLEY_CUTOFF_KM={10,25,50} x MODEL_SPEC={demographic_only,
with_physical}` runs (`AME_BOOT_REPS=5`; only `conley_se`, written before the
bootstrap, is used from these). Columns: `spec, cutoff_km, transition, term,
estimate, clustered_std_error, conley_se, conley_cutoff_km`. Copies of every
`spec_output_name()` file per run are kept under
`outputs/tables/conley_sweep/<cutoff>km_<spec>/`.

**The Conley SEs in this pipeline are not stable and should be reported with
an explicit caveat.**

* **Positive-definiteness.** `fixest::vcov_conley()` returned a
  positive-definite matrix for all seven models only at **10 km, demographic-
  only**. At 10 km with_physical, 2 of 7 were non-PD; at **25 km and 50 km,
  both specifications, all 7 of 7** Conley covariance matrices were not
  positive-definite and were eigenvalue-repaired by fixest (a logged
  warning). Every 25/50 km `conley_se` is therefore from a repaired matrix,
  not a genuine Conley estimator. The likely cause is the panel structure:
  each block group contributes six rows (SLR 1-6 ft) at identical centroid
  coordinates, and a spatial-only kernel with a wide bandwidth over a
  ~200 km study extent easily produces an indefinite estimator.
* **Cutoff sensitivity.** For **44 of 98** (spec, transition, term)
  combinations the ordering of `conley_se` relative to `clustered_std_error`
  flips across the three cutoffs. A single coefficient's `conley_se` moves by
  up to **8.6x** across cutoffs (demographic-only Redundant -> Isolated /
  `z_pct_black_nh`: 0.167 at 10 km, 0.099 at 25 km, 0.019 at 50 km). Median
  `conley_se / clustered_se` is 1.62 at 10 km (Conley larger for 92/98
  terms), 1.42 at 25 km (73/98), and 1.09 at 50 km (59/98; minimum ratio
  0.34 — several repaired Conley SEs collapse to a third of the clustered
  SE).
* **Racial coefficients.** In the one defensible cell (10 km, demographic-
  only) every Black-share / Hispanic-share coefficient's Conley SE is 1.5-4x
  the clustered SE, yet all stay significant at 0.05 (e.g. Redundant ->
  Worse `z_pct_black_nh`: estimate -0.703, clustered SE 0.065, Conley@10 km
  SE 0.193, z approximately -3.6).

Bottom line for the manuscript: if Conley SEs are reported, use the 10 km
demographic-only column and state plainly that wider bandwidths yield
non-positive-definite covariance matrices and are not reliable here; the
block-group cluster bootstrap remains the primary uncertainty measure.

**Follow-up (2026-09-08).** A narrow sweep at 5 / 10 / 15 km
(`outputs/tables/conley_cutoff_comparison_narrow.csv`) plus a root-cause
diagnostic show that **5 km is the only bandwidth where all seven models are
positive-definite under *both* specs**, that the non-PD failure at wider
bandwidths is *not* caused by the repeated-coordinate panel structure (a
single-SLR cross-section is also 0/7 PD at 25/50 km), and that all 28
racial-composition coefficients stay significant under Conley@5 km SEs
(largest p = 0.04). The bandwidth choice and the panel-structure ruling-out
are written up as a candidate manuscript note in
`docs/conley_bandwidth_justification_note.md`.

---

## 9. Population-weighted average marginal effects (additional table)

`AME_POP_WEIGHT=1 Rscript scripts/04_transition_models.R` (MODEL_SPEC
`demographic_only`, no Conley, `AME_BOOT_REPS` default 199) writes
`ame_bootstrap_results_approach_popweighted.{xlsx}` and
`ame_bootstrap_transition_table_approach_popweighted.tex` **in addition to**
the unchanged Table 2 files (verified byte-identical to a no-flag run). The
grouped-binomial fit is unchanged (still weighted by
`baseline_redundant_n` / `baseline_fragile_n`); only `avg_slopes()` is
weighted, by `eligible_pop20` supplied as a numeric vector aligned to each
model's retained rows. This is total-population weighting — every resident
counts once, not weighting by the vulnerability subgroup.

**Population weighting barely moves the racial-composition AMEs.** Count-
weighted values are Table 2; population-weighted are from the 199-replication
bootstrap (`ame_bootstrap_results_approach_popweighted.xlsx`). SE in
parentheses; all entries `***` (p < 0.001) in both columns.

| Transition | z_pct_black_nh count -> pop | % chg | z_pct_hispanic count -> pop | % chg |
|---|---|---|---|---|
| Redundant -> Fragile | -0.0069 (0.0012) -> -0.0065 (0.0011) | -5.7% | -0.0108 (0.0013) -> -0.0102 (0.0011) | -5.7% |
| Redundant -> Isolated | -0.0138 (0.0022) -> -0.0133 (0.0018) | -3.5% | -0.0173 (0.0024) -> -0.0167 (0.0017) | -3.5% |
| Redundant -> Inundated | -0.0289 (0.0034) -> -0.0279 (0.0032) | -3.2% | -0.0349 (0.0041) -> -0.0338 (0.0039) | -3.2% |
| Redundant -> Worse | -0.0490 (0.0048) -> -0.0477 (0.0046) | -2.7% | -0.0628 (0.0056) -> -0.0611 (0.0054) | -2.7% |
| Fragile -> Isolated | -0.0240 (0.0036) -> -0.0226 (0.0027) | -5.9% | -0.0287 (0.0041) -> -0.0270 (0.0031) | -5.9% |
| Fragile -> Inundated | -0.0352 (0.0054) -> -0.0340 (0.0051) | -3.5% | -0.0431 (0.0059) -> -0.0416 (0.0057) | -3.5% |
| Fragile -> Worse | -0.0586 (0.0064) -> -0.0567 (0.0061) | -3.3% | -0.0729 (0.0084) -> -0.0705 (0.0080) | -3.3% |

Every racial AME attenuates toward zero by 2.7-5.9% (mean 4.0%); sign,
magnitude, and significance are unchanged (population-weighted p-values run
from ~1e-30 to 6.6e-10), and bootstrap SEs are slightly smaller under
population weighting. Higher-population block groups carry marginally
shallower demographic gradients, but the manuscript's core finding is robust
to weighting residents rather than block groups.

Two transitions (Redundant/Fragile -> Isolated) needed ~38-39 extra bootstrap
attempts to reach 199 successful population-weighted replications (vs 0-4
elsewhere): a weighted `avg_slopes()` occasionally fails on a resample with a
sparse fixed-effect cell. All seven models still reached 199 successes, so
the SEs are valid, but for those two the resampled block-group sets are not
identical to the count-weighted run's.
