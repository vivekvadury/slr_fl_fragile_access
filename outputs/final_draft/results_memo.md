# Results memo — final-draft freeze (1 October 2026)

Every number here comes from a file in `outputs/final_draft/` (named in
brackets). Inference is **Conley 5 km**, demographic-only specification,
approach bridge rule, unless stated otherwise. Effects are percentage points
per 1 SD.

---

## Decisions needed from you (nothing below has been decided for you)

1. **Primary inference.** The tables are built so that Conley 5 km can be
   primary and the bootstrap supplementary. The manuscript (v42) still says
   bootstrap-primary with a Conley robustness rule. Under either framing, the
   set of "robust" findings is identical: an association is significant under
   Conley 5 km only where it is also significant under the bootstrap (see
   §5.6 below). So the choice changes the narrative, not the findings.
2. **Population-weighted AMEs.** The current implementation reweights the
   *average* only, by block-group eligible population. In a logit without
   interactions, this multiplies every AME in a model by the same factor
   (here +2.7% to +5.9% in magnitude), so it cannot change signs, relative
   sizes or which covariates matter. As implemented, it does not test what
   §4.5 says it tests ("whether the demographic associations change when more
   populous blocks receive greater weight").
   - **(a)** Drop it.
   - **(b)** Keep one honest sentence, e.g. "Averaging over residents rather
     than block groups changes all effects by less than 6% and does not alter
     their pattern."
   - **(c)** Fit population-weighted *models*. This is a different estimand
     (resident-level transition risk) and needs a new bootstrap; not
     recommended before 8 October.
3. **Physical-adjusted extrapolation outlier.** Block group 120860100261
   (Miami-Dade; mean elevation 7.4 m, z = 3.4) loses redundancy in all 12
   redundant blocks at 6 ft. Its fitted probability under the physical model is
   1.2×10⁻⁷. This one row is 99.4% of that model's Pearson χ²: dispersion is
   4,792 with it and 28.9 without it. The same row is also the largest
   contributor in 4 other physical-adjusted models. For example, it is 79% of
   Fragile→Worse χ² (31.5 with it, 6.6 without) [`diagnostics_summary.csv`].
   Sandwich and
   Conley SEs are not driven by it in the same way, since scores are bounded,
   but it shows that elevation enters partly as a mechanical encoding of the
   inundation rule. **Recommend:** report dispersion for the demographic-only
   models only, and describe the physical specification as a sensitivity
   analysis whose elevation coefficient is not interpretable as a physical
   effect. No exclusion or refit has been made.
4. **School layer** (all K–12, unfiltered) and **fire-station vintage**
   (2012 county updates). These are wording decisions only; they are carried
   from earlier.

---

## 5.1 Baseline access is already unequal  [`descriptive_access_by_slr.csv`]

At 0 ft (MHHW), across 68,521 eligible blocks and 6,135,688 residents:

| State | Blocks | % of blocks | Residents | % of residents |
|---|---:|---:|---:|---:|
| Redundant | 50,835 | 74.19 | 3,714,651 | 60.54 |
| Fragile | 17,335 | 25.30 | 2,382,858 | 38.84 |
| Isolated | 163 | 0.24 | 18,126 | 0.30 |
| Inundated | 188 | 0.27 | 20,053 | 0.33 |

Fragile blocks are more populous than redundant ones: 25.3% of blocks hold
38.8% of residents.

## 5.2 Most newly affected residents are not inundated  [`population_thresholds_by_slr.csv`]

Population in blocks making any of the five adverse transitions, and the
share whose transition is *not* inundation:

| SLR | Newly affected residents | Non-inundation share |
|---|---:|---:|
| 1 ft | 5,523 | 12.1% |
| 2 ft | 63,117 | 75.0% |
| 3 ft | 274,793 | 70.3% |
| 4 ft | 540,709 | 55.0% |
| 5 ft | 1,002,449 | 43.5% |
| 6 ft | 2,048,545 | 35.7% |

From 2 ft through 4 ft, most newly affected residents lose access through the
network while their own block stays dry.

## 5.3 The population counted as affected depends on the threshold

The three nested thresholds at 6 ft:

| Threshold | Residents |
|---|---:|
| New inundation | 1,316,751 |
| + new isolation | 1,958,517 |
| + new fragility | 2,048,545 |

Adding fragility counts 90,028 more residents at 6 ft and 12,988 at 2 ft.
Adding fragility is 20.6% of all newly affected residents at 2 ft, but 4.4% at
6 ft.

## 5.4 Social patterning of access degradation  [`main_transition_ames_conley5.csv`]

Significant at p < 0.05 under Conley 5 km, out of 7 models:

| Covariate | Direction | Significant | Range (pp per SD) |
|---|---|---:|---|
| Black share | − | 7/7 | −0.69 (R→F) to −5.86 (F→Worse) |
| Hispanic share | − | 7/7 | −1.08 to −7.29 |
| Renter share | + | 7/7 | +0.85 to +5.10 |
| Age 65+ share | + | 5/7 | +0.44 to +2.23 |
| Log median income | + | 1/7 | only Fragile→Isolated, +0.85 |
| No-vehicle share | — | 0/7 | |

- **Age 65+.** Not significant for Redundant→Inundated (p = 0.055) or
  Fragile→Inundated (p = 0.083). It is significant for every transition that
  runs through network loss.
- **Effect sizes.** They are largest for the composite "worse" outcomes and
  for transitions out of fragile access.
- **Samples.** The redundant-risk models use 20,652 rows (3,442 block groups;
  46,954 baseline-redundant blocks). The fragile-risk models use 19,068 rows
  (3,178 block groups; 16,114 baseline-fragile blocks) [`model_sample_sizes.csv`].

## 5.5 Role of elevation and drainage  [`physical_controls_comparison.csv`]

Adding block-group mean elevation and distance to primary drainage changes the
following (Conley 5 km):

| Covariate | Magnitude change | Significant (physical spec) |
|---|---|---:|
| Black share | shrinks 40–66% | 7/7 |
| Hispanic share | shrinks 30–64% | 7/7 |
| Renter share | shrinks 21–62% | 6/7 |
| Age 65+ share | 75–150% of the estimate removed | 0/7 (3 sign changes) |
| Log median income | — | 0/7 |
| No-vehicle share | — | 1/7 (unstable; 4 sign changes) |

- **Elevation** is negative and very large (−2.0 to −18.7 pp per SD; all
  p < 10⁻⁸).
- **Drainage distance** is significant in 3/7 models.

Interpretation:
- **Race and renter.** These associations are partly, but not wholly,
  explained by physical setting.
- **Age 65+.** The older-adult association is not distinguishable from
  physical setting, i.e. older residents' block groups are lower-lying.
- **Caveat.** See decision 3: elevation partly encodes the outcome-generating
  inundation rule.

## 5.6 Robustness

**Inference method** [`cluster_vs_conley.csv`, `conley_cutoff_sensitivity.csv`]

For the demographic-only AMEs, Conley 5 km SEs are 1.13–2.91× the
block-group-clustered SEs (median 1.69) and 1.12–2.60× the bootstrap SEs
(median 1.64). Agreement at p < 0.05 for the 42 demographic-only AMEs:

| Covariate | Agreement across the three methods |
|---|---|
| Black, Hispanic, renter | 21/21 significant under all three |
| No-vehicle | 0/7 under all three |
| Age 65+ | 5/7 under all three; 2 significant under clustered and bootstrap but not Conley |
| Log median income | 1/7 under all three; 2 lose significance under Conley; 4 never significant |

No AME is significant under Conley but not under the bootstrap, in either
specification.

**Cutoff.** Significance counts are stable across 5, 10 and 15 km: race and
renter 7/7 at every cutoff; age 5, 6 and 6; income 1/7; no-vehicle 0/7. The
10/15 km matrices are not positive definite in several models
[`conley_positive_definiteness.csv`] and were eigenvalue-repaired. Report them
as supplementary only.

**Bridge rule** [`bridge_rule_comparison.csv`; AMEs from the bootstrap tables]

| Rule | Baseline fragile (blocks) | Newly affected, 6 ft | Non-inundation share, 6 ft |
|---|---:|---:|---:|
| Intersect | 25.68% | 2,039,740 | 36.3% |
| Approach | 25.30% | 2,048,545 | 35.7% |
| Retain | 25.33% | 2,050,708 | 35.7% |

AMEs change by at most 0.09 pp under retain, with no sign or significance
changes. Under intersect they change by up to 0.70 pp, and only income and
no-vehicle terms change (bootstrap p-values):
- Log median income loses significance in Redundant→Fragile, Fragile→Isolated
  and Fragile→Worse.
- No-vehicle share becomes significant in Fragile→Isolated (−0.51 pp,
  p = 0.033).
- No-vehicle share changes sign in Redundant→Worse; it is near zero and not
  significant under either rule.

Race, renter and age are unaffected. Income is already significant in only
1/7 models under Conley 5 km, so the intersect result does not change the
primary findings.

**Network attachment** [`outputs/attachment_sensitivity/attach_20260925/REPORT.md`]

Five alternative origin and facility rules:
- baseline fragile 25.06–25.40%;
- newly affected at 6 ft 2.037–2.052 M;
- maximum AME change 0.32 pp;
- race and renter significant in all arms.

Localized effects:
- Fisher Island, which is ferry-only, becomes isolated under a 500 m facility cap.
- Nearest-node facility attachment changes 87 baseline states.

**Spatial residuals** [`diagnostics_summary.csv`]

Residual Moran's I is positive and significant in 40/42 scenario-by-model tests
(demographic-only) and 35/42 with physical controls. Neighboring block groups'
residuals remain correlated after adjustment, which motivates the Conley
standard errors.

**Overdispersion.** Pearson dispersion is 1.83–6.98 in the demographic-only
models. Model-based binomial SEs would therefore be too small; Conley,
clustered and bootstrap SEs do not rely on the binomial variance.

---

## Supplementary Information to-do

- **Conley 10/15 km eigenvalue adjustment (decided 2026-10-01: disclose in the SI, not in §4.7).**
  State that at 10 and 15 km the estimated covariance matrices were not positive
  definite for some models and were made positive definite by eigenvalue
  adjustment (Cameron, Gelbach & Miller, 2011). Counts by model and specification
  are in `conley_positive_definiteness.csv`: 10 km, 0/7 demographic-only and 2/7
  physical-adjusted; 15 km, 5/7 and 6/7.
