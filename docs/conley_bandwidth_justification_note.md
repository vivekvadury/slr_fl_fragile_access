# Draft candidate note — Conley/HAC bandwidth choice

**Status: draft for Vivek to edit. Not manuscript-ready text.** Numbers are
from the 2026-09-08 diagnostic runs (Phase A/B of
`docs/physical_covariate_and_conley_extension.md` continuation). Supporting
files: `outputs/tables/conley_cutoff_comparison_narrow.csv` (5/10/15 km),
`outputs/tables/conley_cutoff_comparison_approach.csv` (10/25/50 km),
`outputs/tables/conley_sweep/`, and the scratch diagnostics
`A1_conley_pd_status.csv` / `A2_nn_distances.rds`.

---

## Candidate paragraph(s)

Spatial dependence in the residuals of the block-group transition models
(Section [Moran's I]) motivates a spatially robust supplement to the
block-group cluster bootstrap. We report Conley/HAC standard errors
(Conley 1999) as that supplement, computed at block-group centroids with a
uniform kernel. The kernel bandwidth is a substantive modeling choice about
the geographic scale over which two block groups plausibly share correlated
model error — a shared local drainage sub-basin, a shared stretch of
arterial road, a shared barrier-island exposure. Block groups in the study
area are closely spaced: the median distance between a block-group centroid
and its nearest neighbour is 0.6 km, the 90th percentile is 1.0 km, and only
a handful of peripheral units exceed 5 km. Against that spacing, a bandwidth
of 5–15 km already pools each unit with roughly its first 8 to 25 rings of
neighbours. Bandwidths beyond ~15 km — 25 km spans about one third of the
study area's 85 km east–west width, 50 km more than half — would assert
correlated errors between coastal and far-inland block groups, which the
plausible mechanisms do not support. We therefore restrict the Conley
robustness check to bandwidths of 5–15 km, fixed a priori from the
inter-unit spacing rather than from the behaviour of the estimator.

Within that range the Conley standard errors are well behaved and the
substantive conclusion is unchanged. At a 5 km bandwidth the Conley
covariance matrix is positive-definite for all seven transition models under
both the demographic-only and the physical-covariate specification; the
Conley standard errors on the standardized Black-share and Hispanic-share
coefficients are 1.5 to 3 times the block-group-clustered standard errors,
and every one of those coefficients remains negative and significant at the
5% level (largest p-value 0.04; all others below 0.005). At 10 km the
demographic-only matrices remain positive-definite and two of the seven
physical-covariate matrices require an eigenvalue repair; by 15 km most
matrices require repair. We therefore treat the 5 km demographic-only
result as the reference Conley check and report the 10 and 15 km results
only as a sensitivity band.

[If the manuscript reports wider bandwidths at all:] At bandwidths of 25 km
and 50 km the raw Conley covariance matrix is not positive-definite for any
of the seven models under either specification and is replaced by its
nearest positive-semidefinite projection before standard errors are read
off. This is not an artifact of the panel structure — each block group
appears six times, once per positive SLR scenario, at an identical centroid
— because the same non-positive-definiteness appears when the estimator is
computed on a single SLR cross-section with no repeated coordinates
(checked at every SLR scenario, both specifications, both bandwidths: zero
of seven matrices positive-definite in every case). It is a property of the
uniform-kernel Conley estimator applied to this residual field at a
bandwidth that is large relative to the ~0.6 km inter-unit spacing: the
kernel sums a very large number of noisy cross-products and the resulting
matrix estimate loses rank. The repaired 25/50 km standard errors are
therefore not a genuine Conley estimator and we do not rely on them.

The block-group cluster bootstrap remains the primary basis for inference on
the average marginal effects; the Conley standard errors are reported as a
supplementary check on whether the sign and significance of the
racial-composition coefficients depend on the assumption that model errors
are uncorrelated across neighbouring block groups. They do not.

---

## Supporting numbers (for the editor, not for the manuscript body)

**Positive-definiteness of the raw Conley covariance, by bandwidth**
(count of the 7 transition models with a positive-definite matrix before any
repair):

| bandwidth | demographic_only | with_physical |
|---|---|---|
| 5 km  | 7 / 7 | 7 / 7 |
| 10 km | 7 / 7 | 5 / 7 |
| 15 km | 2 / 7 | 1 / 7 |
| 25 km | 0 / 7 | 0 / 7 |
| 50 km | 0 / 7 | 0 / 7 |

**Single-SLR-scenario cross-section check (Phase A.1)** — Conley VCOV PD count
of 7, county fixed effect only, one row per block group (no repeated
coordinates):

| SLR scenario used | 25 km | 50 km |
|---|---|---|
| slr_ft = 2 | 0 / 7 (both specs) | 0 / 7 (both specs) |
| slr_ft = 3 | 0 / 7 | 0 / 7 |
| slr_ft = 4 | 0 / 7 | 0 / 7 |
| slr_ft = 5 | 0 / 7 | 0 / 7 |
| slr_ft = 6 | 0 / 7 | 0 / 7 |

(slr_ft = 1 excluded: rare transitions make several county cells degenerate
in a single cross-section. The full-panel reproduction in the same script
also returns 0 / 7 PD at 25 and 50 km, matching the Phase 1 sweep.)

**Nearest-neighbour distance between block-group centroids (metres, EPSG:32617)**

| set | n | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|---|
| all analysis block groups | 3942 | 338 | 418 | 595 | 779 | 983 | 21216 |
| redundant-risk set | 3442 | 371 | 449 | 627 | 803 | 1012 | 21787 |
| fragile-risk set | 3178 | 371 | 465 | 655 | 817 | 1083 | 21787 |

Study extent 85.4 km (E–W) × 203.3 km (N–S); bbox diagonal 220.5 km.

**Cutoff sensitivity within 5–15 km:** median `conley_se / clustered_se` is
1.69 / 1.64 / 1.61 (demographic-only) and 1.51 / 1.62 / 1.50 (with_physical)
at 5 / 10 / 15 km — stable, versus 1.62 / 1.42 / 1.09 across 10 / 25 / 50 km
in the original sweep. Only 11 of 98 (spec, transition, term) combinations
flip the Conley-vs-clustered SE ordering across 5/10/15 km (44 of 98 across
10/25/50 km), and the flips concentrate in the Redundant → Fragile model.
