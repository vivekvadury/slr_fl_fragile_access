# Final methods verification: bridge-rule headlines and uncertainty diagnostics

Date: 2026-08-31

Scope: Steps 0–2 only. No classifier, notebook, model specification, or existing output was modified. All current-run block summaries below use `analysis_eligible == True` and no population filter unless a row explicitly says `POP20 > 0`.

## 1. Step 0 — quick drift spot-check

| Item | Result | Evidence |
|---|---|---|
| Layer gate and cache invalidation | **PASS** | `02_access_flags.py` has `CACHE_SCHEMA_VERSION = 3`; the default cache key is `layer_positive`; `is_positive_layer_value()` tests `float(value) > 0.0`; and the nonzero-layer gate is used only when `legacy_layer_gate` is enabled. |
| Notebook 03 universe and state partition | **PASS** | The notebook applies `analysis_eligible == True` once, explicitly does not filter on `pop20`, and asserts that the five state shares sum to one. Its saved output records 68,521 eligible blocks in each arm. |
| Model-script partition assertion | **PASS** | `04_transition_models.R` calls `assert_eligible_state_partition()` at the start of `prepare_transition_data()`, before constructing risk sets or fitting models. |
| Population-script universe and states | **PASS** | `05_population_figures.py` rejects any row not marked `analysis_eligible == True` and explicitly tabulates unclassified, inundated, isolated, fragile, and redundant. |
| Completed positive-layer runs | **PASS** | The intersect, approach, and retain manifests each record cache schema 3, `legacy_layer_gate = false`, 68,521 eligible blocks, and 70,695 rows at each of 0–6 ft. The block CSVs independently reproduce the 68,521 eligible-block universe in every scenario. |

No requested Step 0 item has drifted. All three manifests identify commit `27ac074753b24e1b68e21597b746682f077931c9`; each also records that the Della working tree was dirty when the run was made.

## 2. Step 1 — three-arm headline comparison

Each baseline state cell gives `count (share)`. Population-weighted state cells give `POP20 in state (share of eligible POP20)`. The two pre-fix values are block shares from `baseline_fragile_shares.csv`, not population-weighted shares, so their denominators are shown explicitly.

| Metric | Pre-fix reference | Intersect | Approach | Retain |
|---|---:|---:|---:|---:|
| Eligible blocks | — | 68,521 | 68,521 | 68,521 |
| Eligible POP20 | — | 6,135,688 | 6,135,688 | 6,135,688 |
| Baseline redundant — blocks | — | 50,303 (73.41%) | 50,835 (74.19%) | 50,835 (74.19%) |
| Baseline redundant — population-weighted | — | 3,683,312 (60.03%) | 3,714,651 (60.54%) | 3,714,651 (60.54%) |
| Baseline fragile — blocks | 17,458 / 70,695 (24.69%) | 17,594 (25.68%) | 17,335 (25.30%) | 17,356 (25.33%) |
| Baseline fragile — population-weighted | — | 2,391,791 (38.98%) | 2,382,858 (38.84%) | 2,385,673 (38.88%) |
| Baseline fragile — blocks with POP20 > 0 | 14,322 / 55,411 (25.85%) | 14,801 / 55,381 (26.73%) | 14,706 / 55,381 (26.55%) | 14,722 / 55,381 (26.58%) |
| Baseline isolated — blocks | — | 436 (0.64%) | 163 (0.24%) | 142 (0.21%) |
| Baseline isolated — population-weighted | — | 40,532 (0.66%) | 18,126 (0.30%) | 15,311 (0.25%) |
| Baseline inundated — blocks | — | 188 (0.27%) | 188 (0.27%) | 188 (0.27%) |
| Baseline inundated — population-weighted | — | 20,053 (0.33%) | 20,053 (0.33%) | 20,053 (0.33%) |
| Baseline unclassified — blocks | — | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| Baseline unclassified — population-weighted | — | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| Non-inundation pathway at 2 ft | — | 555 / 804 (69.03%) | 575 / 824 (69.78%) | 581 / 830 (70.00%) |
| Non-inundation pathway at 6 ft | — | 8,184 / 21,198 (38.61%) | 7,831 / 20,845 (37.57%) | 7,833 / 20,847 (37.57%) |

The non-inundation pathway denominator is all blocks newly degraded relative to baseline at that SLR level: `new fragile + new isolated + new inundated`. Its numerator is `new fragile + new isolated`.

## 3. Step 2a — grouped-binomial overdispersion

The seven approach-arm models were refit without changing the specification: `fixest::feglm`, binomial family, block-group state-count weights, county and SLR fixed effects, and block-group-clustered point-estimate variance. Pearson dispersion is `sum(Pearson residual^2) / residual df`; residual degrees of freedom came from `fixest::degrees_freedom(model, type = "resid")`. All seven fits converged.

| Transition/outcome | Observations | Block-group clusters | Residual df | Pearson chi-square | Dispersion |
|---|---:|---:|---:|---:|---:|
| Redundant → Fragile | 20,652 | 3,442 | 20,638 | 71,248.142 | 3.452 |
| Redundant → Isolated | 20,652 | 3,442 | 20,638 | 87,065.876 | 4.219 |
| Redundant → Inundated | 20,652 | 3,442 | 20,638 | 98,626.880 | 4.779 |
| Redundant → Worse | 20,652 | 3,442 | 20,638 | 144,125.724 | 6.984 |
| Fragile → Isolated | 19,068 | 3,178 | 19,054 | 34,905.915 | 1.832 |
| Fragile → Inundated | 19,068 | 3,178 | 19,054 | 35,569.665 | 1.867 |
| Fragile → Worse | 19,068 | 3,178 | 19,054 | 47,892.074 | 2.513 |

All values materially exceed 1. Cluster-robust and block-group cluster-bootstrap standard errors partially absorb extra-binomial variation within clusters, but a Pearson dispersion statistic is still standard reporting for grouped-binomial models and should be reported here.

## 4. Step 2b — Moran's I of approach-arm Pearson residuals (resolved)

This diagnostic ran entirely locally in R 4.6.1 with `spdep` 1.4.2, `sf` 1.1.2, and `fixest` 0.14.2; neither `dplyr` nor `tidyr` was used.

The seven approach-arm models were refit with the unchanged demographic-only specification (six standardized demographic covariates). The redundant-risk estimation sample contained the same 3,442 block groups at every SLR level for all four redundant-risk outcomes; the fragile-risk sample contained the same 3,178 block groups at every level for all three fragile-risk outcomes. The full 3,942-unit queen graph had 1 zero-neighbor unit, so the prescribed symmetric k = 6 fallback was used once for the full adjacency. The two estimation-sample graphs were then induced with `subset.nb()` from that full topology rather than rebuilding neighbors from subset geometries. Each Pearson-residual vector was joined by `block_group_geoid` and explicitly reordered to its risk-set `listw` GEOID order before testing.

All seven grouped-binomial fits are overdispersed; live Pearson dispersion values range from 1.832–6.984 (`sum(Pearson residual^2) / residual df`).

**Zero-neighbor islands after risk-set subsetting: redundant-risk = 0; fragile-risk = 0.** Both lists were nevertheless constructed and tested with `zero.policy = TRUE` as specified. The subset graphs contained 4 and 3 connected components, respectively.

The tests use the default one-sided `greater` alternative for positive spatial autocorrelation. “0 (underflow)” means R returned a numerical p-value of exactly zero at double precision.

| Transition/outcome | Risk family | SLR scenario | N | Moran's I | p-value | Significant at 0.05 |
|---|---|---:|---:|---:|---:|---|
| Redundant → Fragile | Redundant | 1 ft | 3,442 | 0.066528 | 6.844e-19 | Yes |
| Redundant → Fragile | Redundant | 2 ft | 3,442 | 0.199016 | 2.500e-101 | Yes |
| Redundant → Fragile | Redundant | 3 ft | 3,442 | 0.331448 | 1.850e-266 | Yes |
| Redundant → Fragile | Redundant | 4 ft | 3,442 | 0.199272 | 8.977e-95 | Yes |
| Redundant → Fragile | Redundant | 5 ft | 3,442 | 0.250002 | 4.202e-147 | Yes |
| Redundant → Fragile | Redundant | 6 ft | 3,442 | 0.201477 | 3.421e-97 | Yes |
| Redundant → Isolated | Redundant | 1 ft | 3,442 | -0.001276 | 0.588050 | No |
| Redundant → Isolated | Redundant | 2 ft | 3,442 | 0.304811 | 2.150e-224 | Yes |
| Redundant → Isolated | Redundant | 3 ft | 3,442 | 0.295421 | 9.533e-207 | Yes |
| Redundant → Isolated | Redundant | 4 ft | 3,442 | 0.340274 | 2.447e-268 | Yes |
| Redundant → Isolated | Redundant | 5 ft | 3,442 | 0.315019 | 4.348e-229 | Yes |
| Redundant → Isolated | Redundant | 6 ft | 3,442 | 0.470790 | 0 (underflow) | Yes |
| Redundant → Inundated | Redundant | 1 ft | 3,442 | 0.017059 | 0.025779 | Yes |
| Redundant → Inundated | Redundant | 2 ft | 3,442 | 0.193631 | 5.539e-106 | Yes |
| Redundant → Inundated | Redundant | 3 ft | 3,442 | 0.346365 | 3.738e-290 | Yes |
| Redundant → Inundated | Redundant | 4 ft | 3,442 | 0.397804 | 0 (underflow) | Yes |
| Redundant → Inundated | Redundant | 5 ft | 3,442 | 0.439700 | 0 (underflow) | Yes |
| Redundant → Inundated | Redundant | 6 ft | 3,442 | 0.535805 | 0 (underflow) | Yes |
| Redundant → Worse | Redundant | 1 ft | 3,442 | 0.026870 | 0.001269 | Yes |
| Redundant → Worse | Redundant | 2 ft | 3,442 | 0.377066 | 0 (underflow) | Yes |
| Redundant → Worse | Redundant | 3 ft | 3,442 | 0.469189 | 0 (underflow) | Yes |
| Redundant → Worse | Redundant | 4 ft | 3,442 | 0.480515 | 0 (underflow) | Yes |
| Redundant → Worse | Redundant | 5 ft | 3,442 | 0.511382 | 0 (underflow) | Yes |
| Redundant → Worse | Redundant | 6 ft | 3,442 | 0.631461 | 0 (underflow) | Yes |
| Fragile → Isolated | Fragile | 1 ft | 3,178 | -0.001600 | 0.571023 | No |
| Fragile → Isolated | Fragile | 2 ft | 3,178 | 0.193474 | 2.481e-77 | Yes |
| Fragile → Isolated | Fragile | 3 ft | 3,178 | 0.323642 | 2.354e-204 | Yes |
| Fragile → Isolated | Fragile | 4 ft | 3,178 | 0.283705 | 5.390e-157 | Yes |
| Fragile → Isolated | Fragile | 5 ft | 3,178 | 0.320747 | 1.714e-199 | Yes |
| Fragile → Isolated | Fragile | 6 ft | 3,178 | 0.344013 | 1.424e-228 | Yes |
| Fragile → Inundated | Fragile | 1 ft | 3,178 | 0.142102 | 2.158e-44 | Yes |
| Fragile → Inundated | Fragile | 2 ft | 3,178 | 0.142851 | 3.244e-43 | Yes |
| Fragile → Inundated | Fragile | 3 ft | 3,178 | 0.245323 | 6.552e-120 | Yes |
| Fragile → Inundated | Fragile | 4 ft | 3,178 | 0.280788 | 5.876e-154 | Yes |
| Fragile → Inundated | Fragile | 5 ft | 3,178 | 0.356144 | 1.251e-244 | Yes |
| Fragile → Inundated | Fragile | 6 ft | 3,178 | 0.455005 | 0 (underflow) | Yes |
| Fragile → Worse | Fragile | 1 ft | 3,178 | 0.136931 | 1.490e-41 | Yes |
| Fragile → Worse | Fragile | 2 ft | 3,178 | 0.223669 | 2.648e-100 | Yes |
| Fragile → Worse | Fragile | 3 ft | 3,178 | 0.347571 | 2.243e-234 | Yes |
| Fragile → Worse | Fragile | 4 ft | 3,178 | 0.393880 | 4.566e-299 | Yes |
| Fragile → Worse | Fragile | 5 ft | 3,178 | 0.477135 | 0 (underflow) | Yes |
| Fragile → Worse | Fragile | 6 ft | 3,178 | 0.580119 | 0 (underflow) | Yes |

40 of the 42 tests reject spatial randomness at 0.05: 23 of 24 in the redundant-risk family and 17 of 18 in the fragile-risk family.
The nonsignificant results are Redundant -> Isolated at 1 ft; Fragile -> Isolated at 1 ft.

## 5. Verdict

**For the `demographic_only` specification, `vcov = block_group_geoid` plus block-group cluster-bootstrap uncertainty approach is not sufficient on its own for a defensible Methods section.** Pearson dispersion across the seven current fits ranges from 1.832–6.984, and 40 of 42 Moran tests show significant positive spatial autocorrelation, including 23 of 24 redundant-risk tests and 17 of 18 fragile-risk tests. Clustering by block group handles repeated observations of the same unit across SLR scenarios, but it does not address dependence between neighboring block groups. The point specification need not change, but the reported uncertainty must be supplemented with a spatially robust procedure before the inferential claims are defensible; documenting spatial dependence only as a limitation is not enough.
