## 4. Step 2b — Moran's I of approach-arm Pearson residuals (with physical covariates)

This diagnostic ran entirely locally in R 4.6.1 with `spdep` 1.4.2, `sf` 1.1.2, and `fixest` 0.14.2; neither `dplyr` nor `tidyr` was used.

The seven approach-arm models were refit with the explicitly alternate with-physical specification (the same six standardized demographic covariates plus standardized mean elevation and drainage distance). The redundant-risk estimation sample contained the same 3,442 block groups at every SLR level for all four redundant-risk outcomes; the fragile-risk sample contained the same 3,178 block groups at every level for all three fragile-risk outcomes. The full 3,942-unit queen graph had 1 zero-neighbor unit, so the prescribed symmetric k = 6 fallback was used once for the full adjacency. The two estimation-sample graphs were then induced with `subset.nb()` from that full topology rather than rebuilding neighbors from subset geometries. Each Pearson-residual vector was joined by `block_group_geoid` and explicitly reordered to its risk-set `listw` GEOID order before testing.

All seven grouped-binomial fits are overdispersed; live Pearson dispersion values range from 1.142–4792.382 (`sum(Pearson residual^2) / residual df`).

**Zero-neighbor islands after risk-set subsetting: redundant-risk = 0; fragile-risk = 0.** Both lists were nevertheless constructed and tested with `zero.policy = TRUE` as specified. The subset graphs contained 4 and 3 connected components, respectively.

The tests use the default one-sided `greater` alternative for positive spatial autocorrelation. “0 (underflow)” means R returned a numerical p-value of exactly zero at double precision.

| Transition/outcome | Risk family | SLR scenario | N | Moran's I | p-value | Significant at 0.05 |
|---|---|---:|---:|---:|---:|---|
| Redundant → Fragile | Redundant | 1 ft | 3,442 | 0.039434 | 1.319e-10 | Yes |
| Redundant → Fragile | Redundant | 2 ft | 3,442 | 0.201248 | 2.282e-102 | Yes |
| Redundant → Fragile | Redundant | 3 ft | 3,442 | 0.411858 | 0 (underflow) | Yes |
| Redundant → Fragile | Redundant | 4 ft | 3,442 | 0.159015 | 5.525e-61 | Yes |
| Redundant → Fragile | Redundant | 5 ft | 3,442 | 0.273676 | 4.937e-177 | Yes |
| Redundant → Fragile | Redundant | 6 ft | 3,442 | 0.034315 | 1.097e-09 | Yes |
| Redundant → Isolated | Redundant | 1 ft | 3,442 | -0.006406 | 0.878225 | No |
| Redundant → Isolated | Redundant | 2 ft | 3,442 | 0.341120 | 3.946e-276 | Yes |
| Redundant → Isolated | Redundant | 3 ft | 3,442 | 0.282020 | 1.035e-192 | Yes |
| Redundant → Isolated | Redundant | 4 ft | 3,442 | 0.334116 | 6.212e-259 | Yes |
| Redundant → Isolated | Redundant | 5 ft | 3,442 | 0.243812 | 2.716e-138 | Yes |
| Redundant → Isolated | Redundant | 6 ft | 3,442 | 0.232506 | 2.870e-147 | Yes |
| Redundant → Inundated | Redundant | 1 ft | 3,442 | -0.001199 | 0.551217 | No |
| Redundant → Inundated | Redundant | 2 ft | 3,442 | 0.060304 | 2.267e-12 | Yes |
| Redundant → Inundated | Redundant | 3 ft | 3,442 | 0.150135 | 3.685e-55 | Yes |
| Redundant → Inundated | Redundant | 4 ft | 3,442 | 0.275624 | 1.526e-177 | Yes |
| Redundant → Inundated | Redundant | 5 ft | 3,442 | 0.266611 | 5.918e-165 | Yes |
| Redundant → Inundated | Redundant | 6 ft | 3,442 | 0.051841 | 8.502e-23 | Yes |
| Redundant → Worse | Redundant | 1 ft | 3,442 | -0.001264 | 0.546390 | No |
| Redundant → Worse | Redundant | 2 ft | 3,442 | 0.110948 | 2.829e-39 | Yes |
| Redundant → Worse | Redundant | 3 ft | 3,442 | 0.323341 | 3.417e-251 | Yes |
| Redundant → Worse | Redundant | 4 ft | 3,442 | 0.205644 | 2.383e-104 | Yes |
| Redundant → Worse | Redundant | 5 ft | 3,442 | 0.284062 | 9.925e-188 | Yes |
| Redundant → Worse | Redundant | 6 ft | 3,442 | 0.000524 | 0.207183 | No |
| Fragile → Isolated | Fragile | 1 ft | 3,178 | -0.005342 | 0.746476 | No |
| Fragile → Isolated | Fragile | 2 ft | 3,178 | 0.224319 | 4.929e-103 | Yes |
| Fragile → Isolated | Fragile | 3 ft | 3,178 | 0.300225 | 2.886e-176 | Yes |
| Fragile → Isolated | Fragile | 4 ft | 3,178 | 0.293424 | 2.548e-167 | Yes |
| Fragile → Isolated | Fragile | 5 ft | 3,178 | 0.268017 | 1.036e-139 | Yes |
| Fragile → Isolated | Fragile | 6 ft | 3,178 | 0.267298 | 2.106e-142 | Yes |
| Fragile → Inundated | Fragile | 1 ft | 3,178 | 0.008568 | 0.062218 | No |
| Fragile → Inundated | Fragile | 2 ft | 3,178 | 0.011003 | 0.033908 | Yes |
| Fragile → Inundated | Fragile | 3 ft | 3,178 | 0.081312 | 9.857e-23 | Yes |
| Fragile → Inundated | Fragile | 4 ft | 3,178 | 0.132451 | 4.720e-38 | Yes |
| Fragile → Inundated | Fragile | 5 ft | 3,178 | 0.212395 | 1.890e-89 | Yes |
| Fragile → Inundated | Fragile | 6 ft | 3,178 | 0.187541 | 1.228e-80 | Yes |
| Fragile → Worse | Fragile | 1 ft | 3,178 | 0.003684 | 0.196170 | No |
| Fragile → Worse | Fragile | 2 ft | 3,178 | 0.030308 | 5.661e-05 | Yes |
| Fragile → Worse | Fragile | 3 ft | 3,178 | 0.123775 | 5.763e-36 | Yes |
| Fragile → Worse | Fragile | 4 ft | 3,178 | 0.139893 | 7.439e-43 | Yes |
| Fragile → Worse | Fragile | 5 ft | 3,178 | 0.261386 | 2.084e-134 | Yes |
| Fragile → Worse | Fragile | 6 ft | 3,178 | 0.013572 | 3.186e-04 | Yes |

35 of the 42 tests reject spatial randomness at 0.05: 20 of 24 in the redundant-risk family and 15 of 18 in the fragile-risk family.
The nonsignificant results are Redundant -> Isolated at 1 ft; Redundant -> Inundated at 1 ft; Redundant -> Worse at 1 ft; Redundant -> Worse at 6 ft; Fragile -> Isolated at 1 ft; Fragile -> Inundated at 1 ft; Fragile -> Worse at 1 ft.

## 5. Verdict

**For the `with_physical` specification, `vcov = block_group_geoid` plus block-group cluster-bootstrap uncertainty approach is not sufficient on its own for a defensible Methods section.** Pearson dispersion across the seven current fits ranges from 1.142–4792.382, and 35 of 42 Moran tests show significant positive spatial autocorrelation, including 20 of 24 redundant-risk tests and 15 of 18 fragile-risk tests. Clustering by block group handles repeated observations of the same unit across SLR scenarios, but it does not address dependence between neighboring block groups. The point specification need not change, but the reported uncertainty must be supplemented with a spatially robust procedure before the inferential claims are defensible; documenting spatial dependence only as a limitation is not enough.
