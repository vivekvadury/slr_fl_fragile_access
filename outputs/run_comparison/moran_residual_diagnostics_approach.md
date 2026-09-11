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
