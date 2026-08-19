# Phase 3 — Bridge-rule robustness

> **FLAG — model inference is arm-dependent.** Across the 42 transition–covariate
> AMEs, one sign changes and six p<0.05 classifications change across arms. Every
> difference is between the pessimistic `intersect` arm and the other two arms;
> `approach` and `retain` agree on all 42 signs and significance classifications.
> The sole sign flip is non-significant in every arm. The Black/Hispanic results do
> **not** move: all 14 racial-composition AMEs are negative and p<0.05 in every arm,
> so the RQ3 reversal survives the bracket. Source:
> `reports/03_ame_flag_summary.csv` and
> `reports/03_ame_all_arm_comparison.csv`.

> **DEFERRED-AUDIT TRIGGER MET.** Blocks that differ between `approach` and
> `intersect` are demographically distinctive under the prespecified criterion.
> The opposite-side/dry-surface-component validity audit is therefore no longer
> optional before finalizing the bridge-rule interpretation. Source:
> `reports/03_deferred_dry_landing_trigger_assessment.csv`.

## Inputs and validation

All three models use the identical eligible key universe: 68,521 blocks and
6,135,688 people at each of seven SLR levels. The three key sets, population, and
block-group mappings match exactly; all 13 support checks pass. Source:
`reports/03_support_validation.csv`.

The AME comparison uses **49 successful bootstrap replications** for each of seven
transitions and six covariates in every arm. Each workbook has 42 unique rows,
`n_boot = 49`, and `n_boot_fail = 0`; the joined long table has 126 unique
transition–covariate–arm rows. Source:
`reports/03_ame_all_arm_comparison.csv` and
`outputs/tables/ame_bootstrap_results_intersect.xlsx`,
`outputs/tables/ame_bootstrap_results_approach.xlsx`, and
`outputs/tables/ame_bootstrap_results_retain.xlsx`.

The complete-covariate sample contains 3,604 block groups in every arm. The
positive baseline-risk filters retain 3,413 redundant-risk and 3,184 fragile-risk
groups under `intersect`, versus 3,442 and 3,178 under both `approach` and
`retain`. Sources: `outputs/tables/transition_sample_diagnostics_intersect.csv`,
`outputs/tables/transition_sample_diagnostics_approach.csv`, and
`outputs/tables/transition_sample_diagnostics_retain.csv`.

## Block-level disagreement

“Approach better” means the approach state is less adverse in the ordering
inundated < isolated < fragile < redundant; eligible `unclassified` counts are
zero in every arm. The complete 5×5 state tables are in
`reports/03_status_crosstabs.csv`.

| SLR (ft) | Approach vs intersect: changed blocks | Population | Approach better / worse | Approach vs retain: changed blocks | Population | Approach better / worse |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 744 | 51,526 | 732 / 12 | 21 | 2,815 | 0 / 21 |
| 1 | 744 | 50,980 | 732 / 12 | 21 | 2,815 | 0 / 21 |
| 2 | 666 | 43,305 | 654 / 12 | 15 | 2,731 | 0 / 15 |
| 3 | 545 | 46,149 | 535 / 10 | 14 | 3,781 | 0 / 14 |
| 4 | 558 | 26,220 | 548 / 10 | 11 | 1,951 | 0 / 11 |
| 5 | 694 | 30,158 | 686 / 8 | 10 | 1,004 | 0 / 10 |
| 6 | 681 | 24,088 | 673 / 8 | 9 | 652 | 0 / 9 |

Relative to `approach`, `intersect` changes 545–744 blocks per level
(0.80%–1.09%) and 24,088–51,526 people (0.39%–0.84%). `retain` changes only
9–21 blocks (0.013%–0.031%) and 652–3,781 people (0.011%–0.062%). Nearly all
`intersect` disagreements put `approach` in the better state, while every
`retain` disagreement puts `approach` in the worse state, as expected for the
pessimistic and optimistic bounds. Source:
`reports/03_status_disagreement_summary.csv`.

## Are the disagreeing blocks demographically distinctive?

The comparison covers all six prespecified block-group measures—Black and
Hispanic shares, renter share, log median income, age 65+ share, and no-vehicle
share—using one-block/one-weight descriptive means, medians, standard deviations,
and standardized mean differences (SMDs). It reports missingness explicitly and
does not attach pseudo-replication p-values. The full 84-row comparison is in
`reports/03_disagreement_demographic_comparison.csv`.

For `approach` versus `intersect`, the differing subset satisfies the criterion
|SMD| >= 0.20 with at least 100 nonmissing differing blocks at every SLR level:

| SLR (ft) | Differing blocks | Largest SMD variable | SMD | Covariates with abs(SMD) >= 0.20 |
|---:|---:|---|---:|---:|
| 0 | 744 | Log median income | +0.729 | 5 |
| 1 | 744 | Log median income | +0.713 | 5 |
| 2 | 666 | Hispanic share | -0.726 | 5 |
| 3 | 545 | Hispanic share | -0.727 | 6 |
| 4 | 558 | Black share | -0.287 | 2 |
| 5 | 694 | Renter share | +0.223 | 1 |
| 6 | 681 | Black share | -0.329 | 4 |

The `approach`–`retain` differences involve only 9–21 blocks per level, so they do
not meet the prespecified minimum of 100 even when their raw SMDs are large. Thus
the demographic concern is specifically the pessimistic geometric-removal arm,
not the optimistic bound. Source:
`reports/03_demographic_distinctiveness_summary.csv` and
`reports/03_deferred_dry_landing_trigger_assessment.csv`.

## Do the models agree?

The full result for every transition, covariate, and arm—including estimate,
sign, p-value, p<0.05 classification, and approach-versus-sensitivity change—is in
`reports/03_ame_all_arm_comparison.csv`. Seven transition–covariate pairs require
attention:

| Transition | Covariate | Intersect estimate (p) | Approach estimate (p) | Retain estimate (p) | Flag |
|---|---|---:|---:|---:|---|
| Redundant → Fragile | Log median income | +0.001609 (0.1830) | +0.002512 (0.0070) | +0.002505 (0.0071) | Significant in approach/retain only |
| Redundant → Isolated | Log median income | +0.002267 (0.1412) | +0.003871 (0.0398) | +0.003877 (0.0390) | Significant in approach/retain only |
| Redundant → Isolated | No-vehicle share | -0.002617 (0.0428) | -0.002383 (0.1546) | -0.002388 (0.1536) | Significant in intersect only |
| Redundant → Worse | No-vehicle share | -0.001069 (0.7667) | +0.000126 (0.9788) | +0.000126 (0.9788) | Sign flip; non-significant throughout |
| Fragile → Isolated | Log median income | +0.007774 (0.1072) | +0.008461 (0.0093) | +0.008374 (0.0099) | Significant in approach/retain only |
| Fragile → Isolated | No-vehicle share | -0.005090 (0.0067) | -0.001726 (0.3500) | -0.001700 (0.3558) | Significant in intersect only |
| Fragile → Worse | Log median income | +0.013161 (0.1140) | +0.016151 (0.0043) | +0.016290 (0.0040) | Significant in approach/retain only |

Four positive log-income AMEs that are significant under `approach` and `retain`
are not significant under `intersect`; two negative no-vehicle AMEs become
significant only under `intersect`. The only sign flip is the near-zero,
non-significant no-vehicle AME for Redundant → Worse. No sign or p<0.05
classification differs between `approach` and `retain`. Source:
`reports/03_ame_flag_summary.csv`.

### RQ3 racial-composition result

The result is unambiguous: all 14 Black/Hispanic transition AMEs are negative and
p<0.05 under all three bridge rules. Across the 42 arm-specific racial AMEs,
estimates range from -0.07378 to -0.006422 and the largest p-value is
1.82e-7. There is no racial-composition sign flip or significance loss. The
South Florida reversal central to RQ3 therefore survives the bridge-rule bracket;
it is not an artifact of the approach rule. Source:
`reports/03_ame_all_arm_comparison.csv` and
`reports/03_ame_flag_summary.csv`.

## Does RQ1 survive?

Non-inundation means newly fragile plus newly isolated, divided by all newly
fragile, isolated, or inundated blocks (and analogously for population):

| Arm | 2 ft block share | 2 ft population share | 6 ft block share | 6 ft population share |
|---|---:|---:|---:|---:|
| Intersect | 69.03% | 70.11% | 38.61% | 35.79% |
| Approach | 69.78% | 73.08% | 37.57% | 35.55% |
| Retain | 70.00% | 73.12% | 37.57% | 35.55% |

The qualitative RQ1 pattern holds in every arm: non-inundation pathways account
for roughly 69%–70% of newly affected blocks at 2 ft and decline to roughly
38%–39% at 6 ft. The population-weighted share similarly declines from
70%–73% to about 36%. Denominators are identical across arms and the three new
outcome flags have zero overlap. Source:
`reports/03_non_inundation_pathway_by_arm.csv`.

## Verdict

For the headline RQ1 pathway result and the RQ3 racial-composition reversal, the
bridge rule is a **magnitude question**, not a conclusion question: both survive
the full pessimistic-to-optimistic bracket. It is **not purely a magnitude
question for every secondary social result**, because the pessimistic `intersect`
arm changes six income/no-vehicle p<0.05 classifications. The default `approach`
and optimistic `retain` arms are inferentially identical, while the blocks moved
by `intersect` are demographically distinctive. The central paper claims survive,
but secondary income/no-vehicle claims must be qualified and the triggered
dry-landing validity audit must be completed before treating the pessimistic-arm
difference as resolved. Sources: `reports/03_ame_flag_summary.csv`,
`reports/03_non_inundation_pathway_by_arm.csv`,
`reports/03_demographic_distinctiveness_summary.csv`, and
`reports/03_deferred_dry_landing_trigger_assessment.csv`.
