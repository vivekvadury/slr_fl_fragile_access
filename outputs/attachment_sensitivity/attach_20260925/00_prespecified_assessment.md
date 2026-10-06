# Prespecified assessment plan: attachment sensitivity (experiment `attach_20260925`)

Written 25 September 2026, **before any sensitivity arm was run or any sensitivity
outcome was inspected**. Only the saved production reference run had been read
(inventory and coverage diagnostics). This file is not edited after results exist;
deviations are recorded in the final report instead.

## Arms (bridge rule `approach` in every arm; distinct `attachment_arm` field)

| attachment_arm | origin limit | facility rule |
|---|---:|---|
| reference | 2,000 m | preferred raw-LCC 2ECC(size>=2) node within 1,000 m, else unconstrained nearest within 1,000 m |
| origin_1000 | 1,000 m | reference facility map, unchanged |
| origin_500 | 500 m | reference facility map, unchanged |
| facility_500 | 2,000 m | same preferred-then-fallback algorithm, both capped at 500 m |
| facility_add_uncapped | 2,000 m | reference-valid attachments fixed; reference-excluded finite candidates added at unconstrained nearest raw-graph node, no distance cap |
| facility_nearest_1000 | 2,000 m | every candidate at unconstrained nearest raw-graph node within 1,000 m (no component preference) |

## Manuscript conclusions and metrics to be assessed (all arms, all reported; no pass/fail tolerance)

1. **Baseline fragility**: baseline (0 ft) fragile share of eligible blocks and of eligible
   population; baseline isolated share; reference value 25.30% of blocks.
2. **Added population through non-inundation pathways** (05_population_figures.py definitions,
   baseline-connected origins only), at each of 1-6 ft separately:
   inundation-only population, isolation-or-inundation population, fragility-or-worse
   population, population added by fragility, and the non-inundation pathway share
   (blocks and population).
3. **Five transition totals** (redundant->fragile/isolated/inundated; fragile->isolated/inundated),
   blocks and population, each arm using its own baseline states.
4. **Social associations** under BOTH the demographic-only and physical-adjusted
   specifications: all seven transition models x all six social terms
   (Black share, Hispanic share, renter share, log median income, age 65+ share,
   no-vehicle share). Report sign, AME (probability and percentage points), bootstrap SE,
   percentile 95% interval, p-value, and absolute and relative change vs. reference.
   The age-65+ result is explicitly in scope under both specifications.
5. Coverage: eligible blocks/population, exclusions by reason, facility candidate/valid/
   reattached/fallback/added counts, and geographic/social composition of changed units.

## Interpretation rules fixed in advance

- Full ranges across arms are reported for every metric above, including changes that
  weaken or reverse manuscript claims.
- A p-value crossing 0.05 alone is not described as a material change; absolute AME
  differences are reported alongside relative ones.
- No statement of formal equivalence is made from overlapping intervals.
- Localized changes are reported even when regional totals are stable.
- Origin-only arms: classifications of commonly retained origins are expected to be
  identical; any difference is investigated as a possible defect, not a sensitivity result.
