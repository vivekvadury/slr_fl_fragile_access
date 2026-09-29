# Attachment sensitivity: completed results (experiment `attach_20260925`)

Completed 29 September 2026. All six arms were implemented, tested, run locally at 0-6 ft,
verified, and analyzed; all model tables use 199 successful block-group bootstrap
replications (0 failures) under both specifications. The assessment plan in
`00_prespecified_assessment.md` was written before any arm ran; every metric listed
there is reported below with its full range across arms. Paths are relative to this
directory unless stated.

## 1. What was run

| Arm | Origin limit | Facility rule | Access run | Models |
|---|---:|---|---|---|
| reference | 2,000 m | preferred 2ECC node <=1 km, else nearest <=1 km | completed | fit (both specs) |
| origin_1000 | 1,000 m | reference map | completed | fit |
| origin_500 | 500 m | reference map | completed | fit |
| facility_500 | 2,000 m | preferred-then-fallback, both <=500 m | completed | fit |
| facility_add_uncapped | 2,000 m | reference valid fixed; excluded candidates added at nearest node, no cap | completed | reused from reference (identical model input; `models/facility_add_uncapped/REUSED_FROM_REFERENCE.json`) |
| facility_nearest_1000 | 2,000 m | nearest raw-graph node <=1 km, no preference | completed | fit |

Bridge rule `approach` throughout; same inputs (SHA-256 equal to the production Della
manifest), road graph, NOAA extents, facility candidates (2,184 records in the 10-km
buffer), pooling, and eligibility rule.

**Deviations and execution notes.** (i) The machine shut down during the 6 ft union on
2026-09-26; completed 0-5 ft parts were kept (atomic writes) and the run resumed on
2026-09-29. (ii) The 6 ft union used `shapely.coverage_union_all` instead of
`union_all` (hours faster); it was first checked equal at 0 and 5 ft (identical area,
0 mismatches at all 70,695 origins and 1,659,411 nodes; `equivalence_checks/`), and the
6 ft reference then reproduced production exactly. (iii) Facility-independent graph
computations and the per-scenario union were memoized across arms inside one process;
the production classifier (`scenario_results_for_origins`) ran for every arm and
scenario. (iv) No Della run was needed; peak memory 7.35 GiB.

## 2. Verification (`validation/validation_summary.csv`, 30/30 pass)

- Reference reproduces the production access parquet in **every column and all 494,865
  rows** (0-6 ft), the production facility audit (2,190 rows), and the production bridge
  audits at 0-6 ft.
- Rebuilt reference block-group dataset equals `data/processed/analysis/block_group_analysis_dataset_approach.csv`
  in all columns; rerun reference models equal `outputs/tables/ame_bootstrap_results_approach*.xlsx`
  to <=4e-13 in every field (`reference_vs_production_models.csv`).
- Facility-only arms: origin membership, attachment, and inundation identical to reference.
- Add-uncapped: all reference-valid attachments unchanged; destinations a superset; no
  eligible block-scenario worse than reference.
- Origin arms: every commonly retained origin identical to reference in every field;
  eligible sets are subsets. Differences in those arms are therefore coverage only.
- No arm has an eligible block that improves relative to its own baseline, or an
  unclassified eligible block.

## 3. Results by question

### Q1. Origin limits (1,000 m and 500 m)

- Coverage (`eligibility_comparison.csv`): 1,000 m excludes 196 more blocks (7,251 people;
  68,325 eligible, 3,941 block groups); 500 m excludes 777 (52,758 people; 67,744 eligible,
  3,939 block groups). Eligible-origin attachment distance: median 47.8 m, p95 196.7 m,
  p99 535.8 m, max 1,983 m in the reference (`origin_attachment_summary.csv`).
- The excluded origins are not a random subset (`changed_unit_composition.csv`,
  block-level descriptive means): at 500 m they are 53% Palm Beach (vs 25% of all blocks),
  less Hispanic (mean 33% vs 44%), older (22.9% vs 19.1% age 65+), higher income, and
  lower renter share. They are also disproportionately coastal: at 1 ft, the 500 m arm
  counts 3,907 newly inundated residents vs 4,853 in the reference (-19%).
- Baseline fragile share: 25.22% (1,000 m) and 25.06% (500 m) vs 25.30% of blocks;
  38.80% and 38.70% vs 38.84% of population (`state_comparison.csv`).
- Because retained origins are classified identically, all changes in population totals
  and models in these arms reflect coverage.

### Q2. Tighter facility tolerance (500 m)

- 2 schools excluded (Clewiston High School and Harvest Academy, Hendry County, outside
  the study counties); 2 facilities reattached from preferred nodes 583-707 m away to
  fallback nodes 37-62 m away (node displacement 593-679 m): Fisher Island Day School and
  Miami-Dade Fire Rescue Station 42 (`attachment_audit.csv`).
- Effect: blocks in one tract (12086004500, Fisher Island; 1,028 residents) become
  isolated at 3-6 ft: 10 blocks at 3-4 ft, 9 at 5 ft, 8 at 6 ft (at 6 ft, 7 were
  redundant and 1 fragile in the reference). No other block changes at any scenario
  (`state_crosstab_vs_reference.csv`). Fisher Island has no road bridge (ferry access;
  general knowledge, not verified against imagery here). Its reference facility
  attachments therefore cross Government Cut by straight-line snapping, and so do nine of
  its ten block origins (431-1,044 m, `data/.../reference/origin_snap_audit.csv`); neither
  representation is a verified road connection.

### Q3. Admitting excluded facilities without a distance cap

- All 27 reference-excluded candidates are added (17 schools, 10 fire stations), at
  1.1-12.9 km (median 3.1 km) from their nearest node. All lie outside the retained road
  extract (Clewiston and Big Cypress, Indiantown/Hobe Sound/Jupiter Island in Martin
  County, Key Largo and Tavernier in Monroe County), so these attachments are not
  geographically plausible.
- **No block changes state at any scenario**, and the model input is identical to the
  reference. The distance cap on facilities does not affect any reported result.

### Q4. Preferential vs. nearest-node facility attachment (both 1 km)

- 574 of 2,163 facilities (440 schools, 134 fire stations; 26.5%) reattach. Node
  displacement: median 47.6 m, p90 115.7 m, p99 232.3 m, max 679.2 m. Added snap
  distance is negative by construction (nearest is never farther): median -13.7 m; this
  understates displacement, which is reported separately.
- Access changes: 87 blocks (11,962 residents) change baseline state (net +69 fragile,
  -69 redundant; baseline fragile share 25.40% vs 25.30%, population 38.99% vs 38.84%);
  247 blocks (19,544 residents) differ at one or more scenarios. At 6 ft, 67 more blocks
  are isolated than in the reference (facilities on spurs whose only link floods).
- Largest relative population effect in the experiment: at 2 ft, population added
  through fragility is 9,994 vs 12,988 (-23%), because some blocks that become fragile at
  2 ft in the reference are already fragile at baseline under nearest attachment.
  Non-inundation population share at 2 ft: 73.75% vs 75.00%.
- Changed blocks are concentrated in Miami-Dade (65 of 87 baseline changes) and have
  higher mean Hispanic and renter shares than the full sample (descriptive only).

### Q5. Concentration and plausibility

- Changes are localized: Fisher Island (facility_500), out-of-extent candidates
  (add-uncapped, no effect), 87-247 blocks under nearest attachment, and a socially
  patterned set of long-attachment origins (origin arms).
- `spatial_review/attachment_review.gpkg` contains every changed facility attachment,
  every facility attachment >250 m, and every origin attachment >500 m, as lines from
  the point to its node. **Geographic plausibility is unverified** except where stated
  from general knowledge above; no imagery or entrance-level check was performed.

## 4. Prespecified metrics: full ranges across the six arms

Access (`state_comparison.csv`, `summary_ranges_by_scenario.csv`):

| Metric | Reference | Range across arms |
|---|---:|---:|
| Baseline fragile, share of blocks | 25.30% | 25.06%-25.40% |
| Baseline fragile, share of population | 38.84% | 38.70%-38.99% |
| Newly fragile-or-worse population, 2 ft | 63,117 | 60,123-63,117 |
| Newly fragile-or-worse population, 6 ft | 2,048,545 | 2,036,524-2,052,392 |
| Population added by fragility, 6 ft | 90,028 | 89,134-91,362 |
| Non-inundation share of new population, 2 ft | 75.00% | 73.75%-76.63% |
| Non-inundation share of new population, 6 ft | 35.72% | 35.72%-35.84% |
| Non-inundation share of new population, 1 ft | 12.13% | 12.13%-14.64% |

Social associations (`ame_comparison_demographic_only.csv`, `ame_comparison_with_physical.csv`;
6 arms x 7 transitions = 42 AMEs per term and specification):

| Term | Demographic-only: sign; significant (p<0.05) | Physical-adjusted: sign; significant |
|---|---|---|
| Black share | negative 42/42; 42/42 (max p 3.1e-9) | negative 42/42; 42/42 (max p 3.1e-4) |
| Hispanic share | negative 42/42; 42/42 (max p 1.9e-12) | negative 42/42; 42/42 (max p 2.6e-5) |
| Renter share | positive 42/42; 42/42 | positive 42/42; 42/42 (max p 0.006) |
| Age 65+ share | positive 42/42; 42/42 (max p 0.007) | mixed sign (24 positive); **0/42**; every CI includes 0 |
| Log median income | positive 42/42; 20/42 | mixed sign; 0/42 |
| No-vehicle share | mixed sign; 0/42 | mostly positive (36/42); 20/42 |

- Largest absolute AME change vs. reference, any term: 0.19 pp (demographic-only) and
  0.32 pp (physical-adjusted; Hispanic share, origin_500).
- Sign changes (6): all on near-zero, non-significant AMEs (no-vehicle share in
  demographic-only; log income, Redundant->Isolated, in physical-adjusted; all p > 0.77).
- Significance crossings (4): income, Redundant->Isolated, demographic-only (reference
  p=0.059; facility_500 0.045, nearest 0.046); no-vehicle, Fragile->Worse,
  physical-adjusted (reference 0.057; origin_1000 0.049, nearest 0.036). These are
  borderline shifts, not evidence of materially different effects.
- The income and no-vehicle results already vary in significance across transitions in
  the reference itself; they remain secondary under every attachment rule.

## 5. Interpretation

Across the tested attachment choices, the manuscript's regional conclusions are
unchanged: baseline fragility (25.1-25.4% of blocks), the population added through
non-inundation pathways, and the direction and significance of the Black, Hispanic,
renter, and (demographic-only) age associations. The specification dependence of the
age association is itself unchanged: under physical adjustment it is non-significant in
all arms. The tests do not show that every modeled connection is real. Straight-line
attachment misrepresents at least one ferry-only island in every arm, the preferred rule
bypasses genuine single-access facility sites by design (synthetic test), and the
distant origins that a tighter limit would drop are socially and geographically
distinctive, so the choice of origin limit changes who is represented even though it
does not change how retained blocks are classified.

## 6. Observations about the production pipeline (not repaired)

- Duplicate facility ID: `fire_station_FS_PT_1009` (Doral Fire Headquarters) occurs on
  3 candidate records; the production merge on `service_id` expands them to 9 rows
  (2,184 candidates -> 2,190 audit rows). All 9 share one node, so access states are
  unaffected; `n_reachable_services` is inflated by up to 6 in components containing it.
  Preserved identically in every arm, as required.
- Exclusion-reason precedence: a block with a failed snap and zero land area is recorded
  as `origin_snap_failed`, so zero-land-area counts differ across origin arms
  (1,912 / 1,861 / 1,767) with no change in who is eligible for that reason.
- Earlier memo statistic (13.5 m median "added distance") is an added-distance penalty,
  not node displacement; the displacement for the 574 nearest-vs-preferred changes has
  median 47.6 m and maximum 679 m.

## 7. Suggested manuscript text

Methods (replace the VA TO DO paragraph; numbers in Results):

> We treat the 2-km origin and 1-km facility limits as operational tolerances and
> assessed their influence with five alternative specifications, holding all other
> modeling choices fixed: origin limits of 1 km and 500 m; a 500-m facility limit
> applied to both the preferred and fallback searches; admission of every previously
> excluded candidate facility at its nearest network node without a distance limit,
> with all existing attachments unchanged; and nearest-node facility attachment
> without the preference for two-edge-connected components. For each specification we
> reclassified all blocks at 0-6 ft, rebuilt the block-group transition data, and
> re-estimated all seven transition models under both covariate specifications.

Results (each number traceable to the files named in sections 3-4):

> Attachment choices had little effect on the regional results. Across the six
> specifications, the baseline fragile share ranged from 25.1% to 25.4% of blocks, the
> population newly affected by 6 ft ranged from 2.04 to 2.05 million, and the share of
> newly affected residents reached only through fragility or isolation at 6 ft ranged
> from 35.7% to 35.8%. The Black, Hispanic, and renter associations kept their sign and
> significance in every specification and transition, and the older-adult association
> remained significant without physical controls and non-significant with them. The
> largest change in any average marginal effect was 0.32 percentage points. Effects were
> concentrated in specific places: nearest-node attachment changed the baseline state of
> 87 blocks and reduced the population added through fragility at 2 ft by 23%, and a
> 500-m facility limit isolated up to ten blocks (1,028 residents) on Fisher Island,
> which has no road connection. Tightening the origin limit to 500 m removed 777 blocks (52,758 residents)
> that were older, less Hispanic, and more often coastal than the study area as a whole,
> without changing the classification of any retained block.

What needs revision elsewhere: any claim that attachment choices are immaterial should
be limited to the regional totals and associations above; the Fisher Island and
nearest-attachment cases belong in the limitations on straight-line attachment; the
older-adult association should continue to be reported as specification dependent.
