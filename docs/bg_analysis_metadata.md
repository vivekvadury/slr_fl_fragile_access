# Metadata: `bg_analysis`

This metadata describes `bg_analysis`, saved by
`scripts/03_build_extension_dataset_and_memo.ipynb` as
`data/processed/analysis/block_group_analysis_dataset.csv`.

Generated from the saved dataset and source scripts on 2026-04-07.

## Dataset Scope

- Unit of observation: census block group by SLR scenario.
- Current saved shape: 27,622 rows and 43 columns.
- Spatial scope: Broward, Miami-Dade, and Palm Beach counties, Florida.
- Scenario scope: `slr_ft` values 0 through 6. The `slr_ft == 0` rows are the
  0 ft NOAA SLR layer and are also used as the reference scenario for the
  transition fields.
- Block-group access measures are unweighted block-count aggregates. A block
  group with many small blocks and a block group with a few large blocks are not
  population-weighted unless you add weights downstream.

## Access-Model Assumptions

- Origins are 2020 Census block centroids. A block is treated according to its
  centroid, not according to the full block polygon or residential population
  distribution within the block.
- The road graph is built from retained OSM drivable road segments. It is
  undirected, private-access edges are dropped, and turn restrictions, traffic,
  and one-way rules are not modeled.
- Roads are segmentized using source vertices, but the workflow does not fully
  planarize road crossings. If two road geometries cross without a shared source
  vertex, the graph does not create a new intersection there.
- A road segment is removed from a scenario's dry graph if its geometry
  intersects that scenario's NOAA inundation polygon. Bridges, tunnels, and
  vertical separation are not modeled.
- Services are public primary schools, private primary schools, and fire
  stations. Services must survive the service filtering and snapping steps
  before they count as reachable services.
- Block origins are snapped to the nearest graph node, with invalid origin snaps
  treated as access failures unless the block centroid is inundated. Services
  are also snapped to graph nodes before reachability is computed.
- Status classes are mutually exclusive with this precedence:
  inundated, isolated, redundant, fragile. A centroid-inundated block is
  classified as inundated even if its snapped network node would otherwise be
  connected.
- Redundancy uses the upstream capped path proxy
  `MAX_EDGE_DISJOINT_PATHS_CAP = 2`. The model distinguishes 0, 1, and at least
  2 edge-disjoint dry paths, but it cannot distinguish 2 paths from 3 or more.
- `path_inflation_ratio` is conditional on blocks that remain connected and not
  inundated; isolated and inundated blocks are coded `NaN` before the
  block-group mean and median are calculated.
- Demographic variables come from 2022 ACS 5-year block-group estimates. Census
  suppression code `-666666666` is recoded to missing. The current dataset does
  not include ACS margins of error.
- `poverty_rate` is structurally unusable in the current block-group dataset:
  2022 ACS 5-year `B17001` poverty estimate fields returned missing for all
  Broward, Miami-Dade, and Palm Beach block groups.

## Field Metadata

### Identifiers And Scenario Fields

| Field | What it measures | How it was calculated and assumptions |
|---|---|---|
| `block_group_geoid` | 12-character Census block group GEOID. | Derived from the 2020 Census block GEOID as state + county + tract + block group, and used as the merge key to ACS block-group data. Stored as a zero-padded string. |
| `county_fips` | Three-character county FIPS code. | Derived from the block geography or block GEOID and zero-padded. In this dataset: `011`, `086`, `099`. |
| `county_name` | County name. | Mapped from `county_fips` using the study-area mapping in `02_access_flags.py`: Broward, Miami-Dade, Palm Beach. |
| `tract_geoid` | 11-character Census tract GEOID. | Derived from the 2020 Census block GEOID as state + county + tract and used as a grouping key during aggregation. |
| `slr_ft` | Sea-level-rise scenario in feet. | Scenario key from NOAA SLR layers: 0 through 6 ft. The 0 ft scenario is included and is also used as the reference status for transition fields. |

### Block-Count Aggregates

These fields are counts of Census blocks within each block group and SLR
scenario unless otherwise noted.

| Field | What it measures | How it was calculated and assumptions |
|---|---|---|
| `block_centroid_inundated` | Count of blocks whose centroid is inundated in the scenario. | Sum of the block-level `block_centroid_inundated` flag. The flag is 1 when the block centroid intersects the scenario's NOAA SLR polygon. This is all inundation in the scenario, not only new inundation relative to baseline. |
| `block_centroid_isolated` | Count of blocks that are not inundated but cannot reach any retained essential service on the scenario dry graph. | Sum of the block-level `block_centroid_isolated` flag. A block is isolated if its centroid is not inundated and either the origin snap is invalid or its dry-road connected component has zero reachable services. |
| `block_centroid_fragile` | Count of connected, non-inundated blocks with exactly one capped dry path to reachable services. | Sum of the block-level `block_centroid_fragile` flag. This includes all fragile blocks in the scenario, including blocks already fragile at baseline. It is not limited to blocks made fragile by SLR. |
| `block_centroid_redundant` | Count of connected, non-inundated blocks with at least two capped dry paths to reachable services. | Sum of the block-level `block_centroid_redundant` flag. Because the path count is capped at 2, this means 2 or more; the dataset cannot distinguish 2-path from 3-plus-path redundancy. |
| `fragile_or_worse` | Count of blocks classified as fragile, isolated, or inundated in the scenario. | Derived in the notebook as `scenario_status in ["fragile", "isolated", "inundated"]`, then summed by block group and scenario. This includes persistent baseline fragility as well as SLR-related degradation. |
| `any_loss_of_redundancy` | Count of blocks that were redundant in the 0 ft reference scenario and are fragile, isolated, or inundated in the current scenario. | Derived as `baseline_block_centroid_redundant == 1` and current `scenario_status` in fragile, isolated, or inundated. This captures loss from baseline redundancy to any worse category, but it does not count baseline-fragile blocks that become isolated or inundated. |
| `new_fragile_due_to_slr` | Count of blocks that move from baseline redundant to scenario fragile. | Created upstream for positive SLR scenarios as `slr_ft != 0`, baseline redundant, and current fragile. This is the narrow "newly fragile because of SLR scenario" measure. It excludes blocks that were already fragile at baseline. |
| `new_isolated_due_to_slr` | Count of blocks that were not isolated at baseline and are isolated in the current scenario. | Created upstream for positive SLR scenarios as `slr_ft != 0`, baseline isolated equals 0, and current isolated equals 1. This is not restricted to baseline-redundant blocks. |
| `new_inundated_due_to_slr` | Count of blocks that were not inundated at baseline and are inundated in the current scenario. | Created upstream for positive SLR scenarios as `slr_ft != 0`, baseline inundated equals 0, and current inundated equals 1. |
| `baseline_redundant_to_fragile` | Count of blocks that were baseline redundant and are scenario fragile. | Derived in the notebook from baseline redundant and current fragile block flags. For positive SLR scenarios it should align with `new_fragile_due_to_slr`; it is structurally zero in the 0 ft reference row because statuses are mutually exclusive. |
| `baseline_redundant_to_isolated` | Count of blocks that were baseline redundant and are scenario isolated. | Derived in the notebook from baseline redundant and current isolated block flags. This is a specific transition count from redundancy to isolation. |
| `baseline_redundant_to_inundated` | Count of blocks that were baseline redundant and are scenario inundated. | Derived in the notebook from baseline redundant and current inundated block flags. This is a specific transition count from redundancy to inundation. |
| `baseline_fragile_to_isolated` | Count of blocks that were baseline fragile and are scenario isolated. | Derived in the notebook from baseline fragile and current isolated block flags. This captures deterioration among already-fragile baseline blocks. |
| `baseline_fragile_to_inundated` | Count of blocks that were baseline fragile and are scenario inundated. | Derived in the notebook from baseline fragile and current inundated block flags. This captures inundation among already-fragile baseline blocks. |
| `total_blocks` | Number of Census blocks in the block group for the scenario row. | Count of `block_geoid` within `block_group_geoid`, `county_fips`, `county_name`, `tract_geoid`, and `slr_ft`. This is the denominator for the share fields. It is an unweighted block count, not population. |
| `mean_max_edge_disjoint_paths` | Mean capped dry-path redundancy across blocks in the block group. | Mean of block-level `max_edge_disjoint_paths_any_service`. Isolated and inundated blocks are coded 0 upstream, fragile blocks are 1, and redundant blocks are 2 because of the cap. |
| `median_max_edge_disjoint_paths` | Median capped dry-path redundancy across blocks in the block group. | Median of block-level `max_edge_disjoint_paths_any_service`, with the same 0/1/2 capped interpretation as the mean. |
| `mean_path_inflation_ratio` | Mean dry-path inflation ratio among connected, non-inundated blocks in the block group. | Mean of `path_inflation_ratio`, which is the scenario dry shortest-path distance divided by the baseline shortest-path distance. Before aggregation, the notebook sets this ratio to missing for isolated and inundated blocks, so this is conditional on blocks with a dry route. |
| `median_path_inflation_ratio` | Median dry-path inflation ratio among connected, non-inundated blocks in the block group. | Median of `path_inflation_ratio`, after isolated and inundated blocks are set to missing. It describes the connected subset, not the full block group. |

### Block-Share Aggregates

All share fields divide the corresponding count by `total_blocks`. The
denominator is all Census blocks in the block group, not population and not the
number of baseline-redundant blocks.

| Field | What it measures | How it was calculated and assumptions |
|---|---|---|
| `share_inundated` | Share of blocks in the block group whose centroid is inundated in the scenario. | `block_centroid_inundated / total_blocks`. This is all scenario inundation, not only new inundation. |
| `share_isolated` | Share of blocks in the block group that are isolated in the scenario. | `block_centroid_isolated / total_blocks`. Excludes inundated blocks because the status flags are mutually exclusive. |
| `share_fragile` | Share of blocks in the block group that are fragile in the scenario. | `block_centroid_fragile / total_blocks`. This includes all currently fragile blocks, including those already fragile at baseline. For newly fragile baseline-redundant blocks, use `share_new_fragile`. |
| `share_redundant` | Share of blocks in the block group that are redundant in the scenario. | `block_centroid_redundant / total_blocks`. Redundant means at least two capped edge-disjoint dry paths under the scenario graph. |
| `share_fragile_or_worse` | Share of blocks in the block group that are fragile, isolated, or inundated in the scenario. | `fragile_or_worse / total_blocks`. This is a broad current-condition exposure measure, not a new-exposure-only measure. |
| `share_lost_redundancy` | Share of all blocks in the block group that lost baseline redundancy. | `any_loss_of_redundancy / total_blocks`. The numerator is baseline-redundant blocks that are now fragile, isolated, or inundated. The denominator remains all blocks, not only baseline-redundant blocks. |
| `share_new_fragile` | Share of all blocks in the block group that became fragile from a baseline-redundant state. | `new_fragile_due_to_slr / total_blocks`. This is the narrow new-fragility measure. It excludes persistent baseline-fragile blocks. |
| `share_new_isolated` | Share of all blocks in the block group that became isolated relative to baseline. | `new_isolated_due_to_slr / total_blocks`. The numerator is not limited to baseline-redundant blocks. |
| `share_new_inundated` | Share of all blocks in the block group that became inundated relative to baseline. | `new_inundated_due_to_slr / total_blocks`. This is new inundation relative to the 0 ft reference status. |

### ACS Demographic Fields

These fields are 2022 ACS 5-year block-group estimates merged on
`block_group_geoid`. They are repeated for every SLR scenario row for the same
block group.

| Field | What it measures | How it was calculated and assumptions |
|---|---|---|
| `total_pop` | ACS total population estimate for the block group. | `B01001_001E`, converted to numeric. Census suppression code `-666666666` is recoded to missing. |
| `median_income` | ACS median household income estimate for the block group. | `B19013_001E`, converted to numeric and suppression-recoded to missing. Some block groups can be missing because of ACS suppression or data availability. |
| `median_age` | ACS median age estimate for the block group. | `B01002_001E`, converted to numeric and suppression-recoded to missing. |
| `pct_white_nh` | Share of race/ethnicity total population that is non-Hispanic White alone. | `B03002_003E / B03002_001E`. This is an ACS estimate ratio; no ACS margin-of-error propagation is included in the current dataset. |
| `pct_black_nh` | Share of race/ethnicity total population that is non-Hispanic Black alone. | `B03002_004E / B03002_001E`. This is an ACS estimate ratio. |
| `pct_hispanic` | Share of race/ethnicity total population that is Hispanic or Latino. | `B03002_012E / B03002_001E`. This is an ACS estimate ratio. |
| `pct_nonwhite` | Share of population that is not non-Hispanic White alone. | `1.0 - pct_white_nh`. This includes Hispanic residents and all non-Hispanic race categories other than White alone. |
| `renter_share` | Share of occupied housing units that are renter occupied. | `B25003_003E / B25003_001E`. The denominator is occupied housing units, not people or all housing units. |
| `poverty_rate` | Intended ACS poverty share for the block group. | `B17001_002E / B17001_001E`. In the current South Florida block-group ACS pull, both the numerator and denominator are missing for all rows, so this field is empty throughout the saved dataset. Use a tract-level poverty merge or another available proxy before including poverty in regressions. |
| `log_median_income` | Natural log of median household income. | `np.log(median_income.clip(lower=1))`. Values below 1 would be clipped to 1 before logging. Missing median income remains missing. |

