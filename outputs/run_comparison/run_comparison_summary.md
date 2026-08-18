# Bridge-rule run comparison

## Run identification

| bridge_rule | detected_rule | run_directory | result_files | unique_blocks | slr_levels |
| --- | --- | --- | --- | --- | --- |
| intersect | intersect | C:\Users\Vivek\Dropbox\repos\slr_fl_fragile_access\data\processed\access\edited\della_runs\corrected_intersect | 1 | 70695 | 0,1,2,3,4,5,6 |
| approach | approach | C:\Users\Vivek\Dropbox\repos\slr_fl_fragile_access\data\processed\access\edited\della_runs\corrected_canonical | 1 | 70695 | 0,1,2,3,4,5,6 |
| retain | retain | C:\Users\Vivek\Dropbox\repos\slr_fl_fragile_access\data\processed\access\edited\della_runs\corrected_retain | 1 | 70695 | 0,1,2,3,4,5,6 |

## Verdicts

| check_group | verdict |
| --- | --- |
| A | PASS |
| B | FAIL |
| C | PASS |
| D | PASS |
| E | REVIEW |
| F | PASS |

## Group A — Comparability (PASS)

| record_type | bridge_rule | check | value | detail | verdict | slr_ft | unclassified_count | also_flagged_inundated_count | unclassified_pop20 | unclassified_analysis_eligible_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run_summary | intersect | unique_blocks | 70695 |  | PASS |  |  |  |  |  |
| run_summary | intersect | slr_levels | 7 | 0,1,2,3,4,5,6 | PASS |  |  |  |  |  |
| run_summary | intersect | duplicate_block_slr_rows | 0 |  | PASS |  |  |  |  |  |
| status_reassignment | intersect | scenario_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 0 |  |  |  |  |
| status_reassignment | intersect | scenario_status_unclassified_to_inundated | 106 | input label overridden from flag precedence | REVIEW | 1 |  |  |  |  |
| status_reassignment | intersect | scenario_status_unclassified_to_inundated | 112 | input label overridden from flag precedence | REVIEW | 2 |  |  |  |  |
| status_reassignment | intersect | scenario_status_unclassified_to_inundated | 120 | input label overridden from flag precedence | REVIEW | 3 |  |  |  |  |
| status_reassignment | intersect | scenario_status_unclassified_to_inundated | 121 | input label overridden from flag precedence | REVIEW | 4 |  |  |  |  |
| status_reassignment | intersect | scenario_status_unclassified_to_inundated | 130 | input label overridden from flag precedence | REVIEW | 5 |  |  |  |  |
| status_reassignment | intersect | scenario_status_unclassified_to_inundated | 147 | input label overridden from flag precedence | REVIEW | 6 |  |  |  |  |
| status_reassignment | intersect | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 0 |  |  |  |  |
| status_reassignment | intersect | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 1 |  |  |  |  |
| status_reassignment | intersect | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 2 |  |  |  |  |
| status_reassignment | intersect | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 3 |  |  |  |  |
| status_reassignment | intersect | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 4 |  |  |  |  |
| status_reassignment | intersect | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 5 |  |  |  |  |
| status_reassignment | intersect | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 6 |  |  |  |  |
| assertion | intersect | five_state_partition_within_analysis_universe | 0 | effective inundated > unclassified > isolated > redundant > fragile states must sum to 1 | PASS |  |  |  |  |  |
| descriptive | intersect | five_state_partition_violations_outside_analysis_universe | 0 | outside analysis_eligible & pop20 > 0 & land_area_m2 > 0 | PASS |  |  |  |  |  |
| assertion | intersect | five_state_baseline_partition_within_analysis_universe | 0 | recomputed baseline state must be exactly one of five states | PASS |  |  |  |  |  |
| descriptive | intersect | five_state_baseline_partition_violations_outside_analysis_universe | 0 | outside analysis_eligible & pop20 > 0 & land_area_m2 > 0 | PASS |  |  |  |  |  |
| unclassified_descriptive | intersect | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 0 | 262 | 91 | 1848 | 0 |
| unclassified_descriptive | intersect | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 1 | 262 | 106 | 1848 | 0 |
| unclassified_descriptive | intersect | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 2 | 262 | 112 | 1848 | 0 |
| unclassified_descriptive | intersect | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 3 | 262 | 120 | 1848 | 0 |
| unclassified_descriptive | intersect | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 4 | 262 | 121 | 1848 | 0 |
| unclassified_descriptive | intersect | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 5 | 262 | 130 | 1848 | 0 |
| unclassified_descriptive | intersect | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 6 | 262 | 147 | 1848 | 0 |
| analysis_universe_descriptive | intersect | baseline_blocks_total | 70695 | zero-land and zero-population exclusions reported separately | PASS | 0 |  |  |  |  |
| analysis_universe_descriptive | intersect | baseline_zero_land_area_blocks | 2019 | zero-land and zero-population exclusions reported separately | PASS | 0 |  |  |  |  |
| analysis_universe_descriptive | intersect | baseline_zero_population_blocks | 15284 | zero-land and zero-population exclusions reported separately | PASS | 0 |  |  |  |  |
| analysis_universe_descriptive | intersect | baseline_zero_land_and_zero_population_blocks | 2002 | zero-land and zero-population exclusions reported separately | PASS | 0 |  |  |  |  |
| analysis_universe_descriptive | intersect | baseline_analysis_eligible_before_population_filter | 68521 | zero-land and zero-population exclusions reported separately | PASS | 0 |  |  |  |  |
| analysis_universe_descriptive | intersect | baseline_analysis_universe_after_population_filter | 55381 | zero-land and zero-population exclusions reported separately | PASS | 0 |  |  |  |  |
| run_summary | approach | unique_blocks | 70695 |  | PASS |  |  |  |  |  |
| run_summary | approach | slr_levels | 7 | 0,1,2,3,4,5,6 | PASS |  |  |  |  |  |
| run_summary | approach | duplicate_block_slr_rows | 0 |  | PASS |  |  |  |  |  |
| status_reassignment | approach | scenario_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 0 |  |  |  |  |
| status_reassignment | approach | scenario_status_unclassified_to_inundated | 106 | input label overridden from flag precedence | REVIEW | 1 |  |  |  |  |
| status_reassignment | approach | scenario_status_unclassified_to_inundated | 112 | input label overridden from flag precedence | REVIEW | 2 |  |  |  |  |
| status_reassignment | approach | scenario_status_unclassified_to_inundated | 120 | input label overridden from flag precedence | REVIEW | 3 |  |  |  |  |
| status_reassignment | approach | scenario_status_unclassified_to_inundated | 121 | input label overridden from flag precedence | REVIEW | 4 |  |  |  |  |
| status_reassignment | approach | scenario_status_unclassified_to_inundated | 130 | input label overridden from flag precedence | REVIEW | 5 |  |  |  |  |
| status_reassignment | approach | scenario_status_unclassified_to_inundated | 147 | input label overridden from flag precedence | REVIEW | 6 |  |  |  |  |
| status_reassignment | approach | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 0 |  |  |  |  |
| status_reassignment | approach | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 1 |  |  |  |  |
| status_reassignment | approach | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 2 |  |  |  |  |
| status_reassignment | approach | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 3 |  |  |  |  |
| status_reassignment | approach | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 4 |  |  |  |  |
| status_reassignment | approach | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 5 |  |  |  |  |
| status_reassignment | approach | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 6 |  |  |  |  |
| assertion | approach | five_state_partition_within_analysis_universe | 0 | effective inundated > unclassified > isolated > redundant > fragile states must sum to 1 | PASS |  |  |  |  |  |
| descriptive | approach | five_state_partition_violations_outside_analysis_universe | 0 | outside analysis_eligible & pop20 > 0 & land_area_m2 > 0 | PASS |  |  |  |  |  |
| assertion | approach | five_state_baseline_partition_within_analysis_universe | 0 | recomputed baseline state must be exactly one of five states | PASS |  |  |  |  |  |
| descriptive | approach | five_state_baseline_partition_violations_outside_analysis_universe | 0 | outside analysis_eligible & pop20 > 0 & land_area_m2 > 0 | PASS |  |  |  |  |  |
| unclassified_descriptive | approach | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 0 | 262 | 91 | 1848 | 0 |
| unclassified_descriptive | approach | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 1 | 262 | 106 | 1848 | 0 |
| unclassified_descriptive | approach | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 2 | 262 | 112 | 1848 | 0 |
| unclassified_descriptive | approach | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 3 | 262 | 120 | 1848 | 0 |
| unclassified_descriptive | approach | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 4 | 262 | 121 | 1848 | 0 |
| unclassified_descriptive | approach | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 5 | 262 | 130 | 1848 | 0 |
| unclassified_descriptive | approach | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 6 | 262 | 147 | 1848 | 0 |
| analysis_universe_descriptive | approach | baseline_blocks_total | 70695 | zero-land and zero-population exclusions reported separately | PASS | 0 |  |  |  |  |
| analysis_universe_descriptive | approach | baseline_zero_land_area_blocks | 2019 | zero-land and zero-population exclusions reported separately | PASS | 0 |  |  |  |  |
| analysis_universe_descriptive | approach | baseline_zero_population_blocks | 15284 | zero-land and zero-population exclusions reported separately | PASS | 0 |  |  |  |  |
| analysis_universe_descriptive | approach | baseline_zero_land_and_zero_population_blocks | 2002 | zero-land and zero-population exclusions reported separately | PASS | 0 |  |  |  |  |
| analysis_universe_descriptive | approach | baseline_analysis_eligible_before_population_filter | 68521 | zero-land and zero-population exclusions reported separately | PASS | 0 |  |  |  |  |
| analysis_universe_descriptive | approach | baseline_analysis_universe_after_population_filter | 55381 | zero-land and zero-population exclusions reported separately | PASS | 0 |  |  |  |  |
| run_summary | retain | unique_blocks | 70695 |  | PASS |  |  |  |  |  |
| run_summary | retain | slr_levels | 7 | 0,1,2,3,4,5,6 | PASS |  |  |  |  |  |
| run_summary | retain | duplicate_block_slr_rows | 0 |  | PASS |  |  |  |  |  |
| status_reassignment | retain | scenario_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 0 |  |  |  |  |
| status_reassignment | retain | scenario_status_unclassified_to_inundated | 106 | input label overridden from flag precedence | REVIEW | 1 |  |  |  |  |
| status_reassignment | retain | scenario_status_unclassified_to_inundated | 112 | input label overridden from flag precedence | REVIEW | 2 |  |  |  |  |
| status_reassignment | retain | scenario_status_unclassified_to_inundated | 120 | input label overridden from flag precedence | REVIEW | 3 |  |  |  |  |
| status_reassignment | retain | scenario_status_unclassified_to_inundated | 121 | input label overridden from flag precedence | REVIEW | 4 |  |  |  |  |
| status_reassignment | retain | scenario_status_unclassified_to_inundated | 130 | input label overridden from flag precedence | REVIEW | 5 |  |  |  |  |
| status_reassignment | retain | scenario_status_unclassified_to_inundated | 147 | input label overridden from flag precedence | REVIEW | 6 |  |  |  |  |
| status_reassignment | retain | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 0 |  |  |  |  |
| status_reassignment | retain | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 1 |  |  |  |  |
| status_reassignment | retain | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 2 |  |  |  |  |
| status_reassignment | retain | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 3 |  |  |  |  |
| status_reassignment | retain | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 4 |  |  |  |  |
| status_reassignment | retain | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 5 |  |  |  |  |
| status_reassignment | retain | baseline_status_unclassified_to_inundated | 91 | input label overridden from flag precedence | REVIEW | 6 |  |  |  |  |
| assertion | retain | five_state_partition_within_analysis_universe | 0 | effective inundated > unclassified > isolated > redundant > fragile states must sum to 1 | PASS |  |  |  |  |  |
| descriptive | retain | five_state_partition_violations_outside_analysis_universe | 0 | outside analysis_eligible & pop20 > 0 & land_area_m2 > 0 | PASS |  |  |  |  |  |
| assertion | retain | five_state_baseline_partition_within_analysis_universe | 0 | recomputed baseline state must be exactly one of five states | PASS |  |  |  |  |  |
| descriptive | retain | five_state_baseline_partition_violations_outside_analysis_universe | 0 | outside analysis_eligible & pop20 > 0 & land_area_m2 > 0 | PASS |  |  |  |  |  |
| unclassified_descriptive | retain | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 0 | 262 | 91 | 1848 | 0 |
| unclassified_descriptive | retain | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 1 | 262 | 106 | 1848 | 0 |
| unclassified_descriptive | retain | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 2 | 262 | 112 | 1848 | 0 |
| unclassified_descriptive | retain | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 3 | 262 | 120 | 1848 | 0 |
| unclassified_descriptive | retain | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 4 | 262 | 121 | 1848 | 0 |
| unclassified_descriptive | retain | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 5 | 262 | 130 | 1848 | 0 |
| unclassified_descriptive | retain | raw_unclassified_blocks | 262 | counts use block_centroid_unclassified before precedence | PASS | 6 | 262 | 147 | 1848 | 0 |
| analysis_universe_descriptive | retain | baseline_blocks_total | 70695 | zero-land and zero-population exclusions reported separately | PASS | 0 |  |  |  |  |
| analysis_universe_descriptive | retain | baseline_zero_land_area_blocks | 2019 | zero-land and zero-population exclusions reported separately | PASS | 0 |  |  |  |  |
| analysis_universe_descriptive | retain | baseline_zero_population_blocks | 15284 | zero-land and zero-population exclusions reported separately | PASS | 0 |  |  |  |  |
| analysis_universe_descriptive | retain | baseline_zero_land_and_zero_population_blocks | 2002 | zero-land and zero-population exclusions reported separately | PASS | 0 |  |  |  |  |
| analysis_universe_descriptive | retain | baseline_analysis_eligible_before_population_filter | 68521 | zero-land and zero-population exclusions reported separately | PASS | 0 |  |  |  |  |
| analysis_universe_descriptive | retain | baseline_analysis_universe_after_population_filter | 55381 | zero-land and zero-population exclusions reported separately | PASS | 0 |  |  |  |  |
| assertion | approach | identical_block_universe | 0 | symmetric_difference_vs_intersect; sample_saved=0 | PASS |  |  |  |  |  |
| assertion | approach | identical_slr_levels | 0 | reference=[0, 1, 2, 3, 4, 5, 6]; other=[0, 1, 2, 3, 4, 5, 6] | PASS |  |  |  |  |  |
| assertion | approach | identical_block_scenario_keys | 0 | symmetric_difference_vs_intersect; sample_saved=0 | PASS |  |  |  |  |  |
| assertion | retain | identical_block_universe | 0 | symmetric_difference_vs_intersect; sample_saved=0 | PASS |  |  |  |  |  |
| assertion | retain | identical_slr_levels | 0 | reference=[0, 1, 2, 3, 4, 5, 6]; other=[0, 1, 2, 3, 4, 5, 6] | PASS |  |  |  |  |  |
| assertion | retain | identical_block_scenario_keys | 0 | symmetric_difference_vs_intersect; sample_saved=0 | PASS |  |  |  |  |  |
| assertion | approach | identical_block_centroid_inundated | 0 | differences_vs_intersect | PASS |  |  |  |  |  |
| assertion | retain | identical_block_centroid_inundated | 0 | differences_vs_intersect | PASS |  |  |  |  |  |

## Group B — Bridge fix (FAIL)

The expected control pattern holds: intersect should approximate the pre-fix bridge removals, while approach and retain should sharply reduce them.

| record_type | bridge_rule | slr_ft | category | segment_count | length_m | length_km | parent_way_count | structure_count | inventory_segment_count | inventory_length_m | inventory_length_km | intersecting_segment_count | reference_value | delta | current_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| removed_segments | intersect | 0 | all_segments | 1478 | 77427.7 | 77.4277 | 982 | 469 | 1856748 | 5.42243e+07 | 54224.3 | 1478 |  |  |  |
| removed_segments | intersect | 0 | bridge_tag_present | 1041 | 56164.3 | 56.1643 | 655 | 464 | 9770 | 332937 | 332.937 | 1478 |  |  |  |
| removed_segments | intersect | 0 | movable_tag | 93 | 3853.07 | 3.85307 | 83 | 83 | 95 | 3877.24 | 3.87724 | 1478 |  |  |  |
| removed_segments | intersect | 0 | negative_layer | 9 | 1393.91 | 1.39391 | 5 | 5 | 1436 | 26458.4 | 26.4584 | 1478 |  |  |  |
| removed_segments | intersect | 1 | all_segments | 1872 | 97135.3 | 97.1353 | 1135 | 487 | 1856748 | 5.42243e+07 | 54224.3 | 1872 |  |  |  |
| removed_segments | intersect | 1 | bridge_tag_present | 1074 | 57049.9 | 57.0499 | 678 | 482 | 9770 | 332937 | 332.937 | 1872 |  |  |  |
| removed_segments | intersect | 1 | movable_tag | 93 | 3853.07 | 3.85307 | 83 | 83 | 95 | 3877.24 | 3.87724 | 1872 |  |  |  |
| removed_segments | intersect | 1 | negative_layer | 9 | 1393.91 | 1.39391 | 5 | 5 | 1436 | 26458.4 | 26.4584 | 1872 |  |  |  |
| removed_segments | approach | 0 | all_segments | 1232 | 38739.2 | 38.7392 | 632 | 239 | 1856748 | 5.42243e+07 | 54224.3 | 1478 |  |  |  |
| removed_segments | approach | 0 | bridge_tag_present | 329 | 9442.35 | 9.44235 | 184 | 165 | 9770 | 332937 | 332.937 | 1478 |  |  |  |
| removed_segments | approach | 0 | movable_tag | 10 | 493.477 | 0.493477 | 6 | 6 | 95 | 3877.24 | 3.87724 | 1478 |  |  |  |
| removed_segments | approach | 0 | negative_layer | 144 | 2094.27 | 2.09427 | 45 | 39 | 1436 | 26458.4 | 26.4584 | 1478 |  |  |  |
| removed_segments | approach | 1 | all_segments | 1596 | 57667.1 | 57.6671 | 765 | 242 | 1856748 | 5.42243e+07 | 54224.3 | 1872 |  |  |  |
| removed_segments | approach | 1 | bridge_tag_present | 332 | 9548.29 | 9.54829 | 187 | 168 | 9770 | 332937 | 332.937 | 1872 |  |  |  |
| removed_segments | approach | 1 | movable_tag | 10 | 493.477 | 0.493477 | 6 | 6 | 95 | 3877.24 | 3.87724 | 1872 |  |  |  |
| removed_segments | approach | 1 | negative_layer | 144 | 2094.27 | 2.09427 | 45 | 39 | 1436 | 26458.4 | 26.4584 | 1872 |  |  |  |
| removed_segments | retain | 0 | all_segments | 428 | 19869.5 | 19.8695 | 322 | 0 | 1856748 | 5.42243e+07 | 54224.3 | 1478 |  |  |  |
| removed_segments | retain | 0 | bridge_tag_present | 0 | 0 | 0 | 0 | 0 | 9770 | 332937 | 332.937 | 1478 |  |  |  |
| removed_segments | retain | 0 | movable_tag | 0 | 0 | 0 | 0 | 0 | 95 | 3877.24 | 3.87724 | 1478 |  |  |  |
| removed_segments | retain | 0 | negative_layer | 0 | 0 | 0 | 0 | 0 | 1436 | 26458.4 | 26.4584 | 1478 |  |  |  |
| removed_segments | retain | 1 | all_segments | 789 | 38691.5 | 38.6915 | 452 | 0 | 1856748 | 5.42243e+07 | 54224.3 | 1872 |  |  |  |
| removed_segments | retain | 1 | bridge_tag_present | 0 | 0 | 0 | 0 | 0 | 9770 | 332937 | 332.937 | 1872 |  |  |  |
| removed_segments | retain | 1 | movable_tag | 0 | 0 | 0 | 0 | 0 | 95 | 3877.24 | 3.87724 | 1872 |  |  |  |
| removed_segments | retain | 1 | negative_layer | 0 | 0 | 0 | 0 | 0 | 1436 | 26458.4 | 26.4584 | 1872 |  |  |  |
| pre_fix_comparison | intersect | 0 | bridge_parent_way_count |  |  |  |  |  |  |  |  |  | 660 | -5 | 655 |
| pre_fix_comparison | intersect | 0 | bridge_length_km |  |  |  |  |  |  |  |  |  | 57.6 | -1.43573 | 56.1643 |
| pre_fix_comparison | intersect | 0 | bridge_segment_count_share |  |  |  |  |  |  |  |  |  | 0.71 | -0.00566982 | 0.70433 |
| pre_fix_comparison | intersect | 0 | bridge_length_share |  |  |  |  |  |  |  |  |  | 0.71 | 0.0153769 | 0.725377 |
| assertion | all | 0 | expected_control_and_collapse_pattern |  |  |  |  |  |  |  |  |  | intersect approximately pre-fix; approach lower; retain zero |  | True |
| tag_inventory | all |  | movable_tag_segments | 95 |  |  |  |  |  |  |  |  |  |  |  |
| tag_inventory | all |  | negative_layer_segments | 1436 |  |  |  |  |  |  |  |  |  |  |  |
| pre_fix_comparison | intersect | 0_and_1 | movable_removed_person_scenarios |  |  |  |  |  |  |  |  |  | 186 | 0 | 186 |
| pre_fix_comparison | intersect | 0_and_1 | negative_layer_removed_person_scenarios |  |  |  |  |  |  |  |  |  | 18 | 0 | 18 |
| assertion | all |  | positive_layer_and_fixed_span_gate_matches_cache | 1437 |  |  |  |  |  |  |  |  | 0 | 1437 | False |
| assertion | all |  | negative_layer_without_fixed_bridge_is_removal_eligible | 1436 |  |  |  |  | 1436 |  |  |  | 0 | 1436 | False |
| assertion | all |  | movable_without_positive_layer_is_removal_eligible | 1 |  |  |  |  | 1 |  |  |  | 0 | 1 | False |

## Group C — Service snapping (PASS)

Singleton 2-edge-connected-component shares are compared directly with the pre-fix 26.3% overall and 45.1% fire-station rates.

| record_type | bridge_rule | service_type | service_count | singleton_count | singleton_share | pre_fix_share | share_delta | source | moved_count | moved_share | added_distance_median_m | added_distance_p90_m | added_distance_max_m | exceeds_max_service_snap_count | pre_fix_added_distance_median_m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| singleton_2ecc | intersect | all | 2190 | 29 | 0.013242 | 0.263 | -0.249758 | C:\Users\Vivek\Dropbox\repos\slr_fl_fragile_access\data\processed\access\edited\della_runs\corrected_intersect\service_snapping_audit.csv |  |  |  |  |  |  |  |
| singleton_2ecc | intersect | fire_station | 334 | 11 | 0.0329341 | 0.451 | -0.418066 | C:\Users\Vivek\Dropbox\repos\slr_fl_fragile_access\data\processed\access\edited\della_runs\corrected_intersect\service_snapping_audit.csv |  |  |  |  |  |  |  |
| singleton_2ecc | intersect | school | 1856 | 18 | 0.00969828 |  |  | C:\Users\Vivek\Dropbox\repos\slr_fl_fragile_access\data\processed\access\edited\della_runs\corrected_intersect\service_snapping_audit.csv |  |  |  |  |  |  |  |
| resnapping | intersect | all | 2190 |  |  |  |  | C:\Users\Vivek\Dropbox\repos\slr_fl_fragile_access\data\processed\access\edited\della_runs\corrected_intersect\service_snapping_audit.csv | 574 | 0.2621 | 13.7079 | 40.1361 | 669.924 | 27 | 13.5 |
| singleton_2ecc | approach | all | 2190 | 29 | 0.013242 | 0.263 | -0.249758 | C:\Users\Vivek\Dropbox\repos\slr_fl_fragile_access\data\processed\access\edited\della_runs\corrected_canonical\service_snapping_audit.csv |  |  |  |  |  |  |  |
| singleton_2ecc | approach | fire_station | 334 | 11 | 0.0329341 | 0.451 | -0.418066 | C:\Users\Vivek\Dropbox\repos\slr_fl_fragile_access\data\processed\access\edited\della_runs\corrected_canonical\service_snapping_audit.csv |  |  |  |  |  |  |  |
| singleton_2ecc | approach | school | 1856 | 18 | 0.00969828 |  |  | C:\Users\Vivek\Dropbox\repos\slr_fl_fragile_access\data\processed\access\edited\della_runs\corrected_canonical\service_snapping_audit.csv |  |  |  |  |  |  |  |
| resnapping | approach | all | 2190 |  |  |  |  | C:\Users\Vivek\Dropbox\repos\slr_fl_fragile_access\data\processed\access\edited\della_runs\corrected_canonical\service_snapping_audit.csv | 574 | 0.2621 | 13.7079 | 40.1361 | 669.924 | 27 | 13.5 |
| singleton_2ecc | retain | all | 2190 | 29 | 0.013242 | 0.263 | -0.249758 | C:\Users\Vivek\Dropbox\repos\slr_fl_fragile_access\data\processed\access\edited\della_runs\corrected_retain\service_snapping_audit.csv |  |  |  |  |  |  |  |
| singleton_2ecc | retain | fire_station | 334 | 11 | 0.0329341 | 0.451 | -0.418066 | C:\Users\Vivek\Dropbox\repos\slr_fl_fragile_access\data\processed\access\edited\della_runs\corrected_retain\service_snapping_audit.csv |  |  |  |  |  |  |  |
| singleton_2ecc | retain | school | 1856 | 18 | 0.00969828 |  |  | C:\Users\Vivek\Dropbox\repos\slr_fl_fragile_access\data\processed\access\edited\della_runs\corrected_retain\service_snapping_audit.csv |  |  |  |  |  |  |  |
| resnapping | retain | all | 2190 |  |  |  |  | C:\Users\Vivek\Dropbox\repos\slr_fl_fragile_access\data\processed\access\edited\della_runs\corrected_retain\service_snapping_audit.csv | 574 | 0.2621 | 13.7079 | 40.1361 | 669.924 | 27 | 13.5 |

## Group D — Block universe filter (PASS)

The filter counts and status cross-tabs distinguish the stored analysis_eligible flag from the population and land-area criteria.

| record_type | bridge_rule | universe | baseline_status | criterion | flag_value | block_count |
| --- | --- | --- | --- | --- | --- | --- |
| filter_count | intersect | baseline | all | blocks_before_filtering |  | 70695 |
| filter_count | intersect | baseline | all | blocks_after_analysis_eligible |  | 68521 |
| filter_count | intersect | baseline | all | blocks_after_population_criterion |  | 55411 |
| filter_count | intersect | baseline | all | blocks_after_land_area_criterion |  | 68676 |
| filter_count | intersect | baseline | all | blocks_after_population_and_land_criteria |  | 55394 |
| filter_count | intersect | baseline | all | blocks_after_comparison_analysis_universe |  | 55381 |
| filter_count | intersect | baseline | all | dropped_by_population_from_pipeline_eligible |  | 13140 |
| filter_count | intersect | baseline | all | dropped_by_zero_land_from_full_universe |  | 2019 |
| filter_count | intersect | baseline | all | dropped_population_only |  | 13282 |
| filter_count | intersect | baseline | all | dropped_land_area_only |  | 17 |
| filter_count | intersect | baseline | all | dropped_both_population_and_land |  | 2002 |
| filter_count | approach | baseline | all | blocks_before_filtering |  | 70695 |
| filter_count | approach | baseline | all | blocks_after_analysis_eligible |  | 68521 |
| filter_count | approach | baseline | all | blocks_after_population_criterion |  | 55411 |
| filter_count | approach | baseline | all | blocks_after_land_area_criterion |  | 68676 |
| filter_count | approach | baseline | all | blocks_after_population_and_land_criteria |  | 55394 |
| filter_count | approach | baseline | all | blocks_after_comparison_analysis_universe |  | 55381 |
| filter_count | approach | baseline | all | dropped_by_population_from_pipeline_eligible |  | 13140 |
| filter_count | approach | baseline | all | dropped_by_zero_land_from_full_universe |  | 2019 |
| filter_count | approach | baseline | all | dropped_population_only |  | 13282 |
| filter_count | approach | baseline | all | dropped_land_area_only |  | 17 |
| filter_count | approach | baseline | all | dropped_both_population_and_land |  | 2002 |
| filter_count | retain | baseline | all | blocks_before_filtering |  | 70695 |
| filter_count | retain | baseline | all | blocks_after_analysis_eligible |  | 68521 |
| filter_count | retain | baseline | all | blocks_after_population_criterion |  | 55411 |
| filter_count | retain | baseline | all | blocks_after_land_area_criterion |  | 68676 |
| filter_count | retain | baseline | all | blocks_after_population_and_land_criteria |  | 55394 |
| filter_count | retain | baseline | all | blocks_after_comparison_analysis_universe |  | 55381 |
| filter_count | retain | baseline | all | dropped_by_population_from_pipeline_eligible |  | 13140 |
| filter_count | retain | baseline | all | dropped_by_zero_land_from_full_universe |  | 2019 |
| filter_count | retain | baseline | all | dropped_population_only |  | 13282 |
| filter_count | retain | baseline | all | dropped_land_area_only |  | 17 |
| filter_count | retain | baseline | all | dropped_both_population_and_land |  | 2002 |

## Group E — Headline numbers (REVIEW)

Population metrics are emitted as both person-scenario totals and block-deduplicated unique-person totals.

| bridge_rule | slr_ft | metric | value | unit | universe | population_count_type | reference_value | delta | record_type | baseline_status | scenario_status | block_count | population |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| intersect | 0 | baseline_fragile_share | 0.254459 | share | all_blocks | not_applicable | 0.247 | 0.0074593 | headline |  |  |  |  |
| intersect | 1 | cumulative_newly_affected_population | 5349 | persons | all_blocks | person_scenario |  |  | headline |  |  |  |  |
| intersect | 1 | cumulative_newly_affected_population | 5349 | persons | all_blocks | unique_person |  |  | headline |  |  |  |  |
| intersect | 6 | cumulative_newly_affected_population | 3.97074e+06 | persons | all_blocks | person_scenario | 660000 | 3.31074e+06 | headline |  |  |  |  |
| intersect | 6 | cumulative_newly_affected_population | 2.06837e+06 | persons | all_blocks | unique_person | 660000 | 1.40837e+06 | headline |  |  |  |  |
| intersect | 0 | baseline_fragile_share | 0.267258 | share | analysis_universe | not_applicable | 0.247 | 0.0202577 | headline |  |  |  |  |
| intersect | 1 | cumulative_newly_affected_population | 5349 | persons | analysis_universe | person_scenario |  |  | headline |  |  |  |  |
| intersect | 1 | cumulative_newly_affected_population | 5349 | persons | analysis_universe | unique_person |  |  | headline |  |  |  |  |
| intersect | 6 | cumulative_newly_affected_population | 3.96878e+06 | persons | analysis_universe | person_scenario | 660000 | 3.30878e+06 | headline |  |  |  |  |
| intersect | 6 | cumulative_newly_affected_population | 2.06648e+06 | persons | analysis_universe | unique_person | 660000 | 1.40648e+06 | headline |  |  |  |  |
| approach | 0 | baseline_fragile_share | 0.250569 | share | all_blocks | not_applicable | 0.247 | 0.00356935 | headline |  |  |  |  |
| approach | 1 | cumulative_newly_affected_population | 5523 | persons | all_blocks | person_scenario |  |  | headline |  |  |  |  |
| approach | 1 | cumulative_newly_affected_population | 5523 | persons | all_blocks | unique_person |  |  | headline |  |  |  |  |
| approach | 6 | cumulative_newly_affected_population | 3.97053e+06 | persons | all_blocks | person_scenario | 660000 | 3.31053e+06 | headline |  |  |  |  |
| approach | 6 | cumulative_newly_affected_population | 2.06056e+06 | persons | all_blocks | unique_person | 660000 | 1.40056e+06 | headline |  |  |  |  |
| approach | 0 | baseline_fragile_share | 0.26538 | share | analysis_universe | not_applicable | 0.247 | 0.0183798 | headline |  |  |  |  |
| approach | 1 | cumulative_newly_affected_population | 5523 | persons | analysis_universe | person_scenario |  |  | headline |  |  |  |  |
| approach | 1 | cumulative_newly_affected_population | 5523 | persons | analysis_universe | unique_person |  |  | headline |  |  |  |  |
| approach | 6 | cumulative_newly_affected_population | 3.96858e+06 | persons | analysis_universe | person_scenario | 660000 | 3.30858e+06 | headline |  |  |  |  |
| approach | 6 | cumulative_newly_affected_population | 2.05868e+06 | persons | analysis_universe | unique_person | 660000 | 1.39868e+06 | headline |  |  |  |  |
| retain | 0 | baseline_fragile_share | 0.251036 | share | all_blocks | not_applicable | 0.247 | 0.00403614 | headline |  |  |  |  |
| retain | 1 | cumulative_newly_affected_population | 5523 | persons | all_blocks | person_scenario |  |  | headline |  |  |  |  |
| retain | 1 | cumulative_newly_affected_population | 5523 | persons | all_blocks | unique_person |  |  | headline |  |  |  |  |
| retain | 6 | cumulative_newly_affected_population | 3.97126e+06 | persons | all_blocks | person_scenario | 660000 | 3.31126e+06 | headline |  |  |  |  |
| retain | 6 | cumulative_newly_affected_population | 2.06062e+06 | persons | all_blocks | unique_person | 660000 | 1.40062e+06 | headline |  |  |  |  |
| retain | 0 | baseline_fragile_share | 0.265831 | share | analysis_universe | not_applicable | 0.247 | 0.0188312 | headline |  |  |  |  |
| retain | 1 | cumulative_newly_affected_population | 5523 | persons | analysis_universe | person_scenario |  |  | headline |  |  |  |  |
| retain | 1 | cumulative_newly_affected_population | 5523 | persons | analysis_universe | unique_person |  |  | headline |  |  |  |  |
| retain | 6 | cumulative_newly_affected_population | 3.96931e+06 | persons | analysis_universe | person_scenario | 660000 | 3.30931e+06 | headline |  |  |  |  |
| retain | 6 | cumulative_newly_affected_population | 2.05874e+06 | persons | analysis_universe | unique_person | 660000 | 1.39874e+06 | headline |  |  |  |  |

## Group F — Ordering (PASS)

Ordering is checked as retain ≤ approach ≤ intersect at every SLR level.

| check | bridge_rule | slr_ft | value | previous_value | violation_magnitude | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| fragile_or_worse_order_retain_le_approach_le_intersect | all | 0 | retain=19068;approach=19068;intersect=19606 |  | 0 | PASS |
| fragile_or_worse_order_retain_le_approach_le_intersect | all | 1 | retain=19128;approach=19128;intersect=19670 |  | 0 | PASS |
| fragile_or_worse_order_retain_le_approach_le_intersect | all | 2 | retain=19660;approach=19660;intersect=20151 |  | 0 | PASS |
| fragile_or_worse_order_retain_le_approach_le_intersect | all | 3 | retain=21031;approach=21031;intersect=21433 |  | 0 | PASS |
| fragile_or_worse_order_retain_le_approach_le_intersect | all | 4 | retain=22972;approach=22972;intersect=23391 |  | 0 | PASS |
| fragile_or_worse_order_retain_le_approach_le_intersect | all | 5 | retain=27137;approach=27137;intersect=27693 |  | 0 | PASS |
| fragile_or_worse_order_retain_le_approach_le_intersect | all | 6 | retain=35182;approach=35182;intersect=35674 |  | 0 | PASS |

## What moved

The following pairwise headline values differ by more than 1 percentage point or 1% relative.

| comparison | slr_ft | metric | universe | population_count_type | reference_value | comparison_value | change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| approach_vs_intersect | 0 | baseline_status_fragile_blocks | all_blocks | not_applicable | 17989 | 17714 | -275 (1.53%) |
| retain_vs_intersect | 0 | baseline_status_fragile_blocks | all_blocks | not_applicable | 17989 | 17747 | -242 (1.35%) |
| approach_vs_intersect | 0 | baseline_status_isolated_blocks | all_blocks | not_applicable | 439 | 176 | -263 (59.91%) |
| retain_vs_approach | 0 | baseline_status_isolated_blocks | all_blocks | not_applicable | 176 | 143 | -33 (18.75%) |
| retain_vs_intersect | 0 | baseline_status_isolated_blocks | all_blocks | not_applicable | 439 | 143 | -296 (67.43%) |
| approach_vs_intersect | 0 | baseline_status_isolated_blocks | analysis_universe | not_applicable | 375 | 152 | -223 (59.47%) |
| retain_vs_approach | 0 | baseline_status_isolated_blocks | analysis_universe | not_applicable | 152 | 127 | -25 (16.45%) |
| retain_vs_intersect | 0 | baseline_status_isolated_blocks | analysis_universe | not_applicable | 375 | 127 | -248 (66.13%) |
| approach_vs_intersect | 0 | baseline_status_isolated_population | all_blocks | person_scenario | 40532 | 19842 | -20,690 (51.05%) |
| retain_vs_approach | 0 | baseline_status_isolated_population | all_blocks | person_scenario | 19842 | 15311 | -4,531 (22.84%) |
| retain_vs_intersect | 0 | baseline_status_isolated_population | all_blocks | person_scenario | 40532 | 15311 | -25,221 (62.22%) |
| approach_vs_intersect | 0 | baseline_status_isolated_population | all_blocks | unique_person | 40532 | 19842 | -20,690 (51.05%) |
| retain_vs_approach | 0 | baseline_status_isolated_population | all_blocks | unique_person | 19842 | 15311 | -4,531 (22.84%) |
| retain_vs_intersect | 0 | baseline_status_isolated_population | all_blocks | unique_person | 40532 | 15311 | -25,221 (62.22%) |
| approach_vs_intersect | 0 | baseline_status_isolated_population | analysis_universe | person_scenario | 40532 | 19842 | -20,690 (51.05%) |
| retain_vs_approach | 0 | baseline_status_isolated_population | analysis_universe | person_scenario | 19842 | 15311 | -4,531 (22.84%) |
| retain_vs_intersect | 0 | baseline_status_isolated_population | analysis_universe | person_scenario | 40532 | 15311 | -25,221 (62.22%) |
| approach_vs_intersect | 0 | baseline_status_isolated_population | analysis_universe | unique_person | 40532 | 19842 | -20,690 (51.05%) |
| retain_vs_approach | 0 | baseline_status_isolated_population | analysis_universe | unique_person | 19842 | 15311 | -4,531 (22.84%) |
| retain_vs_intersect | 0 | baseline_status_isolated_population | analysis_universe | unique_person | 40532 | 15311 | -25,221 (62.22%) |
| approach_vs_intersect | 0 | baseline_status_redundant_blocks | all_blocks | not_applicable | 50918 | 51456 | +538 (1.06%) |
| retain_vs_intersect | 0 | baseline_status_redundant_blocks | all_blocks | not_applicable | 50918 | 51456 | +538 (1.06%) |
| approach_vs_intersect | 1 | cumulative_newly_affected_population | all_blocks | person_scenario | 5349 | 5523 | +174 (3.25%) |
| retain_vs_intersect | 1 | cumulative_newly_affected_population | all_blocks | person_scenario | 5349 | 5523 | +174 (3.25%) |
| approach_vs_intersect | 1 | cumulative_newly_affected_population | all_blocks | unique_person | 5349 | 5523 | +174 (3.25%) |
| retain_vs_intersect | 1 | cumulative_newly_affected_population | all_blocks | unique_person | 5349 | 5523 | +174 (3.25%) |
| approach_vs_intersect | 1 | cumulative_newly_affected_population | analysis_universe | person_scenario | 5349 | 5523 | +174 (3.25%) |
| retain_vs_intersect | 1 | cumulative_newly_affected_population | analysis_universe | person_scenario | 5349 | 5523 | +174 (3.25%) |
| approach_vs_intersect | 1 | cumulative_newly_affected_population | analysis_universe | unique_person | 5349 | 5523 | +174 (3.25%) |
| retain_vs_intersect | 1 | cumulative_newly_affected_population | analysis_universe | unique_person | 5349 | 5523 | +174 (3.25%) |
| approach_vs_intersect | 1 | new_fragile_blocks | all_blocks | not_applicable | 11 | 5 | -6 (54.55%) |
| retain_vs_intersect | 1 | new_fragile_blocks | all_blocks | not_applicable | 11 | 5 | -6 (54.55%) |
| approach_vs_intersect | 1 | new_fragile_blocks | analysis_universe | not_applicable | 5 | 3 | -2 (40.00%) |
| retain_vs_intersect | 1 | new_fragile_blocks | analysis_universe | not_applicable | 5 | 3 | -2 (40.00%) |
| approach_vs_intersect | 1 | new_fragile_population | all_blocks | person_scenario | 150 | 56 | -94 (62.67%) |
| retain_vs_intersect | 1 | new_fragile_population | all_blocks | person_scenario | 150 | 56 | -94 (62.67%) |
| approach_vs_intersect | 1 | new_fragile_population | all_blocks | unique_person | 150 | 56 | -94 (62.67%) |
| retain_vs_intersect | 1 | new_fragile_population | all_blocks | unique_person | 150 | 56 | -94 (62.67%) |
| approach_vs_intersect | 1 | new_fragile_population | analysis_universe | person_scenario | 150 | 56 | -94 (62.67%) |
| retain_vs_intersect | 1 | new_fragile_population | analysis_universe | person_scenario | 150 | 56 | -94 (62.67%) |
| approach_vs_intersect | 1 | new_fragile_population | analysis_universe | unique_person | 150 | 56 | -94 (62.67%) |
| retain_vs_intersect | 1 | new_fragile_population | analysis_universe | unique_person | 150 | 56 | -94 (62.67%) |
| approach_vs_intersect | 1 | new_isolated_blocks | all_blocks | not_applicable | 13 | 14 | +1 (7.69%) |
| retain_vs_intersect | 1 | new_isolated_blocks | all_blocks | not_applicable | 13 | 14 | +1 (7.69%) |
| approach_vs_intersect | 1 | new_isolated_blocks | analysis_universe | not_applicable | 11 | 13 | +2 (18.18%) |
| retain_vs_intersect | 1 | new_isolated_blocks | analysis_universe | not_applicable | 11 | 13 | +2 (18.18%) |
| approach_vs_intersect | 1 | new_isolated_population | all_blocks | person_scenario | 346 | 614 | +268 (77.46%) |
| retain_vs_intersect | 1 | new_isolated_population | all_blocks | person_scenario | 346 | 614 | +268 (77.46%) |
| approach_vs_intersect | 1 | new_isolated_population | all_blocks | unique_person | 346 | 614 | +268 (77.46%) |
| retain_vs_intersect | 1 | new_isolated_population | all_blocks | unique_person | 346 | 614 | +268 (77.46%) |
| approach_vs_intersect | 1 | new_isolated_population | analysis_universe | person_scenario | 346 | 614 | +268 (77.46%) |
| retain_vs_intersect | 1 | new_isolated_population | analysis_universe | person_scenario | 346 | 614 | +268 (77.46%) |
| approach_vs_intersect | 1 | new_isolated_population | analysis_universe | unique_person | 346 | 614 | +268 (77.46%) |
| retain_vs_intersect | 1 | new_isolated_population | analysis_universe | unique_person | 346 | 614 | +268 (77.46%) |
| approach_vs_intersect | 2 | cumulative_newly_affected_population | all_blocks | person_scenario | 63689 | 70295 | +6,606 (10.37%) |
| retain_vs_intersect | 2 | cumulative_newly_affected_population | all_blocks | person_scenario | 63689 | 70379 | +6,690 (10.50%) |
| approach_vs_intersect | 2 | cumulative_newly_affected_population | all_blocks | unique_person | 58340 | 64772 | +6,432 (11.03%) |
| retain_vs_intersect | 2 | cumulative_newly_affected_population | all_blocks | unique_person | 58340 | 64856 | +6,516 (11.17%) |
| approach_vs_intersect | 2 | cumulative_newly_affected_population | analysis_universe | person_scenario | 63689 | 70295 | +6,606 (10.37%) |
| retain_vs_intersect | 2 | cumulative_newly_affected_population | analysis_universe | person_scenario | 63689 | 70379 | +6,690 (10.50%) |
| approach_vs_intersect | 2 | cumulative_newly_affected_population | analysis_universe | unique_person | 58340 | 64772 | +6,432 (11.03%) |
| retain_vs_intersect | 2 | cumulative_newly_affected_population | analysis_universe | unique_person | 58340 | 64856 | +6,516 (11.17%) |
| approach_vs_intersect | 2 | new_fragile_blocks | all_blocks | not_applicable | 214 | 204 | -10 (4.67%) |
| retain_vs_intersect | 2 | new_fragile_blocks | all_blocks | not_applicable | 214 | 204 | -10 (4.67%) |
| approach_vs_intersect | 2 | new_fragile_blocks | analysis_universe | not_applicable | 170 | 160 | -10 (5.88%) |
| retain_vs_intersect | 2 | new_fragile_blocks | analysis_universe | not_applicable | 170 | 160 | -10 (5.88%) |
| approach_vs_intersect | 2 | new_fragile_population | all_blocks | person_scenario | 12494 | 12988 | +494 (3.95%) |
| retain_vs_intersect | 2 | new_fragile_population | all_blocks | person_scenario | 12494 | 12988 | +494 (3.95%) |
| approach_vs_intersect | 2 | new_fragile_population | all_blocks | unique_person | 12494 | 12988 | +494 (3.95%) |
| retain_vs_intersect | 2 | new_fragile_population | all_blocks | unique_person | 12494 | 12988 | +494 (3.95%) |
| approach_vs_intersect | 2 | new_fragile_population | analysis_universe | person_scenario | 12494 | 12988 | +494 (3.95%) |
| retain_vs_intersect | 2 | new_fragile_population | analysis_universe | person_scenario | 12494 | 12988 | +494 (3.95%) |
| approach_vs_intersect | 2 | new_fragile_population | analysis_universe | unique_person | 12494 | 12988 | +494 (3.95%) |
| retain_vs_intersect | 2 | new_fragile_population | analysis_universe | unique_person | 12494 | 12988 | +494 (3.95%) |
| approach_vs_intersect | 2 | new_isolated_blocks | all_blocks | not_applicable | 342 | 373 | +31 (9.06%) |
| retain_vs_approach | 2 | new_isolated_blocks | all_blocks | not_applicable | 373 | 379 | +6 (1.61%) |
| retain_vs_intersect | 2 | new_isolated_blocks | all_blocks | not_applicable | 342 | 379 | +37 (10.82%) |
| approach_vs_intersect | 2 | new_isolated_blocks | analysis_universe | not_applicable | 273 | 323 | +50 (18.32%) |
| retain_vs_approach | 2 | new_isolated_blocks | analysis_universe | not_applicable | 323 | 328 | +5 (1.55%) |
| retain_vs_intersect | 2 | new_isolated_blocks | analysis_universe | not_applicable | 273 | 328 | +55 (20.15%) |
| approach_vs_intersect | 2 | new_isolated_population | all_blocks | person_scenario | 28411 | 34349 | +5,938 (20.90%) |
| retain_vs_intersect | 2 | new_isolated_population | all_blocks | person_scenario | 28411 | 34433 | +6,022 (21.20%) |
| approach_vs_intersect | 2 | new_isolated_population | all_blocks | unique_person | 28411 | 34349 | +5,938 (20.90%) |
| retain_vs_intersect | 2 | new_isolated_population | all_blocks | unique_person | 28411 | 34433 | +6,022 (21.20%) |
| approach_vs_intersect | 2 | new_isolated_population | analysis_universe | person_scenario | 28411 | 34349 | +5,938 (20.90%) |
| retain_vs_intersect | 2 | new_isolated_population | analysis_universe | person_scenario | 28411 | 34433 | +6,022 (21.20%) |
| approach_vs_intersect | 2 | new_isolated_population | analysis_universe | unique_person | 28411 | 34349 | +5,938 (20.90%) |
| retain_vs_intersect | 2 | new_isolated_population | analysis_universe | unique_person | 28411 | 34433 | +6,022 (21.20%) |
| approach_vs_intersect | 3 | cumulative_newly_affected_population | all_blocks | person_scenario | 340339 | 349660 | +9,321 (2.74%) |
| retain_vs_intersect | 3 | cumulative_newly_affected_population | all_blocks | person_scenario | 340339 | 349961 | +9,622 (2.83%) |
| retain_vs_intersect | 3 | cumulative_newly_affected_population | all_blocks | unique_person | 276650 | 279582 | +2,932 (1.06%) |
| approach_vs_intersect | 3 | cumulative_newly_affected_population | analysis_universe | person_scenario | 340339 | 349660 | +9,321 (2.74%) |
| retain_vs_intersect | 3 | cumulative_newly_affected_population | analysis_universe | person_scenario | 340339 | 349961 | +9,622 (2.83%) |
| retain_vs_intersect | 3 | cumulative_newly_affected_population | analysis_universe | unique_person | 276650 | 279582 | +2,932 (1.06%) |
| approach_vs_intersect | 3 | new_fragile_blocks | all_blocks | not_applicable | 482 | 477 | -5 (1.04%) |
| approach_vs_intersect | 3 | new_fragile_blocks | analysis_universe | not_applicable | 387 | 380 | -7 (1.81%) |
| retain_vs_intersect | 3 | new_fragile_blocks | analysis_universe | not_applicable | 387 | 381 | -6 (1.55%) |
| approach_vs_intersect | 3 | new_fragile_population | all_blocks | person_scenario | 33977 | 40500 | +6,523 (19.20%) |
| retain_vs_approach | 3 | new_fragile_population | all_blocks | person_scenario | 40500 | 41683 | +1,183 (2.92%) |
| retain_vs_intersect | 3 | new_fragile_population | all_blocks | person_scenario | 33977 | 41683 | +7,706 (22.68%) |
| approach_vs_intersect | 3 | new_fragile_population | all_blocks | unique_person | 33977 | 40500 | +6,523 (19.20%) |
| retain_vs_approach | 3 | new_fragile_population | all_blocks | unique_person | 40500 | 41683 | +1,183 (2.92%) |
| retain_vs_intersect | 3 | new_fragile_population | all_blocks | unique_person | 33977 | 41683 | +7,706 (22.68%) |
| approach_vs_intersect | 3 | new_fragile_population | analysis_universe | person_scenario | 33977 | 40500 | +6,523 (19.20%) |
| retain_vs_approach | 3 | new_fragile_population | analysis_universe | person_scenario | 40500 | 41683 | +1,183 (2.92%) |
| retain_vs_intersect | 3 | new_fragile_population | analysis_universe | person_scenario | 33977 | 41683 | +7,706 (22.68%) |
| approach_vs_intersect | 3 | new_fragile_population | analysis_universe | unique_person | 33977 | 40500 | +6,523 (19.20%) |
| retain_vs_approach | 3 | new_fragile_population | analysis_universe | unique_person | 40500 | 41683 | +1,183 (2.92%) |
| retain_vs_intersect | 3 | new_fragile_population | analysis_universe | unique_person | 33977 | 41683 | +7,706 (22.68%) |
| approach_vs_intersect | 3 | new_isolated_blocks | all_blocks | not_applicable | 1149 | 1201 | +52 (4.53%) |
| retain_vs_intersect | 3 | new_isolated_blocks | all_blocks | not_applicable | 1149 | 1206 | +57 (4.96%) |
| approach_vs_intersect | 3 | new_isolated_blocks | analysis_universe | not_applicable | 980 | 1050 | +70 (7.14%) |
| retain_vs_intersect | 3 | new_isolated_blocks | analysis_universe | not_applicable | 980 | 1054 | +74 (7.55%) |
| approach_vs_intersect | 3 | new_isolated_population | all_blocks | person_scenario | 156368 | 152560 | -3,808 (2.44%) |
| retain_vs_intersect | 3 | new_isolated_population | all_blocks | person_scenario | 156368 | 151594 | -4,774 (3.05%) |
| approach_vs_intersect | 3 | new_isolated_population | all_blocks | unique_person | 156368 | 152560 | -3,808 (2.44%) |
| retain_vs_intersect | 3 | new_isolated_population | all_blocks | unique_person | 156368 | 151594 | -4,774 (3.05%) |
| approach_vs_intersect | 3 | new_isolated_population | analysis_universe | person_scenario | 156368 | 152560 | -3,808 (2.44%) |
| retain_vs_intersect | 3 | new_isolated_population | analysis_universe | person_scenario | 156368 | 151594 | -4,774 (3.05%) |
| approach_vs_intersect | 3 | new_isolated_population | analysis_universe | unique_person | 156368 | 152560 | -3,808 (2.44%) |
| retain_vs_intersect | 3 | new_isolated_population | analysis_universe | unique_person | 156368 | 151594 | -4,774 (3.05%) |
| approach_vs_intersect | 4 | cumulative_newly_affected_population | all_blocks | person_scenario | 883051 | 898199 | +15,148 (1.72%) |
| retain_vs_intersect | 4 | cumulative_newly_affected_population | all_blocks | person_scenario | 883051 | 898701 | +15,650 (1.77%) |
| approach_vs_intersect | 4 | cumulative_newly_affected_population | all_blocks | unique_person | 542712 | 548539 | +5,827 (1.07%) |
| retain_vs_intersect | 4 | cumulative_newly_affected_population | all_blocks | unique_person | 542712 | 548740 | +6,028 (1.11%) |
| approach_vs_intersect | 4 | cumulative_newly_affected_population | analysis_universe | person_scenario | 883020 | 898168 | +15,148 (1.72%) |
| retain_vs_intersect | 4 | cumulative_newly_affected_population | analysis_universe | person_scenario | 883020 | 898670 | +15,650 (1.77%) |
| approach_vs_intersect | 4 | cumulative_newly_affected_population | analysis_universe | unique_person | 542681 | 548508 | +5,827 (1.07%) |
| retain_vs_intersect | 4 | cumulative_newly_affected_population | analysis_universe | unique_person | 542681 | 548709 | +6,028 (1.11%) |
| approach_vs_intersect | 4 | new_fragile_blocks | all_blocks | not_applicable | 783 | 734 | -49 (6.26%) |
| retain_vs_intersect | 4 | new_fragile_blocks | all_blocks | not_applicable | 783 | 735 | -48 (6.13%) |
| approach_vs_intersect | 4 | new_fragile_population | all_blocks | person_scenario | 54298 | 57430 | +3,132 (5.77%) |
| retain_vs_approach | 4 | new_fragile_population | all_blocks | person_scenario | 57430 | 58613 | +1,183 (2.06%) |
| retain_vs_intersect | 4 | new_fragile_population | all_blocks | person_scenario | 54298 | 58613 | +4,315 (7.95%) |
| approach_vs_intersect | 4 | new_fragile_population | all_blocks | unique_person | 54298 | 57430 | +3,132 (5.77%) |
| retain_vs_approach | 4 | new_fragile_population | all_blocks | unique_person | 57430 | 58613 | +1,183 (2.06%) |
| retain_vs_intersect | 4 | new_fragile_population | all_blocks | unique_person | 54298 | 58613 | +4,315 (7.95%) |
| approach_vs_intersect | 4 | new_fragile_population | analysis_universe | person_scenario | 54298 | 57430 | +3,132 (5.77%) |
| retain_vs_approach | 4 | new_fragile_population | analysis_universe | person_scenario | 57430 | 58613 | +1,183 (2.06%) |
| retain_vs_intersect | 4 | new_fragile_population | analysis_universe | person_scenario | 54298 | 58613 | +4,315 (7.95%) |
| approach_vs_intersect | 4 | new_fragile_population | analysis_universe | unique_person | 54298 | 57430 | +3,132 (5.77%) |
| retain_vs_approach | 4 | new_fragile_population | analysis_universe | unique_person | 57430 | 58613 | +1,183 (2.06%) |
| retain_vs_intersect | 4 | new_fragile_population | analysis_universe | unique_person | 54298 | 58613 | +4,315 (7.95%) |
| approach_vs_intersect | 4 | new_isolated_blocks | all_blocks | not_applicable | 2051 | 1981 | -70 (3.41%) |
| retain_vs_intersect | 4 | new_isolated_blocks | all_blocks | not_applicable | 2051 | 1986 | -65 (3.17%) |
| approach_vs_intersect | 4 | new_isolated_blocks | analysis_universe | not_applicable | 1615 | 1639 | +24 (1.49%) |
| retain_vs_intersect | 4 | new_isolated_blocks | analysis_universe | not_applicable | 1615 | 1642 | +27 (1.67%) |
| approach_vs_intersect | 4 | new_isolated_population | all_blocks | person_scenario | 237017 | 239712 | +2,695 (1.14%) |
| approach_vs_intersect | 4 | new_isolated_population | all_blocks | unique_person | 237017 | 239712 | +2,695 (1.14%) |
| approach_vs_intersect | 4 | new_isolated_population | analysis_universe | person_scenario | 237017 | 239712 | +2,695 (1.14%) |
| approach_vs_intersect | 4 | new_isolated_population | analysis_universe | unique_person | 237017 | 239712 | +2,695 (1.14%) |
| approach_vs_intersect | 5 | new_fragile_blocks | all_blocks | not_applicable | 1366 | 1215 | -151 (11.05%) |
| retain_vs_intersect | 5 | new_fragile_blocks | all_blocks | not_applicable | 1366 | 1217 | -149 (10.91%) |
| approach_vs_intersect | 5 | new_fragile_blocks | analysis_universe | not_applicable | 952 | 896 | -56 (5.88%) |
| retain_vs_intersect | 5 | new_fragile_blocks | analysis_universe | not_applicable | 952 | 897 | -55 (5.78%) |
| approach_vs_intersect | 5 | new_fragile_population | all_blocks | person_scenario | 87930 | 82627 | -5,303 (6.03%) |
| retain_vs_intersect | 5 | new_fragile_population | all_blocks | person_scenario | 87930 | 82979 | -4,951 (5.63%) |
| approach_vs_intersect | 5 | new_fragile_population | all_blocks | unique_person | 87930 | 82627 | -5,303 (6.03%) |
| retain_vs_intersect | 5 | new_fragile_population | all_blocks | unique_person | 87930 | 82979 | -4,951 (5.63%) |
| approach_vs_intersect | 5 | new_fragile_population | analysis_universe | person_scenario | 87930 | 82627 | -5,303 (6.03%) |
| retain_vs_intersect | 5 | new_fragile_population | analysis_universe | person_scenario | 87930 | 82979 | -4,951 (5.63%) |
| approach_vs_intersect | 5 | new_fragile_population | analysis_universe | unique_person | 87930 | 82627 | -5,303 (6.03%) |
| retain_vs_intersect | 5 | new_fragile_population | analysis_universe | unique_person | 87930 | 82979 | -4,951 (5.63%) |
| approach_vs_intersect | 5 | new_isolated_blocks | all_blocks | not_applicable | 3632 | 3425 | -207 (5.70%) |
| retain_vs_intersect | 5 | new_isolated_blocks | all_blocks | not_applicable | 3632 | 3428 | -204 (5.62%) |
| approach_vs_intersect | 6 | new_fragile_blocks | all_blocks | not_applicable | 1326 | 1380 | +54 (4.07%) |
| retain_vs_intersect | 6 | new_fragile_blocks | all_blocks | not_applicable | 1326 | 1381 | +55 (4.15%) |
| approach_vs_intersect | 6 | new_fragile_population | all_blocks | person_scenario | 88320 | 90028 | +1,708 (1.93%) |
| retain_vs_intersect | 6 | new_fragile_population | all_blocks | person_scenario | 88320 | 90028 | +1,708 (1.93%) |
| approach_vs_intersect | 6 | new_fragile_population | all_blocks | unique_person | 88320 | 90028 | +1,708 (1.93%) |
| retain_vs_intersect | 6 | new_fragile_population | all_blocks | unique_person | 88320 | 90028 | +1,708 (1.93%) |
| approach_vs_intersect | 6 | new_fragile_population | analysis_universe | person_scenario | 88320 | 90028 | +1,708 (1.93%) |
| retain_vs_intersect | 6 | new_fragile_population | analysis_universe | person_scenario | 88320 | 90028 | +1,708 (1.93%) |
| approach_vs_intersect | 6 | new_fragile_population | analysis_universe | unique_person | 88320 | 90028 | +1,708 (1.93%) |
| retain_vs_intersect | 6 | new_fragile_population | analysis_universe | unique_person | 88320 | 90028 | +1,708 (1.93%) |
| approach_vs_intersect | 6 | new_isolated_blocks | all_blocks | not_applicable | 6871 | 6458 | -413 (6.01%) |
| retain_vs_intersect | 6 | new_isolated_blocks | all_blocks | not_applicable | 6871 | 6459 | -412 (6.00%) |
| approach_vs_intersect | 6 | new_isolated_population | all_blocks | person_scenario | 651276 | 641766 | -9,510 (1.46%) |
| retain_vs_intersect | 6 | new_isolated_population | all_blocks | person_scenario | 651276 | 641821 | -9,455 (1.45%) |
| approach_vs_intersect | 6 | new_isolated_population | all_blocks | unique_person | 651276 | 641766 | -9,510 (1.46%) |
| retain_vs_intersect | 6 | new_isolated_population | all_blocks | unique_person | 651276 | 641821 | -9,455 (1.45%) |
| approach_vs_intersect | 6 | new_isolated_population | analysis_universe | person_scenario | 651276 | 641766 | -9,510 (1.46%) |
| retain_vs_intersect | 6 | new_isolated_population | analysis_universe | person_scenario | 651276 | 641821 | -9,455 (1.45%) |
| approach_vs_intersect | 6 | new_isolated_population | analysis_universe | unique_person | 651276 | 641766 | -9,510 (1.46%) |
| retain_vs_intersect | 6 | new_isolated_population | analysis_universe | unique_person | 651276 | 641821 | -9,455 (1.45%) |
