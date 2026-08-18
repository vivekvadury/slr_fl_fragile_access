# Task 0 - Della run verification

**GATES PASS (items 1 and 3).** All three manifests record legacy_layer_gate=false, and both physical representations of every arm contain all seven SLR levels with unique (block_geoid, slr_ft) keys. Sources: data/processed/access/edited/della_runs/positive_layer_20260814_approach/run_manifest.json:74-95; data/processed/access/edited/della_runs/positive_layer_20260814_intersect/run_manifest.json:74-95; data/processed/access/edited/della_runs/positive_layer_20260814_retain/run_manifest.json:74-95; reports/00_run_files_and_integrity.csv.

**Warnings, neither of which is a stated gate:** the output schema has 55 columns, not the expected 38, and all three manifests record working_tree_dirty=true. The schema contains every expected old/added field plus 17 appended derived/status fields; the dirty-tree flag means the recorded commit does not by itself fully identify the source used for the runs. Sources: reports/00_schema_diff.csv; data/processed/access/edited/della_runs/positive_layer_20260814_approach/run_manifest.json:102-104; data/processed/access/edited/della_runs/positive_layer_20260814_intersect/run_manifest.json:102-104; data/processed/access/edited/della_runs/positive_layer_20260814_retain/run_manifest.json:102-104.

## 1. Manifest gate and resolved arguments

| Arm | legacy_layer_gate | bridge_rule | config_name | Git commit | working_tree_dirty |
|---|---:|---|---|---|---:|
| approach | false | approach | positive_layer_20260814_approach | 27ac074753b24e1b68e21597b746682f077931c9 | true |
| intersect | false | intersect | positive_layer_20260814_intersect | 27ac074753b24e1b68e21597b746682f077931c9 | true |
| retain | false | retain | positive_layer_20260814_retain | 27ac074753b24e1b68e21597b746682f077931c9 | true |

Sources: each arm's run_manifest.json: bridge_rule at line 75, config_name at line 77, legacy_layer_gate at line 81, and Git fields at lines 102-104:

- data/processed/access/edited/della_runs/positive_layer_20260814_approach/run_manifest.json:75-104
- data/processed/access/edited/della_runs/positive_layer_20260814_intersect/run_manifest.json:75-104
- data/processed/access/edited/della_runs/positive_layer_20260814_retain/run_manifest.json:75-104

The complete resolved argument dictionaries are the manifests' cli_flags objects.

### approach

    {
      "bridge_rule": "approach",
      "cache_dir": "data/processed/access/cache",
      "config_name": "positive_layer_20260814_approach",
      "legacy_centroid": false,
      "legacy_centroid_inundation_join": false,
      "legacy_collocated_rule": false,
      "legacy_layer_gate": false,
      "legacy_mode": false,
      "legacy_origin_failure_status": false,
      "legacy_origin_snap": false,
      "legacy_service_snap": false,
      "max_blocks": null,
      "output_suffix": "",
      "rebuild_cache": false,
      "resolved_cache_dir": "data/processed/access/cache",
      "resolved_run_dir": "data/processed/access/edited/della_runs/positive_layer_20260814_approach",
      "resume": true,
      "run_dir": "data/processed/access/edited/della_runs",
      "scenarios": "0,1,2,3,4,5,6",
      "slr_ft": null,
      "smoke": false
    }

Source: data/processed/access/edited/della_runs/positive_layer_20260814_approach/run_manifest.json:74-95.

### intersect

    {
      "bridge_rule": "intersect",
      "cache_dir": "data/processed/access/cache",
      "config_name": "positive_layer_20260814_intersect",
      "legacy_centroid": false,
      "legacy_centroid_inundation_join": false,
      "legacy_collocated_rule": false,
      "legacy_layer_gate": false,
      "legacy_mode": false,
      "legacy_origin_failure_status": false,
      "legacy_origin_snap": false,
      "legacy_service_snap": false,
      "max_blocks": null,
      "output_suffix": "",
      "rebuild_cache": false,
      "resolved_cache_dir": "data/processed/access/cache",
      "resolved_run_dir": "data/processed/access/edited/della_runs/positive_layer_20260814_intersect",
      "resume": true,
      "run_dir": "data/processed/access/edited/della_runs",
      "scenarios": "0,1,2,3,4,5,6",
      "slr_ft": null,
      "smoke": false
    }

Source: data/processed/access/edited/della_runs/positive_layer_20260814_intersect/run_manifest.json:74-95.

### retain

    {
      "bridge_rule": "retain",
      "cache_dir": "data/processed/access/cache",
      "config_name": "positive_layer_20260814_retain",
      "legacy_centroid": false,
      "legacy_centroid_inundation_join": false,
      "legacy_collocated_rule": false,
      "legacy_layer_gate": false,
      "legacy_mode": false,
      "legacy_origin_failure_status": false,
      "legacy_origin_snap": false,
      "legacy_service_snap": false,
      "max_blocks": null,
      "output_suffix": "",
      "rebuild_cache": false,
      "resolved_cache_dir": "data/processed/access/cache",
      "resolved_run_dir": "data/processed/access/edited/della_runs/positive_layer_20260814_retain",
      "resume": true,
      "run_dir": "data/processed/access/edited/della_runs",
      "scenarios": "0,1,2,3,4,5,6",
      "slr_ft": null,
      "smoke": false
    }

Source: data/processed/access/edited/della_runs/positive_layer_20260814_retain/run_manifest.json:74-95.

## 2. Physical file layout

Each arm has exactly two physical block_access_flags_long* files:

| Arm | Physical file | SLR values in that file |
|---|---|---|
| approach | block_access_flags_long.csv | 0, 1, 2, 3, 4, 5, 6 |
| approach | block_access_flags_long.parquet | 0, 1, 2, 3, 4, 5, 6 |
| intersect | block_access_flags_long.csv | 0, 1, 2, 3, 4, 5, 6 |
| intersect | block_access_flags_long.parquet | 0, 1, 2, 3, 4, 5, 6 |
| retain | block_access_flags_long.csv | 0, 1, 2, 3, 4, 5, 6 |
| retain | block_access_flags_long.parquet | 0, 1, 2, 3, 4, 5, 6 |

Source: reports/00_run_files_and_integrity.csv.

**Unambiguous layout verdict:** this is one logical all-level long dataset per arm, serialized twice (CSV and Parquet). It is not a six-scenario-file layout. Notebook 03 should read one unsuffixed representation per arm and must not stack six scenario-suffixed files. Source: reports/00_run_files_and_integrity.csv.

## 3. Seven-level and key-integrity gate

CSV and Parquet were checked independently. Their key sets match within each arm, and the three arms also have the same complete key universe. Source: reports/00_run_files_and_integrity.csv.

| Arm | Rows in each representation | Distinct blocks | SLR values | Rows at each SLR | Duplicate key groups | Rows in duplicate keys | Duplicate excess | Null block_geoid | Null slr_ft |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| approach | 494,865 | 70,695 | 0-6 | 70,695 | 0 | 0 | 0 | 0 | 0 |
| intersect | 494,865 | 70,695 | 0-6 | 70,695 | 0 | 0 | 0 | 0 | 0 |
| retain | 494,865 | 70,695 | 0-6 | 70,695 | 0 | 0 | 0 | 0 | 0 |

Source: reports/00_run_files_and_integrity.csv. The manifests independently report 70,695 rows at every level in each run_manifest.json:326-334.

**Gate verdict:** PASS for all arms. All seven levels are present, and (block_geoid, slr_ft) is unique. Source: reports/00_run_files_and_integrity.csv.

## 4. Schema and old-schema diff

The ordered approach schema is:

    1  block_geoid
    2  block_group_geoid
    3  tract_geoid
    4  block
    5  county_fips
    6  county_name
    7  pop20
    8  land_area_m2
    9  analysis_eligible
    10 exclusion_reason
    11 slr_ft
    12 slr_layer_name
    13 origin_node_id
    14 origin_snap_distance_m
    15 origin_snap_exceeds_threshold
    16 origin_in_lcc
    17 origin_geometry_method
    18 boundary_flag
    19 boundary_distance_m
    20 component_touches_boundary
    21 block_centroid_inundated
    22 block_centroid_isolated
    23 block_centroid_redundant
    24 block_centroid_fragile
    25 block_centroid_unclassified
    26 n_reachable_services
    27 n_reachable_service_nodes
    28 max_edge_disjoint_paths_any_service
    29 nearest_reachable_service_type
    30 nearest_reachable_service_id
    31 service_snap_rule
    32 bridge_rule_applied
    33 nearby_bridge_structure_id
    34 nearby_bridge_structure_distance_m
    35 nearby_bridge_structure_retained
    36 baseline_shortest_path_distance_m
    37 dry_shortest_path_distance_m
    38 detour_ratio
    39 baseline_block_centroid_inundated
    40 baseline_block_centroid_isolated
    41 baseline_block_centroid_redundant
    42 baseline_block_centroid_fragile
    43 baseline_n_reachable_services
    44 baseline_n_reachable_service_nodes
    45 baseline_max_edge_disjoint_paths_any_service
    46 baseline_baseline_shortest_path_distance_m
    47 baseline_dry_shortest_path_distance_m
    48 baseline_detour_ratio
    49 baseline_block_centroid_unclassified
    50 baseline_status
    51 scenario_status
    52 persistent_fragile
    53 new_fragile_due_to_slr
    54 new_isolated_due_to_slr
    55 new_inundated_due_to_slr

Sources: data/processed/access/edited/della_runs/positive_layer_20260814_approach/block_access_flags_long.csv:1; reports/00_schema_diff.csv. CSV line 1 and Parquet metadata agree, and all three arms have the same order; see reports/00_run_files_and_integrity.csv and reports/00_schema_diff.csv.

Diff against the requested old-26-plus-12 expectation:

- All 26 comparison-schema columns are present. Source: rows categorized old_26 in reports/00_schema_diff.csv.
- All 12 expected additions are present: pop20, land_area_m2, analysis_eligible, exclusion_reason, origin_in_lcc, origin_geometry_method, block_centroid_unclassified, service_snap_rule, bridge_rule_applied, nearby_bridge_structure_id, nearby_bridge_structure_distance_m, and nearby_bridge_structure_retained. Source: rows categorized expected_addition_12 in reports/00_schema_diff.csv.
- No expected column is missing. Source: reports/00_schema_diff.csv.
- Seventeen columns are unexpectedly appended after the expected 38: baseline_block_centroid_inundated, baseline_block_centroid_isolated, baseline_block_centroid_redundant, baseline_block_centroid_fragile, baseline_n_reachable_services, baseline_n_reachable_service_nodes, baseline_max_edge_disjoint_paths_any_service, baseline_baseline_shortest_path_distance_m, baseline_dry_shortest_path_distance_m, baseline_detour_ratio, baseline_block_centroid_unclassified, baseline_status, scenario_status, persistent_fragile, new_fragile_due_to_slr, new_isolated_due_to_slr, and new_inundated_due_to_slr. Source: rows categorized unexpected_appended_17 in reports/00_schema_diff.csv.

**Schema verdict: WARN.** Actual width is 55, not 38. The literal double prefix in baseline_baseline_shortest_path_distance_m is present in the files; it is not a transcription error. Source: reports/00_schema_diff.csv.

## 5. Removed-edge counts and arm monotonicity

Each entry below is "bridge edges removed / total edges removed / bridge structures removed." The total includes the non-bridge removal count, which is identical across arms at a given SLR. Source: reports/00_removed_edges.csv.

| SLR ft | intersect | approach | retain |
|---:|---:|---:|---:|
| 0 | 1,041 / 1,478 / 464 | 660 / 1,097 / 200 | 0 / 437 / 0 |
| 1 | 1,074 / 1,872 / 482 | 663 / 1,461 / 203 | 0 / 798 / 0 |
| 2 | 1,127 / 8,558 / 506 | 672 / 8,103 / 212 | 0 / 7,431 / 0 |
| 3 | 1,294 / 38,865 / 545 | 775 / 38,346 / 239 | 0 / 37,571 / 0 |
| 4 | 1,929 / 89,880 / 822 | 891 / 88,842 / 304 | 0 / 87,951 / 0 |
| 5 | 2,474 / 192,700 / 1,065 | 1,050 / 191,276 / 399 | 0 / 190,226 / 0 |
| 6 | 3,440 / 434,690 / 1,408 | 1,578 / 432,828 / 575 | 0 / 431,250 / 0 |

The sums of removed_edge_count in every bridge_structures_slr_{0..6}ft.csv agree with the corresponding manifest n_bridge_edges_removed values, and removed=True counts agree with n_bridge_structures_removed. Each summary contains 3,482 structures. Source paths for all 21 summaries and the matching manifest line spans are recorded row by row in reports/00_removed_edges.csv; the summary schema is on line 1 of each cited bridge CSV.

**Monotonicity verdict: PASS at every SLR level.** intersect >= approach >= retain holds separately for bridge-edge removals, total-edge removals, and removed bridge structures. Source: reports/00_removed_edges.csv.

## 6. Unclassified prevalence, reason, and spatial concentration

### 6.1 Prevalence by arm and SLR

The denominator is 70,695 blocks and pop20=6,138,333 at every arm-level. The three arms produce identical unclassified counts and population totals, but all 21 requested arm-level rows are printed below. Source: reports/00_unclassified_prevalence.csv.

| Arm | SLR ft | Unclassified blocks | Share of blocks | Unclassified pop20 | Share of pop20 |
|---|---:|---:|---:|---:|---:|
| approach | 0 | 171 | 0.241884% | 1,847 | 0.030090% |
| approach | 1 | 156 | 0.220666% | 1,847 | 0.030090% |
| approach | 2 | 150 | 0.212179% | 1,847 | 0.030090% |
| approach | 3 | 142 | 0.200863% | 1,847 | 0.030090% |
| approach | 4 | 141 | 0.199448% | 1,847 | 0.030090% |
| approach | 5 | 132 | 0.186718% | 1,847 | 0.030090% |
| approach | 6 | 115 | 0.162671% | 519 | 0.008455% |
| intersect | 0 | 171 | 0.241884% | 1,847 | 0.030090% |
| intersect | 1 | 156 | 0.220666% | 1,847 | 0.030090% |
| intersect | 2 | 150 | 0.212179% | 1,847 | 0.030090% |
| intersect | 3 | 142 | 0.200863% | 1,847 | 0.030090% |
| intersect | 4 | 141 | 0.199448% | 1,847 | 0.030090% |
| intersect | 5 | 132 | 0.186718% | 1,847 | 0.030090% |
| intersect | 6 | 115 | 0.162671% | 519 | 0.008455% |
| retain | 0 | 171 | 0.241884% | 1,847 | 0.030090% |
| retain | 1 | 156 | 0.220666% | 1,847 | 0.030090% |
| retain | 2 | 150 | 0.212179% | 1,847 | 0.030090% |
| retain | 3 | 142 | 0.200863% | 1,847 | 0.030090% |
| retain | 4 | 141 | 0.199448% | 1,847 | 0.030090% |
| retain | 5 | 132 | 0.186718% | 1,847 | 0.030090% |
| retain | 6 | 115 | 0.162671% | 519 | 0.008455% |

Source: reports/00_unclassified_prevalence.csv.

At every arm and SLR, the exclusion_reason distribution among scenario_status=unclassified is exactly origin_snap_failed: 100% of unclassified rows and 100% of their population; no blank or other reason occurs. Source: the reason_count, reason_count_share, reason_population, and reason_population_share fields in reports/00_unclassified_prevalence.csv.

The decreasing unclassified count does not mean that origin snapping improves with SLR. Status precedence assigns inundated before unclassified, so an origin-snap-failed block that becomes inundated is reported as inundated. Sources: scripts/02_access_flags.py:1582-1593; reports/00_unclassified_prevalence.csv.

### 6.2 County concentration

County rates are identical across arms. Values are "unclassified count (county block prevalence)." Source: reports/00_unclassified_county.csv.

| SLR ft | Broward | Miami-Dade | Palm Beach | Pearson chi-square (df=2) | p | Cramer's V |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16 (0.0764%) | 68 (0.2150%) | 87 (0.4798%) | 67.229 | 2.52e-15 | 0.0308 |
| 1 | 16 (0.0764%) | 53 (0.1676%) | 87 (0.4798%) | 79.122 | 6.59e-18 | 0.0335 |
| 2 | 16 (0.0764%) | 47 (0.1486%) | 87 (0.4798%) | 85.584 | 2.60e-19 | 0.0348 |
| 3 | 16 (0.0764%) | 39 (0.1233%) | 87 (0.4798%) | 96.026 | 1.41e-21 | 0.0369 |
| 4 | 16 (0.0764%) | 38 (0.1202%) | 87 (0.4798%) | 97.493 | 6.75e-22 | 0.0371 |
| 5 | 16 (0.0764%) | 29 (0.0917%) | 87 (0.4798%) | 112.544 | 3.64e-25 | 0.0399 |
| 6 | 16 (0.0764%) | 12 (0.0379%) | 87 (0.4798%) | 152.150 | 9.14e-34 | 0.0464 |

Sources: reports/00_unclassified_county.csv; reports/00_unclassified_county_tests.csv.

County is non-randomly associated with unclassified status, but the global effect is small (Cramer's V 0.0308-0.0464). The concentration is specifically Palm Beach: it has 25.651% of blocks but 50.877% of unclassified blocks at 0 ft and 75.652% at 6 ft, giving representation ratios of 1.983 and 2.949. Sources: reports/00_unclassified_county.csv; reports/00_unclassified_county_tests.csv.

### 6.3 Distance to coast

There is no literal shoreline-distance field in the run outputs. In particular, boundary_distance_m is distance to the boundary of a projected road-network bounding box, not the coast: the code constructs retained_network_bbox from road bounds and then measures point-to-boundary distance. Sources: scripts/02_access_flags.py:722-745; scripts/02_access_flags.py:2321-2323.

I therefore used a fixed, analysis-relevant proxy and label it explicitly as **distance to the NOAA 0-ft coastal-inundation footprint**, not literal shoreline distance:

1. The proxy geometry is FL_SE_slr_0_0ft in data/raw/noaa/FL_SE_slr_final_dist.gpkg. It is the pipeline's baseline layer, and the NOAA README defines "_slr_" polygon layers as ocean-connected inundation. Sources: scripts/02_access_flags.py:167-168; data/raw/noaa/NOAA_OCM_SLR_Data_README.txt:15-19.
2. Block geometries from data/processed/census/blocks/fl_tricounty_blocks_2020.gpkg and the NOAA polygons were projected to EPSG:32617; representative points match the corrected origin construction because legacy_centroid=false in all manifests. Sources: scripts/02_access_flags.py:164; scripts/02_access_flags.py:748-776; scripts/02_access_flags.py:2292-2294; each run_manifest.json:78.
3. Distance is exact Euclidean distance from each representative point to the nearest original NOAA polygon, using an STRtree; there was no geometry union or simplification. The case is scenario_status=unclassified. Source: distance_definition, distance_engine, geometry_handling, and case_definition in reports/00_unclassified_coastal_proxy.csv.
4. To remove the observed county imbalance, each SLR comparison uses deterministic 20:1 within-county controls, excludes every block that is unclassified at any level, and orders eligible controls by SHA256("coast-control-v1:" + block_geoid). Source: control_definition in reports/00_unclassified_coastal_proxy.csv.

Positive rank-biserial values below mean unclassified blocks are farther from the footprint than matched controls. Source: reports/00_unclassified_coastal_proxy.csv.

| SLR ft | Cases / controls | Unclassified median km (IQR) | Unclassified <=5 km | Control median km | Control <=5 km | Rank-biserial | Mann-Whitney p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 171 / 3,420 | 31.668 (16.784-58.775) | 14.620% | 4.240 | 54.942% | 0.6889 | 2.41e-52 |
| 1 | 156 / 3,120 | 35.759 (20.358-62.315) | 6.410% | 4.152 | 55.160% | 0.8177 | 9.48e-67 |
| 2 | 150 / 3,000 | 37.657 (21.965-63.346) | 3.333% | 4.123 | 55.400% | 0.8682 | 3.27e-72 |
| 3 | 142 / 2,840 | 38.897 (24.180-64.523) | 1.408% | 4.056 | 55.880% | 0.9006 | 1.61e-73 |
| 4 | 141 / 2,820 | 38.949 (24.200-64.703) | 0.709% | 4.056 | 55.922% | 0.9073 | 4.52e-74 |
| 5 | 132 / 2,640 | 40.689 (27.249-67.985) | 0.758% | 4.019 | 56.136% | 0.9095 | 8.50e-70 |
| 6 | 115 / 2,300 | 47.848 (31.253-69.139) | 0.870% | 3.849 | 56.913% | 0.9112 | 2.95e-61 |

Source: reports/00_unclassified_coastal_proxy.csv.

The endpoint thresholds reinforce the direction: at 0 ft, 9.357% of cases versus 22.719% of controls are within 1 km, and 20.468% versus 76.988% are within 10 km; at 6 ft the corresponding pairs are 0.870% versus 25.826% and 4.348% versus 77.565%. Source: reports/00_unclassified_coastal_proxy.csv.

**Spatial verdict:** unclassified blocks do cluster by county, chiefly in Palm Beach, but they emphatically do **not** cluster near the coastal-inundation footprint. They are much farther inland than county-matched controls at every SLR level. This diagnostic therefore provides no evidence of a coastal concentration of missing classifications that would bias RQ1 through that mechanism. The limit is explicit: the metric is distance to an ocean-connected NOAA inundation footprint, not distance to a surveyed shoreline. Sources: reports/00_unclassified_county.csv; reports/00_unclassified_county_tests.csv; reports/00_unclassified_coastal_proxy.csv; data/raw/noaa/NOAA_OCM_SLR_Data_README.txt:15-19.

## Bottom line

- **Item 1 gate:** PASS - legacy_layer_gate=false in approach, intersect, and retain. Sources: each run_manifest.json:81.
- **Item 3 gate:** PASS - seven levels, 70,695 blocks, 494,865 unique keys, and zero duplicates in both CSV and Parquet for every arm. Source: reports/00_run_files_and_integrity.csv.
- **Layout:** one logical all-level table per arm, dual-format, not six scenario files. Source: reports/00_run_files_and_integrity.csv.
- **Schema:** WARN - 55 columns rather than 38; all 12 expected additions exist, plus 17 unexpected appended fields. Source: reports/00_schema_diff.csv.
- **Removed edges:** PASS - intersect >= approach >= retain at every level, for bridge and total removals. Source: reports/00_removed_edges.csv.
- **Unclassified:** rare, exclusively origin_snap_failed, county-clustered in Palm Beach, and strongly farther from the NOAA coastal footprint than county-matched controls. Sources: reports/00_unclassified_prevalence.csv; reports/00_unclassified_county.csv; reports/00_unclassified_coastal_proxy.csv.
