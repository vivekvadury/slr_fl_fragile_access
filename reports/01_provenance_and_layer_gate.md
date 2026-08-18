# Phase 1 - Provenance repair and layer-gate exposure

## Prominent findings

**STOP-CONDITION ALERT: the current worktree is clean.** At the pre-report snapshot on 2026-08-18 at 15:28 EDT, git status returned "nothing to commit, working tree clean" on codex/downstream-propagation. The uncommitted checkout recorded by the three Della manifests no longer exists as a dirty working tree, so the exact Aug. 14 patch cannot be recovered from current uncommitted state. Source: reports/01_provenance_status.csv.

The current scripts do contain the producer changes relative to 27ac074, and the current scripts/02_access_flags.py blob exactly matches later commit fcc532b. That makes fcc532b a plausible capture of the producer patch, but it does not prove byte-for-byte identity with the dirty checkout used by the runs because the manifests do not record a patch or code hash. The defensible run description remains "27ac074 plus local changes" unless an independent archived patch or runtime source hash is found. Source: reports/01_provenance_status.csv.

**Baseline-removal warning:** the 200 structures removed by approach at 0 ft are not a benign-stub-only set. Ninety-eight have fewer than two physical landing nodes and are stub-like, but 102 have at least two physical landings, including 45 on high-order roads; those 102 fail because they have fewer than two dry landings at the baseline layer. Source: reports/01_baseline_removed_topology.csv.

## 1. Cache-only layer-gate exposure

### Cache provenance and method

The requested local cache filename containing layer_positive is absent. The only local full edge cache is data/processed/access/cache/segmentized_edges_fe639d75a84cb809_full_v2.parquet (SHA-256 c084c0750f4539b6679f9703dd8cef676ff18b06869215b653937574fb38c716). Its stored bridge_like field exactly implements the legacy nonzero gate. No graph was rebuilt. Source: reports/01_layer_gate_overall.csv.

The v2 cache still contains edge_id, u, v, highway, bridge_tag_present, layer_value, bridge_like, length_m, and geometry, so both definitions are recomputable from the raw segmentized network. Legacy bridge-like is bridge_tag_present OR numeric(layer_value) != 0; corrected bridge-like is bridge_tag_present OR numeric(layer_value) > 0. The definitions match scripts/02_access_flags.py:882-903 and :945-986. All nonmissing cached layer values used here were numeric. Source: reports/01_layer_gate_overall.csv.

As a cross-check, the corrected recomputation produces 3,482 structures and matches the canonical approach 0-ft bridge summary on structure ID, landing nodes/IDs, length, highway classes, bridge-edge count, and OSM-way count. This establishes that the raw v2 edge segmentation corresponds to the run network even though its corrected cache copy is not present locally. Source: reports/01_layer_gate_overall.csv.

### Exposure

The defect exposed 1,436 of 1,856,748 edges (0.07734%), totaling 26,458.416 m across 344 OSM ways. Every exposed edge had layer_value=-1 and no bridge tag. Source: reports/01_layer_gate_overall.csv.

| Highway | Exposed edges | Length m | OSM ways | Share of exposed edges | Share of exposed length |
|---|---:|---:|---:|---:|---:|
| service | 1,176 | 17,922.563 | 306 | 81.894% | 67.739% |
| unclassified | 95 | 1,394.394 | 9 | 6.616% | 5.270% |
| primary | 79 | 2,789.587 | 13 | 5.501% | 10.543% |
| residential | 39 | 951.181 | 6 | 2.716% | 3.595% |
| motorway | 38 | 2,608.142 | 2 | 2.646% | 9.858% |
| primary_link | 5 | 413.674 | 4 | 0.348% | 1.563% |
| secondary | 2 | 35.910 | 2 | 0.139% | 0.136% |
| tertiary | 2 | 342.965 | 2 | 0.139% | 1.296% |

Source: reports/01_layer_gate_by_highway.csv.

### Connected components and landings

Numeric structure IDs were not assumed comparable between gates. Components were matched by shared edge membership, following the connected-component and landing-node construction at scripts/02_access_flags.py:1024-1082. Source: reports/01_layer_gate_components.csv.

| Component result from legacy to corrected | Legacy components | Corrected components | Exposed edges | Exposed length m |
|---|---:|---:|---:|---:|
| unchanged | 3,481 | 3,481 | 0 | 0 |
| membership changed, unsplit | 1 | 1 | 22 | 618.524 |
| split | 0 | 0 | 0 | 0 |
| vanished | 282 | 0 | 1,414 | 25,839.892 |
| total | 3,764 | 3,482 | 1,436 | 26,458.416 |

Source: reports/01_layer_gate_components.csv.

Thus 283 legacy components are affected: 282 vanish and one survives with changed membership; none split. Among the 3,482 structures present under both definitions, 3,481 have no landing-count change. The sole changed survivor goes from seven legacy landings to two corrected landings, a corrected-minus-legacy change of -5. Source: reports/01_layer_gate_components.csv; reports/01_layer_gate_landing_changes.csv.

### Methods-ready summary

Recomputing the bridge-layer gate from the cached segmentized network reclassified 1,436 of 1,856,748 road segments (0.077%; 26.46 km), all untagged segments at layer -1. Service roads accounted for 1,176 exposed segments and 17.92 km. Removing these below-grade segments eliminated 282 bridge-like connected components, changed one surviving component, and split none; 3,481 of the 3,482 corrected structures retained identical membership and landing counts, while one changed from seven to two landing nodes. Sources: reports/01_layer_gate_overall.csv; reports/01_layer_gate_by_highway.csv; reports/01_layer_gate_components.csv; reports/01_layer_gate_landing_changes.csv.

## 2. Baseline structure removal under approach

At 0 ft, approach removes 200 of 3,482 structures, representing 660 bridge edges and 16,775.416 m of structure length. All removed structures have zero or one dry landing: 38 have zero and 162 have one. Source: data/processed/access/edited/della_runs/positive_layer_20260814_approach/run_manifest.json:2-12; reports/01_baseline_removed_landing_counts.csv.

| Total landing nodes | Removed structures | Share | Total length m | Dry=0 | Dry=1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 14 | 7.0% | 1,865.975 | 14 | 0 |
| 1 | 84 | 42.0% | 8,706.746 | 0 | 84 |
| 2 | 100 | 50.0% | 6,008.786 | 24 | 76 |
| 3 | 1 | 0.5% | 39.802 | 0 | 1 |
| 4 | 1 | 0.5% | 154.107 | 0 | 1 |

Source: reports/01_baseline_removed_landing_counts.csv.

The highway_classes distribution is service 86, residential 42, primary 23, tertiary 23, secondary 15, motorway 4, unclassified 3, and one each of trunk, secondary_link, primary|primary_link, and residential|unclassified. Their exact lengths are in reports/01_baseline_removed_highways.csv.

Landing nodes are bridge-component nodes incident to ordinary edges; approach counts how many are dry and retains a structure only when at least two are dry. Sources: scripts/02_access_flags.py:1024-1082; scripts/02_access_flags.py:1137-1171.

The structural verdict is mixed and contradicts a wholesale benign-stub interpretation:

- 98 structures (49%; 10.573 km) have fewer than two total landings and can never pass the rule. None is on a high-order highway, and 78 are single OSM ways. These fit dead-end or stub topology. Source: reports/01_baseline_removed_topology.csv.
- 102 structures (51%; 6.203 km) have at least two total landings but fewer than two dry landings; 85 are single bridge edges, 96 are single OSM ways, and 45 use motorway, trunk, primary, secondary, or link classes. These have through-route-like topology/class mix rather than dead-end topology. Source: reports/01_baseline_removed_topology.csv; reports/01_baseline_removed_highways.csv.

These removals are not driven by added future SLR, but half are driven by the baseline 0-ft dry-landing test rather than an absence of two physical landings. They should be treated as a baseline rule/NOAA-geometry audit issue, not defended collectively as harmless ramp stubs. Sources: reports/01_baseline_removed_landing_counts.csv; reports/01_baseline_removed_topology.csv.

## 3. Redundancy censoring

MAX_EDGE_DISJOINT_PATHS_CAP is 2. The connectivity routine returns 0, 1, or the cap based on origin degree and membership in a 2-edge-connected component containing a service. Redundant is defined as max paths >=2 and fragile as exactly 1. Sources: scripts/02_access_flags.py:193; scripts/02_access_flags.py:1521-1579; scripts/02_access_flags.py:1851-1864.

Therefore the cap does not change the fragile/redundant boundary and cannot change modeled transition outcomes derived from those flags. The transition flags consume baseline redundant/fragile and scenario fragile/isolated/inundated states, and the model specifications use those transitions. Sources: scripts/02_access_flags.py:1691-1705; scripts/04_transition_models.R:276-312.

The eligible-universe groupby confirms the censoring: every redundant block is exactly at the cap at every SLR level. Counts are 50,835 at 0 ft, 50,797 at 1, 50,283 at 2, 48,937 at 3, 47,081 at 4, 43,028 at 5, and 35,088 at 6. Pooled, 326,049 of 326,049 redundant block-scenario rows are exactly 2. Source: reports/01_redundancy_cap.csv.

Limitation sentence: Because edge-disjoint connectivity is censored at two, the analysis distinguishes fragile blocks with one route from redundant blocks with at least two, but cannot assess gradients in redundancy among blocks with two versus many independent service routes. Sources: scripts/02_access_flags.py:1550-1579; reports/01_redundancy_cap.csv.

## 4. Provenance repair

The required pre-report commands were git status and git diff 27ac074 -- scripts/. The status was clean; the scripts diff contains eight paths, 5,730 insertions, and 11 deletions. Sources: reports/01_provenance_status.csv; reports/01_provenance_scripts_diff.csv.

The only canonical-output producer in that broad diff is scripts/02_access_flags.py (+57/-11). Relative to 27ac074, its current producer-relevant changes are:

1. the positive-layer provenance note (current line 30) and CACHE_SCHEMA_VERSION 2 to 3 (line 195);
2. the --legacy-layer-gate CLI flag (lines 318-322) and legacy-mode activation (line 359);
3. cache keys containing layer_positive or layer_nonzero and propagation into cache metadata/building (lines 453-524);
4. the corrected numeric >0 gate, legacy !=0 gate, and gate-aware segmentization (lines 882-953);
5. inundated-before-unclassified status precedence (lines 1582-1612);
6. prevention of flooded failed origins being marked unclassified (lines 1842-1846); and
7. propagation of legacy_layer_gate from main into cache loading/building (line 2335).

Source: reports/01_provenance_scripts_diff.csv; scripts/02_access_flags.py at the cited current lines.

The remaining scripts diff is not the producer patch: scripts/02f_bridge_rule_investigation.ipynb is a later investigation, scripts/02g_compare_bridge_rule_runs.py is a comparison utility, and five __pycache__ files are compiled artifacts. Source: reports/01_provenance_scripts_diff.csv.

History provides partial repair: cdbc0f7 records the status-partition changes and fcc532b records the positive-layer/cache changes; the current producer blob equals fcc532b and differs from the producer blob at 27ac074. Nevertheless, because all Della manifests identify 27ac074 with working_tree_dirty=true rather than either later commit, the exact run state is not provenance-complete. Source: reports/01_provenance_status.csv; data/processed/access/edited/della_runs/positive_layer_20260814_approach/run_manifest.json:102-104; data/processed/access/edited/della_runs/positive_layer_20260814_intersect/run_manifest.json:102-104; data/processed/access/edited/della_runs/positive_layer_20260814_retain/run_manifest.json:102-104.
