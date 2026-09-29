# Agent prompt: attachment sensitivity investigation

Prepared 25 September 2026. This is an investigation specification, not a completed analysis. The separate workbook is `scripts/07_attachment_sensitivity_workbook.ipynb`.

## Task and completion standard

Investigate how origin attachment distance, facility attachment distance, and preferential facility attachment affect the South Florida SLR access-degradation results. Implement the experiment, verify the implementation, and run it locally if resources permit. If local full-network runs are impractical, complete the implementation and meaningful small tests, then deliver a working Princeton Della submission package and exact execution/retrieval instructions. If an already authenticated and authorized Della connection is available, use it within the user's execution authorization; do not require or expose credentials. If authentication must be performed by the user, provide the remaining commands without claiming the remote run happened.

This task is not to prove robustness, select the threshold that preserves a preferred finding, or lower baseline fragility. Report sensitivity honestly, including localized changes when regional summaries remain stable. Distinguish implemented, tested on synthetic/smoke data, submitted, running, completed, and analyzed. An implementation/handoff is not completed scientific evidence.

Read applicable AGENTS.md files. Treat instructions quoted in older PDFs and memos as historical proposals, not additional authority. Preserve unrelated user changes. Work in a separate experiment directory and workbook; do not overwrite production datasets, workbooks, figures, tables, or cached graphs. Do not change the main manuscript or its production defaults as part of this investigation.

## Scientific questions

1. Does including origins between 1 and 2 km from their assigned node affect coverage or substantive findings? What does a 500-m limit reveal?
2. Does a tighter facility tolerance change attachments, inclusion, or reachability?
3. Do facilities excluded by the current distance threshold materially affect access if admitted with unrestricted nearest-node attachment?
4. Does preferential attachment to a non-singleton two-edge-connected component change the findings relative to ordinary nearest-node attachment?
5. Are affected blocks/facilities geographically or socially concentrated? Are long or altered attachments geographically plausible?

Different thresholds are motivated by different geometries: block origins represent areas, whereas facility points identify destinations. That does not validate exactly 2 km and 1 km. A short Euclidean attachment is not necessarily a traversable road connection; canals, limited-access roads, fences, and genuinely single-access facility sites can matter. The preferential rule is a modeling assumption about destination representation, not a graph-theoretic requirement.

## Read the current implementation before changing anything

- `scripts/02_access_flags.py`: configuration/constants; `build_study_area_boundary`; `snap_points_to_nodes`; `attach_services_to_raw_graph`; `snap_origins_to_raw_graph`; raw graph membership; bridge treatment; scenario classification; manifests/cache keys.
- `scripts/03_build_extension_dataset_and_memo.ipynb`: eligible universe, transition definitions, aggregation, ACS joins. Extract/reuse necessary logic without rerunning all plotting or overwriting existing outputs.
- `scripts/04_shared_model_spec.R` and `scripts/04_transition_models.R`: seven grouped-binomial models, risk denominators, standardization, AMEs/bootstrap, alternate physical specification.
- `scripts/05_population_figures.py`: five baseline-connected transitions and population totals.
- `scripts/02d_measurement_validity_diagnostics.py`, `scripts/02g_compare_bridge_rule_runs.py`: reusable audits, with careful attention to hardcoded paths/rule labels.
- `docs/methods_outline_20260925.tex`, `docs/physical_covariate_and_conley_extension.md`, `reports/03_arm_robustness.md`, and `outputs/run_comparison/final_methods_verification.md` for context. Reports can be stale: inspect live output workbooks and manifests.
- Canonical reference access run: `data/processed/access/edited/della_runs/positive_layer_20260814_approach/` (manifest, service audit, access Parquet).
- Primary model tables: `outputs/tables/ame_bootstrap_results_approach.xlsx` and `_approach_with_physical.xlsx`; primary input datasets under `data/processed/analysis/`.

As of preparation, `slurm/run_access_flags.sbatch` referenced by the workflow README does not exist. The supplied `docs/attachment_sensitivity/della_job.sbatch.template` is a generic guarded launcher, not an implemented sensitivity runner. Produce the actual runnable experiment and Slurm files; do not merely repeat nonexistent commands from the README.

## Current behavior to preserve and verify

- Origins: polygon representative points, nearest raw-largest-connected-component node, maximum 2,000 m; attachments fixed across scenarios. Positive land area plus valid origin snap determines eligibility. Do not add a population-positive filter.
- Candidate facilities: the same school/fire-station records within 10 km of the **retained-network bounding rectangle**, not the county polygon. Preserve candidate IDs, geometry, multiplicity, and facility types across arms. Do not silently deduplicate differently.
- Primary facilities: nearest raw-LCC node in a two-edge-connected component with at least two nodes within 1,000 m; fallback to the nearest node of the full raw graph within 1,000 m; otherwise exclude. This is a geometric nearest-node search, not tracing the actual driveway to its street entrance.
- All runs: same retained road PBF and classes; private-access exclusion; graph coordinate construction; representative points; positive layer gate; `approach` bridge rule; same NOAA 0--6 ft extents; fixed geographic extent; same facility pooling and facility-operability assumptions.
- Apply inundation and bridge rules at 0 ft too. Do not treat the raw graph as the baseline dry graph.
- Do not activate `--legacy-mode`. `--legacy-service-snap` may implement one narrow treatment, but verify precisely which behavior it changes before using it.
- Keep invalid/nonfinite facility geometry invalid even in an uncapped arm. No artificial nodes, new roads, or artificial access links are added to the graph.

## Required run matrix

Six arms, each evaluated at 0, 1, 2, 3, 4, 5, and 6 ft:

| ID | Origin limit | Facility treatment | Purpose |
|---|---:|---|---|
| `reference` | 2,000 m | Existing preferential rule, 1,000 m including fallback | Reproduce current behavior |
| `origin_1000` | 1,000 m | Exactly the reference facility map | Test the larger origin tolerance |
| `origin_500` | 500 m | Exactly the reference facility map | Stronger origin restriction |
| `facility_500` | 2,000 m | Existing preferential-plus-fallback algorithm, both capped at 500 m | Tighter facility attachment policy |
| `facility_add_uncapped` | 2,000 m | Preserve every reference-valid attachment; add reference-excluded candidates at their unconstrained nearest raw-graph node, with no finite distance cap | Isolate exclusion of distant facilities |
| `facility_nearest_1000` | 2,000 m | Attach every candidate to its unconstrained nearest full-raw-graph node within 1,000 m; no component preference | Test preferential attachment separately |

`facility_add_uncapped` is NOT permission to increase the preferred-node radius for already included facilities, move all destinations, expand the candidate geography, or download statewide extra facilities. An uncapped preferred-node search would be a different experiment and is outside the required matrix. Store the distance-limit state explicitly (e.g. JSON null plus a named rule), rather than emitting nonstandard JSON Infinity.

Under `facility_500`, an existing preferred snap beyond 500 m may be replaced by a valid unconstrained snap within 500 m; implement the actual algorithm, not just deletion of all current snaps over 500 m. Report additions, exclusions, and reattachments separately. Ordinary nearest-node treatment can also change facility eligibility; distinguish this from pure relocation in reporting.

## Isolation, provenance, and implementation architecture

1. Record git state/source hashes, input hashes, software versions, resolved paths, and existing output hashes before any mutation. Existing Della manifests record a dirty source tree: do not invent byte-for-byte provenance from a commit alone.
2. Put experiment outputs below `outputs/attachment_sensitivity/<experiment_id>/` and new intermediate access data below `data/processed/access/attachment_sensitivity/<experiment_id>/`. Never name a facility treatment as if it were a new bridge arm. Bridge rule stays `approach`; use a distinct `attachment_arm` field.
3. Build a small command-line runner/config interface with explicit treatment, thresholds, input/output/cache paths, scenarios, and reference attachment map. Reuse production functions where safe, preserving defaults. If editing shared functions is necessary, add default-equivalence tests and keep original CLI behavior unchanged.
4. IMPORTANT: `04_transition_models.R --data ...` changes the input but still writes arm-tagged outputs to `outputs/tables/`. Merely passing a different dataset would overwrite primary results. Add safe output-directory/run-tag support or use a verified isolated experiment entry point. Inspect every downstream writer, including notebook exports and population figures, for the same issue.
5. Use stable facility identifiers and retain a reference attachment registry. The production audit includes rejected candidates, not only retained facilities. Separate candidate counts, valid attachments, and in-graph reachable facilities in every scenario.
6. Write run/configuration manifests, stage-completion markers after validation, timings, and resume behavior. Do not treat partially written outputs as complete; writes should be atomic where practical. Cache keys must distinguish anything that affects a cached artifact.

## Efficient execution without changing the experiment

The raw road topology is unchanged across these arms. Verified segmentation and raw-membership caches can be reused. Reconstruct the graph from compatible caches if appropriate; do not rebuild identical geometry six times just to satisfy a phrase in the task. Build/load each scenario's retained graph once per suitable worker, then recompute facility-dependent connected-component/service counts and access classifications for each distinct facility map.

Origin-only thresholds change valid-origin membership, not facility placement or road topology. If code inspection and equivalence tests establish this, their eligible results can be derived from the full reference block records, with correct eligibility/exclusion flags and complete downstream rebuilding. Do not drop rows blindly or claim derived output is a fresh full classifier run. Document the exact reuse and prove shared retained blocks' states match. Facility treatments require new facility-dependent classification unless full relevant equivalence is proved.

Avoid multiple jobs writing one cache simultaneously. Either warm an isolated cache once and use it read-only, or use per-job caches. Avoid `--smoke` for scientific conclusions because it clips the network and changes connectivity; use it only to exercise code paths. A small full-network origin sample is useful for timing but still requires graph memory.

## Verification required before expensive runs

Use small synthetic networks with meaningful cases, not only tests mirroring the code:

- A loop/grid with a dead-end facility spur: nearest and preferred attachment differ for the intended reason.
- A real single-access facility site: document how the preferred rule bypasses that dependency instead of claiming it is universally a mapping error.
- A preferred node outside 500 m but an ordinary node inside 500 m: verify the fallback in `facility_500`.
- A reference-excluded facility with a finite distant nearest node: only the add-uncapped arm admits it, preserving reference-valid node assignments exactly.
- An invalid-coordinate candidate remains excluded when the distance cap is removed.
- Origins with distances on either side of and exactly equal to each threshold: confirm the inclusive <= rule, land-area eligibility, and zero-population handling.
- Two geographic road networks separated by a barrier: expose Euclidean attachment limitations; do not add an unrequested barrier-routing correction.

Reproduce the reference with current source and explain discrepancies before attributing any difference to sensitivity. Required checks include eligible GEOID sets; population totals; facility node assignments; bridge removals; inundation flags; four access-state flags; baseline transitions; and seven-model samples/point estimates. Existing reference totals are 70,695 input blocks, 68,521 eligible blocks, 6,135,688 eligible residents, and 2,163 retained facilities; verify against the actual files. Reproducing aggregate counts alone is insufficient.

For facility-only arms, origin membership and origin inundation flags must be identical. For add-uncapped, reference-valid attachments must match exactly and destinations form a superset; reachability/redundancy at a given water level should not worsen solely from adding facilities. Baseline-relative transition totals need not be monotone because the baseline itself can improve. Investigate any classification invariant violation, including coincident origin/facility nodes, rather than invoking a mathematical theorem to waive a code discrepancy. Unexpected classifier defects are findings: do not silently repair them only in an experimental arm.

## Attachment audit and map review

Export distributions and individual records for origin distance, facility distance, fallback use, candidate exclusion, preferred-node changes, old/new node IDs, and facility component membership. Include median, p90, p95, p99, maximum, affected block counts, and affected population. Separate all blocks from populated blocks for description without changing eligibility.

Important: added snap distance (new facility-to-node distance minus old facility-to-node distance) is NOT the physical displacement between old and new nodes. Compute both. A small difference in the two distances can conceal a move across a canal. Avoid repeating the old memo's interpretation that a median distance penalty proves all changes are tiny or correct.

Produce an inspectable GeoPackage/HTML map or map panels for changed/distant attachments. Include a deterministic sample from short and long distance bands, changed fallback cases, relevant counties/facility types, and the most influential changed areas. Inspect actual imagery/road connections only when suitable data are accessible; otherwise label geographic plausibility unverified. Do not manufacture an audited entrance or claim manual validation based on distances alone.

## Outcomes, denominators, and model comparisons

Produce, by arm and water level:

1. Candidate/retained/reattached/fallback facility counts, origin eligibility counts and population, exclusion reasons, and sample-composition summaries.
2. Baseline and scenario state counts/populations/shares and full reference-vs-arm state cross-tabs by GEOID. Report both absolute changes and changes in percentage points with explicit denominators.
3. The five baseline-connected transitions: redundant to fragile/isolated/inundated; fragile to isolated/inundated. Each arm uses its OWN baseline states. Define composite worse outcomes consistently.
4. Population totals for inundation only, isolation-or-inundation, and fragility-or-worse, and the non-inundation pathway share, separately at each scenario. Follow `05_population_figures.py`: these population summaries concern baseline-connected origins. Do not mix them with broader diagnostic new-inundated flags that include baseline-isolated origins. Do not sum people across scenarios as unique residents.
5. Geographic and demographic summaries of changed/excluded origins; use descriptive comparisons without pretending repeated blocks create independent evidence about ACS attributes.

Report two comparison views: (a) each arm's operational eligible sample and (b) pairwise common eligible origins. Origin restrictions should not change retained common origins' classifications; resulting model/sample differences reflect coverage. For facility arms, compare changes in baseline risk membership as well as subsequent outcomes. Any common-baseline-state regression check must explicitly condition on the intersection of baseline-risk origins; do not silently replace the primary arm-specific risk sets.

Rebuild block-group transition datasets; reuse the same ACS values and physical covariates by GEOID instead of making new API pulls. Reproduce the six social covariates, income transform, standardization order, complete cases, county/scenario fixed effects, and baseline block-count denominators. Zero-population eligible blocks remain included. Main AMEs are equally averaged over model rows, not population weighted. Do not confuse this experiment with the existing population-weighted-AME sensitivity.

For every arm with changed model inputs, fit all seven models under demographic-only AND physical-adjusted specifications. The age finding depends on specification, so do not validate only racial-composition coefficients. Use at least the current 199 successful block-group bootstrap replications for final comparable tables, the documented seeds/retry policy, percentile intervals, and the current normal-approximation p-values based on bootstrap SDs. Short bootstrap runs are smoke tests only. Report realized successful/failed replications and sample counts. Use consistent scaling within clearly labeled common-support comparisons. Where an entire model input and specification is exactly identical, reuse its verified result and document the equality instead of refitting needlessly.

Compare AME signs, magnitudes in probability and percentage-point units, intervals, and sample changes for ALL six social terms. Relative change is unstable near zero; report absolute differences too. A p=0.049 to p=0.051 crossing alone is not evidence of a materially different effect, and overlapping intervals are not a formal equivalence test. Do not claim significant differences between specifications without an appropriate comparison procedure. Sensitivity conclusions concern the tested choices, not proof that the network depicts every real connection correctly.

Before looking at sensitivity outcomes, record which manuscript conclusions and metrics will be assessed (baseline fragility, added population through non-inundation pathways, and social associations under both specs). Report the full ranges. Do not invent a universal pass/fail tolerance or stop checking once selected findings look stable.

## Local execution versus Della

Inspect available RAM, disk space, existing dependencies, compatible caches, and prior timing/resource logs before a full local run. The local environment previously used is `C:/Users/Vivek/miniforge3/envs/research-geo/python.exe`; verify it rather than using the Windows Store python alias. Use explicit interpreters, environment isolation, captured logs, and bounded worker/thread counts. The graph is largely serial NetworkX work; extra allocated CPUs do not automatically accelerate it. Avoid memory overcommit from concurrent graphs. Do not launch an uncontrolled full run to discover whether it fits.

Run locally if preflight and a representative benchmark demonstrate feasibility. If not, finish smoke/equivalence tests locally and prepare a complete Della handoff. Do not keep trying full local runs after reproducible resource failures. A Della benchmark should use a compute allocation and `jobstats`/`sacct` to size the full request. Do not infer peak full-network memory from a clipped smoke graph.

Use current official guidance and verify it again at execution time:

- Della and scheduler policies: https://researchcomputing.princeton.edu/systems/della
- Python environments: https://researchcomputing.princeton.edu/support/knowledge-base/python
- Memory requests: https://researchcomputing.princeton.edu/support/knowledge-base/memory
- First Slurm job: https://researchcomputing.princeton.edu/get-started/guide-princeton-clusters/3-first-slurm-job
- Job profiling: https://researchcomputing.princeton.edu/support/knowledge-base/jobstats

At preparation, Della uses Slurm; full graph/model computations belong on compute nodes. CPU work does not require a GPU. Let the scheduler select QOS unless current documentation requires otherwise; do not invent an account or partition. Discover available modules and reuse a working project environment or create a pinned isolated environment. The project manifest is useful for versions but does not prove a given environment exists on Della.

Historical manifests show `/scratch/gpfs/ERICTATE/vivek/slr_fl_fragile_access`; treat it as a path to verify, not a guaranteed account, current checkout, or writable allocation. Scratch is not a durable backup. Prefer existing cluster inputs with matching checksums to retransferring large files. A fresh git checkout will not contain the ignored `data/` directory. Provide an explicit input inventory and exact staging/symlink plan; do not rewrite canonical paths to absent data or use commands that overwrite unrelated files.

Required Della package:

1. A real, tested experiment CLI and validated configuration manifest. No command may depend on a flag that was merely proposed in this prompt.
2. Pinned environment requirements and an environment-setup script with verified module/env names, or clear user-specific fields if remote discovery is unavailable.
3. An entry-point shell script invoking the actual implemented stages, with failure propagation and logs.
4. An sbatch file (one node, explicit tasks/CPUs/memory/time, log locations created before submission). Explain measured or provisional resource sizing. Supply a small compute-node benchmark first if no full-network profile exists.
5. Separate graph and model stages if this helps resources/restarts, with `afterok` dependencies or a safe sequential pipeline. Use isolated/read-only caches. Serial arm execution or a limited array is preferable to uncontrolled simultaneous graph copies.
6. Copy-paste commands for connecting/staging, preflight, environment preparation, submission (`sbatch --parsable`), monitoring (`squeue`, logs), resource inspection (`sacct`/`jobstats`), cancellation of this investigation's job IDs only, resumption, and fetching the small tables/maps/manifests needed locally.
7. An explicit note that a submitted/pending job is not a completed result and that an afterok postprocessing job will not execute after a failed dependency without resubmission.

The supplied generic template accepts a project environment-setup file and an implemented entry-point file via environment variables. Turn it into the concrete scripts above. If remote execution is unavailable, do not claim to have verified paths, modules, resource sufficiency, or sbatch acceptance. Clearly separate tested local code from user-side cluster steps.

## Deliverables

- Update `scripts/07_attachment_sensitivity_workbook.ipynb` with provenance, run registry, commands/job IDs, validation results, tables, maps, and interpretation. Keep heavy runs in resumable scripts rather than a notebook cell that runs automatically. It must remain safe to open without starting jobs.
- An experiment run matrix/config file and isolated implementation with focused tests.
- `attachment_audit.csv`, `eligibility_comparison.csv`, `state_comparison.csv`, `transition_population_comparison.csv`, model-input/sample diagnostics, complete AME comparison tables for both specifications, and a spatial review artifact. Exact filenames may differ if clearly indexed in a README.
- A completed-results report with quantified changes and limitations, or a candid partial report plus complete Della runbook if runs cannot be completed in-session. Never fill result placeholders with hoped-for robustness.
- The actual local command or Della entry point/sbatch/environment files and step-by-step instructions.
- Suggested final Methods and Results text. Replace VA TO DO only for completed analyses and support every numerical sentence with an output path/table field. If thresholds or the preferred snapping assumption materially affect findings, say so and identify what needs revision.

## Known baseline observations, not sensitivity findings

A read-only audit of the saved primary baseline on 25 September 2026 found eligible-origin snap distances: median 47.827 m; p95 196.682 m; p99 535.754 m. There are 196 eligible blocks (7,251 residents) above 1,000 m and 777 (52,758 residents) above 500 m. These are coverage diagnostics, not rerun results or validated thresholds. Reproduce them from the saved Parquet.

The older facility memo's 13.5-m statistic is an added distance penalty, not measured node displacement. Some older bridge reports describe 49 bootstrap replications, whereas current principal workbooks contain 199. Current physical-adjusted older-adult intervals include zero for all seven models; avoid copying older unqualified claims about that association.

Start by inventorying the existing run and producing the isolated implementation plan, then execute the authorized work without repeatedly asking whether to continue. Request only genuinely missing credentials, account/path choices, or an interpretation decision that cannot be resolved from the repo. Do not select or suppress results to obtain the hoped-for conclusion.
