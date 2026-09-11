# Manuscript Feedback TODO

This file records future extensions to consider after the current cleanup.
Updated 2026-09-09: script-name references corrected to match the current
`scripts/` layout, and each item marked with its current disposition
(closed, updated, or explicitly deferred) rather than left as an undated
blanket TODO.

## 1. Service-Specific Access — DEFERRED, revisit if raised in review

Separate primary school access from fire-station access instead of pooling
both into one essential-services layer. No new analysis has been done on
this; explicitly out of scope for the current submission. Likely code
touchpoints if revisited: `scripts/02_access_flags.py` (service loading,
reachable-service logic, access-state classification),
`scripts/03_build_extension_dataset_and_memo.ipynb` (block-level to
block-group aggregation and figure/table construction), and
`scripts/04_transition_models.R` (grouped binomial models if
service-specific transition outcomes are modeled).

## 2. Estimand / Denominator Sensitivity — CLOSED

Addressed via the 2026-09-02 population-weighted AME table
(`AME_POP_WEIGHT=1` in `scripts/04_transition_models.R`, writing
`ame_bootstrap_*_approach_popweighted.{xlsx,tex}`): weighting `avg_slopes()`
by `eligible_pop20` attenuates the racial AMEs by only 2.7-5.9% (mean ~4%),
all still significant at p<0.001 — the finding is robust to weighting
residents vs. block groups. A full person-level respecification (rebuilding
the model around individual/household denominators rather than
population-weighting the existing block-group-denominator model) is
explicitly out of scope for this submission and not planned.

## 3. Uncertainty / Overdispersion / Spatial Clustering — UPDATED, partially addressed

Spatial clustering is now addressed: `scripts/04_transition_models.R`
supports `CONLEY_CUTOFF_KM` (Conley/HAC spatial-robust SEs alongside the
existing clustered and cluster-bootstrap SEs). A 2026-09-08 bandwidth sweep
found the Conley estimator non-positive-definite at wide cutoffs for this
panel (intrinsic to the uniform-kernel estimator at bandwidths large
relative to the ~0.6 km median nearest-neighbour spacing, not a panel-
structure artifact); **5 km is the validated cutoff** (all seven models
positive-definite, Conley SE ~1.5-3x the clustered SE, all 28 racial
coefficients still significant, largest p=0.04). See
`docs/conley_bandwidth_justification_note.md` and
`docs/physical_covariate_and_conley_extension.md`. The cluster bootstrap
remains the primary manuscript uncertainty estimate; Conley@5km is reported
alongside it, not in place of it.

Overdispersion has been measured, not fixed: `scripts/04b_spatial_residual_diagnostics.R`
computes live Pearson dispersion (`sum(Pearson residual^2) / residual df`)
for the seven grouped-binomial fits. As of this round (2026-09-09,
`demographic_only` spec) dispersion ranges **1.832-6.984** across the seven
models — all seven are overdispersed. This is a distinct concern from the
spatial autocorrelation that Conley SEs target: cluster-bootstrap and
Conley SEs partially absorb extra-binomial variance within/across clusters,
but neither is a substitute for addressing dispersion directly.
**Quasi-binomial refitting itself remains undone** — explicitly out of
scope for this round; the current AME table's point estimates are unchanged
by this diagnostic, only the case for supplementing its uncertainty
treatment is strengthened. Likely touchpoint if taken up:
`scripts/04_transition_models.R` (quasi-binomial or alternative uncertainty
models); spatial joins for model-ready block-group geometries are already
available via `scripts/04_shared_model_spec.R` / `scripts/04b_spatial_residual_diagnostics.R`.

## 4. Network Validation — DEFERRED, revisit if raised in review

Validate graph-based classification against a routing-engine implementation
such as OSRM (one-way restrictions, planar grade separations, travel-time
costs) and test alternative/probabilistic road-removal rules. No new
analysis has been done on this; explicitly out of scope for the current
submission. Likely code touchpoints if revisited:
`scripts/02_access_flags.py` (graph construction and scenario edge-removal
logic), `scripts/02b_diagnose_access_run.py` (comparison summaries for
validation runs), `scripts/02c_graph_component_diagnostics.py`
(graph-structure diagnostics).

## 5. Origin and Inundation Sensitivity — DEFERRED, revisit if raised in review

Compare centroid-based inundation to population-weighted origins or
block-level residential point locations, as an alternative or supplement to
the current representative-point origin geometry. No new analysis has been
done on this; explicitly out of scope for the current submission. Likely
code touchpoints if revisited: `scripts/02_access_flags.py` (origin
construction, snapping, and inundation status assignment),
`scripts/03_build_extension_dataset_and_memo.ipynb` (aggregation and
comparison of sensitivity outputs), `scripts/05_population_figures.py`
(population-weighted affected-population summaries).
