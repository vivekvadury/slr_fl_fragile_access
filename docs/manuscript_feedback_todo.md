# Manuscript Feedback TODO

This file records future extensions to consider after the current cleanup. The
items below are intentionally not implemented in the retained scripts yet.

## 1. Service-Specific Access

- Separate primary school access from fire-station access instead of treating
  both as one essential-services layer.
- Consider presenting service-specific access as a robustness check rather than
  replacing the pooled essential-service measure.
- Likely code touchpoints:
  - `scripts/02_access_flags.py`: service loading, reachable-service logic, and
    access-state classification.
  - `scripts/03_build_extension_dataset_and_memo.ipynb`: block-level to
    block-group aggregation and figure/table construction.
  - `scripts/04_manuscript_transition_models.R`: grouped binomial models if
    service-specific transition outcomes are modeled.

## 2. Estimand / Denominator Sensitivity

- Clarify in the manuscript that grouped binomial denominators are counts of
  baseline-origin blocks, while vulnerability indicators describe people and
  households at the block-group level.
- Consider population-weighted models or population-denominator sensitivity
  checks.
- Likely code touchpoints:
  - `scripts/03_build_extension_dataset_and_memo.ipynb`: construct population
    denominators and block-group analysis variants.
  - `scripts/04_manuscript_transition_models.R`: add alternative model
    specifications after the main manuscript models.
  - `scripts/05_manuscript_population_figures.py`: population-weighted
    descriptive outputs.

## 3. Uncertainty / Overdispersion / Spatial Clustering

- Test quasi-binomial or other overdispersion diagnostics for grouped binomial
  outcomes.
- Consider spatially clustered or spatially explicit uncertainty checks.
- Keep the current cluster-bootstrap AME table as the manuscript baseline until
  a sensitivity design is chosen.
- Likely code touchpoints:
  - `scripts/04_manuscript_transition_models.R`: quasi-binomial or alternative
    uncertainty models.
  - `scripts/03_build_extension_dataset_and_memo.ipynb`: spatial joins if
    model-ready block-group geometries are needed.

## 4. Network Validation

- Validate graph-based classification against a routing-engine implementation,
  such as OSRM, to test whether one-way restrictions, planar grade separations,
  and travel-time costs change fragile/isolated classifications.
- Test directed-network checks.
- Test alternative road-removal rules, including depth/elevation-aware or
  probabilistic passability assumptions.
- Likely code touchpoints:
  - `scripts/02_access_flags.py`: graph construction and scenario edge-removal
    logic.
  - `scripts/02b_diagnose_access_run.py`: comparison summaries for validation
    runs.
  - `scripts/02c_graph_component_diagnostics.py`: graph-structure diagnostics.

## 5. Origin and Inundation Sensitivity

- Compare centroid-based inundation to population-weighted origins or
  block-level residential point locations.
- Consider replacing or supplementing block centroids with representative
  points, residential parcel/address points, or population-weighted origins
  where available.
- Likely code touchpoints:
  - `scripts/02_access_flags.py`: origin construction, snapping, and inundation
    status assignment.
  - `scripts/03_build_extension_dataset_and_memo.ipynb`: aggregation and
    comparison of sensitivity outputs.
  - `scripts/05_manuscript_population_figures.py`: population-weighted
    affected-population summaries.
