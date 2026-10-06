# Future work (deliberately out of scope for the October 2026 draft)

Items here were raised during the analysis freeze and set aside so the draft
could close. None is required for the claims the draft makes. Each can be
framed in the Discussion as future research.

## Inference
- **Network-distance spatial HAC.** Conley standard errors use Euclidean
  (great-circle) distance between block-group centroids. Dependence induced by
  shared road links follows network distance, not straight-line distance. A
  kernel defined on road-network distance, or on shared critical links, would
  match the dependence mechanism more closely.
- **Model-based spatial dependence.** Spatial random effects (e.g., BYM/ICAR)
  or a spatial-lag outcome model would model dependence rather than only
  correcting standard errors.
- **Overdispersion in the mean model.** Beta-binomial or observation-level
  random-effect versions of the grouped models (Pearson dispersion 1.8–7.0
  under the demographic-only specification).

## Measurement
- **Travel time and congestion.** The access states are topological
  (connectivity and edge-disjoint redundancy), not travel-time based.
- **Additional facility types** (pharmacies, grocery, dialysis, shelters) and a
  grade-level filter for schools (the school layer currently includes all K–12
  public schools).
- **Network-distance or road-segment attachment** in place of nearest-node
  attachment with distance caps.
- **Ferry-only and private-access communities** (e.g., Fisher Island) need
  explicit treatment rather than road-network attachment.
- **Dynamic flooding** (tidal timing, compound rainfall/storm surge,
  groundwater emergence) rather than static MHHW-referenced bathtub extents.

## Physical setting
- **Elevation as a mediator.** Elevation partly encodes the inundation rule
  that produces the outcome, so the physical-adjusted models are a sensitivity
  analysis, not a decomposition. A design that separates exposure from network
  position (e.g., modeling only dry-origin transitions) could address this.
