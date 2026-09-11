# Scripts Workflow

This directory now emphasizes the current manuscript workflow for
sea-level-rise-induced transportation access degradation in South Florida.
The workflow classifies blocks as redundant, fragile, isolated, inundated, or
unclassified;
aggregates block-level transitions to block groups; and estimates grouped
binomial transition models linked to social vulnerability indicators.

## Suggested Run Order

1. `01_pull_census_geometries.py`
   - Run only when 2020 TIGER block or block-group geometry needs to be
     downloaded or rebuilt.
   - Writes tri-county processed geometry files under
     `data/processed/census/`.

2. `02_access_flags.py`
   - Core block-level access-state engine.
   - Builds the undirected drivable road graph; attaches services to
     redundancy-eligible nodes in the raw graph's largest connected component;
     snaps within-polygon block origins to the raw LCC; and applies scenario
     inundation and physical-bridge rules.
   - Keeps every block in the long output. `analysis_eligible` and
     `exclusion_reason` identify zero-land-area blocks and failed origin snaps.
   - Main outputs are `block_access_flags_long*.csv` and
     `block_access_flags_long*.parquet` under a directory named by
     `--config-name`. Each run also writes `run_manifest.json`, a service-snap
     audit, and one physical-bridge audit per scenario.
   - Segmentized roads are cached as GeoParquet by highway-filter hash. Use
     `--resume` to reuse the raw-graph component/2ecc cache and
     `--rebuild-cache` to force recomputation.
   - `--scenarios 0` runs baseline only; the default is all seven scenarios.
     `--bridge-rule` accepts `intersect`, `approach` (default), or `retain`.
   - Lightweight corrected smoke run:
     `python scripts/02_access_flags.py --smoke --scenarios 0 --config-name smoke_corrected`.
   - Published-behavior verification:
     `python scripts/02_access_flags.py --legacy-mode --bridge-rule intersect --scenarios 0 --config-name legacy_0ft`.

3. `02b_diagnose_access_run.py` (diagnostic/QA)
   - Reads a completed access run directory, deduplicates block-scenario rows,
     exports status and transition summaries, and writes selected QA maps.
   - Does not rerun the access model.

4. `02c_graph_component_diagnostics.py` (diagnostic/QA)
   - Rebuilds the current `02_access_flags.py` road graph and compares raw
     graph components to the 0 ft dry graph.
   - Useful for validating baseline fragile/isolated classifications and
     component structure.
   - Expensive enough to treat as diagnostic rather than part of every rerun.

5. `03_build_extension_dataset_and_memo.ipynb`
   - Core analysis notebook.
   - Stacks block-level access output, validates transition summaries,
     aggregates to block groups, pulls/merges ACS vulnerability indicators, and
     writes the block-level and block-group analysis datasets under
     `data/processed/analysis/`.
   - Also contains manuscript figure design and map iteration. Notebook figure
     cells were retained intentionally.

6. `03b_join_elevation_drainage.py` (optional; feeds the alternate spec)
   - Builds `data/processed/analysis/block_group_physical_covariates.csv`,
     one row per analysis block group with `elevation_m_mean`,
     `elevation_m_median`, and `drainage_distance_km`.
   - Elevation: pinned, checksum-verified USGS 3DEP seamless 1/3-arc-second
     bare-earth DEM tiles (same USGS/TNM lidar-derived family behind NOAA's
     SLR Viewer DEM); zonal mean/median over the `slr_0ft` block-group
     polygons. Drainage: centroid distance (km) to the nearest SFWMD AHED
     `HYDRO_ORDER = 'PRIMARY'` canal/primary feature (a continuous proxy was
     chosen over a categorical basin fixed effect; see the script docstring).
   - First run downloads ~0.8 GB of DEM inputs to
     `data/raw/physical_covariates/` and reuses the validated cache after
     that; `--download-only` and `--no-download` are available. Requires
     network access to `prd-tnm.s3.amazonaws.com` and `geoweb.sfwmd.gov`.
   - Only needed before running `04`/`04b` with the `with_physical`
     specification. The default demographic-only pipeline does not read it.
   - Warning carried in the docstring: elevation is mechanically close to the
     classifier's inundation rule, so check Inundated-outcome models for
     separation/convergence when using this file.

7. `04_transition_models.R`
   - Core manuscript regression/table script.
   - Estimates grouped binomial transition models for:
     - baseline redundant to fragile, isolated, inundated, or worse;
     - baseline fragile to isolated, inundated, or worse.
   - Uses standardized block-group vulnerability indicators:
     non-Hispanic Black share, Hispanic share, renter share, log median
     household income, age 65+ share, and no-vehicle household share.
   - Exports:
     - `outputs/tables/ame_bootstrap_results.xlsx`
     - `outputs/tables/ame_bootstrap_transition_table.tex`
   - Expensive because it runs a cluster bootstrap. Set `AME_BOOT_REPS` to a
     small value for syntax/runtime smoke checks, but use the manuscript
     default for final table regeneration.
   - Covariate/spec logic and the shared fitting helpers live in
     `04_shared_model_spec.R`, which both this script and `04b` source.
   - `MODEL_SPEC` (default `demographic_only`) selects the specification.
     `with_physical` adds standardized mean elevation and primary-drainage
     distance from `03b`'s CSV as an explicitly labeled, alternate
     specification; it never overwrites the demographic-only outputs
     (arm-tagged filenames gain a `_with_physical` suffix) and, when both
     specs have been run for an arm, also writes
     `transition_model_spec_comparison_<arm>.csv` (point estimate and SE
     under each spec, plus the percent change in the Black and Hispanic
     coefficients). Overcontrol is a known risk with `with_physical`; treat
     it as a sensitivity check, not a replacement.
   - `CONLEY_CUTOFF_KM` (unset by default) additionally reports
     Conley/HAC spatial standard errors (`conley_se`, `conley_cutoff_km`
     columns) alongside the existing clustered and bootstrap SEs, using
     block-group centroids and the cutoff in kilometers you supply. These are
     bandwidth-sensitive and non-positive-definite at wider cutoffs for this
     panel; see `docs/physical_covariate_and_conley_extension.md`.
   - `AME_POP_WEIGHT` (unset by default; `1` resolves to `eligible_pop20`)
     additionally writes `ame_bootstrap_*_approach_popweighted.{xlsx,tex}`,
     a population-weighted variant of the AME table that sits alongside
     Table 2. The model fit is unchanged; only the `avg_slopes()` average is
     population-weighted.

8. `04b_spatial_residual_diagnostics.R` (diagnostic/QA)
   - Refits the seven approach-arm transition models (sourcing
     `04_shared_model_spec.R`, without running `04`'s bootstrap or manuscript
     exports) and tests each model's Pearson residuals for spatial
     autocorrelation with Moran's I at every positive SLR scenario (42 tests).
   - `--spec {demographic_only,with_physical,both}` (default
     `demographic_only`) picks which specification's residuals are tested.
     `demographic_only` keeps the historical
     `outputs/run_comparison/moran_residual_diagnostics_approach.*` filenames
     and refreshes Sections 4-5 of `final_methods_verification.md`;
     `with_physical` writes `_with_physical`-suffixed siblings and leaves the
     report untouched; `both` runs each and also writes
     `moran_residual_diagnostics_spec_comparison.csv`, a 42-row paired table
     with a column counting how many tests lose 0.05 significance under the
     physical-covariate spec.
   - The Pearson-dispersion range quoted in the report is now computed live
     from this script's fitted models.

9. `05_population_figures.py`
   - Population-weighted manuscript table/figure supplement.
   - Joins 2020 Census block population to
     `data/processed/analysis/block_level_long_dataset.csv`.
   - Writes:
     - `outputs/tables/fig4_transition_population_by_slr.csv`
     - `outputs/tables/fig4_cumulative_population_by_slr.csv`
     - `outputs/figures/fig4a_population_transition_decomposition.[png|pdf]`
     - `outputs/figures/fig4b_population_cumulative_adverse_transitions.[png|pdf]`
   - The `fig4_*` filenames are retained for compatibility with existing
     notebooks and draft-placeholder code, even though the current manuscript
     may number these figures differently.

10. `06_placeholders_in_draft.ipynb`
   - Small draft-support notebook that reads existing output tables and prints
     manuscript replacement text for numeric placeholders.
   - Kept because it documents how draft prose numbers were derived.

11. `02e_compare_runs.py` (correction/sensitivity comparison)
   - Takes legacy and corrected run directories.
   - Writes per-scenario old-vs-new status matrices, baseline fragile shares
     for full/populated/eligible universes, and a block-level baseline-change
     file containing population, county, service-snap, and bridge fields.

12. `slurm/run_access_flags.sbatch`
    - Cluster submission template with TODO resource/module settings.
    - Places graph caches on scratch and parameterizes configuration name,
      bridge rule, and scenario list.

## Exploratory Notebooks

`00_data_exploration.ipynb` is an older exploratory notebook. It remains in
place because notebooks are treated conservatively: figure/design exploration
and early data checks can still be useful context, and notebook deletion should
be reviewed explicitly before removal.

## Environment Assumptions

- Run commands from the repository root.
- Python dependencies include `geopandas`, `pyogrio`, `networkx`, `numpy`,
  `pandas`, `pyarrow`, `pyproj`, `scipy`, `shapely`, and `matplotlib`.
- R dependencies include `tidyverse`, `fixest`, `marginaleffects`,
  `openxlsx`, and (for `04b`) `sf` and `spdep`.
- The full access workflow assumes local access to the NOAA SLR geopackage,
  processed service layers, processed Census geometries, and the retained OSM
  road PBF listed in `02_access_flags.py`.
- The `with_physical` specification additionally needs
  `data/processed/analysis/block_group_physical_covariates.csv` from `03b`,
  whose first run downloads pinned USGS 3DEP and SFWMD source data.
- Full access sweeps are intended for the cluster; use
  `slurm/run_access_flags.sbatch` after filling its site-specific TODOs.

## Known Limitations

- The road graph is undirected and does not model one-way restrictions, turn
  restrictions, congestion, speeds, drainage, or road depth.
- Ordinary road segments are removed if their geometry intersects a NOAA SLR
  layer; the workflow does not split roads at flood boundaries. Bridge-like
  edges use the selected connected-structure rule.
- Block inundation is origin-point based. The corrected origin geometry is a
  polygon representative point; legacy centroid behavior remains switchable.
- Service access currently combines primary schools and fire stations into one
  essential-services layer.
- Grouped binomial denominators are block counts, while vulnerability variables
  describe people and households at the block-group level.
- Bootstrap uncertainty is implemented for the manuscript AME table. Spatial
  dependence is now diagnosed by `04b` and can be addressed with the optional
  `CONLEY_CUTOFF_KM` Conley/HAC SEs in `04`; further quasi-binomial and
  denominator sensitivity checks are documented in
  `docs/manuscript_feedback_todo.md` but not implemented here.

## Diagnostics Only

- `02b_diagnose_access_run.py`
- `02c_graph_component_diagnostics.py`
- `02d_measurement_validity_diagnostics.py`
- `02e_compare_runs.py`
- `04b_spatial_residual_diagnostics.R`

These scripts validate outputs and graph structure. They do not replace the
core access run or analysis-dataset construction.
