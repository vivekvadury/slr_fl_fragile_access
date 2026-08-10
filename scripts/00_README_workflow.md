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

6. `04_transition_models.R`
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

7. `05_population_figures.py`
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

8. `06_placeholders_in_draft.ipynb`
   - Small draft-support notebook that reads existing output tables and prints
     manuscript replacement text for numeric placeholders.
   - Kept because it documents how draft prose numbers were derived.

9. `02e_compare_runs.py` (correction/sensitivity comparison)
   - Takes legacy and corrected run directories.
   - Writes per-scenario old-vs-new status matrices, baseline fragile shares
     for full/populated/eligible universes, and a block-level baseline-change
     file containing population, county, service-snap, and bridge fields.

10. `slurm/run_access_flags.sbatch`
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
- R dependencies include `tidyverse`, `fixest`, `marginaleffects`, and
  `openxlsx`.
- The full access workflow assumes local access to the NOAA SLR geopackage,
  processed service layers, processed Census geometries, and the retained OSM
  road PBF listed in `02_access_flags.py`.
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
- Bootstrap uncertainty is implemented for the manuscript AME table; additional
  quasi-binomial, spatial, and denominator sensitivity checks are documented in
  `docs/manuscript_feedback_todo.md` but not implemented here.

## Diagnostics Only

- `02b_diagnose_access_run.py`
- `02c_graph_component_diagnostics.py`
- `02d_measurement_validity_diagnostics.py`
- `02e_compare_runs.py`

These scripts validate outputs and graph structure. They do not replace the
core access run or analysis-dataset construction.
