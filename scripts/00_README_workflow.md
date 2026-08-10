# Scripts Workflow

This directory now emphasizes the current manuscript workflow for
sea-level-rise-induced transportation access degradation in South Florida.
The workflow classifies blocks as redundant, fragile, isolated, or inundated;
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
   - Builds the undirected drivable road graph, removes SLR-intersecting road
     segments by scenario, snaps block centroids and essential services to the
     graph, and classifies each block x SLR scenario as inundated, isolated,
     fragile, or redundant.
   - Main outputs are `block_access_flags_long*.csv` and, when pyarrow is
     available, `block_access_flags_long*.parquet` under
     `data/processed/access/edited/`.
   - Expensive for the full block universe. For a lightweight smoke command,
     use something like:
     `python scripts/02_access_flags.py --max-blocks 500 --slr-ft 1 --output-suffix __smoke`.

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

6. `04_manuscript_transition_models.R`
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

7. `05_manuscript_population_figures.py`
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

8. `07_placeholders_in_draft.ipynb`
   - Small draft-support notebook that reads existing output tables and prints
     manuscript replacement text for numeric placeholders.
   - Kept because it documents how draft prose numbers were derived.

## Exploratory Notebooks

`00_data_exploration.ipynb` is an older exploratory notebook. It remains in
place because notebooks are treated conservatively: figure/design exploration
and early data checks can still be useful context, and notebook deletion should
be reviewed explicitly before removal.

## Environment Assumptions

- Run commands from the repository root.
- Python dependencies include `geopandas`, `pyogrio`, `networkx`, `numpy`,
  `pandas`, `pyproj`, `scipy`, `shapely`, and `matplotlib`.
- R dependencies include `tidyverse`, `fixest`, `marginaleffects`, and
  `openxlsx`.
- The full access workflow assumes local access to the NOAA SLR geopackage,
  processed service layers, processed Census geometries, and the retained OSM
  road PBF listed in `02_access_flags.py`.
- No retained script is clearly cluster-only. The full access run and bootstrap
  table can be slow locally, but the retained code does not encode a cluster
  submission workflow.

## Known Limitations

- The road graph is undirected and does not model one-way restrictions, turn
  restrictions, congestion, speeds, bridges/tunnels, drainage, or road depth.
- Road segments are removed if their geometry intersects a NOAA SLR layer; the
  workflow does not split roads at flood boundaries.
- Inundation is centroid-based at the block level.
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

These scripts validate outputs and graph structure. They do not replace the
core access run or analysis-dataset construction.
