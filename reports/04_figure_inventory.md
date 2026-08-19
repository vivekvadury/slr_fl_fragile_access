# Phase 4 — Figure inventory

**Inventory only: no figure, notebook cell, or PNG was deleted or modified in this
phase.** Cell indexes below are zero-based. Current notebook filenames append the arm;
the approach path is shown, and the same template exists for `intersect` and `retain`.
The complete 59-PNG producer classification is in `reports/04_png_inventory.csv`.

## Notebook figures and recommendations

| Figure | Producing cell and current approach output | What it shows | Research question | Duplication | Recommendation |
|---|---|---|---|---|---|
| Figure 1 | Cell 25; `outputs/figures/fig1_status_shares_by_slr_approach.png` | Five-state shares of eligible blocks at SLR 1–6 ft for the three-county study area; the stack is asserted to sum to one. | RQ1 | No direct duplicate; it is the broad state-composition overview. | **KEEP.** It gives the cleanest complete-stage overview and explicitly includes `unclassified`. |
| Figure 2 | Cell 26; `outputs/figures/fig2_new_transitions_by_slr_approach.png` | Mutually exclusive newly fragile, isolated, and inundated outcomes as counts and shares of eligible blocks at SLR 1–6 ft, study-area aggregate. | RQ1 | Substantially repeats Figure 4b, but uniquely shows eligible-universe shares rather than cumulative counts. | **MERGE into Figure 4b.** Preserve the normalized-share information as a panel or secondary axis instead of carrying a separate figure. |
| Figure 3 | Cell 27; `outputs/figures/fig3_county_comparison_approach.png` | County lines for fragile-or-worse share and lost-redundancy share at SLR 1–6 ft. | none / exploratory | County stratification is unique, but it is not a pathway decomposition or a demographic test. | **CUT.** It describes county heterogeneity without directly answering RQ1–RQ3. |
| Figure 4, combined | Cell 29; `outputs/figures/fig4_network_transition_decomposition_approach.png` | Left: baseline-origin-to-scenario transition stacks; right: cumulative inundated, isolated-or-worse, and fragile-or-worse counts, SLR 1–6 ft. | RQ1 | Exact combined-layout duplicate of the two manuscript-width outputs from cell 30. | **CUT.** Retain Figures 4a and 4b rather than a third binary containing both panels. |
| Figure 4a | Cell 30; `outputs/figures/fig4a_transition_decomposition_approach.png` and `.pdf` | Redundant-to-fragile/isolated/inundated and fragile-to-isolated/inundated transition counts by SLR, three-county aggregate. | RQ1 | It is the left half of combined Figure 4, but not a duplicate of 4b. | **KEEP with Figure 4b.** It is the detailed origin-to-destination pathway panel. |
| Figure 4b | Cell 30; `outputs/figures/fig4b_cumulative_adverse_transitions_approach.png` and `.pdf` | Cumulative new inundated, isolated-or-worse, and fragile-or-worse counts by SLR, showing what inundation-only analysis omits. | RQ1 | It is the right half of combined Figure 4 and overlaps Figures 2 and 5. | **KEEP with Figure 4a.** It most directly visualizes the non-inundation-pathway argument. |
| Figure 5 | Cell 31; `outputs/figures/fig5_isolation_vs_redundancy_approach.png` | Eligible-block shares isolated, fragile-or-worse, and losing baseline redundancy at SLR 1–6 ft, with the isolation-to-fragility gap shaded. | RQ1 | Its central gap repeats Figure 4b; the isolated-share line is the main distinct element. | **MERGE into Figure 4b.** Preserve the isolated-share comparison, then remove the standalone figure if the merged panel is approved. |
| Figure A | Cell 37; `outputs/figures/figA_baseline_fragility_map_approach.png` | Block-group share fragile at baseline (0 ft), all three counties, with county outlines and a 95th-percentile color cap. | RQ1 | No exact duplicate, but it is baseline context rather than an SLR result. | **CUT from the main figure set.** It is useful appendix context but does not show an SLR pathway or change. |
| Figure B | Cell 38; `outputs/figures/figB_lost_redundancy_small_multiples_approach.png` | Block-group share losing redundancy at 4, 5, and 6 ft, three-county active-area zoom. | RQ1 | Figure C maps the same quantity at 3 ft under an algebraically equivalent label. | **KEEP AFTER REVISION.** It is the strongest spatial RQ1 progression, but its `total_pop > 0` display filter must be removed or explicitly reconciled with the eligible universe. |
| Figure C | Cell 39; `outputs/figures/figC_delta_fragile_or_worse_slr3_approach.png` | Change in block-group fragile-or-worse share at 3 ft versus baseline, all three counties. | RQ1 | Exact numerical duplicate of `share_lost_redundancy` at 3 ft: 0 of 3,942 block groups differ and the maximum floating-point difference is 2.22e-16. | **CUT.** It adds an earlier SLR panel to Figure B but no distinct metric or conclusion. |
| Figure D | **No producing cell**; the referenced arm-tagged file does not exist. Only legacy `outputs/figures/figD_vulnerability_demography_slr3.png` remains. | Intended side-by-side lost redundancy and `pct_nonwhite` at 3 ft. | none / exploratory | Repackages a network map beside one demographic surface without the multivariable controls used for RQ2/RQ3. | **CUT.** The notebook itself calls it exploratory/descriptive and disclaims causation; it is the visual form of the composite-index approach that the transition regressions replace. |
| Figure E | Cell 40; `outputs/figures/figE_transition_progression_slr4_approach.png` | Miami-Dade block-group shares of baseline-redundant blocks progressing to inundated, isolated-or-worse, and fragile-or-worse at **4 ft**, zoomed to active geography. | RQ1 | Related to Figure 4b, but uniquely spatializes its nested pathway. | **KEEP AFTER REVISION.** Correct the 3-ft/4-ft labels and remove or disclose the `total_pop > 0` display filter before use. |

The producing statements are recorded in
`scripts/03_build_extension_dataset_and_memo.ipynb:3855-4351` for Figures 1–5 and
`scripts/03_build_extension_dataset_and_memo.ipynb:4905-5252` for Figures A–E.
Figure 4's split outputs are created at lines 4276–4283. Figure C's exact equivalence
to Figure B's metric at 3 ft is sourced by
`reports/04_figure_metric_equivalence.csv`.

## Required flags before any cleanup

### Figure D is not a model-based social result

The Section 7A table describes Figure D as `share_lost_redundancy + pct_nonwhite`, and
the prose explicitly calls it exploratory and descriptive and says spatial co-occurrence
does not imply causation (`scripts/03_build_extension_dataset_and_memo.ipynb:4373` and
`:5301-5304`). No code cell currently produces it; the remaining untagged PNG is therefore
an orphan. This figure should not stand in for the adjusted RQ2/RQ3 transition models.

### Figure C does not add a different measure

Figure C is described as the 3-ft change in `share_fragile_or_worse`; Figure B maps
`share_lost_redundancy` at 4–6 ft. On the eligible approach block-group table, those two
metrics are equal for every block group at 3 ft, to floating-point precision. Figure C
therefore says only what a 3-ft panel added to Figure B would say, not something
conceptually different. Source: `reports/04_figure_metric_equivalence.csv`.

### Figure E is 4 ft, not 3 ft

Cell 40 sets `SLR_FOR_E = 4`, filters 4-ft rows, and saves
`figE_transition_progression_slr4_<arm>.png`
(`scripts/03_build_extension_dataset_and_memo.ipynb:5152-5252`). In contrast, the
Section 7A table says 3 ft, its explanatory markdown names the `slr3` file, and the
printed saved-file summary checks that nonexistent `slr3` path
(`scripts/03_build_extension_dataset_and_memo.ipynb:4374`, `:5268`, and `:5307`). The
stored output consequently reports the real `slr4` file as saved and the false `slr3`
entry as not saved (`scripts/03_build_extension_dataset_and_memo.ipynb:5116-5125`). This
must be corrected in Phase 6 regardless of the keep/cut decision.

### Two maps narrow the decided universe for display

Figure B filters to block groups with `total_pop > 0`, and Figure E applies the same
filter in Miami-Dade (`scripts/03_build_extension_dataset_and_memo.ipynb:4949-4953` and
`:5169`). Their mapped shares were calculated using eligible-block denominators, but the
display omits zero-population eligible geography. Any retained version should remove the
filter or label this as a display-only restriction; it must not be described as the full
eligible spatial universe.

## PNGs with no current producer

`outputs/figures/` contains 59 PNGs: 33 match the 11 current notebook templates for all
three arms, two approach PNGs are currently produced by
`scripts/05_population_figures.py:306-355`, and the following 24 have no current
producing notebook cell or script. Source: `reports/04_png_inventory.csv`.

- `outputs/figures/fig1_status_shares_by_slr.png`
- `outputs/figures/fig2_new_transitions_by_slr.png`
- `outputs/figures/fig3.png`
- `outputs/figures/fig3_county_comparison.png`
- `outputs/figures/fig3_redundancy_loss_maps.png`
- `outputs/figures/fig4_network_transition_decomposition.png`
- `outputs/figures/fig4_path_inflation_ratio_dist.png`
- `outputs/figures/fig4a_population_transition_decomposition.png`
- `outputs/figures/fig4a_transition_decomposition.png`
- `outputs/figures/fig4b_cumulative_adverse_transitions.png`
- `outputs/figures/fig4b_population_cumulative_adverse_transitions.png`
- `outputs/figures/fig4c_population_threshold_ladder.png`
- `outputs/figures/fig5_isolation_vs_redundancy.png`
- `outputs/figures/fig6_demographic_scatter_slr3.png`
- `outputs/figures/figA_baseline_fragility_map.png`
- `outputs/figures/figC_delta_fragile_or_worse_slr3.png`
- `outputs/figures/figD_vulnerability_demography_slr3.png`
- `outputs/figures/figE.png`
- `outputs/figures/figE_isolation_vs_fragile_or_worse_map.png`
- `outputs/figures/figE_transition_progression_slr3.png`
- `outputs/figures/figE_transition_progression_slr4.png`
- `outputs/figures/figE1_redundant_to_fragile_small_multiples.png`
- `outputs/figures/figE2_redundant_to_isolated_or_inundated_small_multiples.png`
- `outputs/figures/figE3_redundant_to_inundated_small_multiples.png`

The two current non-notebook PNGs are
`outputs/figures/fig4a_population_transition_decomposition_approach.png` and
`outputs/figures/fig4b_population_cumulative_adverse_transitions_approach.png`; they are
not orphans because script 05 produces them (`scripts/05_population_figures.py:306-355`).

## Proposed cut/merge list for approval

- **Keep:** Figures 1, 4a, and 4b.
- **Keep after revision:** Figures B and E.
- **Merge:** Figure 2 into 4b; Figure 5's isolated-share comparison into 4b.
- **Cut:** Figure 3, combined Figure 4, Figure A, Figure C, and Figure D.

This is a recommendation only. Phase 6 was not run, and no listed file or cell has been
deleted.
