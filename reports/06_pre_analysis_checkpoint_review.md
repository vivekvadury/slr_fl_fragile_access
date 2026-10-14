# Pre-analysis checkpoint review

Review date: 14 October 2026. Scope: pending changes on `main` since `d139f38`,
including notebook 06, origin-placement sensitivity scripts and outputs, the
social-vulnerability audit, and the elevation/drainage audit bundle.

## Code finding fixed

`scripts/origin_placement_sensitivity/07_compare_models.py` previously discarded
failed fits and merely logged canonical reproduction differences. A rerun could
therefore publish sensitivity summaries from an incomplete experiment or a
canonical fit that no longer matched production.

The comparison now stops before writing summaries if fits fail, replications
do not match the classification manifest, coefficient keys are missing or
duplicated, coefficients/standard errors are non-finite, or canonical
coefficients/standard errors differ from production by more than 1e-10.
Seven synthetic regression tests cover the gate.

## Checks performed

- Parsed all Python scripts and Python code cells in notebook 06.
- Parsed all eight R scripts in `scripts/` and the elevation/drainage audit.
- Executed all 13 code cells of notebook 06 against the local manuscript PDF
  and analysis inputs, without rewriting the notebook. All assertions passed.
  The manuscript comparison reported 30 matching claims and three discrepancies.
- Ran the existing attachment/network synthetic suite: 11/11 passed.
- Ran the new model-validation suite: 7/7 passed.
- Validated the saved canonical fit and all 50 Monte Carlo model replications
  with the new gate, and executed the full comparison stage with writes suppressed.
- Inspected saved validation manifests: zero reported canonical state,
  production pilot, node-state, bridge, inundation, or transition mismatches;
  178 figure-table values reproduced; block-group counts reproduced for 27,594
  rows; canonical model coefficient differences below 1.5e-15.

The road-network builds and R model fits were not rerun. Saved manifests are
historical validation evidence, not new full-pipeline verification. Original
outputs and run hashes remain intact; their script hashes describe the code
used for those runs, before this review's comparison-gate change.

## Findings relevant to the next manuscript edit

Notebook 06 identifies Section 5.3 / Figure 4 counts of 3,738 / 7,788 / 15,717
at 4 / 5 / 6 ft as counts restricted to block groups with ACS population above
zero. Counts across all eligible blocks are 3,754 / 7,807 / 15,747. The prose
should specify the restricted universe or use the study-wide counts.

Notebook 06 is deliberately tied to local manuscript version (45), including
its SHA-256 and line numbers. A different draft requires updating that audit.
Raw data and large regenerable sensitivity intermediates remain Git-ignored;
the repository alone is not sufficient for a full rerun.
