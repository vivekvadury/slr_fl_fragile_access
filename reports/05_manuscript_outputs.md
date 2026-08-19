# Phase 5 — Manuscript-facing outputs

## Outcome

**PASS.** Exactly two Phase 5 artifacts were written under `outputs/tables/`:

1. `outputs/tables/bridge_rule_sensitivity.tex`
2. `outputs/tables/service_snapping_robustness.csv`

No manuscript prose or analysis source was changed.

## Bridge-rule sensitivity table

The LaTeX table orders the arms from pessimistic `intersect` to default `approach` to
optimistic `retain`, uses the common 68,521-block eligible universe, retains eligible
zero-population blocks in block denominators, and uses `pop20` for population totals.
It contains baseline fragile share; newly fragile blocks at 2, 4, and 6 ft; newly
affected population at 6 ft; and non-inundation block share at 2 and 6 ft. Sources:
`outputs/tables/bridge_rule_sensitivity.tex`,
`data/processed/access/edited/della_runs/positive_layer_20260814_intersect/block_access_flags_long.parquet`,
`data/processed/access/edited/della_runs/positive_layer_20260814_approach/block_access_flags_long.parquet`,
and
`data/processed/access/edited/della_runs/positive_layer_20260814_retain/block_access_flags_long.parquet`.

The row values independently close to the common eligible denominator, the three new
adverse-outcome flags never overlap within a row, and the pathway fractions reproduce
`reports/03_non_inundation_pathway_by_arm.csv`. Each rule is defined in one clause in the
table note. Source: `outputs/tables/bridge_rule_sensitivity.tex` and
`reports/03_non_inundation_pathway_by_arm.csv`.

## Service-snapping robustness

Row-level recomputation gives 2,163 services. Initially, 569 (26.306%) were assigned to
singleton two-edge-connected components: 136 of 324 fire stations (41.975%) and 433 of
1,839 schools (23.545%). Re-snapping recovered 566 of 569 (99.473%), leaving three
unrecoverable. Among recovered services, added snap distance was 13.497 m at the median,
38.958 m at p90, and 582.072 m at the maximum. Source:
`outputs/tables/service_snapping_robustness.csv`.

Rounded as requested, these are 2,163; 569 and 26.3%; 42.0% versus 23.5%; 566 and 99.5%;
three; and 13.5 m / 39.0 m / 582.1 m. **No prompt value differs from the source CSVs.**
The recomputation reproduces the summary files at
`data/processed/access/edited/2026-04-03_run/diagnostics/validity/service_singleton_2ecc_summary.csv:2-4`
and
`data/processed/access/edited/2026-04-03_run/diagnostics/validity/service_resnapping_summary.csv:2`.
It was derived from the row-level
`data/processed/access/edited/2026-04-03_run/diagnostics/validity/service_2ecc_assignments.csv`
and
`data/processed/access/edited/2026-04-03_run/diagnostics/validity/service_resnapping.csv`;
none of the retired `block_access_flags_longslr*` files was read.
