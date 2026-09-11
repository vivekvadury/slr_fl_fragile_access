# Baseline snapshot — 2026-09-02

Canonical **no-Conley, default AME_BOOT_REPS=199** output set, captured before the
Conley cutoff sweep (Phase 1) and the population-weighted AME work (Phase 2).

Source runs (BRIDGE_ARM=approach, no CONLEY_CUTOFF_KM, AME_BOOT_REPS unset):
- `Rscript scripts/04_transition_models.R`                        (MODEL_SPEC=demographic_only)
- `MODEL_SPEC=with_physical Rscript scripts/04_transition_models.R`

Verified against docs/physical_covariate_and_conley_extension.md Section 6:
all 14 pct_black_nh / pct_hispanic attenuation percentages reproduce to 4 dp;
every with_physical racial coefficient stays negative and significant (p <= 8.3e-6).

`transition_model_coefficients_approach.csv` and
`transition_sample_diagnostics_approach.csv` are byte-identical to the committed
HEAD copies. Do not overwrite this directory.
