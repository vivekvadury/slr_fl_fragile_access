"""Build the Section 5.5 audit deliverables after current-model verification passes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path.cwd()
OUT = ROOT / "outputs/elevation_drainage_audit_20261006"
checks = pd.read_csv(OUT / "saved_vs_current_checks.csv")
assert checks.passed.all()
diagnostics = pd.read_csv(OUT / "verified_adjusted_diagnostics.csv")
assert len(diagnostics)==7 and diagnostics.converged.all()
fresh = pd.read_csv(OUT / "reproduced_ames_conley5.csv")
terms = dict(z_pct_black_nh="Black share",z_pct_hispanic="Hispanic share",
             z_renter_share="Renter share",z_log_median_income="Log median income",
             z_pct_age_65plus="Age 65+ share",z_no_vehicle_share="No-vehicle household share")
transitions = ["Redundant -> Fragile","Redundant -> Isolated","Redundant -> Inundated",
               "Redundant -> Worse","Fragile -> Isolated","Fragile -> Inundated","Fragile -> Worse"]
for spec,file in [("demographic_only","main_transition_ames_conley5.csv"),
                  ("with_physical","transition_ames_conley5_with_physical.csv")]:
    stored = pd.read_csv(ROOT / "outputs/final_draft" / file)
    current = fresh[fresh.spec.eq(spec)].merge(stored,on=["spec","transition","term"],suffixes=("_fresh","_saved"),validate="one_to_one")
    assert len(current)==(42 if spec=="demographic_only" else 56)
    for col in ["estimate","std_error","p_value","conf_low","conf_high"]:
        assert np.allclose(current[col+"_fresh"],current[col+"_saved"],rtol=0,atol=1e-8)
    assert np.allclose(stored.p_value,2*norm.sf(abs(stored.estimate/stored.std_error)),rtol=1e-8,atol=1e-12)
    assert np.allclose(stored.conf_low,stored.estimate-norm.ppf(.975)*stored.std_error,atol=1e-10)
    assert np.allclose(stored.conf_high,stored.estimate+norm.ppf(.975)*stored.std_error,atol=1e-10)

left=fresh[fresh.spec.eq("demographic_only") & fresh.term.isin(terms)].drop(columns="spec")
right=fresh[fresh.spec.eq("with_physical") & fresh.term.isin(terms)].drop(columns="spec")
merged=left.merge(right,on=["transition","term"],suffixes=("_primary","_adjusted"),validate="one_to_one")
rows=[]
for term,label in terms.items():
    for transition in transitions:
        r=merged[(merged.term==term)&(merged.transition==transition)].iloc[0]
        p,a=r.estimate_primary,r.estimate_adjusted
        sign_change=bool(np.sign(p)!=np.sign(a))
        raw_pct=100*(1-abs(a)/abs(p)) if abs(p)>1e-12 else np.nan
        # This is a reporting rule only, not a modeling or statistical test.
        # Avoid calling a reversal "attenuation" or dividing by <0.1 pp.
        note=("sign reversal: report signed changes" if sign_change else
              "primary magnitude below 0.1 pp: unstable relative change" if abs(p)<.001 else
              "both estimates same sign; positive=attenuation, negative=amplification")
        pct=raw_pct if not sign_change and abs(p)>=.001 else np.nan
        ps,ads=bool(r.p_value_primary<.05),bool(r.p_value_adjusted<.05)
        row=dict(covariate=label,term=term,transition=transition,bridge_arm="approach",
                 vcov="Conley spatial HAC, uniform kernel, 5 km",ame_se_method="delta method",
                 primary_ame_pp=100*p,adjusted_ame_pp=100*a,
                 primary_se_pp=100*r.std_error_primary,adjusted_se_pp=100*r.std_error_adjusted,
                 primary_ci95_low_pp=100*r.conf_low_primary,primary_ci95_high_pp=100*r.conf_high_primary,
                 adjusted_ci95_low_pp=100*r.conf_low_adjusted,adjusted_ci95_high_pp=100*r.conf_high_adjusted,
                 primary_p_value=r.p_value_primary,adjusted_p_value=r.p_value_adjusted,
                 primary_significant_05=ps,adjusted_significant_05=ads,
                 signed_change_pp=100*(a-p),absolute_change_pp=100*abs(a-p),
                 magnitude_reduction_pp=100*(abs(p)-abs(a)),
                 raw_percent_reduction_absolute_magnitude=raw_pct,
                 percent_attenuation_absolute_magnitude=pct,percent_interpretation=note,
                 sign_change=sign_change,significance_change=ps!=ads,
                 significance_transition=("lost" if ps and not ads else "gained" if ads and not ps else "unchanged"))
        rows.append(row)
df=pd.DataFrame(rows)
assert len(df)==42
df.to_csv(OUT/"primary_vs_elevation_drainage_ames.csv",index=False,float_format="%.12g")
summary=[]
for label,g in df.groupby("covariate",sort=False):
    valid=g.percent_attenuation_absolute_magnitude.dropna()
    summary.append(dict(covariate=label,primary_significant=int(g.primary_significant_05.sum()),
                        adjusted_significant=int(g.adjusted_significant_05.sum()),
                        adjusted_negative=int((g.adjusted_ame_pp<0).sum()),
                        adjusted_positive=int((g.adjusted_ame_pp>0).sum()),
                        sign_changes=int(g.sign_change.sum()),
                        attenuation_min_percent=valid.min(),attenuation_max_percent=valid.max()))
pd.DataFrame(summary).to_csv(OUT/"indicator_summary.csv",index=False)

intro="""# Elevation and drainage adjustment: Section 5.5 audit

Audit date: 6 October 2026. Primary arm: `approach`. Manuscript reviewed:
`SLR_Access_Degredation_IDJRR_Manuscript (32).pdf`, Sections 4.7-4.8 (PDF pages 18-20),
Table 5 (page 26), and empty Section 5.5 (page 29). The manuscript was not modified.

## Verdict

**Current and directly comparable: yes. Both final-draft specifications use 5-km
Conley spatial HAC covariance and delta-method AME uncertainty. No estimation
rerun is required to update Section 5.5.** This audit reproduced all 14 production
fits from the current local inputs, without a bootstrap, in an isolated folder.
All 98 AMEs (42 social-only and 56 including physical terms), standard errors,
p-values, z statistics and confidence limits matched the final-draft export
within the recorded numerical tolerance. See `saved_vs_current_checks.csv`.
The 42-row comparison includes only the six requested social indicators.

No model, input, manuscript, or production output was changed. Existing unrelated
social-vulnerability audit work was left alone. Results here apply to the current
local finalized analysis inputs, not a newly rebuilt road network.

## Exact pipeline and files

All paths below are relative to the repository root.

1. `scripts/02_access_flags.py` produces the designated approach access run:
   `data/processed/access/edited/della_runs/positive_layer_20260814_approach/block_access_flags_long.parquet`
   (CSV sibling and `run_manifest.json` also present). Manifest: `approach`, 0-6 ft,
   all legacy flags false. This audit did not rebuild the graph.
2. `scripts/03_build_extension_dataset_and_memo.ipynb` retains eligible blocks,
   constructs transition indicators, aggregates them, joins ACS covariates and writes
   `data/processed/analysis/block_level_long_dataset_approach.csv` and
   **`data/processed/analysis/block_group_analysis_dataset_approach.csv`**.
   The latter is the direct regression input for both specifications.
3. `scripts/03b_join_elevation_drainage.py` writes the second adjusted-model input:
   **`data/processed/analysis/block_group_physical_covariates.csv`**. The fitted
   additions are `elevation_m_mean` and `drainage_distance_km`; `elevation_m_median`
   is validated/read but is not a regression covariate.
   - Geometry: `outputs/spatial/slr_block_group_analysis_approach.gpkg`, layer `slr_0ft`.
   - Elevation: valid-cell-center zonal mean from pinned USGS 3DEP 1/3-arc-second
     tiles `USGS_13_n26w081_20260225.tif` and `USGS_13_n27w081_20251216.tif`, cached
     under `data/raw/physical_covariates/usgs_3dep_13_arc_second/`.
   - Drainage: geometric-centroid distance to the nearest SFWMD AHED feature with
     `HYDRO_ORDER = 'PRIMARY'`, measured in EPSG:32617 and converted to km.
     Cache: `data/raw/physical_covariates/sfwmd/sfwmd_ahed_primary_drainage.geojson`
     and `.metadata.json` (7,803 features; retrieved 2 September 2026).
4. `scripts/04_shared_model_spec.R` defines the seven formulas, the covariates,
   grouped-binomial fits, risk-set weights and Conley covariance helper.
5. `scripts/04_transition_models.R` reads the direct inputs, standardizes variables,
   constructs estimation samples and fits the models; `MODEL_SPEC=with_physical`
   adds the two physical controls to the same primary specification.
6. **`scripts/08_final_draft_inference.R` is the final-draft AME exporter.** It fixes
   `ARM='approach'`, evaluates script 04 up to its bootstrap call for both specs,
   then computes Conley/delta-method AMEs. Its relevant outputs are:
   - `outputs/final_draft/main_transition_ames_conley5.csv` (42 primary AMEs);
   - `outputs/final_draft/transition_ames_conley5_with_physical.csv` (56 adjusted AMEs);
   - `outputs/final_draft/main_transition_ames_conley5.xlsx` (both specifications);
   - `outputs/final_draft/all_ames_all_vcovs.csv` (all covariance variants);
   - `outputs/final_draft/physical_controls_comparison.csv` (older comparison layout);
   - `outputs/final_draft/model_run/transition_model_coefficients_approach{,_with_physical}.csv`;
   - `outputs/final_draft/model_sample_sizes.csv`, `model_covariate_filter_diagnostics.csv`,
     `conley_positive_definiteness.csv`, `conley_verification.csv`, `diagnostics_summary.csv`.

Hashes of current pipeline scripts, direct inputs, access states, geometry and final
AME exports are saved in `input_provenance_sha256.csv`. The CSV comparison is generated
from the freshly reproduced AMEs, after comparison with both final-draft tables.

## Comparability checks

| Component | Primary vs adjusted |
| --- | --- |
| Seven outcomes | Identical: redundant to fragile, isolated, inundated, worse; fragile to isolated, inundated, worse. |
| Worse | Redundant: fragile + isolated + inundated; fragile: isolated + inundated. |
| Estimation rows | Identical row by row within every corresponding model, not merely equal sample sizes. |
| Baseline denominators | Identical counts of eligible blocks in the appropriate 0-ft baseline state; also binomial trial weights. |
| Social indicators | Identical standardized Black share, Hispanic share, renter share, log median household income, age-65+ share and no-vehicle household share. |
| Standardization | `scale()` before complete-case and risk-set filters on the stacked 0-6-ft panel; shared social z-scores are exactly equal. |
| Fixed effects | Identical county and SLR-scenario factors; estimation uses the six positive 1-6-ft scenarios. |
| Bridge treatment | Identical `approach`. |
| Access/transition data | Identical current finalized block-group input; independently traced to the designated source run. |
| AME averaging | Equal weight for each estimation-sample block-group/scenario row; model trial weights remain baseline block counts. |
| Only added regressors | Standardized mean elevation and standardized nearest-primary-drainage distance. |

The physical join is complete for all 3,942 block groups. Both models exclude the
same 338 groups for missing social covariates, leaving 3,604 complete-case groups
before baseline-state risk-set filtering. Final samples:

| Baseline family | Models | Block groups | BG-scenario observations | Baseline blocks | Block-scenario trials |
| --- | ---: | ---: | ---: | ---: | ---: |
| Redundant | 4 | 3,442 | 20,652 | 46,954 | 281,724 |
| Fragile | 3 | 3,178 | 19,068 | 16,114 | 96,684 |

`sample_identity_checks.csv` verifies exact row keys, outcomes, denominators, social
z-scores, county/scenario values and coordinates. `data_lineage_checks.csv` verifies
479,647 eligible block-scenario rows (68,521 blocks x 7 scenarios), plus exact
reaggregation of all relevant state/transition counts in the 27,594 block-group rows.

## Variance-estimator audit

The final-draft exports use `fixest::vcov_conley()` with a **5-km cutoff, uniform
kernel**, block-group centroid coordinates, the default triangular distance
approximation (this is a distance algorithm, not a triangular kernel), and fixest's
small-sample adjustment. Centroids come from the same approach GeoPackage.
Repeated scenario observations at the same centroid are included in the spatial
covariance calculation. The final-draft implementation verification documents a
small-cutoff equivalence to unadjusted block-group clustering and a hand-built
Conley reproduction; that independent kernel reconstruction was not rerun here.

AMEs use `marginaleffects::avg_slopes(model, vcov=V)` with the supplied Conley matrix,
delta-method SEs, normal-theory 95% intervals and two-sided normal p-values. All 14
raw 5-km covariance matrices are positive definite in the fresh audit; no eigenvalue
repair is needed at 5 km. The audit also checked p-values and confidence bounds
algebraically. Significance below means nominal p<0.05, without multiplicity adjustment.

**Do not confuse other files with these outputs.** The `std.error`, p-value and CI
columns in `outputs/tables/ame_bootstrap_results_approach_with_physical.xlsx` are from
the 199-successful-replication block-group cluster bootstrap: SE is the SD of bootstrap
AME draws, CI is percentile-based, and p is a normal approximation using the bootstrap
SE. When Conley is requested, script 04 adds separate `conley_se`/`conley_cutoff_km`
columns; it does not turn those bootstrap CI/p columns into Conley inference.
Coefficient diagnostic CSVs likewise have clustered `std.error` and separate
coefficient-level `conley_se`; those are not AME SEs. Earlier `conley_sweep/` files
are superseded for the current manuscript. No such older uncertainty columns were
used in this comparison.

The current final-draft files agree with manuscript Sections 4.7-4.8, which explicitly
specify 5-km Conley inference and delta-method AME SEs for these adjusted models.
The old workflow README and script-04 log wording still emphasize bootstrap or
clustered inference; this documentation is not a reliable guide to the final-draft table.

## Results to prioritize for Section 5.5

- **Black share:** negative and significant in all seven adjusted models, with
  magnitude attenuation of 40.1-65.6%. The least decisive is redundant-to-isolated
  (p=0.0412); do not describe every association as strongly significant.
- **Hispanic share:** negative and significant in all seven, attenuating 30.0-64.0%.
- **Renter share:** positive in all seven, attenuating 20.5-61.5%; significant in
  six, not seven. Redundant-to-isolated changes from +1.045 to +0.403 pp (95% CI
  -0.018 to +0.823; p=0.0606).
- **Age 65+:** five significant primary associations become zero significant
  adjusted associations. The two primary inundation associations were already
  nonsignificant. Adjusted inundation effects and fragile-to-worse change sign,
  but no adjusted age association is distinguishable from zero at 5%.
- **Income:** no adjusted association is significant. The sole significant primary
  association, fragile-to-isolated, falls from +0.846 to +0.171 pp (p=0.470).
  Five estimates reverse sign; all five are nonsignificant before and after.
- **No-vehicle share:** none is significant in the primary models; after adjustment,
  redundant-to-inundated is positive and significant (+0.602 pp; 95% CI +0.071 to
  +1.133; p=0.0264). The other six remain nonsignificant. Four signs change.

Useful overall-worsening anchors (all in percentage points): Black share changes
from -4.904 to -2.382 for redundant-to-worse and -5.860 to -2.667 for fragile-to-worse;
Hispanic share from -6.280 to -3.837 and -7.289 to -3.656; renter share from +4.290
to +2.250 and +5.098 to +2.272. All six adjusted worsening associations remain significant.

### Prose-ready summary

Adding block-group mean elevation and distance to the nearest primary drainage
feature attenuated, but did not eliminate, several social associations with access
degradation. Black and Hispanic population shares remained negatively associated
with all seven transitions, with AME magnitudes declining by approximately 40-66%
and 30-64%, respectively. Renter share remained positively associated with every
transition, although redundant-to-isolated no longer met the 5% significance
threshold; the other six associations remained significant. In contrast, none of
the age-65+ associations remained statistically significant, and the sole significant
primary income association also became nonsignificant. No-vehicle share was
positively associated with redundant-to-inundated transitions after adjustment,
while its other associations remained nonsignificant. Thus, several social
associations persist after conditioning on these two physical characteristics,
while the older-adult pattern is substantially attenuated.

Interpret these as conditional block-group associations. Attenuation is descriptive,
not the percentage causally explained by elevation/drainage. A change in whether
p crosses 0.05 is not a formal test that two AMEs differ. This audit does not estimate
a paired SE/CI for their difference, and does not interpret the physical coefficients.

## Caveats and actions before writing

1. **No freshness or variance-estimator rerun is needed.** Use the verified final-draft
   Conley tables or this audit CSV, not older bootstrap or preliminary outputs.
2. **Model diagnostics deserve review.** A separate current-input refit reproduced the
   saved final-draft diagnostics (`verified_adjusted_diagnostics.csv`), including
   adjusted redundant-to-worse Pearson dispersion around 4,792, with about 99.4% of
   the Pearson statistic from BG `120860100261` at 6 ft (fitted p about 1.22e-7).
   Dispersion remains about 28.95 after removing that row's contribution. This is
   a warning about fit/extreme predictions, not evidence that its removal would
   change the social AMEs by a known amount. All current fits converge and retain
   all coefficients; convergence alone does not resolve this diagnostic. No row
   was dropped or specification changed. A formal influence/refit sensitivity is
   a separate research decision, not required to make the existing tables current.
3. **Upstream source provenance is incomplete.** The access manifest records commit
   `27ac074` with a dirty working tree. `reports/01_provenance_and_layer_gate.md`
   already documents that no runtime source hash/patch identifies that historical
   checkout exactly. Current transition data match that designated run exactly,
   but this audit cannot certify a fresh road-network rebuild would be byte-identical.
4. **Existing comparison percent field is misleading if called attenuation.**
   `outputs/final_draft/physical_controls_comparison.csv` defines `pct_change` as
   100*(adjusted-primary)/abs(primary). That is a signed change normalized by primary
   magnitude, not generally percent attenuation. Use the explicitly defined fields below.

## CSV definitions and complete social-indicator comparisons

File: `primary_vs_elevation_drainage_ames.csv`. All AMEs, SEs, CIs and changes use
**percentage points per one-SD increase** in the indicator (income uses standardized
log median income). Confidence intervals are for individual AMEs, not their differences.

- `signed_change_pp` = adjusted-primary; `absolute_change_pp` = abs(adjusted-primary).
- `magnitude_reduction_pp` = abs(primary)-abs(adjusted).
- `raw_percent_reduction_absolute_magnitude` = 100*(1-abs(adjusted)/abs(primary)).
- `percent_attenuation_absolute_magnitude` reports that ratio only for same-sign
  estimates with abs(primary)>=0.1 pp; otherwise blank, with an explanation. The
  0.1-pp threshold is solely a transparent reporting guard against near-zero ratios.
  Negative values mean amplification. Ratios from nonsignificant primary estimates
  remain descriptive and should not be emphasized as evidence of an underlying effect.
- `significance_change` and `significance_transition` compare nominal p<0.05.

In the tables below, stars are * p<0.05, ** p<0.01, *** p<0.001. Delta is the signed
change in pp; the CSV additionally supplies its absolute value, both SEs and both CIs.
"NA" attenuation denotes a sign change or a near-zero primary estimate as defined above.
"Lost/gained" refers only to crossing p=0.05, not a test of the between-model difference.
"""
def stars(p): return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
parts=[intro]
for label,g in df.groupby("covariate",sort=False):
    parts.append(f"\n### {label}\n\n| Transition | Primary AME | Adjusted AME | Adjusted SE | Adjusted 95% CI | Adjusted p | Delta pp | Attenuation % | Sign flips? | Significance |\n| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |\n")
    for _,r in g.iterrows():
        pct="NA" if pd.isna(r.percent_attenuation_absolute_magnitude) else f"{r.percent_attenuation_absolute_magnitude:.1f}"
        parts.append(f"| {r.transition} | {r.primary_ame_pp:+.3f}{stars(r.primary_p_value)} | {r.adjusted_ame_pp:+.3f}{stars(r.adjusted_p_value)} | {r.adjusted_se_pp:.3f} | [{r.adjusted_ci95_low_pp:+.3f}, {r.adjusted_ci95_high_pp:+.3f}] | {r.adjusted_p_value:.4g} | {r.signed_change_pp:+.3f} | {pct} | {'yes' if r.sign_change else 'no'} | {r.significance_transition} |\n")
parts.append("\n## Verification artifacts and rerun commands\n\n"
    "Run from the repository root with the existing R/Python research environments:\n\n"
    "```text\nRscript outputs/elevation_drainage_audit_20261006/verify_current_models.R\n"
    "Rscript outputs/elevation_drainage_audit_20261006/verify_adjusted_diagnostics.R\n"
    "python outputs/elevation_drainage_audit_20261006/verify_data_lineage.py\n"
    "python outputs/elevation_drainage_audit_20261006/build_comparison.py\n```\n\n"
    "These audit scripts write only in this audit folder. They do not regenerate production outputs. "
    "Verification tables, source hashes, run logs and R session details are saved beside this report.\n")
(OUT/"AUDIT_REPORT.md").write_text("".join(parts),encoding="utf-8")
before=json.loads((OUT/"before_sha256.json").read_text())
changed=[p for p,h in before.items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
assert not changed,changed
(OUT/"preservation_check.txt").write_text(f"PASS: all {len(before)} pre-existing tracked files and selected input/manuscript files match their before-audit SHA-256 hashes.\n")
print(pd.DataFrame(summary).to_string(index=False))
print(f"Saved {len(df)} social-indicator comparisons and AUDIT_REPORT.md. Production files and manuscript unchanged.")
