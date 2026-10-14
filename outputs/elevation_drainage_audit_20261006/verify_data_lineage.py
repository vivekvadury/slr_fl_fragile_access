"""Verify saved approach states -> analysis rows -> transition counts, without writes upstream."""
from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np

ROOT = Path.cwd()
OUT = ROOT / "outputs/elevation_drainage_audit_20261006"
RUN = ROOT / "data/processed/access/edited/della_runs/positive_layer_20260814_approach"
checks = []
def check(name, passed, detail):
    checks.append(dict(check=name, passed=bool(passed), detail=detail))
    if not passed:
        pd.DataFrame(checks).to_csv(OUT / "data_lineage_checks.csv", index=False)
        raise AssertionError(name)

manifest = json.loads((RUN / "run_manifest.json").read_text())
check("Approach run; all legacy behavior disabled", manifest["cli_flags"]["bridge_rule"] == "approach"
      and not any(v for k,v in manifest["cli_flags"].items() if k.startswith("legacy_")),
      "Read canonical approach run_manifest.json; 0-6 ft scenarios.")
src = pd.read_parquet(RUN / "block_access_flags_long.parquet")
src = src[src.analysis_eligible].copy()
check("Eligible source blocks", src.block_geoid.nunique() == 68521 and len(src) == 68521*7,
      f"{src.block_geoid.nunique()} unique eligible blocks; {len(src)} block-scenario rows")
ids = {"block_geoid":str, "block_group_geoid":str, "tract_geoid":str, "county_fips":str, "block":str}
long = pd.read_csv(ROOT / "data/processed/analysis/block_level_long_dataset_approach.csv", dtype=ids, low_memory=False)
bg = pd.read_csv(ROOT / "data/processed/analysis/block_group_analysis_dataset_approach.csv", dtype=ids)
key = ["block_geoid", "slr_ft"]
state_cols = [f"block_centroid_{s}" for s in ["unclassified","inundated","isolated","fragile","redundant"]]
base_cols = ["baseline_"+s for s in state_cols]
compare_cols = ["block_group_geoid", "scenario_status", "baseline_status", "analysis_eligible",
                "pop20", "bridge_rule_applied", "origin_geometry_method"] + state_cols + base_cols
for d in [src,long]:
    d["block_geoid"] = d.block_geoid.astype(str)
    d["block_group_geoid"] = d.block_group_geoid.astype(str)
    check("Unique block-scenario keys: "+("source" if d is src else "analysis"),
          not d.duplicated(key).any(), f"{len(d)} rows")
a = src.set_index(key).sort_index()[compare_cols]
b = long.set_index(key).sort_index()[compare_cols]
pd.testing.assert_frame_equal(a,b,check_dtype=False)
check("Every saved analysis classification matches source access run",True,
      f"All {len(a)} rows: statuses, eligibility, five state flags, baseline flags, population, origin method, bridge arm")
check("Approach and representative-point origins",long.bridge_rule_applied.eq("approach").all()
      and long.origin_geometry_method.eq("representative_point").all(), "All eligible block-scenario rows")

derived = {}
for origin,dest in [("redundant","fragile"),("redundant","isolated"),("redundant","inundated"),
                    ("fragile","isolated"),("fragile","inundated")]:
    col = f"baseline_{origin}_to_{dest}"
    derived[col] = (long[f"baseline_block_centroid_{origin}"].eq(1) & long[f"block_centroid_{dest}"].eq(1)).astype(int)
derived["any_loss_of_redundancy"] = (long.baseline_block_centroid_redundant.eq(1)
    & long.scenario_status.isin(["fragile","isolated","inundated"])).astype(int)
for col,values in derived.items():
    check("Transition definition: "+col,np.array_equal(values,long[col]), "Re-derived from finalized state indicators")
group_key = ["block_group_geoid","slr_ft"]
cols = state_cols+base_cols+list(derived)
agg = long.groupby(group_key)[cols].sum()
agg["total_blocks"] = long.groupby(group_key).size()
agg["eligible_pop20"] = long.groupby(group_key).pop20.sum()
saved_bg = bg.set_index(group_key).sort_index()
pd.testing.assert_frame_equal(agg.sort_index(),saved_bg[agg.columns],check_dtype=False)
check("Block-group inputs equal reaggregated current blocks",True,
      f"{len(agg)} rows, {bg.block_group_geoid.nunique()} block groups; all state/transition counts, denominators, population")
physical = pd.read_csv(ROOT / "data/processed/analysis/block_group_physical_covariates.csv",dtype={"block_group_geoid":str})
check("Physical covariates complete and one-to-one",not physical.block_group_geoid.duplicated().any()
      and set(physical.block_group_geoid)==set(bg.block_group_geoid)
      and np.isfinite(physical[["elevation_m_mean","elevation_m_median","drainage_distance_km"]]).all().all(),
      f"All {len(physical)} analysis block groups; no extra complete-case exclusions")
pd.DataFrame(checks).to_csv(OUT / "data_lineage_checks.csv",index=False)
paths = [ROOT / f for f in ["scripts/02_access_flags.py","scripts/03_build_extension_dataset_and_memo.ipynb",
    "scripts/03b_join_elevation_drainage.py","scripts/04_shared_model_spec.R","scripts/04_transition_models.R",
    "scripts/08_final_draft_inference.R","data/processed/analysis/block_group_analysis_dataset_approach.csv",
    "data/processed/analysis/block_group_physical_covariates.csv","data/processed/analysis/block_level_long_dataset_approach.csv",
    "outputs/spatial/slr_block_group_analysis_approach.gpkg","outputs/final_draft/all_ames_all_vcovs.csv",
    "outputs/final_draft/main_transition_ames_conley5.csv","outputs/final_draft/transition_ames_conley5_with_physical.csv"]]
paths += [RUN/"run_manifest.json",RUN/"block_access_flags_long.parquet"]
pd.DataFrame([dict(path=str(p.relative_to(ROOT)).replace("\\","/"),bytes=p.stat().st_size,
                   sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]).to_csv(OUT/"input_provenance_sha256.csv",index=False)
print(pd.DataFrame(checks).to_string(index=False))
