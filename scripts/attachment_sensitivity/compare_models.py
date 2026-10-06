#!/usr/bin/env python
"""AME comparison tables across attachment arms, both specifications.

Reads outputs/attachment_sensitivity/<exp>/models/<arm>/ame_bootstrap_results_approach[_with_physical].xlsx
and transition_sample_diagnostics / transition_model_coefficients files, and writes:
  ame_comparison_demographic_only.csv, ame_comparison_with_physical.csv
  model_sample_comparison.csv
  reference_vs_production_models.csv   (does the rerun reference reproduce outputs/tables?)
All six social terms and all seven transitions are reported; nothing is filtered.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attachment_lib as al  # noqa: E402

PROJECT_ROOT = al.PROJECT_ROOT
SOCIAL = ["z_pct_black_nh", "z_pct_hispanic", "z_renter_share", "z_log_median_income",
          "z_pct_age_65plus", "z_no_vehicle_share"]
SPEC_FILES = {"demographic_only": "approach", "with_physical": "approach_with_physical"}


def read_ame(dir_: Path, spec: str) -> pd.DataFrame | None:
    """Read an arm's AME workbook; follow a documented reuse marker if present."""
    reuse = dir_ / "REUSED_FROM_REFERENCE.json"
    if reuse.exists():
        ame = read_ame(dir_.parent / "reference", spec)
        if ame is not None:
            ame = ame.copy()
            ame["reused_from_reference"] = True
        return ame
    path = dir_ / f"ame_bootstrap_results_{SPEC_FILES[spec]}.xlsx"
    if not path.exists():
        return None
    ame = pd.read_excel(path)
    ame["reused_from_reference"] = False
    return ame


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "outputs/attachment_sensitivity")
    args = p.parse_args()
    out = args.out_root / args.experiment_id
    models = out / "models"

    # Reference rerun vs production tables (same data, same seeds -> should be identical).
    rep_rows = []
    for spec, suffix in SPEC_FILES.items():
        prod = pd.read_excel(PROJECT_ROOT / "outputs/tables" / f"ame_bootstrap_results_{suffix}.xlsx")
        ref = read_ame(models / "reference", spec)
        if ref is None:
            continue
        m = prod.merge(ref, on=["transition", "term"], suffixes=("_prod", "_ref"))
        for c in ["estimate", "std.error", "conf.low", "conf.high", "p.value", "n_boot"]:
            rep_rows.append({"spec": spec, "field": c, "rows": len(m),
                             "max_abs_diff": float(np.nanmax(np.abs(m[f"{c}_prod"] - m[f"{c}_ref"])))})
    al.atomic_to_csv(pd.DataFrame(rep_rows), out / "reference_vs_production_models.csv")

    sample_rows = []
    for spec in SPEC_FILES:
        frames = []
        for arm in al.ARM_ORDER:
            ame = read_ame(models / arm, spec)
            if ame is None:
                continue
            ame = ame.loc[ame["term"].isin(SOCIAL)].copy()
            ame.insert(0, "attachment_arm", arm)
            frames.append(ame)
            diag = models / arm / f"transition_sample_diagnostics_{SPEC_FILES[spec]}.csv"
            if diag.exists():
                d = pd.read_csv(diag)
                d.insert(0, "spec", spec)
                d["attachment_arm"] = arm
                sample_rows.append(d)
        if not frames:
            continue
        allf = pd.concat(frames, ignore_index=True)
        ref = allf.loc[allf["attachment_arm"].eq("reference"), ["transition", "term", "estimate", "p.value"]].rename(
            columns={"estimate": "ref_estimate", "p.value": "ref_p_value"})
        allf = allf.merge(ref, on=["transition", "term"], how="left")
        allf["estimate_pp"] = 100 * allf["estimate"]
        allf["abs_diff_pp"] = 100 * (allf["estimate"] - allf["ref_estimate"])
        allf["rel_diff_pct"] = np.where(allf["ref_estimate"].abs() > 0,
                                        100 * (allf["estimate"] - allf["ref_estimate"]) / allf["ref_estimate"].abs(), np.nan)
        allf["sign_change"] = np.sign(allf["estimate"]) != np.sign(allf["ref_estimate"])
        allf["sig05"] = allf["p.value"] < 0.05
        allf["ref_sig05"] = allf["ref_p_value"] < 0.05
        allf["sig05_classification_change"] = allf["sig05"] != allf["ref_sig05"]
        allf["ci_includes_zero"] = (allf["conf.low"] <= 0) & (allf["conf.high"] >= 0)
        al.atomic_to_csv(allf, out / f"ame_comparison_{spec}.csv")
    if sample_rows:
        al.atomic_to_csv(pd.concat(sample_rows, ignore_index=True), out / "model_sample_comparison.csv")
    print("wrote model comparisons to", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
