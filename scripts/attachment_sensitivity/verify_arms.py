#!/usr/bin/env python
"""Reference reproduction and invariant checks for the attachment arms.

Writes outputs/attachment_sensitivity/<exp>/validation/*.csv and prints a summary.
A failed check is reported, not repaired: classifier defects are findings.
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
REFERENCE_RUN = PROJECT_ROOT / "data/processed/access/edited/della_runs/positive_layer_20260814_approach"
KEY = ["block_geoid", "slr_ft"]
STATE_COLS = ["block_centroid_inundated", "block_centroid_isolated", "block_centroid_fragile",
              "block_centroid_redundant", "block_centroid_unclassified"]
RANK = {"inundated": 0, "isolated": 1, "fragile": 2, "redundant": 3, "unclassified": -1}


def compare_frames(a: pd.DataFrame, b: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    m = a[KEY + columns].merge(b[KEY + columns], on=KEY, suffixes=("_a", "_b"), how="outer", indicator=True)
    rows = [{"column": "_row_alignment", "n_compared": int(len(m)),
             "n_mismatch": int((m["_merge"] != "both").sum())}]
    m = m.loc[m["_merge"] == "both"]
    for c in columns:
        x, y = m[f"{c}_a"], m[f"{c}_b"]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(x):
            eq = np.isclose(x.astype(float), y.astype(float), rtol=1e-9, atol=1e-6, equal_nan=True)
        else:
            eq = (x.astype("string").fillna("<NA>") == y.astype("string").fillna("<NA>")).to_numpy()
        rows.append({"column": c, "n_compared": int(len(m)), "n_mismatch": int((~eq).sum())})
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/access/attachment_sensitivity")
    p.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "outputs/attachment_sensitivity")
    args = p.parse_args()
    data_dir = args.data_root / args.experiment_id
    out_dir = args.out_root / args.experiment_id / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = {a: pd.read_parquet(data_dir / a / "block_access_flags_long.parquet")
            for a in al.ARM_ORDER if (data_dir / a / "_COMPLETE.json").exists()}
    summary = []

    # 1. Reference reproduction: every production column, every row.
    prod = pd.read_parquet(REFERENCE_RUN / "block_access_flags_long.parquet")
    ref = arms["reference"]
    cols = [c for c in prod.columns if c not in KEY]
    missing_cols = sorted(set(prod.columns) - set(ref.columns))
    rep = compare_frames(prod, ref, [c for c in cols if c in ref.columns])
    al.atomic_to_csv(rep, out_dir / "reference_reproduction_by_column.csv")
    summary.append({"check": "reference reproduces production parquet (all columns, all rows)",
                    "passed": bool(rep["n_mismatch"].sum() == 0 and not missing_cols),
                    "detail": f"{int(rep['n_mismatch'].gt(0).sum())} columns with mismatches; missing={missing_cols}"})

    # 1b. Facility attachment registry and bridge audits vs production.
    prod_fac = pd.read_csv(REFERENCE_RUN / "service_snapping_audit.csv")
    ref_fac = pd.read_csv(data_dir / "reference" / "service_attachment_audit.csv")
    fac_cols = ["service_id", "node_id", "snap_valid", "service_snap_rule", "unconstrained_node_id"]
    fac_eq = (prod_fac[fac_cols].astype(str).reset_index(drop=True) == ref_fac[fac_cols].astype(str).reset_index(drop=True))
    summary.append({"check": "reference facility registry equals production service audit",
                    "passed": bool(len(prod_fac) == len(ref_fac) and fac_eq.all(axis=None)),
                    "detail": f"prod={len(prod_fac)}, ref={len(ref_fac)}, mismatched cells={int((~fac_eq).sum().sum())}"})
    bridge_ok = []
    for s in range(7):
        pb = pd.read_csv(REFERENCE_RUN / f"bridge_structures_slr_{s}ft.csv")
        nb = pd.read_csv(data_dir / "shared" / f"bridge_structures_slr_{s}ft.csv")
        c = ["structure_id", "landing_node_ids", "dry_landing_count", "retained", "removed_edge_count"]
        # fillna: zero-landing structures have empty landing lists (NaN) in both
        # files, and pandas 3 keeps NaN missing under astype(str), where NaN != NaN.
        same = pb[c].astype("string").fillna("<NA>").values == nb[c].astype("string").fillna("<NA>").values
        bridge_ok.append(bool(len(pb) == len(nb) and same.all()))
    summary.append({"check": "bridge-structure audits equal production at 0-6 ft",
                    "passed": all(bridge_ok), "detail": str(bridge_ok)})

    # 2. Facility-only arms: origin membership and origin inundation identical.
    origin_cols = ["analysis_eligible", "exclusion_reason", "origin_node_id", "origin_snap_distance_m",
                   "block_centroid_inundated"]
    for arm in ["facility_500", "facility_add_uncapped", "facility_nearest_1000"]:
        if arm in arms:
            r = compare_frames(ref, arms[arm], origin_cols)
            summary.append({"check": f"{arm}: origin membership and inundation identical to reference",
                            "passed": bool(r["n_mismatch"].sum() == 0),
                            "detail": r.set_index("column")["n_mismatch"].to_dict()})

    # 3. add_uncapped: reference attachments preserved; destinations superset; no block worse.
    if "facility_add_uncapped" in arms:
        add_fac = pd.read_csv(data_dir / "facility_add_uncapped" / "service_attachment_audit.csv")
        ref_valid = ref_fac["snap_valid"].astype(str).eq("True")
        same = (add_fac.loc[ref_valid, ["service_record_id", "node_id"]].values
                == ref_fac.loc[ref_valid, ["service_record_id", "node_id"]].values).all()
        superset = bool(add_fac.loc[ref_valid, "snap_valid"].astype(str).eq("True").all())
        m = ref[KEY + ["scenario_status", "analysis_eligible"]].merge(
            arms["facility_add_uncapped"][KEY + ["scenario_status"]], on=KEY, suffixes=("_ref", "_add"))
        m = m.loc[m["analysis_eligible"]]
        worse = m["scenario_status_add"].map(RANK) < m["scenario_status_ref"].map(RANK)
        al.atomic_to_csv(m.loc[worse], out_dir / "add_uncapped_worse_than_reference.csv")
        summary.append({"check": "facility_add_uncapped: reference-valid attachments unchanged and retained",
                        "passed": bool(same and superset), "detail": f"same_nodes={same}, superset={superset}"})
        summary.append({"check": "facility_add_uncapped: no eligible block-scenario classified worse than reference",
                        "passed": bool(not worse.any()), "detail": f"worse rows={int(worse.sum())}"})

    # 4. Origin arms: common eligible origins classified identically.
    all_cols = [c for c in ref.columns if c not in KEY + ["attachment_arm", "analysis_eligible", "exclusion_reason",
                                                          "origin_snap_exceeds_threshold"]]
    for arm in ["origin_1000", "origin_500"]:
        if arm in arms:
            a = arms[arm]
            common = set(a.loc[a["analysis_eligible"], "block_geoid"])
            r = compare_frames(ref.loc[ref["block_geoid"].isin(common)], a.loc[a["block_geoid"].isin(common)], all_cols)
            al.atomic_to_csv(r, out_dir / f"{arm}_common_origin_comparison.csv")
            excluded = ref.loc[ref["analysis_eligible"] & ~ref["block_geoid"].isin(common) & ref["slr_ft"].eq(0)]
            summary.append({"check": f"{arm}: common eligible origins identical to reference (all fields)",
                            "passed": bool(r["n_mismatch"].sum() == 0),
                            "detail": f"common={len(common):,}; newly excluded={len(excluded):,}; mismatched columns="
                                      f"{r.loc[r['n_mismatch'] > 0, 'column'].tolist()}"})
            subset_ok = common <= set(ref.loc[ref["analysis_eligible"], "block_geoid"])
            summary.append({"check": f"{arm}: eligible set is a subset of reference eligible set",
                            "passed": bool(subset_ok), "detail": ""})

    # 5. Structural invariants in every arm.
    for arm, frame in arms.items():
        states = frame[STATE_COLS].sum(axis=1)
        dup = frame.duplicated(KEY).any()
        summary.append({"check": f"{arm}: exactly one state per row, unique keys, 7 scenarios",
                        "passed": bool(states.eq(1).all() and not dup and frame["slr_ft"].nunique() == 7),
                        "detail": f"rows={len(frame):,}"})
        el = frame.loc[frame["analysis_eligible"] & frame["slr_ft"].gt(0)]
        improve = el["scenario_status"].map(RANK) > el["baseline_status"].map(RANK)
        summary.append({"check": f"{arm}: no eligible block improves relative to its own baseline",
                        "passed": bool(not improve.any()), "detail": f"improving rows={int(improve.sum())}"})
        unc = frame.loc[frame["analysis_eligible"], "block_centroid_unclassified"].sum()
        summary.append({"check": f"{arm}: no eligible block is unclassified", "passed": bool(unc == 0),
                        "detail": f"{int(unc)}"})

    out = pd.DataFrame(summary)
    al.atomic_to_csv(out, out_dir / "validation_summary.csv")
    with pd.option_context("display.max_colwidth", 120, "display.width", 250):
        print(out.to_string(index=False))
    return 0 if out["passed"].all() else 1


if __name__ == "__main__":
    sys.exit(main())
