#!/usr/bin/env python
"""Bridge-rule comparison on the manuscript's population definitions.

outputs/tables/bridge_rule_sensitivity.tex predates the manuscript's current
definitions: its newly affected population includes baseline-isolated blocks
that later become inundated, and its non-inundation share is computed over
blocks. Section 4.5 of the manuscript defines both over the five adverse
transitions from baseline-redundant and baseline-fragile blocks, with the
non-inundation share computed over population.

This script reuses the production table builders in 05_population_figures.py
(load, audit, and cumulative-population logic) for each bridge arm, redirecting
their CSV writes to outputs/final_draft/bridge_arms. Plotting is never called.
The approach arm must reproduce the production fig4 tables exactly.

Run from the repository root with the research-geo environment:
  MPLBACKEND=Agg python scripts/08b_final_draft_bridge_arms.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "final_draft"
ARM_DIR = OUT_DIR / "bridge_arms"


def load_population_module():
    path = PROJECT_ROOT / "scripts" / "05_population_figures.py"
    spec = importlib.util.spec_from_file_location("population_figures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    pf = load_population_module()
    ARM_DIR.mkdir(parents=True, exist_ok=True)
    pf.TABLES_DIR = ARM_DIR  # redirect every CSV the builders write

    rows = []
    for arm in pf.ARMS:
        df = pf.load_long_with_population(arm)
        status = pf.build_status_population_table(df, arm)
        cumulative = pf.build_cumulative_population_table(df, arm).set_index("slr_ft")
        base = status.loc[status["slr_ft"].eq(0)].set_index("scenario_status")
        row = {
            "arm": arm,
            "eligible_blocks": int(base["n_blocks"].sum()),
            "eligible_pop20": int(base["pop20"].sum()),
            "baseline_fragile_share_blocks": base.at["fragile", "share_of_blocks"],
            "baseline_fragile_share_pop20": base.at["fragile", "share_of_pop20"],
        }
        for s in pf.SLR_LEVELS:
            c = cumulative.loc[s]
            row[f"newly_fragile_blocks_{s}ft"] = int(c["added_by_fragile_blocks"])
            row[f"newly_affected_pop20_{s}ft"] = int(c["new_fragile_or_worse_pop20"])
            row[f"non_inundation_share_pop20_{s}ft"] = (
                1 - c["new_inundated_pop20"] / c["new_fragile_or_worse_pop20"]
            )
        rows.append(row)

        if arm == "approach":
            for stem in ("fig4_status_population_by_slr", "fig4_cumulative_population_by_slr"):
                prod = pd.read_csv(PROJECT_ROOT / "outputs" / "tables" / f"{stem}_approach.csv")
                new = pd.read_csv(ARM_DIR / f"{stem}_approach.csv")
                pd.testing.assert_frame_equal(prod, new, check_exact=False, rtol=1e-12)
            print("approach arm reproduces production fig4 tables exactly")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "bridge_rule_comparison.csv", index=False)
    print(summary.T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
