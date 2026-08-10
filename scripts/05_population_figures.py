#!/usr/bin/env python
"""Population-weighted manuscript tables and figures.

This script joins 2020 Census block population (POP20) to the block-level
SLR transition dataset, then exports the population summaries and figures used
for the manuscript's affected-population results.

The output filenames retain their earlier `fig4_*` stems because notebooks and
draft-placeholder code already read those files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "data" / "processed" / "analysis"
RAW_BLOCKS_PATH = PROJECT_ROOT / "data" / "raw" / "census" / "blocks" / "2020" / "tl_2020_12_tabblock20.shp"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

SLR_LEVELS = [1, 2, 3, 4, 5, 6]
TRI_COUNTY_FIPS = {"011", "086", "099"}

TRANSITIONS = [
    ("baseline_redundant_to_fragile", "Redundant to fragile", "#fec44f", "///"),
    ("baseline_redundant_to_isolated", "Redundant to isolated", "#fc8d59", "///"),
    ("baseline_redundant_to_inundated", "Redundant to inundated", "#74add1", "///"),
    ("baseline_fragile_to_isolated", "Fragile to isolated", "#d7301f", None),
    ("baseline_fragile_to_inundated", "Fragile to inundated", "#2171b5", None),
]


def load_block_population() -> pd.DataFrame:
    """Load POP20 from the raw 2020 Census block file."""
    try:
        import pyogrio

        blocks = pyogrio.read_dataframe(
            RAW_BLOCKS_PATH,
            columns=["GEOID20", "COUNTYFP20", "POP20"],
            read_geometry=False,
        )
    except Exception:
        import geopandas as gpd

        blocks = gpd.read_file(
            RAW_BLOCKS_PATH,
            columns=["GEOID20", "COUNTYFP20", "POP20"],
            ignore_geometry=True,
        )

    blocks = blocks.loc[blocks["COUNTYFP20"].astype(str).str.zfill(3).isin(TRI_COUNTY_FIPS)].copy()
    blocks["block_geoid"] = blocks["GEOID20"].astype(str).str.zfill(15)
    blocks["pop20"] = pd.to_numeric(blocks["POP20"], errors="coerce").fillna(0).astype(int)
    return blocks[["block_geoid", "pop20"]].drop_duplicates("block_geoid")


def load_long_with_population() -> pd.DataFrame:
    usecols = [
        "block_geoid",
        "block_group_geoid",
        "county_name",
        "slr_ft",
        "baseline_status",
        "scenario_status",
        "baseline_redundant_to_fragile",
        "baseline_redundant_to_isolated",
        "baseline_redundant_to_inundated",
        "baseline_fragile_to_isolated",
        "baseline_fragile_to_inundated",
    ]
    long_df = pd.read_csv(
        ANALYSIS_DIR / "block_level_long_dataset.csv",
        dtype={"block_geoid": "string", "block_group_geoid": "string"},
        usecols=usecols,
    )
    pop = load_block_population()
    merged = long_df.merge(pop, on="block_geoid", how="left", validate="many_to_one")
    missing = int(merged["pop20"].isna().sum())
    if missing:
        raise ValueError(f"{missing:,} block-scenario rows are missing POP20 after merge.")
    return merged


def build_transition_population_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scenarios = df.loc[df["slr_ft"].isin(SLR_LEVELS)].copy()
    for slr_ft, group in scenarios.groupby("slr_ft", sort=True):
        total_blocks = 0
        total_pop = 0
        scenario_rows = []
        for col, label, _, _ in TRANSITIONS:
            mask = group[col].eq(1)
            n_blocks = int(mask.sum())
            pop20 = int(group.loc[mask, "pop20"].sum())
            scenario_rows.append(
                {
                    "slr_ft": int(slr_ft),
                    "transition": label,
                    "n_blocks": n_blocks,
                    "pop20": pop20,
                }
            )
            total_blocks += n_blocks
            total_pop += pop20
        for row in scenario_rows:
            row["share_of_transition_blocks"] = row["n_blocks"] / total_blocks if total_blocks else 0.0
            row["share_of_transition_pop20"] = row["pop20"] / total_pop if total_pop else 0.0
            rows.append(row)
    output = pd.DataFrame(rows)
    output.to_csv(TABLES_DIR / "fig4_transition_population_by_slr.csv", index=False)
    return output


def build_cumulative_population_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scenarios = df.loc[df["slr_ft"].isin(SLR_LEVELS)].copy()
    for slr_ft, group in scenarios.groupby("slr_ft", sort=True):
        new_inundated = group["baseline_redundant_to_inundated"].eq(1) | group["baseline_fragile_to_inundated"].eq(1)
        new_iso_plus = new_inundated | group["baseline_redundant_to_isolated"].eq(1) | group["baseline_fragile_to_isolated"].eq(1)
        new_all = new_iso_plus | group["baseline_redundant_to_fragile"].eq(1)
        added_by_fragile = new_all & ~new_iso_plus

        rows.append(
            {
                "slr_ft": int(slr_ft),
                "new_inundated_blocks": int(new_inundated.sum()),
                "new_inundated_pop20": int(group.loc[new_inundated, "pop20"].sum()),
                "new_isolated_or_inundated_blocks": int(new_iso_plus.sum()),
                "new_isolated_or_inundated_pop20": int(group.loc[new_iso_plus, "pop20"].sum()),
                "new_fragile_or_worse_blocks": int(new_all.sum()),
                "new_fragile_or_worse_pop20": int(group.loc[new_all, "pop20"].sum()),
                "added_by_fragile_blocks": int(added_by_fragile.sum()),
                "added_by_fragile_pop20": int(group.loc[added_by_fragile, "pop20"].sum()),
            }
        )
    output = pd.DataFrame(rows)
    output["added_by_fragile_share_of_all_blocks"] = (
        output["added_by_fragile_blocks"] / output["new_fragile_or_worse_blocks"]
    )
    output["added_by_fragile_share_of_all_pop20"] = (
        output["added_by_fragile_pop20"] / output["new_fragile_or_worse_pop20"]
    )
    output.to_csv(TABLES_DIR / "fig4_cumulative_population_by_slr.csv", index=False)
    return output


def save_figure(fig: plt.Figure, stem: str, dpi: int = 300) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.05)


def plot_population_transition_figure(transition_pop: pd.DataFrame) -> None:
    x = np.array(SLR_LEVELS)
    fig, ax = plt.subplots(figsize=(6.7, 4.8), constrained_layout=True)
    bottoms = np.zeros(len(x))
    for _, label, color, hatch in TRANSITIONS:
        vals = (
            transition_pop.loc[transition_pop["transition"].eq(label)]
            .set_index("slr_ft")
            .reindex(SLR_LEVELS)["pop20"]
            .fillna(0)
            .to_numpy()
        )
        ax.bar(x, vals, bottom=bottoms, width=0.62, color=color, hatch=hatch, edgecolor="white", linewidth=0.5)
        bottoms += vals

    legend = [
        Patch(facecolor="none", edgecolor="none", label="Redundant baseline"),
        Patch(facecolor="#fec44f", hatch="///", edgecolor="gray", label="  to fragile"),
        Patch(facecolor="#fc8d59", hatch="///", edgecolor="gray", label="  to isolated"),
        Patch(facecolor="#74add1", hatch="///", edgecolor="gray", label="  to inundated"),
        Patch(facecolor="none", edgecolor="none", label="Fragile baseline"),
        Patch(facecolor="#d7301f", label="  to isolated"),
        Patch(facecolor="#2171b5", label="  to inundated"),
    ]
    ax.legend(handles=legend, fontsize=9.5, loc="upper left", framealpha=0.9)
    ax.set_title("Population in transition categories\nHatched = redundant baseline; solid = fragile baseline", fontsize=12)
    ax.set_xlabel("Sea-level rise (ft)", fontsize=12)
    ax.set_ylabel("2020 population in blocks", fontsize=12)
    ax.set_xticks(SLR_LEVELS)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000_000:.1f}M"))
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, "fig4a_population_transition_decomposition")
    plt.close(fig)


def plot_population_cumulative_figure(cumulative: pd.DataFrame) -> None:
    x = cumulative["slr_ft"].to_numpy()
    y_in = cumulative["new_inundated_pop20"].to_numpy()
    y_iso = cumulative["new_isolated_or_inundated_pop20"].to_numpy()
    y_all = cumulative["new_fragile_or_worse_pop20"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.7, 4.8), constrained_layout=True)
    ax.fill_between(x, 0, y_in, color="#1f78b4", alpha=0.15)
    ax.fill_between(x, y_in, y_iso, color="#d7301f", alpha=0.15)
    ax.fill_between(x, y_iso, y_all, color="#fdae61", alpha=0.20)
    ax.plot(x, y_in, color="#1f78b4", linewidth=2.3, marker="^", markersize=7, label="New inundated")
    ax.plot(x, y_iso, color="#d7301f", linewidth=2.3, marker="s", markersize=7, label="+ New isolated")
    ax.plot(x, y_all, color="#e6550d", linewidth=2.3, marker="o", markersize=7, label="+ New fragile")
    ax.set_title("Population added by broader access-loss definitions\nCumulative new adverse transitions", fontsize=12)
    ax.set_xlabel("Sea-level rise (ft)", fontsize=12)
    ax.set_ylabel("2020 population in blocks", fontsize=12)
    ax.set_xticks(SLR_LEVELS)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000_000:.1f}M"))
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, "fig4b_population_cumulative_adverse_transitions")
    plt.close(fig)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_long_with_population()
    transition_pop = build_transition_population_table(df)
    cumulative = build_cumulative_population_table(df)

    plot_population_transition_figure(transition_pop)
    plot_population_cumulative_figure(cumulative)

    print("Wrote:")
    for path in [
        TABLES_DIR / "fig4_transition_population_by_slr.csv",
        TABLES_DIR / "fig4_cumulative_population_by_slr.csv",
        FIGURES_DIR / "fig4a_population_transition_decomposition.png",
        FIGURES_DIR / "fig4b_population_cumulative_adverse_transitions.png",
    ]:
        print(f"  {path.relative_to(PROJECT_ROOT)}")

    print("\nCumulative population:")
    print(cumulative.to_string(index=False))


if __name__ == "__main__":
    main()
