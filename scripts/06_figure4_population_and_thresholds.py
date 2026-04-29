#!/usr/bin/env python
"""Population-weighted supplements for Figure 4 transition analysis.

This script adds 2020 Census block population (POP20) to the block-level
SLR transition dataset, then exports population-weighted Figure 4 summaries
and a scenario-threshold "early warning" figure.
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
        "baseline_isolated_to_inundated",
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
    rows = []
    scen = df.loc[df["slr_ft"].isin(SLR_LEVELS)].copy()
    for slr_ft, group in scen.groupby("slr_ft", sort=True):
        total_blocks = 0
        total_pop = 0
        local_rows = []
        for col, label, _, _ in TRANSITIONS:
            mask = group[col].eq(1)
            n_blocks = int(mask.sum())
            pop20 = int(group.loc[mask, "pop20"].sum())
            local_rows.append(
                {
                    "slr_ft": int(slr_ft),
                    "transition": label,
                    "n_blocks": n_blocks,
                    "pop20": pop20,
                }
            )
            total_blocks += n_blocks
            total_pop += pop20
        for row in local_rows:
            row["share_of_transition_blocks"] = row["n_blocks"] / total_blocks if total_blocks else 0.0
            row["share_of_transition_pop20"] = row["pop20"] / total_pop if total_pop else 0.0
            rows.append(row)
    output = pd.DataFrame(rows)
    output.to_csv(TABLES_DIR / "fig4_transition_population_by_slr.csv", index=False)
    return output


def build_cumulative_population_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scen = df.loc[df["slr_ft"].isin(SLR_LEVELS)].copy()
    for slr_ft, group in scen.groupby("slr_ft", sort=True):
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


def build_transition_probability_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scen = df.loc[df["slr_ft"].isin(SLR_LEVELS)].copy()
    for slr_ft, group in scen.groupby("slr_ft", sort=True):
        for baseline_status in ["redundant", "fragile"]:
            denom = group["baseline_status"].eq(baseline_status)
            denom_blocks = int(denom.sum())
            denom_pop = int(group.loc[denom, "pop20"].sum())
            relevant = [
                item for item in TRANSITIONS if item[0].startswith(f"baseline_{baseline_status}_to")
            ]
            for col, label, _, _ in relevant:
                mask = denom & group[col].eq(1)
                n_blocks = int(mask.sum())
                pop20 = int(group.loc[mask, "pop20"].sum())
                rows.append(
                    {
                        "slr_ft": int(slr_ft),
                        "baseline_status": baseline_status,
                        "transition": label,
                        "n_blocks": n_blocks,
                        "denominator_blocks": denom_blocks,
                        "block_transition_probability": n_blocks / denom_blocks if denom_blocks else np.nan,
                        "pop20": pop20,
                        "denominator_pop20": denom_pop,
                        "pop20_transition_share": pop20 / denom_pop if denom_pop else np.nan,
                    }
                )
    output = pd.DataFrame(rows)
    output.to_csv(TABLES_DIR / "fig4_transition_probabilities_by_slr.csv", index=False)
    return output


def build_added_community_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scen = df.loc[df["slr_ft"].isin(SLR_LEVELS)].copy()
    for slr_ft, group in scen.groupby("slr_ft", sort=True):
        flags = group.assign(
            new_inundated=(
                group["baseline_redundant_to_inundated"].eq(1)
                | group["baseline_fragile_to_inundated"].eq(1)
            ),
            new_iso_plus=(
                group["baseline_redundant_to_inundated"].eq(1)
                | group["baseline_fragile_to_inundated"].eq(1)
                | group["baseline_redundant_to_isolated"].eq(1)
                | group["baseline_fragile_to_isolated"].eq(1)
            ),
            new_all=(
                group["baseline_redundant_to_inundated"].eq(1)
                | group["baseline_fragile_to_inundated"].eq(1)
                | group["baseline_redundant_to_isolated"].eq(1)
                | group["baseline_fragile_to_isolated"].eq(1)
                | group["baseline_redundant_to_fragile"].eq(1)
            ),
        )
        by_bg = (
            flags.groupby("block_group_geoid", as_index=False)
            .agg(
                has_new_inundated=("new_inundated", "max"),
                has_new_isolated_or_inundated=("new_iso_plus", "max"),
                has_new_fragile_or_worse=("new_all", "max"),
                pop20_new_fragile_only=(
                    "pop20",
                    lambda s: int(s[flags.loc[s.index, "new_all"] & ~flags.loc[s.index, "new_iso_plus"]].sum()),
                ),
            )
        )
        added_bg = by_bg["has_new_fragile_or_worse"] & ~by_bg["has_new_isolated_or_inundated"]
        rows.append(
            {
                "slr_ft": int(slr_ft),
                "block_groups_with_new_inundated": int(by_bg["has_new_inundated"].sum()),
                "block_groups_with_new_isolated_or_inundated": int(by_bg["has_new_isolated_or_inundated"].sum()),
                "block_groups_with_new_fragile_or_worse": int(by_bg["has_new_fragile_or_worse"].sum()),
                "block_groups_added_by_fragile_only": int(added_bg.sum()),
                "pop20_in_fragile_only_added_block_groups": int(by_bg.loc[added_bg, "pop20_new_fragile_only"].sum()),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(TABLES_DIR / "fig4_block_groups_added_by_fragility.csv", index=False)
    return output


def first_scenario_status(row: pd.Series, statuses: set[str]) -> int | None:
    for slr_ft in SLR_LEVELS:
        value = row.get(slr_ft)
        if value in statuses:
            return slr_ft
    return None


def build_threshold_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    status_wide = (
        df.loc[df["slr_ft"].isin([0, *SLR_LEVELS]), ["block_geoid", "slr_ft", "scenario_status"]]
        .pivot(index="block_geoid", columns="slr_ft", values="scenario_status")
        .reset_index()
    )
    block_meta = (
        df.loc[df["slr_ft"].eq(0), ["block_geoid", "baseline_status", "pop20"]]
        .drop_duplicates("block_geoid")
        .copy()
    )
    status_wide = status_wide.merge(block_meta, on="block_geoid", how="left", validate="one_to_one")
    status_wide["first_fragile_slr"] = status_wide.apply(
        lambda row: first_scenario_status(row, {"fragile"}), axis=1
    )
    status_wide["first_severe_slr"] = status_wide.apply(
        lambda row: first_scenario_status(row, {"isolated", "inundated"}), axis=1
    )
    status_wide["status_6ft"] = status_wide[6]

    redundant = status_wide.loc[status_wide["baseline_status"].eq("redundant")].copy()
    conditions = [
        redundant["first_fragile_slr"].notna()
        & redundant["first_severe_slr"].notna()
        & (redundant["first_fragile_slr"] < redundant["first_severe_slr"]),
        redundant["first_fragile_slr"].notna() & redundant["first_severe_slr"].isna(),
        redundant["first_severe_slr"].notna()
        & (
            redundant["first_fragile_slr"].isna()
            | (redundant["first_severe_slr"] <= redundant["first_fragile_slr"])
        ),
    ]
    labels = [
        "Fragile before severe loss",
        "Fragile by 6 ft, not severe",
        "Directly severe before fragile",
    ]
    redundant["threshold_path"] = np.select(conditions, labels, default="No adverse transition by 6 ft")

    path_summary = (
        redundant.groupby("threshold_path", as_index=False)
        .agg(n_blocks=("block_geoid", "size"), pop20=("pop20", "sum"))
        .sort_values("pop20", ascending=False)
    )

    severe_label = redundant["first_severe_slr"].fillna("Not severe by 6 ft").astype(str)
    redundant["first_severe_label"] = severe_label
    matrix = (
        redundant.loc[
            redundant["first_fragile_slr"].notna()
            & redundant["first_severe_slr"].notna()
            & (redundant["first_fragile_slr"] < redundant["first_severe_slr"])
        ]
        .groupby(["first_fragile_slr", "first_severe_slr"], as_index=False)
        .agg(n_blocks=("block_geoid", "size"), pop20=("pop20", "sum"))
    )

    fragile = status_wide.loc[status_wide["baseline_status"].eq("fragile")].copy()
    fragile["first_severe_label"] = fragile["first_severe_slr"].fillna("Not severe by 6 ft").astype(str)
    fragile_summary = (
        fragile.groupby("first_severe_label", as_index=False)
        .agg(n_blocks=("block_geoid", "size"), pop20=("pop20", "sum"))
    )

    path_summary.to_csv(TABLES_DIR / "fig4_threshold_path_summary_population.csv", index=False)
    matrix.to_csv(TABLES_DIR / "fig4_redundant_fragile_then_severe_threshold_matrix.csv", index=False)
    fragile_summary.to_csv(TABLES_DIR / "fig4_baseline_fragile_severe_threshold_population.csv", index=False)
    return path_summary, matrix, fragile_summary


def save_figure(fig: plt.Figure, stem: str, dpi: int = 300) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.05)


def plot_population_transition_figure(transition_pop: pd.DataFrame) -> None:
    x = np.array(SLR_LEVELS)
    fig, ax = plt.subplots(figsize=(6.7, 4.8), constrained_layout=True)
    bottoms = np.zeros(len(x))
    for col, label, color, hatch in TRANSITIONS:
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


def plot_threshold_ladder(
    path_summary: pd.DataFrame,
    matrix: pd.DataFrame,
    fragile_summary: pd.DataFrame,
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15, 0.85]},
    )

    fragile_levels = [1, 2, 3, 4, 5]
    severe_levels = [2, 3, 4, 5, 6]
    grid = pd.DataFrame(0, index=severe_levels, columns=fragile_levels, dtype=float)
    for _, row in matrix.iterrows():
        grid.loc[int(row["first_severe_slr"]), int(row["first_fragile_slr"])] = row["pop20"]

    im = ax1.imshow(grid.values, origin="lower", cmap="YlOrRd", aspect="auto")
    ax1.set_xticks(np.arange(len(fragile_levels)), fragile_levels)
    ax1.set_yticks(np.arange(len(severe_levels)), severe_levels)
    ax1.set_xlabel("First fragile SLR scenario (ft)")
    ax1.set_ylabel("First isolated/inundated SLR scenario (ft)")
    ax1.set_title("Baseline redundant blocks\nfragile before severe loss", fontsize=12)
    for i, severe in enumerate(severe_levels):
        for j, fragile in enumerate(fragile_levels):
            value = grid.loc[severe, fragile]
            if value > 0:
                ax1.text(j, i, f"{value:,.0f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("2020 population")

    path_lookup = path_summary.set_index("threshold_path")["pop20"].to_dict()
    severe_by_6 = int(
        fragile_summary.loc[
            fragile_summary["first_severe_label"].ne("Not severe by 6 ft"),
            "pop20",
        ].sum()
    )
    not_severe_by_6 = int(
        fragile_summary.loc[
            fragile_summary["first_severe_label"].eq("Not severe by 6 ft"),
            "pop20",
        ].sum()
    )

    rows = [
        (
            "Baseline redundant\nnewly affected by 6 ft",
            [
                ("Directly severe", path_lookup.get("Directly severe before fragile", 0), "#74add1"),
                ("Fragile before severe", path_lookup.get("Fragile before severe loss", 0), "#fdae61"),
                ("Fragile only by 6 ft", path_lookup.get("Fragile by 6 ft, not severe", 0), "#fee08b"),
            ],
        ),
        (
            "Baseline fragile\nalready at risk",
            [
                ("Severe by 6 ft", severe_by_6, "#d7301f"),
                ("Still fragile at 6 ft", not_severe_by_6, "#bdbdbd"),
            ],
        ),
    ]

    y_positions = np.arange(len(rows))
    for y, (_, segments) in zip(y_positions, rows):
        left = 0
        for label, value, color in segments:
            ax2.barh(y, value, left=left, color=color, label=label)
            if value >= 200_000:
                x_text = left + value / 2
                ax2.text(x_text, y, f"{value / 1_000_000:.2f}M", ha="center", va="center", fontsize=8)
            left += value

    handles = []
    seen = set()
    for _, segments in rows:
        for label, _, color in segments:
            if label not in seen:
                handles.append(Patch(facecolor=color, label=label))
                seen.add(label)
    ax2.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.9)
    ax2.set_yticks(y_positions, [row[0] for row in rows])
    ax2.set_xlabel("2020 population in affected blocks")
    ax2.set_title("Who is affected by 6 ft?\nPopulation-weighted paths", fontsize=12)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000_000:.1f}M"))
    ax2.grid(axis="x", alpha=0.3)

    save_figure(fig, "fig4c_population_threshold_ladder")
    plt.close(fig)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_long_with_population()
    transition_pop = build_transition_population_table(df)
    cumulative = build_cumulative_population_table(df)
    probabilities = build_transition_probability_table(df)
    added_communities = build_added_community_table(df)
    path_summary, matrix, fragile_summary = build_threshold_tables(df)

    plot_population_transition_figure(transition_pop)
    plot_population_cumulative_figure(cumulative)
    plot_threshold_ladder(path_summary, matrix, fragile_summary)

    print("Wrote:")
    for path in [
        TABLES_DIR / "fig4_transition_population_by_slr.csv",
        TABLES_DIR / "fig4_cumulative_population_by_slr.csv",
        TABLES_DIR / "fig4_transition_probabilities_by_slr.csv",
        TABLES_DIR / "fig4_block_groups_added_by_fragility.csv",
        TABLES_DIR / "fig4_threshold_path_summary_population.csv",
        TABLES_DIR / "fig4_redundant_fragile_then_severe_threshold_matrix.csv",
        TABLES_DIR / "fig4_baseline_fragile_severe_threshold_population.csv",
        FIGURES_DIR / "fig4a_population_transition_decomposition.png",
        FIGURES_DIR / "fig4b_population_cumulative_adverse_transitions.png",
        FIGURES_DIR / "fig4c_population_threshold_ladder.png",
    ]:
        print(f"  {path.relative_to(PROJECT_ROOT)}")

    print("\nTransition probabilities by SLR:")
    print(
        probabilities.pivot_table(
            index=["slr_ft", "baseline_status"],
            columns="transition",
            values="block_transition_probability",
        )
        .fillna(0)
        .round(4)
        .to_string()
    )
    print("\nCumulative population:")
    print(cumulative.to_string(index=False))
    print("\nBlock groups added by fragile-only screening:")
    print(added_communities.to_string(index=False))


if __name__ == "__main__":
    main()
