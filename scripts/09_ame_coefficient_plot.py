#!/usr/bin/env python
"""Two-panel coefficient plot of the final transition-model AMEs.

Reads the frozen final-draft estimates written by scripts/08_final_draft_inference.R
(approach arm, demographic-only specification, Conley 5 km SEs, delta-method AMEs)
and plots them; nothing is re-estimated. The same CSV generates
Table tab:ame_transition_probabilities (main_transition_ames_conley5.tex), and the
script checks the plotted estimates against that .tex before drawing.

95% CIs are the stored conf_low/conf_high columns, which equal
estimate +/- qnorm(0.975) * std_error (checked below).

Outputs: figs/fig_ame_coefficients.pdf (vector) and figs/fig_ame_coefficients.png.

Run with the research-geo env and MKL_THREADING_LAYER=SEQUENTIAL set.
"""

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AME_CSV = PROJECT_ROOT / "outputs" / "final_draft" / "main_transition_ames_conley5.csv"
AME_TEX = PROJECT_ROOT / "outputs" / "final_draft" / "main_transition_ames_conley5.tex"
FIG_DIR = PROJECT_ROOT / "figs"
FIG_STEM = "fig_ame_coefficients"

EXPECTED_SPEC = "demographic_only"
EXPECTED_VCOV = "Conley (1999), uniform kernel, 5 km cutoff; delta-method AME"
Z_975 = 1.959963984540054

# Covariates top to bottom, in table order.
COVARIATES = [
    ("Black share", "Black share"),
    ("Hispanic share", "Hispanic share"),
    ("Renter share", "Renter share"),
    ("Log median income", "Log median income"),
    ("Age 65+ share", "Age 65+ share"),
    ("No-vehicle household share", "No-vehicle household share"),
]

# Destination states, less to more severe. One color and marker per destination,
# identical in both panels. Single-hue ordinal blue ramp (steps 250/400/550/700)
# so the order survives grayscale; marker shape is the redundant cue.
DESTINATIONS = {
    "Fragile": {"color": "#86b6ef", "marker": "o", "label": "Fragile"},
    "Isolated": {"color": "#3987e5", "marker": "s", "label": "Isolated"},
    "Inundated": {"color": "#1c5cab", "marker": "^", "label": "Inundated"},
    "Worse": {"color": "#0d366b", "marker": "D", "label": "Any worse state"},
}

PANELS = [
    ("A", "From baseline-redundant access", "Redundant",
     ["Fragile", "Isolated", "Inundated", "Worse"]),
    ("B", "From baseline-fragile access", "Fragile",
     ["Isolated", "Inundated", "Worse"]),
]

INK = "#1f1f1f"
INK_MUTED = "#5a5a5a"
GRID = "#e6e6e6"
BAND = "#f5f5f3"
ZERO = "#8a8a8a"


def load_estimates() -> pd.DataFrame:
    d = pd.read_csv(AME_CSV)
    if set(d["spec"]) != {EXPECTED_SPEC} or set(d["vcov"]) != {EXPECTED_VCOV}:
        raise ValueError(f"{AME_CSV} is not the primary demographic-only Conley 5 km export.")
    if len(d) != 42:
        raise ValueError(f"Expected 42 AME rows (7 transitions x 6 covariates), found {len(d)}.")
    ci_gap = max(
        (d["conf_low"] - (d["estimate"] - Z_975 * d["std_error"])).abs().max(),
        (d["conf_high"] - (d["estimate"] + Z_975 * d["std_error"])).abs().max(),
    )
    if ci_gap > 1e-9:
        raise ValueError(f"Stored CIs differ from estimate +/- 1.96 SE by {ci_gap:.2e}.")
    d["origin"] = d["transition"].str.split(" -> ").str[0]
    d["destination"] = d["transition"].str.split(" -> ").str[1]
    for col in ("estimate", "conf_low", "conf_high"):
        d[f"{col}_pp"] = 100 * d[col]
    return d


def check_against_table(d: pd.DataFrame) -> None:
    """Every plotted point estimate must match the published .tex to 3 decimals."""
    tex = AME_TEX.read_text(encoding="utf-8")
    columns = [
        "Redundant -> Fragile", "Redundant -> Isolated", "Redundant -> Inundated",
        "Redundant -> Worse", "Fragile -> Isolated", "Fragile -> Inundated",
        "Fragile -> Worse",
    ]
    for covariate, _ in COVARIATES:
        line = next(l for l in tex.splitlines() if l.startswith(covariate + " &"))
        tex_vals = [float(v) for v in re.findall(r"-?\d+\.\d{3}", line)]
        csv_vals = [
            round(d.loc[(d["covariate"] == covariate) & (d["transition"] == t), "estimate_pp"].item(), 3)
            for t in columns
        ]
        if not np.allclose(tex_vals, csv_vals, atol=5e-4):
            raise ValueError(f"{covariate}: CSV {csv_vals} does not match table {tex_vals}.")


def draw(d: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), sharex=True, sharey=True)
    n_cov = len(COVARIATES)
    lo = d["conf_low_pp"].min()
    hi = d["conf_high_pp"].max()
    xlim = (lo - 0.5, hi + 0.5)
    # Fixed vertical slot per destination, shared by both panels, so a given
    # outcome sits at the same height in A and B.
    slot = dict(zip(DESTINATIONS, np.linspace(-0.27, 0.27, len(DESTINATIONS))))

    for ax, (letter, title, origin, dests) in zip(axes, PANELS):
        for i in range(n_cov):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0, lw=0)
        ax.axvline(0, color=ZERO, lw=0.8, zorder=1)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)

        for dest in dests:
            off = slot[dest]
            style = DESTINATIONS[dest]
            sub = d[(d["origin"] == origin) & (d["destination"] == dest)]
            for i, (covariate, _) in enumerate(COVARIATES):
                row = sub[sub["covariate"] == covariate].iloc[0]
                y = i + off
                ax.plot(
                    [row["conf_low_pp"], row["conf_high_pp"]], [y, y],
                    color=style["color"], lw=1.3, solid_capstyle="butt", zorder=2,
                )
                significant = bool(row["sig_05"])
                ax.plot(
                    row["estimate_pp"], y, marker=style["marker"], ms=5.2 if style["marker"] != "D" else 4.6,
                    mfc=style["color"] if significant else "white",
                    mec=style["color"], mew=1.1, zorder=3, ls="none",
                )

        ax.set_title(f"{letter}.  {title}", loc="center", fontsize=9, fontweight="bold",
                     color=INK, pad=6)
        ax.set_xlim(*xlim)
        ax.set_ylim(n_cov - 0.5, -0.5)
        ax.set_yticks(range(n_cov))
        ax.set_yticklabels([label for _, label in COVARIATES])
        ax.tick_params(axis="y", length=0, pad=6)
        ax.tick_params(axis="x", length=3, width=0.6)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.6)

    handles = [
        Line2D([0], [0], color=s["color"], lw=1.3, marker=s["marker"],
               ms=5.2 if s["marker"] != "D" else 4.6, mfc=s["color"], mec=s["color"], label=s["label"])
        for s in DESTINATIONS.values()
    ]
    handles.append(Line2D([0], [0], color=INK_MUTED, lw=0, marker="o", ms=5.2, mfc="white",
                          mec=INK_MUTED, mew=1.1, label="95% CI includes 0"))
    fig.legend(
        handles=handles, loc="upper center", ncol=len(handles), frameon=False,
        title="Transition to:", title_fontsize=8.5, fontsize=8.5,
        bbox_to_anchor=(0.56, 1.0), handlelength=1.8, columnspacing=1.3,
        handletextpad=0.5,
    )
    fig.supxlabel(
        "Change in transition probability (percentage points per 1 SD increase in covariate)",
        fontsize=8.5, color=INK, x=0.56, y=0.035,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.92), w_pad=1.5)
    return fig


def main() -> None:
    d = load_estimates()
    check_against_table(d)
    FIG_DIR.mkdir(exist_ok=True)
    fig = draw(d)
    fig.savefig(FIG_DIR / f"{FIG_STEM}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{FIG_STEM}.png", dpi=300, bbox_inches="tight")
    print(f"Saved {FIG_DIR / FIG_STEM}.pdf and .png")


if __name__ == "__main__":
    main()
