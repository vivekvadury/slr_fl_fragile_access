#!/usr/bin/env python3
"""Generate 03_build_extension_dataset_and_memo.ipynb"""
import json, uuid
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "03_build_extension_dataset_and_memo.ipynb"

def md(source):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8],
            "metadata": {}, "outputs": [], "source": source}

# ============================================================
# CELL SOURCES
# ============================================================

TITLE = """\
# Extension Memo: Network Fragility and Redundancy Under Sea-Level Rise
### South Florida Tri-County Study Area (Broward, Miami-Dade, Palm Beach)

**Status:** Working draft — not for citation

---

This notebook builds a cohesive analysis dataset and preliminary findings for an extension of
Best et al. (2023). Rather than focusing only on *binary isolation* (is there any dry path?),
this project measures *network fragility* and *redundancy* under sea-level rise:

- **Isolated** — no dry route to services exists
- **Fragile** — connected, but only via a single dry path (edge-disjoint paths = 1)
- **Redundant** — connected with 2+ edge-disjoint dry paths

The central analytic argument: many blocks lose network redundancy — becoming fragile —
**before** they become fully isolated. Standard binary-isolation metrics miss this earlier
deterioration in network resilience."""

MOTIVATION = """\
## Motivation

Standard access-under-SLR analyses define a binary outcome: is the block reachable via some
dry path? This treats "barely connected through a single corridor" the same as "connected via
multiple independent routes." Network redundancy is a core resilience concept — a fragile
connection (one dry path) is qualitatively different from a redundant one, because a single
additional inundated edge can sever it completely.

### Conceptual Status Hierarchy

```
Redundant  →  Fragile  →  Isolated  →  Inundated
(2+ paths)   (1 path)    (0 paths)   (origin submerged)
```

As SLR increases, blocks may transition through this hierarchy. `Redundant → Fragile` is an
analytically distinct event — a loss of resilience — that binary isolation measures cannot detect.

### On Renaming `detour_ratio` to `path_inflation_ratio`

The raw data contain `detour_ratio = dry_shortest_path / baseline_shortest_path`. We rename
this to **`path_inflation_ratio`** in the analysis, with **"dry-path inflation ratio"** as the
preferred narrative label.

**Why not "detour ratio"?** The term implies a guaranteed commute-time effect. The network is
undirected, uses OSM segments intersected with NOAA inundation polygons, and does not model
traffic or turn restrictions. "Dry-path inflation ratio" is more precise: it measures how much
longer the *modeled* dry path is under the assumption that all inundated segments are
impassable. The raw `detour_ratio` column is preserved for traceability."""

SETUP_SECTION = "## 1. Setup and Project Paths\n\nImport packages, configure paths, create output directories."

SETUP_CODE = """\
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path
import json
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_theme(style="whitegrid", font_scale=0.95)
except ImportError:
    HAS_SEABORN = False
    print("NOTE: seaborn not installed — using matplotlib defaults.")

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed — regression section will be skipped.")

print("Imports complete.")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.4f}".format)"""

PATHS_CODE = """\
# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# This notebook is expected to live in scripts/ within the project root.
# If your working directory differs, update PROJECT_ROOT below.
NOTEBOOK_DIR = Path().resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent

# Precomputed access flag run
RUN_DATE = "2026-04-03_run"
RUN_DIR  = PROJECT_ROOT / "data" / "processed" / "access" / "edited" / RUN_DATE
DIAG_DIR = RUN_DIR / "diagnostics"

# Analysis outputs
ANALYSIS_DIR = PROJECT_ROOT / "data" / "processed" / "analysis"
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR   = PROJECT_ROOT / "outputs" / "tables"

for d in [ANALYSIS_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Census API
CENSUS_API_KEY = "ff5d487d0a2a22c658bf319ba136c27db32aa0be"
CENSUS_YEAR    = "2022"

# Study area
STUDY_STATE    = "12"   # Florida
STUDY_COUNTIES = {"011": "Broward", "086": "Miami-Dade", "099": "Palm Beach"}
SLR_LEVELS     = list(range(1, 7))

# Dtypes for reading raw files — preserves leading zeros in GEOIDs
RESULT_DTYPE = {
    "block_geoid":                    "string",
    "block_group_geoid":              "string",
    "tract_geoid":                    "string",
    "block":                          "string",
    "county_fips":                    "string",
    "county_name":                    "string",
    "slr_layer_name":                 "string",
    "nearest_reachable_service_type": "string",
    "nearest_reachable_service_id":   "string",
    "baseline_status":                "string",
    "scenario_status":                "string",
}

# Visualization constants
STATUS_ORDER  = ["inundated", "isolated", "fragile", "redundant"]
STATUS_COLORS = {
    "inundated": "#1f78b4",
    "isolated":  "#d7301f",
    "fragile":   "#fdae61",
    "redundant": "#bdbdbd",
}
COUNTY_COLORS = {
    "Broward":    "#4daf4a",
    "Miami-Dade": "#984ea3",
    "Palm Beach": "#ff7f00",
}

print(f"Project root : {PROJECT_ROOT}")
print(f"Run directory: {RUN_DIR}")
assert RUN_DIR.exists(), f"Run directory not found: {RUN_DIR}"
print("Paths OK. Output directories ready.")"""

INSPECT_MD = """\
## 2. Inspect Scenario Files

Find all `block_access_flags_long*.csv` files, inventory their contents, and verify that the
baseline rows (`slr_ft == 0`) are consistent across files before extracting one canonical copy.

Each file contains:
- **Baseline rows** (`slr_ft == 0`) — identical across all 6 files (to be verified below)
- **Scenario rows** (`slr_ft == X`) — unique to each file"""

FILE_INVENTORY_CODE = """\
# Find scenario files
csv_files = sorted(
    p for p in RUN_DIR.glob("block_access_flags_long*.csv")
    if "qa_sample" not in p.stem
)
print(f"Found {len(csv_files)} scenario file(s):")
for f in csv_files:
    print(f"  {f.name}")

# Build file inventory
inventory_records = []
for path in csv_files:
    slim = pd.read_csv(path, usecols=["slr_ft"], dtype={"slr_ft": "Int64"})
    counts = slim["slr_ft"].value_counts().sort_index()
    inventory_records.append({
        "file_name":         path.name,
        "total_rows":        len(slim),
        "slr_ft_values":     ", ".join(str(int(x)) for x in counts.index.tolist()),
        "n_baseline_rows":   int(counts.get(0, 0)),
        "n_scenario_rows":   int(slim["slr_ft"].ne(0).sum()),
    })

inventory_df = pd.DataFrame(inventory_records)
display(inventory_df)"""

BASELINE_CHECK_CODE = """\
# ---------------------------------------------------------------------------
# Verify baseline consistency: are the slr_ft==0 rows identical across files?
# ---------------------------------------------------------------------------
BASELINE_CHECK_COLS = [
    "block_geoid", "block_centroid_inundated", "block_centroid_isolated",
    "block_centroid_redundant", "block_centroid_fragile",
    "n_reachable_services", "max_edge_disjoint_paths_any_service",
    "baseline_status", "scenario_status",
]

baseline_frames = {}
for path in csv_files:
    df = pd.read_csv(path, usecols=["slr_ft"] + BASELINE_CHECK_COLS,
                     dtype={**RESULT_DTYPE, "slr_ft": "Int64"})
    df["slr_ft"] = df["slr_ft"].astype(int)
    bl = (df.loc[df["slr_ft"] == 0, BASELINE_CHECK_COLS]
           .sort_values("block_geoid")
           .reset_index(drop=True))
    baseline_frames[path.name] = bl
    print(f"  {path.name}: {len(bl):,} baseline rows")

ref_name = list(baseline_frames.keys())[0]
ref_bl   = baseline_frames[ref_name]

all_equal = True
for name, bl in baseline_frames.items():
    if name == ref_name:
        continue
    try:
        pd.testing.assert_frame_equal(ref_bl, bl, check_dtype=False)
    except AssertionError as e:
        print(f"WARNING — {name} baseline differs from {ref_name}: {e}")
        all_equal = False

if all_equal:
    print(f"\\nBaseline consistency check PASSED across all {len(baseline_frames)} files.")
    print(f"Canonical baseline size: {len(ref_bl):,} blocks")
else:
    print("\\nWARNING: Baseline inconsistency detected. Investigate before proceeding.")"""

BUILD_MD = """\
## 3. Build Canonical Block-Level Long Dataset

**Design:**
1. Extract baseline (`slr_ft == 0`) from exactly one file (verified identical across all files above).
2. Load scenario-only rows (`slr_ft != 0`) from each file and stack.
3. Assert uniqueness on `(block_geoid, slr_ft)` before proceeding.
4. Standardize GEOID columns as zero-padded strings.
5. Create derived analysis variables.

### Key Variable Notes

**`path_inflation_ratio`** — renamed from raw `detour_ratio`.
- Definition: `dry_shortest_path_distance_m / baseline_shortest_path_distance_m`
- `= 1.0` if the dry route is unchanged; `> 1.0` if a longer route is needed due to inundated segments
- `NaN` if the block is isolated or inundated (no dry path exists — ratio is undefined)
- The raw `detour_ratio` column is preserved for traceability.

> **TODO**: The upstream code caps `max_edge_disjoint_paths_any_service` at 2
> (`MAX_EDGE_DISJOINT_PATHS_CAP = 2`). This means the model cannot distinguish 2-path from
> 3+ path redundancy. The fragile/redundant categories are therefore coarser than they appear."""

EXTRACT_BASELINE_CODE = """\
# Extract one canonical baseline (slr_ft == 0) from the first file
ref_path = csv_files[0]
print(f"Loading baseline from: {ref_path.name}")

full_ref = pd.read_csv(ref_path, dtype=RESULT_DTYPE, low_memory=False)
full_ref["slr_ft"] = full_ref["slr_ft"].astype(int)

baseline_df = (full_ref.loc[full_ref["slr_ft"] == 0]
               .copy()
               .sort_values("block_geoid")
               .reset_index(drop=True))

assert baseline_df["block_geoid"].is_unique, "Duplicate block_geoid in baseline."
assert set(baseline_df["slr_ft"].unique()) == {0}, "Unexpected slr_ft in baseline."

print(f"Canonical baseline: {len(baseline_df):,} unique blocks")
del full_ref"""

LOAD_SCENARIOS_CODE = """\
# Load scenario rows (slr_ft != 0) from each file and stack
scenario_frames = []
for path in csv_files:
    df = pd.read_csv(path, dtype=RESULT_DTYPE, low_memory=False)
    df["slr_ft"] = df["slr_ft"].astype(int)
    scen = df.loc[df["slr_ft"] != 0].copy()
    scenario_frames.append(scen)
    print(f"  {path.name}: {len(scen):,} scenario rows")
    del df

scenarios_df = pd.concat(scenario_frames, ignore_index=True)
del scenario_frames

dupes = scenarios_df.duplicated(subset=["block_geoid", "slr_ft"])
if dupes.sum() > 0:
    print(f"WARNING: {dupes.sum()} duplicate (block_geoid, slr_ft) rows — keeping first.")
    scenarios_df = scenarios_df.loc[~dupes].copy()
else:
    print("No duplicate (block_geoid, slr_ft) rows. Good.")

scenarios_df = scenarios_df.sort_values(["slr_ft", "block_geoid"]).reset_index(drop=True)
print(f"\\nScenario rows: {len(scenarios_df):,}")
print(f"SLR levels: {sorted(scenarios_df['slr_ft'].unique().tolist())}")
print(f"Unique blocks in scenarios: {scenarios_df['block_geoid'].nunique():,}")"""

STACK_CODE = """\
# Stack baseline + scenarios into one canonical long dataset
long_df = pd.concat([baseline_df, scenarios_df], ignore_index=True)
long_df = long_df.sort_values(["slr_ft", "block_geoid"]).reset_index(drop=True)

# Standardize GEOIDs
for col, width in [("block_geoid", 15), ("block_group_geoid", 12),
                   ("tract_geoid", 11), ("county_fips", 3)]:
    if col in long_df.columns:
        long_df[col] = long_df[col].astype(str).str.zfill(width)

# Row count checks
n_blocks   = baseline_df["block_geoid"].nunique()
n_expected = n_blocks * (1 + len(SLR_LEVELS))
n_actual   = len(long_df)
n_keys     = long_df.drop_duplicates(subset=["block_geoid", "slr_ft"]).shape[0]

print(f"Long dataset: {n_actual:,} rows")
print(f"  Unique (block_geoid, slr_ft) pairs: {n_keys:,}")
print(f"  Expected: {n_expected:,} ({n_blocks:,} blocks x {1 + len(SLR_LEVELS)} SLR levels)")

if n_actual != n_keys:
    print(f"WARNING: {n_actual - n_keys} duplicate rows detected.")
if n_actual != n_expected:
    print(f"WARNING: Row count {n_actual} differs from expected {n_expected}.")
else:
    print("Row count checks passed.")

display(long_df[["block_geoid", "block_group_geoid", "county_name",
                  "slr_ft", "scenario_status", "baseline_status"]].head(6))"""

DERIVED_VARS_CODE = """\
# ---------------------------------------------------------------------------
# Derived analysis variables
# ---------------------------------------------------------------------------

# 1. Ordered 4-category status
long_df["status_4cat"] = pd.Categorical(
    long_df["scenario_status"],
    categories=["inundated", "isolated", "fragile", "redundant"],
    ordered=True,
)

# 2. Fragile-or-worse: fragile | isolated | inundated
long_df["fragile_or_worse"] = (
    long_df["scenario_status"].isin(["fragile", "isolated", "inundated"])
).astype("Int8")

# 3. Any loss of redundancy: baseline-redundant → any worse outcome
long_df["any_loss_of_redundancy"] = (
    long_df["baseline_block_centroid_redundant"].eq(1) &
    long_df["scenario_status"].isin(["fragile", "isolated", "inundated"])
).astype("Int8")

# 4. Granular transition indicators
long_df["baseline_redundant_to_fragile"] = (
    long_df["baseline_block_centroid_redundant"].eq(1) & long_df["block_centroid_fragile"].eq(1)
).astype("Int8")
long_df["baseline_redundant_to_isolated"] = (
    long_df["baseline_block_centroid_redundant"].eq(1) & long_df["block_centroid_isolated"].eq(1)
).astype("Int8")
long_df["baseline_redundant_to_inundated"] = (
    long_df["baseline_block_centroid_redundant"].eq(1) & long_df["block_centroid_inundated"].eq(1)
).astype("Int8")
long_df["baseline_fragile_persisted"]    = long_df["persistent_fragile"].astype("Int8")
long_df["baseline_fragile_to_isolated"]  = (
    long_df["baseline_block_centroid_fragile"].eq(1) & long_df["block_centroid_isolated"].eq(1)
).astype("Int8")
long_df["baseline_fragile_to_inundated"] = (
    long_df["baseline_block_centroid_fragile"].eq(1) & long_df["block_centroid_inundated"].eq(1)
).astype("Int8")

# 5. Convenience aliases
long_df["became_fragile"]   = long_df["new_fragile_due_to_slr"].astype("Int8")
long_df["became_isolated"]  = long_df["new_isolated_due_to_slr"].astype("Int8")
long_df["became_inundated"] = long_df["new_inundated_due_to_slr"].astype("Int8")

# 6. Path inflation ratio — renamed from detour_ratio
# Raw detour_ratio is preserved. path_inflation_ratio is set to NaN for
# isolated/inundated blocks where no dry path exists.
long_df["path_inflation_ratio"] = pd.to_numeric(long_df["detour_ratio"], errors="coerce")

iso_or_inh = (long_df["block_centroid_isolated"].eq(1) | long_df["block_centroid_inundated"].eq(1))
long_df.loc[iso_or_inh, "path_inflation_ratio"] = np.nan
long_df["path_inflation_ratio"] = long_df["path_inflation_ratio"].replace([np.inf, -np.inf], np.nan)

# Sanity: connected blocks should have ratio >= 1.0
conn_mask  = long_df["path_inflation_ratio"].notna()
below_one  = (long_df.loc[conn_mask, "path_inflation_ratio"] < 0.99).sum()
if below_one > 0:
    print(f"WARNING: {below_one} connected blocks have path_inflation_ratio < 1.0 — investigate.")
else:
    print(f"path_inflation_ratio OK: {conn_mask.sum():,} connected, "
          f"{long_df['path_inflation_ratio'].isna().sum():,} NaN (isolated/inundated).")

print("Derived variables created.")
display(long_df[["block_geoid", "slr_ft", "scenario_status", "fragile_or_worse",
                  "any_loss_of_redundancy", "became_fragile", "path_inflation_ratio"]].head(8))"""

VALIDATION_MD = """\
## 4. Validation and Diagnostics

Compare our canonical dataset against the precomputed diagnostic files in `diagnostics/`.
Any discrepancies are flagged explicitly — nothing is silently ignored."""

CROSSTAB_CODE = """\
# Reproduce baseline_vs_scenario_crosstab and compare to precomputed file
positive = long_df.loc[long_df["slr_ft"] != 0].copy()

crosstab_own = (
    positive
    .groupby(["slr_ft", "baseline_status", "scenario_status"], as_index=False)
    .agg(n_blocks_own=("block_geoid", "size"))
    .sort_values(["slr_ft", "baseline_status", "scenario_status"])
    .reset_index(drop=True)
)

crosstab_ref = (
    pd.read_csv(DIAG_DIR / "baseline_vs_scenario_crosstab.csv")
    .sort_values(["slr_ft", "baseline_status", "scenario_status"])
    .reset_index(drop=True)
)

comp = crosstab_ref.merge(
    crosstab_own,
    on=["slr_ft", "baseline_status", "scenario_status"],
    how="outer",
    validate="one_to_one",
).fillna(0)
comp["diff"] = comp["n_blocks"] - comp["n_blocks_own"]

if comp["diff"].abs().max() == 0:
    print("VALIDATION PASSED: crosstab matches precomputed diagnostic exactly.")
else:
    print(f"DISCREPANCY: max |diff| = {comp['diff'].abs().max()}")
    display(comp.loc[comp["diff"].ne(0)])

print("\\nBaseline→Scenario crosstab from canonical dataset:")
display(crosstab_own)"""

TRANSITION_VALID_CODE = """\
# Reproduce transition_summary_by_slr and cross-check
transition_own = (
    positive
    .groupby("slr_ft", as_index=False)
    .agg(
        n_blocks                          = ("block_geoid",                   "size"),
        n_new_fragile                     = ("new_fragile_due_to_slr",        "sum"),
        n_new_isolated                    = ("new_isolated_due_to_slr",       "sum"),
        n_new_inundated                   = ("new_inundated_due_to_slr",      "sum"),
        n_baseline_redundant_to_fragile   = ("baseline_redundant_to_fragile", "sum"),
        n_baseline_redundant_to_isolated  = ("baseline_redundant_to_isolated","sum"),
        n_baseline_redundant_to_inundated = ("baseline_redundant_to_inundated","sum"),
        n_any_loss_of_redundancy          = ("any_loss_of_redundancy",        "sum"),
    )
    .sort_values("slr_ft")
    .reset_index(drop=True)
)

transition_ref = pd.read_csv(DIAG_DIR / "transition_summary_by_slr.csv")

# Cross-check key columns
check_pairs = [
    ("new_fragile_due_to_slr",   "n_new_fragile"),
    ("new_isolated_due_to_slr",  "n_new_isolated"),
    ("new_inundated_due_to_slr", "n_new_inundated"),
]
all_ok = True
for ref_col, own_col in check_pairs:
    if ref_col not in transition_ref.columns:
        continue
    for slr in SLR_LEVELS:
        ref_val = int(transition_ref.loc[transition_ref["slr_ft"] == slr, ref_col].iloc[0])
        own_val = int(transition_own.loc[transition_own["slr_ft"] == slr, own_col].iloc[0])
        if ref_val != own_val:
            print(f"MISMATCH slr={slr} {ref_col}: ref={ref_val}, own={own_val}")
            all_ok = False
if all_ok:
    print("VALIDATION PASSED: transition counts match precomputed file.")

print("\\nTransition summary from canonical dataset:")
display(transition_own)"""

COUNTY_STATUS_CODE = """\
# County-level status summary
county_status = (
    long_df.loc[long_df["slr_ft"] != 0]
    .groupby(["slr_ft", "county_name", "scenario_status"], as_index=False)
    .agg(n_blocks=("block_geoid", "size"))
)
county_total = county_status.groupby(["slr_ft", "county_name"])["n_blocks"].transform("sum")
county_status["share"] = county_status["n_blocks"] / county_total

print("County-level status shares at SLR = 3 ft:")
piv = county_status.loc[county_status["slr_ft"] == 3].pivot_table(
    index="county_name", columns="scenario_status", values="share", fill_value=0
)
display(piv.round(4))

print("\\nCounty-level status shares at SLR = 6 ft:")
piv6 = county_status.loc[county_status["slr_ft"] == 6].pivot_table(
    index="county_name", columns="scenario_status", values="share", fill_value=0
)
display(piv6.round(4))"""

BG_MD = """\
## 5. Aggregate to Block Group for Analysis

Build a block-group × SLR-scenario dataset for visualization and regression.

> **Note on path_inflation_ratio:** mean and median are computed only over connected
> (non-isolated, non-inundated) blocks, since the ratio is undefined when no dry path exists.
> Interpret the conditional mean/median with care — it describes blocks that *can* reach
> services, not unconditional average network stress."""

BG_AGG_CODE = """\
# ---------------------------------------------------------------------------
# Block-group × scenario aggregation
# ---------------------------------------------------------------------------
GROUP_KEYS = ["block_group_geoid", "county_fips", "county_name", "tract_geoid", "slr_ft"]

count_sum_cols = [
    "block_centroid_inundated", "block_centroid_isolated",
    "block_centroid_fragile", "block_centroid_redundant",
    "fragile_or_worse", "any_loss_of_redundancy",
    "new_fragile_due_to_slr", "new_isolated_due_to_slr", "new_inundated_due_to_slr",
    "baseline_redundant_to_fragile", "baseline_redundant_to_isolated",
    "baseline_redundant_to_inundated", "baseline_fragile_to_isolated",
    "baseline_fragile_to_inundated",
]

# Step 1: Counts and sums
count_sum_agg = {col: "sum" for col in count_sum_cols if col in long_df.columns}
count_sum_agg["block_geoid"] = "count"

bg_counts = (
    long_df.groupby(GROUP_KEYS)
    .agg(count_sum_agg)
    .reset_index()
    .rename(columns={"block_geoid": "total_blocks"})
)
# Remove Int8 ambiguity: cast sum columns to int
for col in count_sum_cols:
    if col in bg_counts.columns:
        bg_counts[col] = bg_counts[col].astype(int)

# Step 2: Mean/median of max_edge_disjoint_paths
bg_paths = (
    long_df.groupby(GROUP_KEYS)["max_edge_disjoint_paths_any_service"]
    .agg(["mean", "median"])
    .rename(columns={"mean":   "mean_max_edge_disjoint_paths",
                     "median": "median_max_edge_disjoint_paths"})
    .reset_index()
)

# Step 3: Mean/median path_inflation_ratio (NaN-safe — skips isolated/inundated)
bg_pir = (
    long_df.groupby(["block_group_geoid", "slr_ft"])["path_inflation_ratio"]
    .agg(["mean", "median"])
    .rename(columns={"mean":   "mean_path_inflation_ratio",
                     "median": "median_path_inflation_ratio"})
    .reset_index()
)

# Merge
bg_df = (bg_counts
         .merge(bg_paths, on=GROUP_KEYS, how="left")
         .merge(bg_pir,   on=["block_group_geoid", "slr_ft"], how="left"))

# Compute share columns
share_map = {
    "block_centroid_inundated":      "share_inundated",
    "block_centroid_isolated":       "share_isolated",
    "block_centroid_fragile":        "share_fragile",
    "block_centroid_redundant":      "share_redundant",
    "fragile_or_worse":              "share_fragile_or_worse",
    "any_loss_of_redundancy":        "share_lost_redundancy",
    "new_fragile_due_to_slr":        "share_new_fragile",
    "new_isolated_due_to_slr":       "share_new_isolated",
    "new_inundated_due_to_slr":      "share_new_inundated",
}
for count_col, share_col in share_map.items():
    if count_col in bg_df.columns:
        bg_df[share_col] = bg_df[count_col] / bg_df["total_blocks"]

print(f"Block-group dataset: {len(bg_df):,} rows")
print(f"  Block groups: {bg_df['block_group_geoid'].nunique():,}")
print(f"  SLR levels: {sorted(bg_df['slr_ft'].unique().tolist())}")
display(bg_df.head(3))"""

TRACT_AGG_CODE = """\
# Optional: tract-level aggregation for robustness
TRACT_KEYS = ["tract_geoid", "county_fips", "county_name", "slr_ft"]

t_count_sum_agg = {col: "sum" for col in count_sum_cols if col in long_df.columns}
t_count_sum_agg["block_geoid"] = "count"

tract_counts = (
    long_df.groupby(TRACT_KEYS)
    .agg(t_count_sum_agg)
    .reset_index()
    .rename(columns={"block_geoid": "total_blocks"})
)
for col in count_sum_cols:
    if col in tract_counts.columns:
        tract_counts[col] = tract_counts[col].astype(int)

tract_pir = (
    long_df.groupby(["tract_geoid", "slr_ft"])["path_inflation_ratio"]
    .agg(["mean", "median"])
    .rename(columns={"mean": "mean_path_inflation_ratio",
                     "median": "median_path_inflation_ratio"})
    .reset_index()
)
tract_df = tract_counts.merge(tract_pir, on=["tract_geoid", "slr_ft"], how="left")
for count_col, share_col in share_map.items():
    if count_col in tract_df.columns:
        tract_df[share_col] = tract_df[count_col] / tract_df["total_blocks"]

print(f"Tract dataset: {len(tract_df):,} rows ({tract_df['tract_geoid'].nunique():,} tracts)")"""

CENSUS_MD = """\
## 6. Census Demographics via API

Pull ACS 5-year 2022 data for block groups in the three study-area counties.

| Variable | Description |
|---|---|
| B01001_001E | Total population |
| B03002_001E | Race/ethnicity total |
| B03002_003E | Non-Hispanic white alone |
| B03002_004E | Non-Hispanic Black alone |
| B03002_012E | Hispanic or Latino |
| B19013_001E | Median household income |
| B25003_001E | Occupied housing units |
| B25003_003E | Renter-occupied |
| B17001_001E | Poverty denominator |
| B17001_002E | Below poverty level |
| B01002_001E | Median age |

GEOID construction: `state(2) + county(3) + tract(6) + block_group(1)` = 12-char string
matching `block_group_geoid` in our data.

> Census API returns `-666666666` for suppressed/missing values — recoded to NaN below."""

CENSUS_PULL_CODE = """\
ACS_VARS = [
    "B01001_001E", "B03002_001E", "B03002_003E", "B03002_004E", "B03002_012E",
    "B19013_001E", "B25003_001E", "B25003_003E",
    "B17001_001E", "B17001_002E", "B01002_001E",
]

def fetch_acs5(state, county, variables, year, api_key):
    var_str = ",".join(["NAME"] + variables)
    url = (
        f"https://api.census.gov/data/{year}/acs/acs5"
        f"?get={var_str}&for=block+group:*&in=state:{state}+county:{county}"
        f"&key={api_key}"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        df = pd.DataFrame(data[1:], columns=data[0])
        df["block_group_geoid"] = df["state"] + df["county"] + df["tract"] + df["block group"]
        return df
    except Exception as exc:
        print(f"  ERROR fetching county {county}: {exc}")
        return pd.DataFrame()

acs_frames = []
for county_fips, county_name in STUDY_COUNTIES.items():
    print(f"Fetching ACS5 for {county_name} ({county_fips})...")
    df = fetch_acs5(STUDY_STATE, county_fips, ACS_VARS, CENSUS_YEAR, CENSUS_API_KEY)
    if not df.empty:
        df["county_name_acs"] = county_name
        acs_frames.append(df)
        print(f"  {len(df):,} block groups")

acs_raw = pd.concat(acs_frames, ignore_index=True) if acs_frames else pd.DataFrame()
print(f"\\nTotal ACS block groups: {len(acs_raw):,}")"""

CENSUS_MERGE_CODE = """\
if acs_raw.empty:
    print("WARNING: No ACS data. Skipping demographic merge.")
    print("TODO: Census API failed — rerun cell above to attempt again.")
    HAS_DEMOGRAPHICS = False
    bg_analysis = bg_df.copy()
else:
    HAS_DEMOGRAPHICS = True
    demo = acs_raw.copy()

    # Recode suppressed/missing to NaN
    for col in ACS_VARS:
        if col in demo.columns:
            demo[col] = pd.to_numeric(demo[col], errors="coerce")
            demo.loc[demo[col] == -666666666, col] = np.nan

    # Derived variables
    demo["total_pop"]            = demo["B01001_001E"]
    demo["median_income"]        = demo["B19013_001E"]
    demo["median_age"]           = demo["B01002_001E"]
    demo["pct_white_nh"]         = demo["B03002_003E"] / demo["B03002_001E"]
    demo["pct_black_nh"]         = demo["B03002_004E"] / demo["B03002_001E"]
    demo["pct_hispanic"]         = demo["B03002_012E"] / demo["B03002_001E"]
    demo["pct_nonwhite"]         = 1.0 - demo["pct_white_nh"]
    demo["renter_share"]         = demo["B25003_003E"] / demo["B25003_001E"]
    demo["poverty_rate"]         = demo["B17001_002E"] / demo["B17001_001E"]
    demo["log_median_income"]    = np.log(demo["median_income"].clip(lower=1))

    keep = [
        "block_group_geoid", "total_pop", "median_income", "median_age",
        "pct_white_nh", "pct_black_nh", "pct_hispanic", "pct_nonwhite",
        "renter_share", "poverty_rate", "log_median_income",
    ]
    demo_slim = demo[[c for c in keep if c in demo.columns]].copy()
    demo_slim["block_group_geoid"] = demo_slim["block_group_geoid"].astype(str).str.zfill(12)

    bg_analysis = bg_df.merge(demo_slim, on="block_group_geoid", how="left", validate="many_to_one")

    n_matched = int(bg_analysis["median_income"].notna().sum())
    n_total   = len(bg_analysis)
    print(f"Merge: {n_matched:,}/{n_total:,} rows matched demographics ({100*n_matched/n_total:.1f}%)")
    display(bg_analysis[["block_group_geoid", "county_name", "slr_ft",
                           "total_pop", "median_income", "pct_nonwhite",
                           "renter_share", "poverty_rate"]].head(5))"""

VIZ_MD = "## 7. Visualizations\n\nFigures saved to `outputs/figures/`."

FIG1_CODE = """\
# Figure 1: Share of blocks by scenario status across SLR levels
status_by_slr = (
    long_df.loc[long_df["slr_ft"] != 0]
    .groupby(["slr_ft", "scenario_status"])
    .size().reset_index(name="n")
)
total_slr = status_by_slr.groupby("slr_ft")["n"].transform("sum")
status_by_slr["share"] = status_by_slr["n"] / total_slr

pivot = (status_by_slr.pivot_table(index="slr_ft", columns="scenario_status",
                                    values="share", fill_value=0)
                       .reindex(SLR_LEVELS).fillna(0))

fig, ax = plt.subplots(figsize=(9, 5))
x      = np.arange(len(SLR_LEVELS))
bottom = np.zeros(len(SLR_LEVELS))
for status in STATUS_ORDER:
    if status in pivot.columns:
        vals = pivot[status].values
        ax.bar(x, vals, bottom=bottom, label=status,
               color=STATUS_COLORS[status], width=0.65, edgecolor="white", linewidth=0.3)
        bottom += vals

ax.set_xticks(x)
ax.set_xticklabels([f"{ft} ft" for ft in SLR_LEVELS])
ax.set_xlabel("Sea-Level Rise Scenario")
ax.set_ylabel("Share of Blocks")
ax.set_title("Block Status by SLR Scenario\\n(Broward, Miami-Dade, Palm Beach combined)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig1_status_shares_by_slr.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: fig1_status_shares_by_slr.png")"""

FIG2_CODE = """\
# Figure 2: New adverse transitions by SLR level
n_all = int(transition_own["n_blocks"].iloc[0])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, normalize, ylabel in [
    (axes[0], False, "Block count"),
    (axes[1], True,  "Share of all blocks"),
]:
    for col, label, marker, color in [
        ("n_new_fragile",   "New fragile",    "o", "#fdae61"),
        ("n_new_isolated",  "New isolated",   "s", "#d7301f"),
        ("n_new_inundated", "New inundated",  "^", "#1f78b4"),
    ]:
        vals = transition_own[col] / n_all if normalize else transition_own[col]
        ax.plot(transition_own["slr_ft"], vals, marker=marker,
                color=color, label=label, linewidth=1.8)
    ax.set_xlabel("SLR (ft)")
    ax.set_ylabel(ylabel)
    ax.set_title("New adverse transitions by SLR level")
    if normalize:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_xticks(SLR_LEVELS)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("New Adverse Transitions Under SLR", fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig2_new_transitions_by_slr.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: fig2_new_transitions_by_slr.png")"""

FIG3_CODE = """\
# Figure 3: County comparison
county_slr = (
    long_df.loc[long_df["slr_ft"] != 0]
    .groupby(["county_name", "slr_ft"])
    .agg(
        total_blocks      = ("block_geoid",           "size"),
        n_fragile_worse   = ("fragile_or_worse",      "sum"),
        n_lost_redundancy = ("any_loss_of_redundancy","sum"),
    ).reset_index()
)
county_slr["share_fragile_or_worse"] = county_slr["n_fragile_worse"]   / county_slr["total_blocks"]
county_slr["share_lost_redundancy"]  = county_slr["n_lost_redundancy"] / county_slr["total_blocks"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, col, title in [
    (axes[0], "share_fragile_or_worse", "Share fragile or worse"),
    (axes[1], "share_lost_redundancy",  "Share lost redundancy"),
]:
    for county, color in COUNTY_COLORS.items():
        sub = county_slr.loc[county_slr["county_name"] == county]
        ax.plot(sub["slr_ft"], sub[col], marker="o", color=color,
                label=county, linewidth=1.8)
    ax.set_xlabel("SLR (ft)")
    ax.set_ylabel(title)
    ax.set_title(title + " by county")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_xticks(SLR_LEVELS)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("County Comparison: Network Degradation Under SLR", fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig3_county_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: fig3_county_comparison.png")"""

FIG4_CODE = """\
# Figure 4: Distribution of dry-path inflation ratio for connected blocks
pir_plot = long_df.loc[
    (long_df["slr_ft"] != 0) & long_df["path_inflation_ratio"].notna()
].copy()

fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True, sharex=True)
axes = axes.flatten()

for i, slr_ft in enumerate(SLR_LEVELS):
    ax   = axes[i]
    vals = pir_plot.loc[pir_plot["slr_ft"] == slr_ft, "path_inflation_ratio"].clip(upper=5)
    if HAS_SEABORN:
        sns.histplot(vals, ax=ax, bins=40, color="#74c476", alpha=0.8)
    else:
        ax.hist(vals.dropna().values, bins=40, color="#74c476", alpha=0.8)
    med = vals.median()
    ax.axvline(med, color="#d7301f", linewidth=1.5, linestyle="--", label=f"Median={med:.2f}")
    ax.set_title(f"SLR = {slr_ft} ft  (n={len(vals):,})", fontsize=9)
    ax.set_xlabel("Ratio (clipped at 5)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Dry-Path Inflation Ratio Distribution for Connected Blocks", fontsize=11)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig4_path_inflation_ratio_dist.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: fig4_path_inflation_ratio_dist.png")"""

FIG5_CODE = """\
# Figure 5: Why redundancy measures add information beyond binary isolation
compare_df = (
    long_df.loc[long_df["slr_ft"] != 0]
    .groupby("slr_ft")
    .agg(
        n_blocks           = ("block_geoid",             "size"),
        n_isolated         = ("block_centroid_isolated",  "sum"),
        n_fragile_or_worse = ("fragile_or_worse",         "sum"),
        n_lost_redundancy  = ("any_loss_of_redundancy",   "sum"),
    ).reset_index()
)
compare_df["share_isolated"]         = compare_df["n_isolated"]         / compare_df["n_blocks"]
compare_df["share_fragile_or_worse"] = compare_df["n_fragile_or_worse"] / compare_df["n_blocks"]
compare_df["share_lost_redundancy"]  = compare_df["n_lost_redundancy"]  / compare_df["n_blocks"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(compare_df["slr_ft"], compare_df["share_isolated"],
        marker="s", color="#d7301f", linewidth=2, label="Isolated (binary measure)")
ax.plot(compare_df["slr_ft"], compare_df["share_fragile_or_worse"],
        marker="o", color="#fdae61", linewidth=2,
        label="Fragile or worse (fragile + isolated + inundated)")
ax.plot(compare_df["slr_ft"], compare_df["share_lost_redundancy"],
        marker="^", color="#807dba", linewidth=2, linestyle="--",
        label="Lost redundancy (baseline-redundant now fragile/worse)")
ax.fill_between(
    compare_df["slr_ft"],
    compare_df["share_isolated"],
    compare_df["share_fragile_or_worse"],
    alpha=0.12, color="#fdae61",
    label="Gap: fragile blocks missed by binary isolation measure"
)
ax.set_xlabel("Sea-Level Rise (ft)")
ax.set_ylabel("Share of all blocks")
ax.set_title("What Binary Isolation Misses:\\nFragile Blocks and Lost Redundancy Under SLR")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
ax.set_xticks(SLR_LEVELS)
ax.legend(fontsize=9, loc="upper left")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig5_isolation_vs_redundancy.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: fig5_isolation_vs_redundancy.png")"""

FIG6_CODE = """\
# Figure 6: Demographic scatter (SLR = 3 ft)
if not HAS_DEMOGRAPHICS:
    print("Demographic data not available — skipping Figure 6.")
    print("TODO: Re-run Section 6 (Census API) before this cell.")
else:
    bg_slr3 = bg_analysis.loc[bg_analysis["slr_ft"] == 3].dropna(
        subset=["pct_nonwhite", "median_income", "share_fragile_or_worse"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for county, color in COUNTY_COLORS.items():
        sub = bg_slr3.loc[bg_slr3["county_name"] == county]
        ax.scatter(sub["pct_nonwhite"], sub["share_fragile_or_worse"],
                   color=color, alpha=0.4, s=14, label=county)
    ax.set_xlabel("Share non-white (ACS 2022)")
    ax.set_ylabel("Share fragile or worse (SLR 3 ft)")
    ax.set_title("Racial composition vs. network fragility")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for county, color in COUNTY_COLORS.items():
        sub = bg_slr3.loc[bg_slr3["county_name"] == county]
        ax.scatter(sub["log_median_income"], sub["share_fragile_or_worse"],
                   color=color, alpha=0.4, s=14, label=county)
    ax.set_xlabel("Log median household income (ACS 2022)")
    ax.set_ylabel("Share fragile or worse (SLR 3 ft)")
    ax.set_title("Income vs. network fragility")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(alpha=0.25)

    plt.suptitle("Block-Group Demographics and Network Fragility (SLR = 3 ft)", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_demographic_scatter_slr3.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: fig6_demographic_scatter_slr3.png")"""

REG_MD = """\
## 8. Regression-Ready Memo Analysis

### Why Redundancy/Fragility Adds Information Beyond Binary Isolation

Binary isolation treats the first connection as all that matters. But two blocks that are both
"connected" may have very different resilience:

- **Block A:** 2 edge-disjoint dry paths at baseline → 1 at SLR 3 ft (now fragile)
- **Block B:** 1 edge-disjoint dry path at baseline → 1 at SLR 3 ft (unchanged fragile)

Both appear "connected" under binary isolation. Block A has lost half its network redundancy —
a qualitatively different deterioration that standard binary measures cannot detect.

### Interpretive Cautions for Path Inflation Ratio

1. **This is a model output, not a guaranteed traveler experience.** The network is undirected
   and does not model traffic, signal timing, or emergency routing.
2. **It is undefined for isolated/inundated blocks** — those blocks have no dry route, so the
   ratio is meaningless. These are coded as NaN.
3. **The upstream path-count cap is 2** (`MAX_EDGE_DISJOINT_PATHS_CAP = 2`). The model cannot
   distinguish 2-path from 3+ path redundancy.

### Descriptive vs. Causal Claims

The regressions below are **descriptive** — they characterize the correlation between block-group
demographics and SLR-related network outcomes. They do *not* support causal claims. The
underlying variation in exposure is driven by geography and network topology, not demographics
directly. A positive coefficient on `pct_nonwhite` does not mean race *causes* higher exposure;
it means non-white block groups tend to be located in areas with more topographic/network
vulnerability, which warrants further investigation.

### Limitation: Fixed Network Under SLR

The road network is held constant across all scenarios. No infrastructure adaptation is modeled.
The counterfactual is "given today's road network, what if sea levels rise?" — useful but
idealized. Results would change substantially if future road investments or land-use changes
are incorporated."""

REG_CODE = """\
if not HAS_STATSMODELS:
    print("statsmodels not available — skipping regression section.")
elif not HAS_DEMOGRAPHICS:
    print("Demographic data not available — re-run Section 6 first.")
else:
    SLR_CROSS = 3   # illustrative cross-sectional scenario

    bg_reg = bg_analysis.loc[bg_analysis["slr_ft"] == SLR_CROSS].dropna(
        subset=["pct_nonwhite", "log_median_income", "renter_share",
                "poverty_rate", "share_fragile_or_worse",
                "share_lost_redundancy", "share_new_isolated"]
    ).copy()

    print(f"Regression sample: {len(bg_reg):,} block groups at SLR = {SLR_CROSS} ft")

    outcomes = {
        "share_fragile_or_worse": "Share fragile or worse",
        "share_lost_redundancy":  "Share lost redundancy",
        "share_new_isolated":     "Share new isolated due to SLR",
    }
    results_records = []

    for outcome, label in outcomes.items():
        formula = (
            f"{outcome} ~ pct_nonwhite + log_median_income + renter_share "
            f"+ poverty_rate + C(county_name)"
        )
        try:
            m = smf.ols(formula, data=bg_reg).fit(cov_type="HC3")
            print(f"\\n{'='*60}")
            print(f"Outcome: {label}  |  N={int(m.nobs):,}  |  R²={m.rsquared:.4f}")
            print(m.summary2().tables[1].to_string())
            results_records.append({
                "outcome":           label,
                "slr_ft":            SLR_CROSS,
                "N":                 int(m.nobs),
                "R2":                round(m.rsquared, 4),
                "coef_pct_nonwhite": round(m.params.get("pct_nonwhite", np.nan), 5),
                "pval_pct_nonwhite": round(m.pvalues.get("pct_nonwhite", np.nan), 4),
                "coef_log_income":   round(m.params.get("log_median_income", np.nan), 5),
                "pval_log_income":   round(m.pvalues.get("log_median_income", np.nan), 4),
                "coef_renter":       round(m.params.get("renter_share", np.nan), 5),
                "pval_renter":       round(m.pvalues.get("renter_share", np.nan), 4),
                "coef_poverty":      round(m.params.get("poverty_rate", np.nan), 5),
                "pval_poverty":      round(m.pvalues.get("poverty_rate", np.nan), 4),
            })
        except Exception as exc:
            print(f"ERROR fitting {outcome}: {exc}")

    results_df = pd.DataFrame(results_records)
    print("\\nRegression summary table:")
    display(results_df)"""

COEF_PLOT_CODE = """\
# Coefficient plot: how do coefficients on pct_nonwhite change across SLR scenarios?
if not HAS_STATSMODELS or not HAS_DEMOGRAPHICS:
    print("Skipping coefficient plot.")
else:
    OUTCOME_PANEL = "share_fragile_or_worse"
    coef_records  = []

    for slr in SLR_LEVELS:
        bg_slr = bg_analysis.loc[bg_analysis["slr_ft"] == slr].dropna(
            subset=["pct_nonwhite", "log_median_income", "renter_share",
                    "poverty_rate", OUTCOME_PANEL]
        )
        if len(bg_slr) < 30:
            continue
        try:
            formula = (
                f"{OUTCOME_PANEL} ~ pct_nonwhite + log_median_income "
                f"+ renter_share + poverty_rate + C(county_name)"
            )
            m  = smf.ols(formula, data=bg_slr).fit(cov_type="HC3")
            ci = m.conf_int()
            for var in ["pct_nonwhite", "log_median_income", "renter_share", "poverty_rate"]:
                if var in m.params:
                    coef_records.append({
                        "slr_ft":   slr,
                        "variable": var,
                        "coef":     m.params[var],
                        "ci_low":   ci.loc[var, 0],
                        "ci_high":  ci.loc[var, 1],
                    })
        except Exception as exc:
            print(f"  slr={slr}: {exc}")

    if coef_records:
        coef_df = pd.DataFrame(coef_records)
        var_labels = {
            "pct_nonwhite":      "% Non-white",
            "log_median_income": "Log median income",
            "renter_share":      "Renter share",
            "poverty_rate":      "Poverty rate",
        }
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes = axes.flatten()
        for i, var in enumerate(["pct_nonwhite", "log_median_income",
                                  "renter_share", "poverty_rate"]):
            ax  = axes[i]
            sub = coef_df.loc[coef_df["variable"] == var]
            ax.plot(sub["slr_ft"], sub["coef"], marker="o", color="#2166ac", linewidth=2)
            ax.fill_between(sub["slr_ft"], sub["ci_low"], sub["ci_high"],
                            alpha=0.2, color="#2166ac", label="95% CI (HC3)")
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(var_labels.get(var, var))
            ax.set_xlabel("SLR (ft)")
            ax.set_ylabel("Coefficient")
            ax.set_xticks(SLR_LEVELS)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        plt.suptitle(
            f"OLS Coefficients on {OUTCOME_PANEL}\\nCounty FE, HC3 SEs",
            fontsize=11
        )
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "fig7_coef_plot.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved: fig7_coef_plot.png")"""

MEMO_MD = """\
## 9. Memo Structure

---

### Motivation

This project extends Best et al. (2023) to capture the *spectrum* of network resilience loss
under SLR, rather than the binary connected/isolated divide. By tracking edge-disjoint path
counts, we can identify blocks that become network-fragile (one dry route) before they become
fully isolated (no dry route).

---

### Data

- **Block-level access flags:** 70,695 blocks in Broward, Miami-Dade, Palm Beach across 6 SLR
  scenarios (1–6 ft) plus baseline (0 ft). Source: `02_access_flags.py`.
- **Demographic data:** ACS 5-year 2022, block-group level, via Census API.
- **Analysis units:** Block × SLR scenario (block-level); block group × scenario (aggregated).

---

### Variable Construction

| Variable | Source | Notes |
|---|---|---|
| `fragile_or_worse` | Derived | fragile ∪ isolated ∪ inundated |
| `any_loss_of_redundancy` | Derived | Baseline-redundant → any worse outcome |
| `path_inflation_ratio` | Renamed from `detour_ratio` | NaN for isolated/inundated |
| Transition indicators | Derived | e.g., `baseline_redundant_to_fragile` |

---

### Key Descriptive Findings

*(Fill in after running the notebook and reviewing figures)*

1. At SLR = X ft, approximately Y% of blocks are fragile-or-worse vs. Z% isolated.
2. The gap between fragile-or-worse and isolated-only shares grows at higher SLR levels,
   suggesting fragility precedes isolation for many blocks.
3. County-level patterns: [fill in from Figure 3].
4. The dry-path inflation ratio grows substantially above SLR = X ft for connected blocks.

---

### Preliminary Regression Results

*(Fill in after running Section 8)*

---

### Interpretation

Results are **descriptive** — they characterize geographic correlations between network
fragility and demographic composition, given the current road network. They do not establish
causation. See Section 8 for full discussion.

---

### Limitations

1. **Fixed network:** Road network held constant. No adaptation modeled.
2. **Path-count cap at 2:** Cannot distinguish 2-path from 3+ path redundancy.
3. **Block centroid measure:** Sub-block variation and population weighting not addressed.
4. **Undirected network:** One-way streets and turn restrictions not modeled.
5. **OSM segment intersection:** No bridge/tunnel handling; segment-level inundation only.

---

### Next Steps

- [ ] Weight blocks by residential population rather than treating blocks equally
- [ ] Design a DiD / event-study comparing high- vs. low-redundancy areas
- [ ] Sensitivity: relax path-count cap in upstream `02_access_flags.py`
- [ ] Spatial mapping of fragile-block hotspots vs. flood infrastructure investments
- [ ] Consider directed network for the path analysis"""

SAVE_CODE = """\
# ---------------------------------------------------------------------------
# Save all outputs to disk
# ---------------------------------------------------------------------------

# 1. Block-level long dataset
block_out = ANALYSIS_DIR / "block_level_long_dataset.parquet"
long_df.to_parquet(block_out, index=False)
print(f"Saved: {block_out}")

# Also a CSV sample for quick inspection
sample_out = ANALYSIS_DIR / "block_level_long_dataset_sample5k.csv"
long_df.sample(min(5000, len(long_df)), random_state=42).to_csv(sample_out, index=False)
print(f"Saved sample: {sample_out}")

# 2. Block-group analysis dataset
bg_out     = ANALYSIS_DIR / "block_group_analysis_dataset.parquet"
bg_csv_out = ANALYSIS_DIR / "block_group_analysis_dataset.csv"
bg_analysis.to_parquet(bg_out, index=False)
bg_analysis.to_csv(bg_csv_out, index=False)
print(f"Saved: {bg_out}")
print(f"Saved: {bg_csv_out}")

# 3. Tract-level dataset
if "tract_df" in dir():
    tract_out = ANALYSIS_DIR / "tract_analysis_dataset.parquet"
    tract_df.to_parquet(tract_out, index=False)
    print(f"Saved: {tract_out}")

# 4. Transition summary tables
transition_own.to_csv(TABLES_DIR / "transition_summary_by_slr.csv", index=False)
crosstab_own.to_csv(TABLES_DIR / "baseline_vs_scenario_crosstab_validated.csv", index=False)
print(f"Saved transition tables to {TABLES_DIR}")

# 5. Regression results (if available)
try:
    results_df.to_csv(TABLES_DIR / "regression_summary.csv", index=False)
    print(f"Saved regression summary to {TABLES_DIR}")
except NameError:
    print("No regression results to save (Section 8 not run or demographics unavailable).")

print("\\nAll outputs saved.")
print(f"  Analysis data : {ANALYSIS_DIR}")
print(f"  Figures       : {FIGURES_DIR}")
print(f"  Tables        : {TABLES_DIR}")"""

# ============================================================
# Assemble cells
# ============================================================
cells = [
    md(TITLE),
    md(MOTIVATION),
    md(SETUP_SECTION),
    code(SETUP_CODE),
    code(PATHS_CODE),
    md(INSPECT_MD),
    code(FILE_INVENTORY_CODE),
    code(BASELINE_CHECK_CODE),
    md(BUILD_MD),
    code(EXTRACT_BASELINE_CODE),
    code(LOAD_SCENARIOS_CODE),
    code(STACK_CODE),
    code(DERIVED_VARS_CODE),
    md(VALIDATION_MD),
    code(CROSSTAB_CODE),
    code(TRANSITION_VALID_CODE),
    code(COUNTY_STATUS_CODE),
    md(BG_MD),
    code(BG_AGG_CODE),
    code(TRACT_AGG_CODE),
    md(CENSUS_MD),
    code(CENSUS_PULL_CODE),
    code(CENSUS_MERGE_CODE),
    md(VIZ_MD),
    code(FIG1_CODE),
    code(FIG2_CODE),
    code(FIG3_CODE),
    code(FIG4_CODE),
    code(FIG5_CODE),
    code(FIG6_CODE),
    md(REG_MD),
    code(REG_CODE),
    code(COEF_PLOT_CODE),
    md(MEMO_MD),
    md("## 10. Save All Outputs\n\nWrite analysis-ready files to disk."),
    code(SAVE_CODE),
]

NOTEBOOK = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(NOTEBOOK, f, indent=2, ensure_ascii=False)

print(f"Saved: {OUTPUT_PATH}")
print(f"Total cells: {len(cells)}")
