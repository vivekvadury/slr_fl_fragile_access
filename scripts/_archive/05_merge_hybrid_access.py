#!/usr/bin/env python
"""
Merge the OSRM-isolation and graph-redundancy branches into one long table.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

import hybrid_access_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge hybrid OSRM and graph access outputs.")
    parser.add_argument("--output-suffix", default="", help="Optional suffix appended to output filenames.")
    return parser.parse_args()


def add_osrm_baseline_fields(df: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        df.loc[
            df["slr_ft"] == 0,
            [
                "block_geoid",
                "has_school_access",
                "has_fire_station_access",
                "has_required_service_access",
                "has_any_essential_access",
                "isolated_missing_required_service_access",
                "isolated_missing_any_essential_access",
            ],
        ]
        .drop_duplicates("block_geoid")
        .rename(
            columns={
                "has_school_access": "baseline_has_school_access",
                "has_fire_station_access": "baseline_has_fire_station_access",
                "has_required_service_access": "baseline_has_required_service_access",
                "has_any_essential_access": "baseline_has_any_essential_access",
                "isolated_missing_required_service_access": "baseline_isolated_missing_required_service_access",
                "isolated_missing_any_essential_access": "baseline_isolated_missing_any_essential_access",
            }
        )
    )
    output = df.merge(baseline, on="block_geoid", how="left")
    output["new_required_service_isolation_due_to_slr"] = (
        (output["slr_ft"] != 0)
        & output["baseline_has_required_service_access"].astype(bool)
        & output["isolated_missing_required_service_access"].astype(bool)
    ).astype(int)
    output["new_any_essential_isolation_due_to_slr"] = (
        (output["slr_ft"] != 0)
        & output["baseline_has_any_essential_access"].astype(bool)
        & output["isolated_missing_any_essential_access"].astype(bool)
    ).astype(int)
    return output


def main() -> int:
    args = parse_args()
    start = time.time()
    output_suffix = args.output_suffix or ""

    osrm_stem = common.OSRM_DIR / f"{common.DEFAULT_OSRM_OUTPUT_STEM}{output_suffix}"
    graph_stem = common.GRAPH_DIR / f"{common.DEFAULT_GRAPH_OUTPUT_STEM}{output_suffix}"
    merged_stem = common.MERGED_DIR / f"{common.DEFAULT_MERGED_OUTPUT_STEM}{output_suffix}"

    osrm = common.read_table(
        osrm_stem,
        dtype={
            "block_geoid": "string",
            "block_group_geoid": "string",
            "tract_geoid": "string",
            "block": "string",
            "county_fips": "string",
        },
    )
    graph = common.read_table(
        graph_stem,
        dtype={
            "block_geoid": "string",
            "block_group_geoid": "string",
            "tract_geoid": "string",
            "block": "string",
            "county_fips": "string",
        },
    )

    common.ensure_dir(common.MERGED_DIR)
    merged = osrm.merge(
        graph,
        on=[
            "block_geoid",
            "block_group_geoid",
            "tract_geoid",
            "block",
            "county_fips",
            "county_name",
            "slr_ft",
            "slr_layer_name",
        ],
        how="outer",
        validate="one_to_one",
    )
    merged = add_osrm_baseline_fields(merged)
    common.write_table(merged, merged_stem)

    summary = (
        merged.groupby("slr_ft", as_index=False)
        .agg(
            n_blocks=("block_geoid", "size"),
            n_required_isolated=("isolated_missing_required_service_access", "sum"),
            n_any_isolated=("isolated_missing_any_essential_access", "sum"),
            n_graph_fragile=("graph_fragile", "sum"),
            n_graph_redundant=("graph_redundant", "sum"),
            n_new_required_isolation=("new_required_service_isolation_due_to_slr", "sum"),
            n_new_graph_fragile=("new_graph_fragile_due_to_slr", "sum"),
            n_new_graph_unreachable=("new_graph_unreachable_due_to_slr", "sum"),
        )
    )
    print("\nHybrid summary by SLR level")
    print(summary.to_string(index=False))

    common.log(f"Saved merged hybrid output to {merged_stem}")
    common.log(f"Finished in {time.time() - start:.1f} seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
