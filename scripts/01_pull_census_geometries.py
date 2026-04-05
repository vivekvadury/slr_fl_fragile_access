#!/usr/bin/env python
"""
Pull 2020 Florida Census blocks and block groups, then save tri-county
processed geometry files for Miami-Dade, Broward, and Palm Beach.

This script keeps the original statewide TIGER/Line downloads in
`data/raw/census/` and writes filtered tri-county geometry outputs to
`data/processed/census/`.

Important note
- This script intentionally pulls 2020 TIGER/Line geometries because the
  current extension workflow is anchored to 2020 geography.
- If you later want exact geography-vintage alignment with the 2020-2024 ACS
  5-year release, you may also want a separate 2024 block-group pull, because
  ACS tabulations are released on contemporary geography vintages.
- Census API credentials are not required for these geometry downloads because
  TIGER/Line files are public.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

YEAR = 2020
STATE_FIPS = "12"
STATE_ABBR = "fl"
TRI_COUNTY_FIPS = {
    "086": "Miami-Dade",
    "011": "Broward",
    "099": "Palm Beach",
}

RAW_CENSUS_DIR = PROJECT_ROOT / "data" / "raw" / "census"
RAW_BG_DIR = RAW_CENSUS_DIR / "block_groups" / str(YEAR)
RAW_BLOCK_DIR = RAW_CENSUS_DIR / "blocks" / str(YEAR)

PROCESSED_CENSUS_DIR = PROJECT_ROOT / "data" / "processed" / "census"
PROCESSED_BG_DIR = PROCESSED_CENSUS_DIR / "block_groups"
PROCESSED_BLOCK_DIR = PROCESSED_CENSUS_DIR / "blocks"

BG_ZIP_NAME = f"tl_{YEAR}_{STATE_FIPS}_bg.zip"
BLOCK_ZIP_NAME = f"tl_{YEAR}_{STATE_FIPS}_tabblock20.zip"

BG_URL = f"https://www2.census.gov/geo/tiger/TIGER{YEAR}/BG/{BG_ZIP_NAME}"
BLOCK_URL = f"https://www2.census.gov/geo/tiger/TIGER{YEAR}/TABBLOCK20/{BLOCK_ZIP_NAME}"


def log(message: str) -> None:
    print(f"[pull_census_geometries] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull 2020 Florida TIGER blocks and block groups, then save tri-county processed outputs."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload and rewrite raw/processed files even if they already exist.",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [RAW_BG_DIR, RAW_BLOCK_DIR, PROCESSED_BG_DIR, PROCESSED_BLOCK_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        log(f"Using existing download: {destination}")
        return

    log(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as out_file:
        shutil.copyfileobj(response, out_file)


def extract_zip(zip_path: Path, destination_dir: Path, overwrite: bool) -> None:
    shp_candidates = list(destination_dir.glob("*.shp"))
    if shp_candidates and not overwrite:
        log(f"Using existing extracted files in {destination_dir}")
        return

    if overwrite:
        for child in destination_dir.iterdir():
            if child.is_file():
                child.unlink()

    log(f"Extracting {zip_path.name} to {destination_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(destination_dir)


def get_single_shapefile(directory: Path) -> Path:
    shapefiles = sorted(directory.glob("*.shp"))
    if len(shapefiles) != 1:
        raise FileNotFoundError(f"Expected exactly one shapefile in {directory}, found {len(shapefiles)}")
    return shapefiles[0]


def read_vector(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"No features found in {path}")
    return gdf


def prepare_block_groups(block_groups: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    output = block_groups.copy()
    output["STATEFP"] = output["STATEFP"].astype(str).str.zfill(2)
    output["COUNTYFP"] = output["COUNTYFP"].astype(str).str.zfill(3)
    output["TRACTCE"] = output["TRACTCE"].astype(str).str.zfill(6)
    output["BLKGRPCE"] = output["BLKGRPCE"].astype(str)
    output["GEOID"] = output["GEOID"].astype(str)

    output = output.loc[output["COUNTYFP"].isin(TRI_COUNTY_FIPS)].copy()
    output["county_name"] = output["COUNTYFP"].map(TRI_COUNTY_FIPS)
    output["tract_geoid"] = output["STATEFP"] + output["COUNTYFP"] + output["TRACTCE"]

    output = output.rename(
        columns={
            "GEOID": "geoid",
            "STATEFP": "state_fips",
            "COUNTYFP": "county_fips",
            "BLKGRPCE": "block_group",
            "NAMELSAD": "name",
        }
    )

    keep_cols = [
        "geoid",
        "state_fips",
        "county_fips",
        "county_name",
        "tract_geoid",
        "block_group",
        "name",
        "geometry",
    ]
    return output[keep_cols].sort_values("geoid").reset_index(drop=True)


def prepare_blocks(blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    output = blocks.copy()
    output["STATEFP20"] = output["STATEFP20"].astype(str).str.zfill(2)
    output["COUNTYFP20"] = output["COUNTYFP20"].astype(str).str.zfill(3)
    output["TRACTCE20"] = output["TRACTCE20"].astype(str).str.zfill(6)
    output["BLOCKCE20"] = output["BLOCKCE20"].astype(str).str.zfill(4)
    output["GEOID20"] = output["GEOID20"].astype(str)

    output = output.loc[output["COUNTYFP20"].isin(TRI_COUNTY_FIPS)].copy()
    output["county_name"] = output["COUNTYFP20"].map(TRI_COUNTY_FIPS)
    output["tract_geoid"] = output["STATEFP20"] + output["COUNTYFP20"] + output["TRACTCE20"]
    output["block_group_geoid"] = output["GEOID20"].str.slice(0, 12)

    output = output.rename(
        columns={
            "GEOID20": "geoid",
            "STATEFP20": "state_fips",
            "COUNTYFP20": "county_fips",
            "BLOCKCE20": "block",
        }
    )

    keep_cols = [
        "geoid",
        "state_fips",
        "county_fips",
        "county_name",
        "tract_geoid",
        "block_group_geoid",
        "block",
        "geometry",
    ]
    return output[keep_cols].sort_values("geoid").reset_index(drop=True)


def save_processed_outputs(
    block_groups: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    bg_gpkg_path = PROCESSED_BG_DIR / f"{STATE_ABBR}_tricounty_block_groups_{YEAR}.gpkg"
    bg_geojson_path = PROCESSED_BG_DIR / f"{STATE_ABBR}_tricounty_block_groups_{YEAR}.geojson"
    block_gpkg_path = PROCESSED_BLOCK_DIR / f"{STATE_ABBR}_tricounty_blocks_{YEAR}.gpkg"

    output_paths = [bg_gpkg_path, bg_geojson_path, block_gpkg_path]
    if not overwrite and all(path.exists() for path in output_paths):
        log("Using existing processed geometry outputs")
        return bg_gpkg_path, bg_geojson_path, block_gpkg_path

    for path in output_paths:
        if path.exists():
            path.unlink()

    block_groups.to_file(bg_gpkg_path, driver="GPKG")
    block_groups.to_file(bg_geojson_path, driver="GeoJSON")
    blocks.to_file(block_gpkg_path, driver="GPKG")

    return bg_gpkg_path, bg_geojson_path, block_gpkg_path


def print_summary(block_groups: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame) -> None:
    bg_counts = (
        block_groups.groupby("county_name")
        .agg(n_block_groups=("geoid", "size"))
        .reset_index()
        .sort_values("county_name")
    )
    block_counts = (
        blocks.groupby("county_name")
        .agg(n_blocks=("geoid", "size"))
        .reset_index()
        .sort_values("county_name")
    )

    summary = pd.merge(bg_counts, block_counts, on="county_name", how="outer").fillna(0)
    log("Tri-county geometry counts:")
    print(summary.to_string(index=False), flush=True)


def main() -> int:
    args = parse_args()
    ensure_dirs()

    bg_zip_path = RAW_BG_DIR / BG_ZIP_NAME
    block_zip_path = RAW_BLOCK_DIR / BLOCK_ZIP_NAME

    download_file(BG_URL, bg_zip_path, overwrite=args.overwrite)
    download_file(BLOCK_URL, block_zip_path, overwrite=args.overwrite)

    extract_zip(bg_zip_path, RAW_BG_DIR, overwrite=args.overwrite)
    extract_zip(block_zip_path, RAW_BLOCK_DIR, overwrite=args.overwrite)

    bg_shp_path = get_single_shapefile(RAW_BG_DIR)
    block_shp_path = get_single_shapefile(RAW_BLOCK_DIR)

    log(f"Reading raw block groups from {bg_shp_path}")
    raw_block_groups = read_vector(bg_shp_path)
    log(f"Reading raw blocks from {block_shp_path}")
    raw_blocks = read_vector(block_shp_path)

    tri_block_groups = prepare_block_groups(raw_block_groups)
    tri_blocks = prepare_blocks(raw_blocks)

    bg_gpkg_path, bg_geojson_path, block_gpkg_path = save_processed_outputs(
        tri_block_groups,
        tri_blocks,
        overwrite=args.overwrite,
    )

    print_summary(tri_block_groups, tri_blocks)
    log(f"Saved processed block groups to {bg_gpkg_path}")
    log(f"Saved processed block groups to {bg_geojson_path}")
    log(f"Saved processed blocks to {block_gpkg_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted.")
        raise SystemExit(1)
