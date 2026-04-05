#!/usr/bin/env python
"""
Shared helpers for the hybrid OSRM-isolation / graph-redundancy workflow.

The goal is to keep the new scripts small and consistent with the existing
`01_bg_access_flags_adroit.py` exploratory workflow while avoiding a large
refactor of the current code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "01_bg_access_flags_adroit.py"

HYBRID_DIR = PROJECT_ROOT / "data" / "processed" / "access" / "hybrid"
PREP_DIR = HYBRID_DIR / "prep"
OSRM_DIR = HYBRID_DIR / "osrm"
GRAPH_DIR = HYBRID_DIR / "graph"
MERGED_DIR = HYBRID_DIR / "merged"
OSRM_UPDATE_DIR = OSRM_DIR / "updates"
OSRM_RUNTIME_DIR = OSRM_DIR / "runtime"

PREP_ORIGINS_GPKG = PREP_DIR / "hybrid_origins_rep_points.gpkg"
PREP_ORIGIN_ACCESS_CSV = PREP_DIR / "hybrid_origin_access_points"
PREP_SERVICES_GPKG = PREP_DIR / "hybrid_services_essential.gpkg"
PREP_SERVICE_COUNTY_CSV = PREP_DIR / "hybrid_service_county_membership"
PREP_COUNTY_BOUNDS_GPKG = PREP_DIR / "hybrid_county_buffered_bounds.gpkg"
PREP_OSRM_EDGES_GPKG = PREP_DIR / "hybrid_osrm_driving_edges.gpkg"

DEFAULT_OSRM_OUTPUT_STEM = "block_osrm_isolation_long"
DEFAULT_GRAPH_OUTPUT_STEM = "block_graph_redundancy_long"
DEFAULT_MERGED_OUTPUT_STEM = "block_access_hybrid_long"

COUNTY_BUFFER_M = 5_000
OSRM_REQUIRED_SERVICE_TYPES = ("school", "fire_station")
OSRM_CANDIDATE_RADIUS_M = 3_000
OSRM_CANDIDATE_K = 5
OSRM_MAX_COORDS_PER_REQUEST = 150
OSRM_MAX_ORIGINS_PER_BATCH = 40


def log(message: str) -> None:
    print(f"[hybrid_access] {message}", flush=True)


def warn(message: str) -> None:
    print(f"[hybrid_access][WARN] {message}", flush=True)


def load_base_module():
    spec = importlib.util.spec_from_file_location("bg_access_flags_adroit_base", BASE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base script from {BASE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()
PROJECTED_CRS = BASE.PROJECTED_CRS


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_output_suffix(max_blocks: int | None, output_suffix: str) -> str:
    if output_suffix:
        return output_suffix
    if max_blocks is not None:
        return "__subset"
    return ""


def write_table(df: pd.DataFrame, output_stem: Path) -> Path:
    ensure_dir(output_stem.parent)
    csv_path = output_stem.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    try:
        parquet_path = output_stem.with_suffix(".parquet")
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        return csv_path


def read_table(output_stem: Path, *, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    parquet_path = output_stem.with_suffix(".parquet")
    csv_path = output_stem.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, dtype=dtype)
    raise FileNotFoundError(f"Could not find table at {parquet_path} or {csv_path}")


def load_prepared_origins() -> gpd.GeoDataFrame:
    if not PREP_ORIGINS_GPKG.exists():
        raise FileNotFoundError(f"Prepared origins file not found: {PREP_ORIGINS_GPKG}")
    return gpd.read_file(PREP_ORIGINS_GPKG)


def load_prepared_services() -> gpd.GeoDataFrame:
    if not PREP_SERVICES_GPKG.exists():
        raise FileNotFoundError(f"Prepared services file not found: {PREP_SERVICES_GPKG}")
    return gpd.read_file(PREP_SERVICES_GPKG)


def load_prepared_county_bounds() -> gpd.GeoDataFrame:
    if not PREP_COUNTY_BOUNDS_GPKG.exists():
        raise FileNotFoundError(f"Prepared county bounds file not found: {PREP_COUNTY_BOUNDS_GPKG}")
    return gpd.read_file(PREP_COUNTY_BOUNDS_GPKG)


def load_prepared_osrm_edges() -> gpd.GeoDataFrame:
    if not PREP_OSRM_EDGES_GPKG.exists():
        raise FileNotFoundError(f"Prepared OSRM edge catalog not found: {PREP_OSRM_EDGES_GPKG}")
    return gpd.read_file(PREP_OSRM_EDGES_GPKG)


def load_prepared_service_county_membership() -> pd.DataFrame:
    return read_table(PREP_SERVICE_COUNTY_CSV, dtype={"service_id": "string", "county_fips": "string", "county_name": "string"})


def load_prepared_origin_access_points() -> pd.DataFrame:
    return read_table(
        PREP_ORIGIN_ACCESS_CSV,
        dtype={"block_geoid": "string", "county_fips": "string", "county_name": "string"},
    )


def build_county_buffered_bounds(blocks: gpd.GeoDataFrame, *, buffer_m: float = COUNTY_BUFFER_M) -> gpd.GeoDataFrame:
    bounds_records: list[dict[str, object]] = []
    for row in blocks.dissolve(by=["county_fips", "county_name"]).reset_index().itertuples(index=False):
        xmin, ymin, xmax, ymax = row.geometry.bounds
        bounds_records.append(
            {
                "county_fips": row.county_fips,
                "county_name": row.county_name,
                "geometry": box(xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m),
            }
        )
    return gpd.GeoDataFrame(bounds_records, geometry="geometry", crs=blocks.crs)


def prepare_representative_point_origins(blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    origins = blocks[
        ["block_geoid", "block_group_geoid", "tract_geoid", "block", "county_fips", "county_name", "geometry"]
    ].copy()
    origins["geometry"] = origins.geometry.representative_point()
    origins_lonlat = origins.to_crs("OGC:CRS84")
    origins["origin_lon"] = origins_lonlat.geometry.x.to_numpy()
    origins["origin_lat"] = origins_lonlat.geometry.y.to_numpy()
    origins["origin_x_m"] = origins.geometry.x.to_numpy()
    origins["origin_y_m"] = origins.geometry.y.to_numpy()
    return origins


def filter_services_to_county_buffers(
    services: gpd.GeoDataFrame,
    county_bounds: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    membership = gpd.sjoin(
        services,
        county_bounds[["county_fips", "county_name", "geometry"]],
        how="inner",
        predicate="within",
    ).drop(columns=["index_right"])
    membership = pd.DataFrame(membership[["service_id", "service_type", "county_fips", "county_name"]]).drop_duplicates()

    retained_ids = membership["service_id"].drop_duplicates().astype(str)
    filtered = services.loc[services["service_id"].astype(str).isin(retained_ids)].copy()
    return filtered, membership


def add_lon_lat_columns(gdf: gpd.GeoDataFrame, *, lon_col: str, lat_col: str) -> gpd.GeoDataFrame:
    output = gdf.copy()
    lonlat = output.to_crs("OGC:CRS84")
    output[lon_col] = lonlat.geometry.x.to_numpy()
    output[lat_col] = lonlat.geometry.y.to_numpy()
    return output


def requested_positive_slr_levels(values: list[int] | None) -> list[int]:
    if values is None:
        return sorted(BASE.SLR_LAYER_MAP)
    invalid = sorted(set(values) - set(BASE.SLR_LAYER_MAP))
    if invalid:
        raise ValueError(f"Unsupported SLR level(s): {invalid}. Supported values are {sorted(BASE.SLR_LAYER_MAP)}")
    return sorted(dict.fromkeys(values))


def scenario_records(values: list[int] | None) -> list[tuple[int, str | None]]:
    positive = requested_positive_slr_levels(values)
    records = [(0, None)]
    records.extend((slr_ft, BASE.SLR_LAYER_MAP[slr_ft]) for slr_ft in positive)
    return records


def load_projected_slr_layer(slr_ft: int, clip_polygon=None) -> gpd.GeoDataFrame | None:
    if slr_ft == 0:
        return None
    clipper = clip_polygon
    if clipper is None:
        roads = BASE.load_roads()
        clipper = box(*roads.total_bounds)
    slr = BASE.load_slr_layer(BASE.SLR_LAYER_MAP[slr_ft], clipper)
    if slr is None or slr.empty:
        return None
    return BASE.maybe_to_projected(slr)


def geometries_intersecting_slr(gdf: gpd.GeoDataFrame, slr_layer: gpd.GeoDataFrame | None) -> pd.Series:
    if slr_layer is None or slr_layer.empty:
        return pd.Series(False, index=gdf.index)
    joined = gpd.sjoin(gdf[["geometry"]], slr_layer[["geometry"]], how="left", predicate="intersects")
    return joined["index_right"].notna().astype(bool).reset_index(drop=True)


def require_inputs(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n" + "\n".join(str(path) for path in missing))
