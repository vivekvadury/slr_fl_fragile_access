#!/usr/bin/env python
"""Build time-invariant physical covariates for transition models.

This script creates one row per block group in
``data/processed/analysis/block_group_physical_covariates.csv`` with the
following exact columns:

* ``block_group_geoid``
* ``elevation_m_mean``
* ``elevation_m_median``
* ``drainage_distance_km``

Elevation source and resolution
--------------------------------
Elevation comes from the USGS 3D Elevation Program (3DEP) seamless
1/3-arc-second bare-earth DEM (approximately 10 m), in NAD83 horizontally and
meters above NAVD88 vertically.  Two immutable, dated GeoTIFF products cover
the analysis area:

* ``USGS_13_n26w081_20260225.tif``
* ``USGS_13_n27w081_20251216.tif``

The dated historical URLs, byte lengths, and USGS-provided MD5 checksums are
pinned below.  This avoids silently changing the covariates when USGS updates
its ``current`` tiles.  The 1/3-arc-second product is fine enough for block-
group zonal summaries while remaining practical to download and rerun; the
1-m product would require many very large project tiles.  It is also in the
same USGS/TNM lidar-derived elevation family as several source DEMs used by
NOAA OCM to construct the Southeast Florida 3-m conditioned DEM behind the
Sea Level Rise Viewer.  It is not, however, NOAA's specially conditioned DEM,
so this script does not claim the two elevation surfaces are identical.

For each block-group polygon in layer ``slr_0ft`` of
``outputs/spatial/slr_block_group_analysis_approach.gpkg``, elevation mean and
median are computed over valid DEM cell centers.  The six-pixel overlap in
adjacent 3DEP distribution tiles is removed at the nominal 26-degree seam so
no elevation cell is counted twice.

Drainage decision
-----------------
The implemented drainage measure is the distance in kilometers from the
block group's geometric centroid to the nearest SFWMD Arc Hydro Enhanced
Database (AHED) feature classified ``HYDRO_ORDER = 'PRIMARY'``.  Distances are
calculated after transforming both layers to EPSG:32617 (meters).  The live
SFWMD FeatureServer response is cached with retrieval metadata and a SHA-256
digest; use ``--refresh-canals`` to request a newer snapshot.

A categorical drainage fixed effect was considered but not selected.  In the
study extent, the SFWMD AHED Basin (HUC6) layer supplies only one basin and
therefore no useful identifying variation.  An audit of the complete USGS WBD
HUC12 layer assigned the 3,942 analysis block-group centroids to 43 HUC12s;
three were singletons and 11 had fewer than 10 block groups.  Adding that many
fixed-effect levels to rare transition-outcome models creates avoidable
separation/incidental-parameter risk and requires an arbitrary assignment for
block groups crossing watershed boundaries.  The continuous primary-drainage
distance is complete, parsimonious, and preserves variation within counties.

Important modeling warning
--------------------------
Elevation is mechanically close to the classifier's own inundation rule.
Downstream models using this alternate physical-covariate specification must
be checked explicitly for separation and convergence problems, particularly
every model whose outcome is ``Inundated``.  These covariates are intended for
a clearly labeled sensitivity specification, not as a silent replacement for
the demographic-only specification.

Run from the repository root with::

    python scripts/03b_join_elevation_drainage.py

The first run downloads about 0.84 GB of pinned DEM inputs.  Subsequent runs
validate and reuse the cache.  Use ``--download-only`` to populate/validate the
cache without computing covariates, or ``--no-download`` for an offline run
that fails clearly if any cache input is absent.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLOCK_GROUP_PATH = (
    PROJECT_ROOT / "outputs" / "spatial" / "slr_block_group_analysis_approach.gpkg"
)
DEFAULT_BLOCK_GROUP_LAYER = "slr_0ft"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "physical_covariates"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "analysis"
    / "block_group_physical_covariates.csv"
)

OUTPUT_COLUMNS = [
    "block_group_geoid",
    "elevation_m_mean",
    "elevation_m_median",
    "drainage_distance_km",
]
GEOID_PATTERN = r"^[0-9]{12}$"
DISTANCE_CRS = "EPSG:32617"
DEM_APPROX_RESOLUTION_DEGREES = 1.0 / 10_800.0
DEFAULT_WINDOW_SIZE = 2_048
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
NETWORK_TIMEOUT_SECONDS = 180
NETWORK_ATTEMPTS = 4
# A browser-style User-Agent is required: the SFWMD ArcGIS host sits behind a
# Web Application Firewall that returns HTTP 403 to non-browser agents. The
# USGS S3 tile bucket is indifferent to the header, so one value is used for
# every request. Override with the SLR_FL_HTTP_USER_AGENT environment variable
# if a future WAF rule requires a different string.
USER_AGENT = os.environ.get(
    "SLR_FL_HTTP_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
)


@dataclass(frozen=True)
class DemTile:
    """Pinned metadata for one immutable 3DEP distribution tile."""

    tile_id: str
    publication_date: str
    filename: str
    url: str
    metadata_url: str
    expected_bytes: int
    expected_md5: str
    nominal_lat_min: float
    nominal_lat_max: float


DEM_TILES = (
    DemTile(
        tile_id="n26w081",
        publication_date="2026-02-25",
        filename="USGS_13_n26w081_20260225.tif",
        url=(
            "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/"
            "TIFF/historical/n26w081/USGS_13_n26w081_20260225.tif"
        ),
        metadata_url=(
            "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/"
            "TIFF/historical/n26w081/USGS_13_n26w081_20260225.xml"
        ),
        expected_bytes=364_091_865,
        expected_md5="65dbd55b1cfab770111ea513f327ffbd",
        nominal_lat_min=25.0,
        nominal_lat_max=26.0,
    ),
    DemTile(
        tile_id="n27w081",
        publication_date="2025-12-16",
        filename="USGS_13_n27w081_20251216.tif",
        url=(
            "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/"
            "TIFF/historical/n27w081/USGS_13_n27w081_20251216.tif"
        ),
        metadata_url=(
            "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/"
            "TIFF/historical/n27w081/USGS_13_n27w081_20251216.xml"
        ),
        expected_bytes=474_839_904,
        expected_md5="81f0d97961dd16c175ba17fa9b76920c",
        nominal_lat_min=26.0,
        nominal_lat_max=27.0,
    ),
)

SFWMD_LAYER_URL = (
    "https://geoweb.sfwmd.gov/agsext1/rest/services/"
    "WaterManagementSystem/Canals/FeatureServer/0"
)
SFWMD_WHERE = "HYDRO_ORDER = 'PRIMARY'"
SFWMD_FIELDS = (
    "OBJECTID",
    "HYDROID",
    "NAME",
    "HYDRO_ORDER",
    "FLOWLINETYPE",
)
SFWMD_CACHE_FILENAME = "sfwmd_ahed_primary_drainage.geojson"
SFWMD_CACHE_METADATA_FILENAME = "sfwmd_ahed_primary_drainage.metadata.json"


def log(message: str) -> None:
    """Print a flushed progress message."""

    print(f"[03b] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    """Resolve a CLI path relative to the repository root."""

    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_dependencies() -> SimpleNamespace:
    """Load geospatial dependencies only after CLI parsing (so --help works)."""

    required = {
        "geopandas": "geopandas",
        "numpy": "numpy",
        "pandas": "pandas",
        "rasterio": "rasterio",
        "shapely": "shapely",
    }
    modules: dict[str, Any] = {}
    failures: list[str] = []
    for label, module_name in required.items():
        try:
            modules[label] = importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            failures.append(f"{label}: {exc}")

    if failures:
        details = "\n  - ".join(failures)
        raise RuntimeError(
            "Required Python geospatial packages are unavailable:\n"
            f"  - {details}\n"
            "Install geopandas, numpy, pandas, rasterio, shapely, and a "
            "GeoPackage reader such as pyogrio, then rerun."
        )

    rasterio = modules["rasterio"]
    return SimpleNamespace(
        gpd=modules["geopandas"],
        np=modules["numpy"],
        pd=modules["pandas"],
        rasterio=rasterio,
        geometry_mask=importlib.import_module("rasterio.features").geometry_mask,
        geometry_window=importlib.import_module("rasterio.features").geometry_window,
        Window=importlib.import_module("rasterio.windows").Window,
        WindowError=importlib.import_module("rasterio.errors").WindowError,
        STRtree=importlib.import_module("shapely.strtree").STRtree,
    )


def md5_file(path: Path) -> str:
    """Return the hexadecimal MD5 digest of a local file."""

    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of bytes."""

    return hashlib.sha256(payload).hexdigest()


def validate_dem_cache(path: Path, tile: DemTile) -> None:
    """Validate a cached DEM against its pinned size and USGS checksum."""

    if not path.is_file():
        raise FileNotFoundError(f"Pinned DEM cache is missing: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != tile.expected_bytes:
        raise ValueError(
            f"Cached {tile.tile_id} has {actual_bytes:,} bytes; expected "
            f"{tile.expected_bytes:,}: {path}"
        )
    actual_md5 = md5_file(path)
    if actual_md5.lower() != tile.expected_md5.lower():
        raise ValueError(
            f"Cached {tile.tile_id} failed MD5 validation: got {actual_md5}, "
            f"expected {tile.expected_md5}: {path}"
        )


def open_url(request: Request, timeout: int = NETWORK_TIMEOUT_SECONDS):
    """Open a URL with bounded retries and an informative final error."""

    last_error: Exception | None = None
    for attempt in range(1, NETWORK_ATTEMPTS + 1):
        try:
            return urlopen(request, timeout=timeout)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == NETWORK_ATTEMPTS:
                break
            delay = min(2 ** (attempt - 1), 8)
            log(f"Network attempt {attempt} failed ({exc}); retrying in {delay}s.")
            time.sleep(delay)
    raise RuntimeError(f"Unable to retrieve {request.full_url}: {last_error}")


def download_dem_tile(
    tile: DemTile,
    destination: Path,
    *,
    force: bool,
    no_download: bool,
) -> Path:
    """Download, resume, checksum, and atomically install one pinned DEM."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        log(f"Validating cached DEM {destination.name} ({tile.tile_id}).")
        try:
            validate_dem_cache(destination, tile)
        except (ValueError, OSError) as exc:
            raise RuntimeError(
                f"Existing DEM cache is invalid: {exc}. Rerun with "
                "--force-dem-download to replace only the invalid cache file."
            ) from exc
        return destination

    if no_download:
        raise FileNotFoundError(
            f"Offline mode requested but valid DEM cache is unavailable: {destination}"
        )

    partial = destination.with_name(destination.name + ".part")
    if force and partial.exists():
        partial.unlink()

    if partial.exists() and partial.stat().st_size == tile.expected_bytes:
        log(f"Validating completed partial download {partial.name}.")
        try:
            validate_dem_cache(partial, tile)
        except (ValueError, OSError):
            if not force:
                raise RuntimeError(
                    f"Completed partial cache is invalid: {partial}. Rerun with "
                    "--force-dem-download."
                )
            partial.unlink()
        else:
            os.replace(partial, destination)
            return destination

    log(
        f"Downloading pinned USGS 3DEP tile {tile.tile_id} "
        f"({tile.expected_bytes / 1_000_000:.1f} MB)."
    )
    for attempt in range(1, NETWORK_ATTEMPTS + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > tile.expected_bytes:
            raise RuntimeError(
                f"Partial file is larger than the pinned object: {partial}. "
                "Rerun with --force-dem-download."
            )

        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(tile.url, headers=headers)

        try:
            with open_url(request) as response:
                status = getattr(response, "status", response.getcode())
                if offset and status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise RuntimeError(
                            f"Unexpected Content-Range while resuming {tile.tile_id}: "
                            f"{content_range!r}"
                        )
                    mode = "ab"
                    completed = offset
                elif status == 200:
                    mode = "wb"
                    completed = 0
                else:
                    raise RuntimeError(
                        f"Unexpected HTTP {status} while downloading {tile.url}"
                    )

                next_report = completed + 100_000_000
                with partial.open(mode) as stream:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        stream.write(chunk)
                        completed += len(chunk)
                        if completed >= next_report:
                            log(
                                f"{tile.tile_id}: {completed / 1_000_000:.0f} / "
                                f"{tile.expected_bytes / 1_000_000:.0f} MB"
                            )
                            next_report += 100_000_000
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
            if attempt == NETWORK_ATTEMPTS:
                raise RuntimeError(
                    f"Download failed after {NETWORK_ATTEMPTS} attempts for "
                    f"{tile.tile_id}; partial data remain at {partial}: {exc}"
                ) from exc
            delay = min(2 ** (attempt - 1), 8)
            log(
                f"Download stream attempt {attempt} stopped ({exc}); "
                f"resuming in {delay}s."
            )
            time.sleep(delay)
            continue

        if partial.stat().st_size == tile.expected_bytes:
            break
        if partial.stat().st_size > tile.expected_bytes:
            raise RuntimeError(
                f"Downloaded {partial.stat().st_size:,} bytes for {tile.tile_id}, "
                f"more than expected {tile.expected_bytes:,}. Rerun with "
                "--force-dem-download."
            )
        log(
            f"{tile.tile_id} download ended at {partial.stat().st_size:,} bytes; "
            "resuming."
        )
    else:  # pragma: no cover - the loop always breaks or raises
        raise RuntimeError(f"Unable to finish {tile.tile_id}")

    log(f"Checking USGS MD5 for {tile.tile_id}.")
    validate_dem_cache(partial, tile)
    os.replace(partial, destination)
    log(f"Cached validated DEM at {destination}.")
    return destination


def request_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Request a JSON object and surface ArcGIS errors returned with HTTP 200."""

    full_url = url
    if params:
        full_url += ("&" if "?" in full_url else "?") + urlencode(params)
    request = Request(full_url, headers={"User-Agent": USER_AGENT})
    with open_url(request) as response:
        payload = response.read()
    try:
        result = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Non-JSON response from {full_url}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"Expected a JSON object from {full_url}")
    if "error" in result:
        raise RuntimeError(
            f"ArcGIS service error from {full_url}: "
            f"{json.dumps(result['error'], ensure_ascii=False)}"
        )
    return result


def feature_property(properties: dict[str, Any], name: str) -> Any:
    """Read an ArcGIS/GeoJSON property without relying on field-name case."""

    target = name.casefold()
    for key, value in properties.items():
        if key.casefold() == target:
            return value
    return None


def validate_canal_feature_collection(collection: dict[str, Any]) -> int:
    """Validate the cached SFWMD primary-drainage FeatureCollection."""

    if collection.get("type") != "FeatureCollection":
        raise ValueError("SFWMD cache is not a GeoJSON FeatureCollection")
    features = collection.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("SFWMD cache contains no features")

    object_ids: set[str] = set()
    for index, feature in enumerate(features):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError(f"Malformed SFWMD feature at index {index}")
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict) or geometry.get("type") not in {
            "LineString",
            "MultiLineString",
        }:
            raise ValueError(f"Non-line SFWMD geometry at feature index {index}")
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            raise ValueError(f"Missing SFWMD properties at feature index {index}")
        order = feature_property(properties, "HYDRO_ORDER")
        if str(order).upper() != "PRIMARY":
            raise ValueError(
                f"Non-primary SFWMD feature at index {index}: HYDRO_ORDER={order!r}"
            )
        object_id = feature_property(properties, "OBJECTID")
        if object_id is None:
            raise ValueError(f"Missing OBJECTID at SFWMD feature index {index}")
        object_id_text = str(object_id)
        if object_id_text in object_ids:
            raise ValueError(f"Duplicate SFWMD OBJECTID: {object_id_text}")
        object_ids.add(object_id_text)
    return len(features)


def read_json_file(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object."""

    with path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return result


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write bytes to a sibling temporary path and replace atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def download_sfwmd_primary_drainage(cache_path: Path) -> tuple[int, str]:
    """Download all primary AHED drainage features with stable pagination."""

    layer_metadata = request_json(SFWMD_LAYER_URL, {"f": "json"})
    max_record_count = int(layer_metadata.get("maxRecordCount") or 2_000)
    object_id_field = str(layer_metadata.get("objectIdField") or "OBJECTID")
    page_size = max(1, min(max_record_count, 2_000))
    query_url = SFWMD_LAYER_URL + "/query"

    common_params: dict[str, Any] = {
        "where": SFWMD_WHERE,
        "outFields": ",".join(SFWMD_FIELDS),
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }
    count_result = request_json(
        query_url,
        {
            "where": SFWMD_WHERE,
            "returnCountOnly": "true",
            "f": "json",
        },
    )
    expected_count = int(count_result.get("count", -1))
    if expected_count <= 0:
        raise RuntimeError(
            f"SFWMD primary-drainage query returned invalid count {expected_count}"
        )

    log(
        f"Downloading {expected_count:,} SFWMD AHED primary-drainage features "
        f"in pages of {page_size:,}."
    )
    features: list[dict[str, Any]] = []
    response_crs: dict[str, Any] | None = None
    offset = 0
    while offset < expected_count:
        params = dict(common_params)
        params.update(
            {
                "orderByFields": object_id_field,
                "resultOffset": str(offset),
                "resultRecordCount": str(page_size),
            }
        )
        page = request_json(query_url, params)
        page_features = page.get("features")
        if not isinstance(page_features, list) or not page_features:
            raise RuntimeError(
                f"SFWMD pagination returned no features at offset {offset:,} of "
                f"{expected_count:,}"
            )
        if response_crs is None and isinstance(page.get("crs"), dict):
            response_crs = page["crs"]
        features.extend(page_features)
        offset += len(page_features)
        log(f"Retrieved {min(offset, expected_count):,} / {expected_count:,} features.")

    if len(features) != expected_count:
        raise RuntimeError(
            f"SFWMD feature-count mismatch: retrieved {len(features):,}, "
            f"service reported {expected_count:,}"
        )

    collection: dict[str, Any] = {
        "type": "FeatureCollection",
        "name": "sfwmd_ahed_primary_drainage",
        "features": features,
    }
    if response_crs is not None:
        collection["crs"] = response_crs
    validate_canal_feature_collection(collection)

    payload = json.dumps(
        collection,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = sha256_bytes(payload)
    atomic_write_bytes(cache_path, payload)

    metadata = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "service_layer": SFWMD_LAYER_URL,
        "where": SFWMD_WHERE,
        "out_fields": list(SFWMD_FIELDS),
        "output_crs": "EPSG:4326",
        "feature_count": expected_count,
        "geojson_sha256": digest,
    }
    metadata_payload = (
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    atomic_write_bytes(
        cache_path.with_name(SFWMD_CACHE_METADATA_FILENAME), metadata_payload
    )
    return expected_count, digest


def ensure_sfwmd_cache(
    cache_path: Path,
    *,
    refresh: bool,
    no_download: bool,
) -> Path:
    """Validate/reuse the canal cache, or fetch a complete new snapshot."""

    if cache_path.exists() and not refresh:
        log(f"Validating cached SFWMD drainage features at {cache_path}.")
        collection = read_json_file(cache_path)
        feature_count = validate_canal_feature_collection(collection)
        metadata_path = cache_path.with_name(SFWMD_CACHE_METADATA_FILENAME)
        if metadata_path.exists():
            metadata = read_json_file(metadata_path)
            expected_digest = metadata.get("geojson_sha256")
            if expected_digest:
                actual_digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
                if actual_digest != expected_digest:
                    raise RuntimeError(
                        f"Cached SFWMD GeoJSON failed its recorded SHA-256: {cache_path}"
                    )
            recorded_count = metadata.get("feature_count")
            if recorded_count is not None and int(recorded_count) != feature_count:
                raise RuntimeError(
                    "Cached SFWMD feature count does not match its metadata: "
                    f"{feature_count} versus {recorded_count}"
                )
        log(f"Reusing {feature_count:,} validated cached primary features.")
        return cache_path

    if no_download:
        raise FileNotFoundError(
            f"Offline mode requested but a usable SFWMD cache is unavailable: {cache_path}"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    count, digest = download_sfwmd_primary_drainage(cache_path)
    log(
        f"Cached {count:,} SFWMD primary features at {cache_path} "
        f"(SHA-256 {digest})."
    )
    return cache_path


def read_block_groups(deps: SimpleNamespace, path: Path, layer: str):
    """Read and validate one unique polygon per analysis block group."""

    if not path.is_file():
        raise FileNotFoundError(f"Block-group GeoPackage not found: {path}")
    log(f"Reading block-group geometry from {path} (layer {layer!r}).")
    try:
        block_groups = deps.gpd.read_file(
            path,
            layer=layer,
            columns=["block_group_geoid"],
        )
    except TypeError:
        block_groups = deps.gpd.read_file(path, layer=layer)
        block_groups = block_groups[["block_group_geoid", "geometry"]]

    if "block_group_geoid" not in block_groups.columns:
        raise ValueError(f"Layer {layer!r} lacks block_group_geoid")
    if block_groups.crs is None:
        raise ValueError(f"Layer {layer!r} has no CRS")
    if block_groups.empty:
        raise ValueError(f"Layer {layer!r} contains no block groups")
    if block_groups.geometry.isna().any() or block_groups.geometry.is_empty.any():
        bad = block_groups.loc[
            block_groups.geometry.isna() | block_groups.geometry.is_empty,
            "block_group_geoid",
        ].astype(str)
        raise ValueError(f"Null/empty block-group geometry: {bad.head(10).tolist()}")
    if not block_groups.geometry.is_valid.all():
        bad = block_groups.loc[
            ~block_groups.geometry.is_valid, "block_group_geoid"
        ].astype(str)
        raise ValueError(f"Invalid block-group geometry: {bad.head(10).tolist()}")

    block_groups = block_groups[["block_group_geoid", "geometry"]].copy()
    block_groups["block_group_geoid"] = (
        block_groups["block_group_geoid"].astype("string").str.strip()
    )
    if block_groups["block_group_geoid"].isna().any():
        raise ValueError("Missing block_group_geoid values")
    malformed = ~block_groups["block_group_geoid"].str.fullmatch(GEOID_PATTERN)
    if malformed.any():
        raise ValueError(
            "Malformed 12-digit block_group_geoid values: "
            f"{block_groups.loc[malformed, 'block_group_geoid'].head(10).tolist()}"
        )
    duplicated = block_groups["block_group_geoid"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            "Duplicate block_group_geoid values: "
            f"{block_groups.loc[duplicated, 'block_group_geoid'].head(10).tolist()}"
        )

    block_groups = block_groups.sort_values("block_group_geoid").reset_index(drop=True)
    log(f"Validated {len(block_groups):,} unique block-group polygons.")
    return block_groups


def validate_raster_source(source: Any, tile: DemTile) -> None:
    """Check that a DEM is a plausible 3DEP 1/3-arc-second source."""

    if source.count != 1:
        raise ValueError(f"{tile.filename} has {source.count} bands; expected one")
    if source.crs is None or not source.crs.is_geographic:
        raise ValueError(f"{tile.filename} must have a geographic CRS")
    epsg = source.crs.to_epsg()
    if epsg not in {4269}:
        raise ValueError(
            f"{tile.filename} has CRS {source.crs}; expected NAD83 (EPSG:4269)"
        )
    x_resolution = abs(float(source.transform.a))
    y_resolution = abs(float(source.transform.e))
    tolerance = DEM_APPROX_RESOLUTION_DEGREES * 1e-4
    if not math.isclose(
        x_resolution,
        DEM_APPROX_RESOLUTION_DEGREES,
        abs_tol=tolerance,
    ) or not math.isclose(
        y_resolution,
        DEM_APPROX_RESOLUTION_DEGREES,
        abs_tol=tolerance,
    ):
        raise ValueError(
            f"{tile.filename} resolution is {x_resolution}, {y_resolution} degrees; "
            "expected 1/3 arc-second"
        )
    if not math.isclose(float(source.transform.b), 0.0, abs_tol=1e-12) or not math.isclose(
        float(source.transform.d), 0.0, abs_tol=1e-12
    ):
        raise ValueError(f"Rotated DEM grids are unsupported: {tile.filename}")
    if source.bounds.bottom > tile.nominal_lat_min or source.bounds.top < tile.nominal_lat_max:
        raise ValueError(
            f"{tile.filename} bounds {source.bounds} do not cover its nominal "
            f"latitude interval [{tile.nominal_lat_min}, {tile.nominal_lat_max}]"
        )


def subdivide_window(window: Any, window_size: int, Window: Any) -> Iterable[Any]:
    """Yield bounded integer subwindows to cap peak raster memory."""

    col_start = int(window.col_off)
    row_start = int(window.row_off)
    col_stop = col_start + int(window.width)
    row_stop = row_start + int(window.height)
    for row_off in range(row_start, row_stop, window_size):
        height = min(window_size, row_stop - row_off)
        for col_off in range(col_start, col_stop, window_size):
            width = min(window_size, col_stop - col_off)
            yield Window(col_off, row_off, width, height)


def values_in_geometry(
    deps: SimpleNamespace,
    source: Any,
    geometry: Any,
    tile: DemTile,
    window_size: int,
) -> list[Any]:
    """Return valid pixel-center values for one polygon in one nominal tile."""

    try:
        full_window = deps.geometry_window(
            source,
            [geometry.__geo_interface__],
            pad_x=0,
            pad_y=0,
        )
        full_window = full_window.intersection(
            deps.Window(0, 0, source.width, source.height)
        )
    except (deps.WindowError, ValueError):
        return []

    values: list[Any] = []
    for window in subdivide_window(full_window, window_size, deps.Window):
        data = source.read(1, window=window, masked=True)
        window_transform = source.window_transform(window)
        inside = deps.geometry_mask(
            [geometry.__geo_interface__],
            out_shape=data.shape,
            transform=window_transform,
            all_touched=False,
            invert=True,
        )
        row_numbers = deps.np.arange(data.shape[0], dtype="float64") + 0.5
        row_latitudes = window_transform.f + row_numbers * window_transform.e
        in_nominal_tile = (row_latitudes >= tile.nominal_lat_min) & (
            row_latitudes < tile.nominal_lat_max
        )
        valid = (
            inside
            & in_nominal_tile[:, None]
            & ~deps.np.ma.getmaskarray(data)
            & deps.np.isfinite(data.data)
        )
        if valid.any():
            values.append(deps.np.asarray(data.data[valid], dtype="float64"))
    return values


def compute_elevation_statistics(
    deps: SimpleNamespace,
    block_groups: Any,
    dem_paths: dict[str, Path],
    window_size: int,
) -> tuple[Any, Any]:
    """Compute deterministic, overlap-free mean and median elevation."""

    sources: list[tuple[DemTile, Any]] = []
    try:
        for tile in DEM_TILES:
            source = deps.rasterio.open(dem_paths[tile.tile_id])
            validate_raster_source(source, tile)
            sources.append((tile, source))
        dem_crs = sources[0][1].crs
        if any(source.crs != dem_crs for _, source in sources[1:]):
            raise ValueError("Pinned DEM tiles do not share a CRS")

        polygons = block_groups.to_crs(dem_crs)
        means = deps.np.full(len(polygons), deps.np.nan, dtype="float64")
        medians = deps.np.full(len(polygons), deps.np.nan, dtype="float64")
        missing_geoids: list[str] = []

        log(
            "Computing 3DEP zonal mean and median using valid pixel centers "
            f"(window cap {window_size:,} x {window_size:,})."
        )
        for position, row in enumerate(polygons.itertuples(index=False), start=0):
            geometry = row.geometry
            value_parts: list[Any] = []
            for tile, source in sources:
                value_parts.extend(
                    values_in_geometry(deps, source, geometry, tile, window_size)
                )
            if not value_parts:
                missing_geoids.append(str(row.block_group_geoid))
            else:
                values = (
                    value_parts[0]
                    if len(value_parts) == 1
                    else deps.np.concatenate(value_parts)
                )
                means[position] = float(deps.np.mean(values, dtype="float64"))
                medians[position] = float(deps.np.median(values))
            if (position + 1) % 250 == 0 or position + 1 == len(polygons):
                log(f"Elevation complete for {position + 1:,} / {len(polygons):,} block groups.")

        if missing_geoids:
            raise RuntimeError(
                f"No valid 3DEP cells were found for {len(missing_geoids):,} block "
                f"groups; examples: {missing_geoids[:10]}. Do not write an output "
                "with silent elevation NAs."
            )
        if not deps.np.isfinite(means).all() or not deps.np.isfinite(medians).all():
            raise RuntimeError("Elevation calculation produced non-finite values")
        if deps.np.any(deps.np.abs(means) > 10_000) or deps.np.any(
            deps.np.abs(medians) > 10_000
        ):
            raise RuntimeError(
                "Implausible elevation magnitude detected; a DEM nodata sentinel may "
                "have escaped masking"
            )
        return means, medians
    finally:
        for _, source in sources:
            source.close()


def load_primary_drainage(deps: SimpleNamespace, path: Path):
    """Load and validate cached SFWMD primary drainage as projected lines."""

    drainage = deps.gpd.read_file(path)
    if drainage.empty:
        raise ValueError(f"No drainage features in {path}")
    if drainage.crs is None:
        # ArcGIS GeoJSON is requested explicitly in EPSG:4326.  GeoJSON readers
        # usually assign this automatically, but make the contract explicit.
        drainage = drainage.set_crs("EPSG:4326", allow_override=True)
    order_column = next(
        (column for column in drainage.columns if column.casefold() == "hydro_order"),
        None,
    )
    if order_column is None:
        raise ValueError(f"Cached drainage lacks HYDRO_ORDER: {path}")
    if not drainage[order_column].astype("string").str.upper().eq("PRIMARY").all():
        raise ValueError("Cached drainage includes non-primary features")
    bad_geometry = (
        drainage.geometry.isna()
        | drainage.geometry.is_empty
        | ~drainage.geometry.is_valid
    )
    if bad_geometry.any():
        raise ValueError(
            f"Cached drainage has {int(bad_geometry.sum())} null, empty, or invalid "
            "geometries"
        )
    line_types = drainage.geometry.geom_type.isin(["LineString", "MultiLineString"])
    if not line_types.all():
        raise ValueError("Cached drainage contains non-line geometry")
    projected = drainage[["geometry"]].to_crs(DISTANCE_CRS)
    log(f"Loaded {len(projected):,} validated primary-drainage lines.")
    return projected


def compute_drainage_distances(
    deps: SimpleNamespace,
    block_groups: Any,
    drainage: Any,
) -> Any:
    """Compute centroid-to-nearest-primary-feature distances in kilometers."""

    projected_groups = block_groups.to_crs(DISTANCE_CRS)
    centroids = projected_groups.geometry.centroid
    if centroids.isna().any() or centroids.is_empty.any():
        raise RuntimeError("Unable to construct all block-group centroids")

    drainage_geometries = drainage.geometry.to_numpy()
    tree = deps.STRtree(drainage_geometries)
    distances_m = deps.np.empty(len(centroids), dtype="float64")
    log("Computing centroid distance to nearest SFWMD AHED primary feature.")
    for position, centroid in enumerate(centroids):
        nearest_index = tree.nearest(centroid)
        if nearest_index is None:
            raise RuntimeError(
                f"No nearest drainage feature for block group at row {position}"
            )
        distances_m[position] = float(
            centroid.distance(drainage_geometries[int(nearest_index)])
        )

    distances_km = distances_m / 1_000.0
    if not deps.np.isfinite(distances_km).all():
        raise RuntimeError("Drainage-distance calculation produced non-finite values")
    if deps.np.any(distances_km < 0):
        raise RuntimeError("Drainage-distance calculation produced negative values")
    return distances_km


def validate_output_frame(deps: SimpleNamespace, output: Any, block_groups: Any) -> None:
    """Enforce the exact output contract before and after writing."""

    if list(output.columns) != OUTPUT_COLUMNS:
        raise ValueError(
            f"Output schema is {list(output.columns)}; expected {OUTPUT_COLUMNS}"
        )
    if len(output) != len(block_groups):
        raise ValueError(
            f"Output has {len(output):,} rows for {len(block_groups):,} block groups"
        )
    if output["block_group_geoid"].duplicated().any():
        raise ValueError("Output contains duplicate block_group_geoid rows")
    output_geoids = set(output["block_group_geoid"].astype(str))
    input_geoids = set(block_groups["block_group_geoid"].astype(str))
    if output_geoids != input_geoids:
        raise ValueError("Output GEOID set differs from the source block-group GEOID set")
    if not output["block_group_geoid"].astype("string").str.fullmatch(GEOID_PATTERN).all():
        raise ValueError("Output contains malformed block_group_geoid values")
    numeric = output[OUTPUT_COLUMNS[1:]].to_numpy(dtype="float64")
    if not deps.np.isfinite(numeric).all():
        raise ValueError("Output contains missing or non-finite physical covariates")
    if (output["drainage_distance_km"] < 0).any():
        raise ValueError("Output contains a negative drainage distance")


def atomic_write_output(
    deps: SimpleNamespace,
    output: Any,
    block_groups: Any,
    path: Path,
) -> None:
    """Write the CSV atomically and read it back to verify its contract."""

    validate_output_frame(deps, output, block_groups)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    output.to_csv(temporary, index=False, lineterminator="\n")
    reread = deps.pd.read_csv(
        temporary,
        dtype={"block_group_geoid": "string"},
    )
    validate_output_frame(deps, reread, block_groups)
    os.replace(temporary, path)
    log(f"Wrote {len(output):,} rows with exact schema to {path}.")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Join pinned USGS 3DEP elevation summaries and SFWMD primary-"
            "drainage distance to analysis block groups."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--block-groups",
        type=Path,
        default=DEFAULT_BLOCK_GROUP_PATH,
        help="GeoPackage containing the analysis block-group polygons.",
    )
    parser.add_argument(
        "--block-group-layer",
        default=DEFAULT_BLOCK_GROUP_LAYER,
        help="GeoPackage layer containing one row per block group.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for validated downloaded source data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Physical-covariate CSV to create.",
    )
    parser.add_argument(
        "--force-dem-download",
        action="store_true",
        help="Redownload pinned DEMs, replacing them only after validation.",
    )
    parser.add_argument(
        "--refresh-canals",
        action="store_true",
        help="Replace the cached SFWMD primary-feature snapshot.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use validated cache inputs only; fail instead of using the network.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Populate and validate source caches, then stop before spatial work.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Maximum raster read-window width and height in pixels.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Download/cache inputs, compute covariates, validate, and write output."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.window_size < 128:
        parser.error("--window-size must be at least 128 pixels")
    if args.no_download and (args.force_dem_download or args.refresh_canals):
        parser.error(
            "--no-download cannot be combined with --force-dem-download or "
            "--refresh-canals"
        )

    block_group_path = resolve_path(args.block_groups)
    cache_dir = resolve_path(args.cache_dir)
    output_path = resolve_path(args.output)
    dem_cache_dir = cache_dir / "usgs_3dep_13_arc_second"
    sfwmd_cache_path = cache_dir / "sfwmd" / SFWMD_CACHE_FILENAME

    log(
        "USGS source: 3DEP seamless 1/3-arc-second bare-earth DEM, "
        "NAD83 / NAVD88 meters."
    )
    dem_paths: dict[str, Path] = {}
    for tile in DEM_TILES:
        destination = dem_cache_dir / tile.filename
        dem_paths[tile.tile_id] = download_dem_tile(
            tile,
            destination,
            force=args.force_dem_download,
            no_download=args.no_download,
        )
    ensure_sfwmd_cache(
        sfwmd_cache_path,
        refresh=args.refresh_canals,
        no_download=args.no_download,
    )

    if args.download_only:
        log("Source caches are complete and validated; --download-only requested.")
        return 0

    deps = load_dependencies()
    block_groups = read_block_groups(deps, block_group_path, args.block_group_layer)
    elevation_mean, elevation_median = compute_elevation_statistics(
        deps,
        block_groups,
        dem_paths,
        args.window_size,
    )
    drainage = load_primary_drainage(deps, sfwmd_cache_path)
    drainage_distance = compute_drainage_distances(deps, block_groups, drainage)

    output = deps.pd.DataFrame(
        {
            "block_group_geoid": block_groups["block_group_geoid"].astype("string"),
            "elevation_m_mean": elevation_mean,
            "elevation_m_median": elevation_median,
            "drainage_distance_km": drainage_distance,
        }
    )
    atomic_write_output(deps, output, block_groups, output_path)
    log(
        "Reminder: elevation is mechanically close to inundation classification; "
        "check separation and convergence, especially for Inundated outcomes."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[03b] Interrupted; resumable cache partials were retained.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[03b] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
