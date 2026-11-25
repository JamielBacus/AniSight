"""Province boundaries (geoBoundaries PHL ADM2/ADM3) and ESA WorldCover cropland masks.

Steps:
1. `load_boundaries` downloads geoBoundaries PHL ADM2 (cached under `data/raw/boundaries/`),
   keeps the provinces in `configs/provinces.yaml` (matching names and aliases) and
   returns one polygon per province in EPSG:4326. `save_boundaries` writes
   `data/interim/boundaries.parquet`.
2. `cropland_mask` loads ESA WorldCover from the Planetary Computer for each province's
   bounding box, clips it to the polygon and keeps the cropland class (40). Each mask
   carries its cropland area in hectares. `save_cropland` writes one GeoTIFF per province
   under `data/interim/cropland/` and `data/interim/cropland_area.parquet`.

Davao City is its own unit (PSA reports "City of Davao" separately). Units with
`part_of` in `provinces.yaml` are matched in ADM2 first; if ADM2 has no polygon for them,
geoBoundaries PHL ADM3 (`sources.boundaries_adm3_url`, downloaded only in that case) is
used. Their polygon is then subtracted from the `part_of` province, so Davao del Sur
never includes the city.

`reporting_unit_geometries` dissolves host + absorbed provinces into the combined PSA
reporting units of `reporting_units` in `provinces.yaml` (e.g. "Davao del Sur+Davao
Occidental"), keyed like the ground truth's `reporting_unit` column.

Usage (fetches real data; WorldCover for the whole region takes a while):
    python -m cropyield.boundaries                       # boundaries + cropland
    python -m cropyield.boundaries --step boundaries
    python -m cropyield.boundaries --step cropland --config-dir configs
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

# geoBoundaries PHL polygons are very detailed; GDAL's GeoJSON reader rejects single
# features above 200 MB by default. 0 removes the limit. Must be set before any read.
os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rioxarray  # noqa: F401  (registers the .rio accessor used below)
import xarray as xr
from shapely.geometry.base import BaseGeometry

from cropyield.config import (DEFAULT_CONFIG_DIR, Config, Province, ProvincesConfig,
                              load_config)

log = logging.getLogger(__name__)

CRS = "EPSG:4326"
WORLDCOVER_RES_DEG = 1 / 12000  # ESA WorldCover native grid, ~10 m
EARTH_RADIUS_M = 6_371_008.8
HEADERS = {"User-Agent": "cropyield-student-project/0.1 (+python-requests)"}

# A loader returns a 2-D land-cover DataArray (dims y, x, CRS set via rioxarray) that
# covers the given (minx, miny, maxx, maxy) bbox in EPSG:4326.
WorldCoverLoader = Callable[[tuple[float, float, float, float]], xr.DataArray]


# --------------------------------------------------------------------------- boundaries

def _norm(name: str) -> str:
    """Normalise a place name for matching: casefold, drop 'province of', squash spaces."""
    s = re.sub(r"[^a-z ]", " ", str(name).casefold())
    s = re.sub(r"\bprovince of\b", " ", s)
    return " ".join(s.split())


def _get(url: str, retries: int = 3) -> requests.Response:
    """GET with retries and backoff on network errors, rate limits and server errors."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=(10, 120))
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Could not reach {url}: {e}") from e
            log.warning("network error on %s (%s), retrying", url, e)
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError("unreachable")


def download_geoboundaries(api_url: str, dest: Path) -> Path:
    """Download the geoBoundaries GeoJSON described by `api_url` to `dest`.

    Skips the download if `dest` already exists, so re-runs are cheap.
    """
    if dest.exists():
        log.info("using cached %s", dest)
        return dest
    meta = _get(api_url).json()
    gj_url = meta.get("gjDownloadURL")
    if not gj_url:
        raise RuntimeError(f"{api_url}: response has no 'gjDownloadURL' field")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(_get(gj_url).content)
    tmp.replace(dest)
    log.info("saved %s", dest)
    return dest


def _match(gdf: gpd.GeoDataFrame, prov: Province, name_col: str) -> gpd.GeoDataFrame:
    """Rows of `gdf` whose `name_col` matches the province's name or an alias."""
    wanted = {_norm(n) for n in (prov.name, *prov.aliases)}
    return gdf[gdf[name_col].map(_norm).isin(wanted)]


def _to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return (gdf.set_crs(CRS) if gdf.crs is None else gdf).to_crs(CRS)


def select_provinces(
    raw: gpd.GeoDataFrame,
    provinces: ProvincesConfig,
    name_col: str = "shapeName",
    adm3: Callable[[], gpd.GeoDataFrame] | None = None,
) -> gpd.GeoDataFrame:
    """Keep the configured provinces from an ADM2 GeoDataFrame.

    Matches `name_col` against each province's canonical name and aliases (case- and
    punctuation-insensitive). Multiple matching features are dissolved into one polygon.
    A unit with `part_of` (Davao City) that has no ADM2 polygon is taken from `adm3()`
    instead, and every `part_of` unit is subtracted from its parent province.

    Args:
        raw: ADM2 boundaries.
        provinces: configured provinces.
        name_col: name column in `raw` and in the ADM3 table.
        adm3: returns ADM3 boundaries; called only when a `part_of` unit is not in ADM2.

    Returns a GeoDataFrame with columns `province`, `source_name`, `geometry` in EPSG:4326,
    in the order of `provinces.yaml`.

    Raises:
        ValueError: naming every configured province with no matching polygon, or if a
            `part_of` parent is not a configured province.
    """
    raw = _to_wgs84(raw)
    adm3_gdf: gpd.GeoDataFrame | None = None

    rows, missing = [], []
    for prov in provinces.provinces:
        hit, level = _match(raw, prov, name_col), "ADM2"
        if hit.empty and prov.part_of is not None and adm3 is not None:
            if adm3_gdf is None:
                adm3_gdf = _to_wgs84(adm3())
            hit, level = _match(adm3_gdf, prov, name_col), "ADM3"
        if hit.empty:
            missing.append(prov.name)
            continue
        source_name = "; ".join(sorted(set(hit[name_col])))
        if prov.part_of is not None:
            log.info("%s: using %s polygon %r", prov.name, level, source_name)
        rows.append({
            "province": prov.name,
            "source_name": source_name if level == "ADM2" else f"{source_name} (ADM3)",
            "geometry": hit.geometry.union_all(),
        })
    if missing:
        raise ValueError(
            f"No boundary polygon matches configured province(s): {', '.join(missing)}. "
            f"Add the geoBoundaries spelling as an alias in configs/provinces.yaml.")

    by_name = {r["province"]: r for r in rows}
    for prov in provinces.provinces:
        if prov.part_of is None:
            continue
        parent = by_name.get(prov.part_of)
        if parent is None:
            raise ValueError(
                f"{prov.name}: part_of {prov.part_of!r} is not a configured province")
        parent["geometry"] = parent["geometry"].difference(by_name[prov.name]["geometry"])
        log.info("Subtracted %s from %s.", prov.name, prov.part_of)

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)


def load_boundaries(
    cfg: Config,
    source: str | Path | None = None,
    adm3_source: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Return configured province polygons from geoBoundaries PHL ADM2 in EPSG:4326.

    PHL ADM3 is downloaded (and cached) only if a `part_of` unit is missing from ADM2.

    Args:
        cfg: project config; uses `sources.boundaries_url`, `sources.boundaries_adm3_url`,
            `paths.raw` and provinces.
        source: optional local ADM2 GeoJSON/GeoPackage to read instead of downloading.
        adm3_source: optional local ADM3 file to read instead of downloading.
    """
    raw_dir = cfg.pipeline.paths.raw / "boundaries"
    if source is None:
        source = download_geoboundaries(
            cfg.pipeline.sources.boundaries_url, raw_dir / "geoBoundaries-PHL-ADM2.geojson")

    def adm3() -> gpd.GeoDataFrame:
        src = adm3_source or download_geoboundaries(
            cfg.pipeline.sources.boundaries_adm3_url,
            raw_dir / "geoBoundaries-PHL-ADM3.geojson")
        return gpd.read_file(src)

    return select_provinces(gpd.read_file(source), cfg.provinces, adm3=adm3)


def reporting_unit_geometries(gdf: gpd.GeoDataFrame, cfg: Config) -> gpd.GeoDataFrame:
    """Polygons of the combined PSA reporting units from `reporting_units` in provinces.yaml.

    Inside a reporting window PSA reports a host province together with the units it
    absorbs (see `cropyield.ground_truth`), so its figure covers their union. This returns
    one dissolved polygon per combined unit, keyed by the same `reporting_unit` name the
    ground truth uses (host first, then absorbed units in `provinces.yaml` order, joined
    by "+"). Single-province footprints are the rows of `gdf` itself. Cropland masks for
    combined units are not built here; union the members' pixel masks instead.

    Args:
        gdf: province polygons from `load_boundaries` (columns `province`, `geometry`).
        cfg: project config; uses `provinces.reporting_units`.

    Returns a GeoDataFrame with columns `reporting_unit`, `members` (tuple of provinces)
    and `geometry` in EPSG:4326; empty when no reporting windows are configured.

    Raises:
        ValueError: if a member province has no polygon in `gdf`.
    """
    gdf = gdf.to_crs(CRS)
    by_name = dict(zip(gdf["province"], gdf.geometry))
    rows = []
    for name, members in cfg.provinces.combined_reporting_units().items():
        absent = [m for m in members if m not in by_name]
        if absent:
            raise ValueError(f"{name}: no polygon for {', '.join(absent)} in boundaries")
        rows.append({"reporting_unit": name, "members": members,
                     "geometry": gpd.GeoSeries([by_name[m] for m in members]).union_all()})
    return gpd.GeoDataFrame(rows, columns=["reporting_unit", "members", "geometry"],
                            geometry="geometry", crs=CRS)


def save_boundaries(gdf: gpd.GeoDataFrame, cfg: Config) -> Path:
    """Write province polygons to `data/interim/boundaries.parquet`."""
    out = cfg.pipeline.paths.interim / "boundaries.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_crs(CRS).to_parquet(out)
    return out


# ------------------------------------------------------------------------- cropland mask

def pixel_area_ha(da: xr.DataArray) -> xr.DataArray:
    """Area of each pixel in hectares (varies by latitude for geographic grids)."""
    x, y = da["x"].values, da["y"].values
    dx = abs(float(x[1] - x[0])) if x.size > 1 else abs(da.rio.resolution()[0])
    dy = abs(float(y[1] - y[0])) if y.size > 1 else abs(da.rio.resolution()[1])
    crs = da.rio.crs
    if crs is not None and not crs.is_geographic:
        row = np.full(y.shape, dx * dy / 1e4)
    else:
        lat = np.radians(y)
        half = math.radians(dy) / 2
        row = (EARTH_RADIUS_M ** 2 * math.radians(dx)
               * np.abs(np.sin(lat + half) - np.sin(lat - half)) / 1e4)
    return xr.DataArray(row, coords={"y": y}, dims="y")


def mask_from_landcover(
    landcover: xr.DataArray, geometry: BaseGeometry, cropland_class: int
) -> xr.DataArray:
    """Boolean cropland mask (True = cropland inside `geometry`) clipped to the polygon.

    `landcover` must be 2-D with dims (y, x) and a CRS set via rioxarray. The returned
    mask has `attrs["cropland_ha"]` (cropland area in hectares) and keeps the CRS.
    """
    clipped = landcover.rio.clip([geometry], crs=CRS, drop=True, all_touched=False)
    mask = (clipped == cropland_class).compute()
    mask = mask.rio.write_crs(landcover.rio.crs)
    mask.name = "cropland"
    mask.attrs = {"cropland_ha": float((mask * pixel_area_ha(mask)).sum())}
    return mask


def stac_worldcover_loader(cfg: Config) -> WorldCoverLoader:
    """Return a loader that reads the latest ESA WorldCover map from the Planetary Computer."""

    def load(bbox: tuple[float, float, float, float]) -> xr.DataArray:
        import odc.stac
        import planetary_computer
        import pystac_client

        src = cfg.pipeline.sources
        catalog = pystac_client.Client.open(
            src.stac_url, modifier=planetary_computer.sign_inplace)
        items = list(catalog.search(collections=[src.worldcover_collection],
                                    bbox=bbox).items())
        if not items:
            raise RuntimeError(f"No {src.worldcover_collection} items for bbox {bbox}")
        # The collection has 2020 (v100) and 2021 (v200) maps; use the latest year only.
        year = lambda it: str(it.properties.get("start_datetime") or it.datetime)[:4]
        latest = max(year(it) for it in items)
        items = [it for it in items if year(it) == latest]
        ds = odc.stac.load(items, bands=["map"], bbox=bbox, crs=CRS,
                           resolution=WORLDCOVER_RES_DEG, chunks={"x": 4096, "y": 4096})
        da = ds["map"].max("time") if "time" in ds["map"].dims else ds["map"]
        da = da.rename({"longitude": "x", "latitude": "y"}) if "longitude" in da.dims else da
        return da.rio.write_crs(CRS)

    return load


def cropland_mask(
    gdf: gpd.GeoDataFrame, cfg: Config, loader: WorldCoverLoader | None = None
) -> dict[str, xr.DataArray]:
    """Per-province ESA WorldCover cropland masks.

    Args:
        gdf: province polygons from `load_boundaries` (columns `province`, `geometry`).
        cfg: project config; uses `cropland_class` and the WorldCover STAC source.
        loader: function returning land cover for a bbox; defaults to the Planetary
            Computer STAC loader. Tests pass a synthetic loader.

    Returns:
        `{province: mask}` where each mask is a boolean (y, x) DataArray in EPSG:4326,
        True on cropland pixels inside the province, with `attrs["cropland_ha"]`.
    """
    loader = loader or stac_worldcover_loader(cfg)
    gdf = gdf.to_crs(CRS)
    masks: dict[str, xr.DataArray] = {}
    for province, geom in zip(gdf["province"], gdf.geometry):
        log.info("cropland mask: %s", province)
        landcover = loader(tuple(geom.bounds))
        masks[province] = mask_from_landcover(landcover, geom, cfg.pipeline.cropland_class)
        log.info("  %s: %.0f ha cropland", province, masks[province].attrs["cropland_ha"])
    return masks


def cropland_area_table(masks: dict[str, xr.DataArray]) -> pd.DataFrame:
    """Table of cropland hectares per province: columns `province`, `cropland_ha`."""
    return pd.DataFrame(
        [{"province": p, "cropland_ha": m.attrs["cropland_ha"]} for p, m in masks.items()])


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")


def save_cropland(masks: dict[str, xr.DataArray], cfg: Config) -> Path:
    """Write masks to `data/interim/cropland/<province>.tif` (uint8, 1 = cropland) and
    hectares to `data/interim/cropland_area.parquet`. Returns the parquet path."""
    out_dir = cfg.pipeline.paths.interim / "cropland"
    out_dir.mkdir(parents=True, exist_ok=True)
    for province, mask in masks.items():
        mask.astype("uint8").rio.to_raster(out_dir / f"{_slug(province)}.tif", compress="deflate")
    out = cfg.pipeline.paths.interim / "cropland_area.parquet"
    cropland_area_table(masks).to_parquet(out, index=False)
    return out


def load_cropland(cfg: Config, provinces: list[str]) -> dict[str, xr.DataArray]:
    """Read masks written by `save_cropland` back as `{province: boolean (y, x) mask}`.

    Each mask gets `attrs["cropland_ha"]` from `cropland_area.parquet` when listed there.

    Raises:
        FileNotFoundError: naming the first province whose GeoTIFF is missing.
    """
    out_dir = cfg.pipeline.paths.interim / "cropland"
    area_path = cfg.pipeline.paths.interim / "cropland_area.parquet"
    area = (pd.read_parquet(area_path).set_index("province")["cropland_ha"].to_dict()
            if area_path.exists() else {})
    masks: dict[str, xr.DataArray] = {}
    for province in provinces:
        path = out_dir / f"{_slug(province)}.tif"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found for {province}; run "
                f"`python -m cropyield.boundaries --step cropland` first")
        da = rioxarray.open_rasterio(path).squeeze("band", drop=True)
        mask = (da == 1).rio.write_crs(da.rio.crs)
        mask.name = "cropland"
        mask.attrs = {"cropland_ha": float(area[province])} if province in area else {}
        masks[province] = mask
    return masks


# ------------------------------------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> None:
    """Fetch boundaries and/or WorldCover cropland masks and write them under data/interim."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR),
                    help="folder with pipeline.yaml and provinces.yaml (default: repo configs/)")
    ap.add_argument("--step", choices=["boundaries", "cropland", "all"], default="all",
                    help="'cropland' reuses data/interim/boundaries.parquet (default: all)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config_dir)

    if args.step in ("boundaries", "all"):
        gdf = load_boundaries(cfg)
        print(f"wrote {save_boundaries(gdf, cfg)} ({len(gdf)} provinces)")
    else:
        gdf = gpd.read_parquet(cfg.pipeline.paths.interim / "boundaries.parquet")
    if args.step in ("cropland", "all"):
        out = save_cropland(cropland_mask(gdf, cfg), cfg)
        print(f"wrote {out}")
        print(pd.read_parquet(out).to_string(index=False))


if __name__ == "__main__":
    main()
