"""Sentinel-2 L2A NDVI/EVI per province and month, over cropland only, with SCL cloud masking.

For each (province, month) chunk:
1. Search `sentinel-2-l2a` on the Planetary Computer for scenes over the province bbox.
2. Load B02 (blue), B04 (red), B08 (NIR) and SCL on the province's cropland-mask grid
   (the ESA WorldCover ~10 m grid from `cropyield.boundaries.cropland_mask`), one time
   step per solar day. Reflectance is DN / 10000, after the -1000 offset that ESA added
   from processing baseline 04.00 (25 Jan 2022) onwards.
3. Drop observations whose SCL class is a cloud / shadow class (`s2_cloud_scl_classes`,
   default 3, 8, 9, 10) or SCL 0 (no data), compute NDVI and EVI per observation, and
   take the per-pixel median of the clear observations as the monthly composite.
4. Aggregate the composite over cropland pixels: mean, median, max, `n_pixels` (cropland
   pixels with a valid composite value) and `valid_frac` (`n_pixels` / cropland pixels).
   When `valid_frac` is below `min_valid_fraction` (default 0.10) the month is marked
   missing: mean, median and max are NaN; `valid_frac` and `n_pixels` are kept.

Each finished chunk is cached as Parquet under `data/interim/cache/s2/` so an interrupted
run resumes where it stopped. Chunks that keep failing after retries are logged and
skipped (not cached), so the next run tries them again. The current, incomplete month is
not cached and months after today are skipped.

Usage (fetches real data; needs `python -m cropyield.boundaries` to have run first):
    python -m cropyield.satellite.sentinel2 --start 2023-01-01 --end 2023-03-31
    python -m cropyield.satellite.sentinel2 --config-dir configs   # full configured range
"""
from __future__ import annotations

import argparse
import logging
import time
import warnings
from collections.abc import Callable
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pystac
import xarray as xr

from cropyield.boundaries import CRS, _slug
from cropyield.config import DEFAULT_CONFIG_DIR, Config, load_config
from cropyield.satellite.stac import fresh_sign, search_items

log = logging.getLogger(__name__)

COLUMNS = ["province", "date", "index", "mean", "median", "max", "valid_frac", "n_pixels"]
INDICES = ("NDVI", "EVI")
BANDS = {"blue": "B02", "red": "B04", "nir": "B08", "scl": "SCL"}
SCL_NODATA = 0
BOA_OFFSET_BASELINE = 4.0  # processing baseline from which BOA_ADD_OFFSET applies
BOA_OFFSET_DN = -1000
DEFAULT_MAX_SCENE_CLOUD = 80.0  # skip scenes with more cloud than this (eo:cloud_cover, %)
# Province averages don't need 10 m pixels. Reading on a 6x coarser grid (~60 m) lets
# GDAL use the COG overviews and cuts the download roughly 36x. 1 = full resolution.
DEFAULT_COARSEN = 6
MIN_CROPLAND_FRAC = 0.5  # a coarse cell counts as cropland when half its 10 m cells are
RETRIES = 3
RETRY_DELAY_S = 5.0

# A loader returns a Dataset with float reflectance `blue`, `red`, `nir` (0..1, NaN = no
# data) and integer `scl` (0 = no data), dims (time, y, x), on the grid of `template`.
S2Loader = Callable[[list[pystac.Item], xr.DataArray], xr.Dataset]
# A search function: (collection, bbox, start, end, query) -> items.
Search = Callable[..., list[pystac.Item]]


# ----------------------------------------------------------------------------- loading
def reflectance_offset(item: pystac.Item) -> int:
    """DN offset to add before scaling: -1000 from processing baseline 04.00, else 0."""
    baseline = item.properties.get("s2:processing_baseline")
    try:
        return BOA_OFFSET_DN if float(baseline) >= BOA_OFFSET_BASELINE else 0
    except (TypeError, ValueError):
        # Unknown baseline: scenes from 25 Jan 2022 onwards carry the offset.
        return BOA_OFFSET_DN if item.datetime and item.datetime.date() >= date(2022, 1, 25) else 0


def stac_s2_loader(chunk_px: int = 2048) -> S2Loader:
    """Return a loader that reads Sentinel-2 L2A bands with odc-stac onto the mask grid.

    Items are grouped by reflectance offset (processing baseline), loaded lazily with
    dask (one time step per solar day), scaled to reflectance and concatenated.
    """
    def load(items: list[pystac.Item], template: xr.DataArray) -> xr.Dataset:
        import odc.stac
        from odc.geo.geobox import GeoBox

        geobox = GeoBox(template.shape, template.rio.transform(), template.rio.crs)
        items = fresh_sign(items)  # avoid HTTP 403 from SAS tokens expiring mid-run
        groups: dict[int, list[pystac.Item]] = {}
        for it in items:
            groups.setdefault(reflectance_offset(it), []).append(it)
        parts = []
        for offset, group in groups.items():
            raw = odc.stac.load(group, bands=list(BANDS.values()), geobox=geobox,
                                groupby="solar_day", chunks={"x": chunk_px, "y": chunk_px})
            raw = raw.rename({d: n for d, n in (("latitude", "y"), ("longitude", "x"))
                              if d in raw.dims})
            ds = xr.Dataset({
                name: (raw[band].where(raw[band] != 0) + offset).astype("float32") / 10_000
                for name, band in BANDS.items() if name != "scl"
            })
            ds["scl"] = raw[BANDS["scl"]]
            parts.append(ds.assign_coords(y=template["y"].values, x=template["x"].values))
        return xr.concat(parts, dim="time").sortby("time")
    return load


# ------------------------------------------------------------------------- computation
def vegetation_indices(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    """NDVI and EVI per observation from reflectance in 0..1.

    EVI = 2.5 (NIR - red) / (NIR + 6 red - 7.5 blue + 1). Values outside -1..1 (EVI blows
    up on bright or very dark surfaces) are set to NaN.
    """
    nir, red, blue = ds["nir"], ds["red"], ds["blue"]
    ndvi = (nir - red) / (nir + red)
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    return {name: v.where((v >= -1) & (v <= 1))
            for name, v in (("NDVI", ndvi), ("EVI", evi))}


def clear_sky(scl: xr.DataArray, cloud_classes: tuple[int, ...]) -> xr.DataArray:
    """True where the SCL class is neither a cloud/shadow class nor no data."""
    return ~scl.isin(list(cloud_classes)) & (scl != SCL_NODATA)


def monthly_composite(ds: xr.Dataset, cloud_classes: tuple[int, ...]) -> dict[str, xr.DataArray]:
    """Per-pixel median over time of clear-sky NDVI and EVI (NaN where never clear)."""
    clear = clear_sky(ds["scl"], cloud_classes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels
        return {name: v.where(clear).median("time", skipna=True)
                for name, v in vegetation_indices(ds).items()}


def coarsen_mask(mask: xr.DataArray, factor: int,
                 min_frac: float = MIN_CROPLAND_FRAC) -> xr.DataArray:
    """Block-average a boolean (y, x) cropland mask by `factor`; True where the cropland
    share is at least `min_frac`. Edge rows/columns that don't fill a block are dropped.
    `factor <= 1` returns the mask unchanged. The CRS is kept, so the result can serve
    as the loader's output grid."""
    if factor <= 1:
        return mask
    crs = mask.rio.crs
    frac = mask.astype("float32").coarsen(y=factor, x=factor, boundary="trim").mean()
    out = frac >= min_frac
    return out.rio.write_crs(crs) if crs is not None else out


def aggregate(
    composites: dict[str, xr.DataArray] | None,
    cropland: xr.DataArray,
    province: str,
    month: pd.Timestamp,
    min_valid_fraction: float,
) -> pd.DataFrame:
    """Province-month statistics over cropland pixels, one row per index.

    `composites=None` (no scenes in the month) gives `valid_frac` 0 and NaN statistics.
    Months with `valid_frac < min_valid_fraction` get NaN mean, median and max.
    """
    crop = np.asarray(cropland.values, dtype=bool)
    n_crop = int(crop.sum())
    rows = []
    for name in INDICES:
        if composites is None or n_crop == 0:
            vals = np.empty(0, dtype="float64")
        else:
            arr = np.asarray(composites[name].values, dtype="float64")
            vals = arr[crop & np.isfinite(arr)]
        n = int(vals.size)
        frac = n / n_crop if n_crop else 0.0
        ok = n > 0 and frac >= min_valid_fraction
        rows.append({
            "province": province, "date": month, "index": name,
            "mean": float(vals.mean()) if ok else np.nan,
            "median": float(np.median(vals)) if ok else np.nan,
            "max": float(vals.max()) if ok else np.nan,
            "valid_frac": frac, "n_pixels": n,
        })
    return _typed(pd.DataFrame(rows, columns=COLUMNS))


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype({"province": "string", "index": "string", "mean": "float64",
                      "median": "float64", "max": "float64", "valid_frac": "float64",
                      "n_pixels": "int64"}).assign(date=pd.to_datetime(df["date"]))


def _with_retry(fn: Callable[[], pd.DataFrame], what: str) -> pd.DataFrame | None:
    """Call `fn` up to RETRIES times with exponential backoff; None if it keeps failing."""
    for attempt in range(RETRIES):
        try:
            return fn()
        except Exception as e:  # network, signing, read errors: retry, then skip
            if attempt == RETRIES - 1:
                log.error("%s failed after %d attempts, skipping: %s", what, RETRIES, e)
                return None
            log.warning("%s failed (%s), retrying", what, e)
            time.sleep(RETRY_DELAY_S * 2 ** attempt)
    return None


# ------------------------------------------------------------------------- extraction
def extract_s2(
    gdf: gpd.GeoDataFrame,
    mask: dict[str, xr.DataArray],
    start: str | date,
    end: str | date,
    *,
    cfg: Config | None = None,
    search: Search | None = None,
    loader: S2Loader | None = None,
    max_scene_cloud: float | None = DEFAULT_MAX_SCENE_CLOUD,
    today: date | None = None,
    coarsen: int = 1,
) -> pd.DataFrame:
    """Monthly Sentinel-2 NDVI/EVI per province over cropland, with SCL cloud masking.

    Args:
        gdf: province polygons from `load_boundaries` (columns `province`, `geometry`).
        mask: `{province: cropland mask}` from `cropland_mask` (boolean (y, x) DataArray
            in EPSG:4326, True on cropland). Its grid is the output grid.
        start, end: ISO dates; every calendar month overlapping the range is processed.
        cfg: project config (collection, SCL classes, `min_valid_fraction`, cache path).
            Defaults to `load_config()`.
        search: STAC search function; defaults to `cropyield.satellite.stac.search_items`.
        loader: scene loader; defaults to `stac_s2_loader()`. Tests pass synthetic ones.
        max_scene_cloud: skip scenes with `eo:cloud_cover` above this percentage
            (None = no filter).
        today: reference date for skipping future months (defaults to today).
        coarsen: read on a grid this many times coarser than the mask (see
            `coarsen_mask`); 1 keeps the mask grid. Each factor has its own cache folder.

    Returns:
        DataFrame with columns `province, date, index, mean, median, max, valid_frac,
        n_pixels`; `date` is the first day of the month, `index` is NDVI or EVI.
    """
    cfg = cfg or load_config()
    search = search or search_items
    loader = loader or stac_s2_loader()
    today = today or date.today()
    p = cfg.pipeline
    cache_dir = p.paths.interim / "cache" / ("s2" if coarsen <= 1 else f"s2_x{coarsen}")
    query = {"eo:cloud_cover": {"lt": max_scene_cloud}} if max_scene_cloud is not None else None
    months = [m for m in pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
              if m.start_time.date() <= today]

    frames = []
    for province, geom in zip(gdf["province"], gdf.to_crs(CRS).geometry):
        if province not in mask:
            raise ValueError(f"No cropland mask for province {province!r}")
        cropland = coarsen_mask(mask[province].astype(bool), coarsen)
        bbox = tuple(float(v) for v in geom.bounds)
        for month in months:
            m_start, m_end = month.start_time.date(), month.end_time.date()
            cache = cache_dir / f"{_slug(province)}_{month.strftime('%Y-%m')}.parquet"
            if cache.exists():
                frames.append(pd.read_parquet(cache))
                continue

            def run(month=month, m_start=m_start, m_end=m_end) -> pd.DataFrame:
                items = search(p.sources.sentinel2_collection, bbox,
                               m_start.isoformat(), m_end.isoformat(), query)
                composites = None
                if items:
                    composites = monthly_composite(loader(items, cropland),
                                                   p.s2_cloud_scl_classes)
                    composites = {k: v.compute() for k, v in composites.items()}
                return aggregate(composites, cropland, province, month.start_time,
                                 p.min_valid_fraction)

            what = f"sentinel-2 {province} {month}"
            log.info(what)
            df = _with_retry(run, what)
            if df is None:
                continue
            if m_end < today:  # the current month is incomplete: don't cache it
                cache.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache.with_suffix(".parquet.part")
                df.to_parquet(tmp, index=False)
                tmp.replace(cache)
            frames.append(df)

    if not frames:
        return _typed(pd.DataFrame(columns=COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    return _typed(out).sort_values(["province", "date", "index"], ignore_index=True)


def save_s2(df: pd.DataFrame, cfg: Config) -> Path:
    """Write the Sentinel-2 table to `data/interim/s2_vi.parquet`."""
    out = cfg.pipeline.paths.interim / "s2_vi.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def load_masks(gdf: gpd.GeoDataFrame, cfg: Config) -> dict[str, xr.DataArray]:
    """Read the cropland masks written by `cropyield.boundaries.save_cropland` (lazily)."""
    import rioxarray

    masks = {}
    for province in gdf["province"]:
        path = cfg.pipeline.paths.interim / "cropland" / f"{_slug(province)}.tif"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing; run `python -m cropyield.boundaries --step cropland` first")
        da = rioxarray.open_rasterio(path, chunks={"x": 2048, "y": 2048})
        masks[province] = (da.squeeze("band", drop=True) == 1)
    return masks


# ------------------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> None:
    """Extract monthly Sentinel-2 NDVI/EVI per province and write data/interim/s2_vi.parquet."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR),
                    help="folder with pipeline.yaml and provinces.yaml (default: repo configs/)")
    ap.add_argument("--start", help="first date, YYYY-MM-DD (default: dates.sentinel2_start)")
    ap.add_argument("--end", help="last date, YYYY-MM-DD (default: dates.end)")
    ap.add_argument("--coarsen", type=int, default=DEFAULT_COARSEN,
                    help=f"read on a grid N times coarser than the 10 m cropland mask "
                         f"(default {DEFAULT_COARSEN}, about 60 m; 1 = full 10 m)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config_dir)
    start = args.start or cfg.pipeline.dates.sentinel2_start.isoformat()
    end = args.end or cfg.pipeline.dates.end.isoformat()
    gdf = gpd.read_parquet(cfg.pipeline.paths.interim / "boundaries.parquet")
    df = extract_s2(gdf, load_masks(gdf, cfg), start, end, cfg=cfg, coarsen=args.coarsen)
    out = save_s2(df, cfg)
    missing = int(df["mean"].isna().sum())
    print(f"wrote {out} ({len(df)} rows, {missing} marked missing)")


if __name__ == "__main__":
    main()
