"""MODIS MOD13Q1 v6.1 NDVI/EVI per province over cropland, from the Planetary Computer.

For each province and 16-day composite:
1. The 10 m WorldCover cropland mask (from `cropyield.boundaries.cropland_mask`) is
   block-averaged to a ~231 m EPSG:4326 grid (25 x 25 WorldCover pixels = 1/480 degree,
   the MODIS 250 m pixel size). A grid cell counts as cropland when at least
   `min_cropland_frac` of it is cropland.
2. NDVI, EVI and pixel reliability are loaded onto that grid. Pixels are kept when the
   reliability band is 0 (good) or 1 (marginal) and the VI value is in its valid range
   (-2000..10000). Values are scaled by 0.0001 to -1..1.
3. One row per province, composite and index (NDVI, EVI) with mean, median, max over
   valid cropland cells, `n_pixels` (valid cells) and `valid_frac` (valid / cropland cells).

`date` is the composite's END date (`end_datetime`, the last day of its 16-day window), and
composites are selected by end date. A cutoff on `date` (quarter end, `--end`) therefore
never admits imagery acquired after the cutoff.

Each (province, composite end date) is cached as Parquet under
`data/interim/cache/modis/<province>/<date>.parquet`; re-runs skip cached chunks. A chunk
that still fails after retries is logged and skipped (not cached), so the next run tries it
again. Delete the cache folder after changing `min_cropland_frac` or the quality rules.

Usage (fetches real data; needs data/interim/boundaries.parquet and cropland masks from
`python -m cropyield.boundaries`):
    python -m cropyield.satellite.modis --start 2020-01-01 --end 2020-12-31
    python -m cropyield.satellite.modis --config-dir configs   # full configured range
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pystac
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from cropyield.config import DEFAULT_CONFIG_DIR, Config, load_config
from cropyield.satellite.stac import (Bbox, cache_dir, call_with_retry, chunk_path, fresh_sign,
                                      item_end_date, open_catalog, read_chunk, search_items,
                                      write_chunk)

log = logging.getLogger(__name__)

CRS = "EPSG:4326"
NDVI = "250m_16_days_NDVI"
EVI = "250m_16_days_EVI"
RELIABILITY = "250m_16_days_pixel_reliability"
BANDS = (NDVI, EVI, RELIABILITY)
INDEX_BANDS = {"NDVI": NDVI, "EVI": EVI}
SCALE = 0.0001
VALID_RANGE = (-2000, 10000)
RELIABLE = (0, 1)  # 0 good, 1 marginal; -1 fill, 2 snow/ice, 3 cloudy are dropped
MODIS_RES_DEG = 1 / 480  # 231.656 m MODIS "250 m" pixel at the equator, in degrees
COLUMNS = ["province", "date", "index", "mean", "median", "max", "valid_frac", "n_pixels"]

# A loader returns a Dataset with the BANDS as variables, dims (y, x) or (time, y, x) and
# x/y coordinates in EPSG:4326, covering `bbox` at roughly `resolution` degrees.
ModisLoader = Callable[[list[pystac.Item], Bbox, float], xr.Dataset]


# ------------------------------------------------------------------------ cropland grid
def cropland_fraction(mask: xr.DataArray, res_deg: float = MODIS_RES_DEG) -> xr.DataArray:
    """Block-average a boolean (y, x) cropland mask to a coarser grid of about `res_deg`.

    The block size is the whole number of mask pixels closest to `res_deg`. Edges are
    padded with non-cropland. Returns a float (y, x) DataArray of cropland fractions 0..1
    with cell-centre coordinates.
    """
    mask = mask.transpose("y", "x")
    x, y = mask["x"].values, mask["y"].values
    dx = float(x[1] - x[0]) if x.size > 1 else float(mask.rio.resolution()[0])
    dy = float(y[1] - y[0]) if y.size > 1 else float(mask.rio.resolution()[1])
    f = max(1, round(res_deg / abs(dx)))
    m = np.asarray(mask.values, dtype=bool)
    ny, nx = m.shape
    m = np.pad(m, ((0, -ny % f), (0, -nx % f)))
    by, bx = m.shape[0] // f, m.shape[1] // f
    frac = m.reshape(by, f, bx, f).sum(axis=(1, 3), dtype=np.uint32) / f ** 2
    xs = x[0] + dx * ((f - 1) / 2 + f * np.arange(bx))
    ys = y[0] + dy * ((f - 1) / 2 + f * np.arange(by))
    out = xr.DataArray(frac, coords={"y": ys, "x": xs}, dims=("y", "x"), name="cropland_frac")
    return out.rio.write_crs(CRS)


def _grid_bbox(grid: xr.DataArray) -> Bbox:
    """Bounds of a grid's cell centres, padded by one cell on each side."""
    x, y = grid["x"].values, grid["y"].values
    r = abs(float(x[1] - x[0])) if x.size > 1 else MODIS_RES_DEG
    return (float(x.min()) - r, float(y.min()) - r, float(x.max()) + r, float(y.max()) + r)


# ------------------------------------------------------------------------ statistics
def composite_stats(
    ds: xr.Dataset, cropland: xr.DataArray, reliable: tuple[int, ...] = RELIABLE
) -> list[dict]:
    """NDVI/EVI statistics for one composite over the cropland cells of one province.

    Args:
        ds: BANDS on the same (y, x) grid as `cropland`, optionally with a `time` dim
            (several tiles of the same composite; they are merged).
        cropland: boolean (y, x) cropland cells.
        reliable: pixel reliability values to keep.

    Returns:
        One dict per index with keys `index, mean, median, max, valid_frac, n_pixels`.
        Stats are NaN when no cropland cell is valid.
    """
    crop = cropland.values.astype(bool)
    n_crop = int(crop.sum())
    ok = ds[RELIABILITY].isin(list(reliable))
    rows = []
    for index, band in INDEX_BANDS.items():
        raw = ds[band]
        keep = ok & (raw >= VALID_RANGE[0]) & (raw <= VALID_RANGE[1])
        scaled = xr.where(keep, raw.astype("float64") * SCALE, np.nan)
        if "time" in scaled.dims:
            scaled = scaled.max("time", skipna=True)
        vals = scaled.transpose("y", "x").values[crop]
        vals = vals[~np.isnan(vals)]
        n = int(vals.size)
        rows.append({
            "index": index,
            "mean": float(vals.mean()) if n else np.nan,
            "median": float(np.median(vals)) if n else np.nan,
            "max": float(vals.max()) if n else np.nan,
            "valid_frac": n / n_crop if n_crop else np.nan,
            "n_pixels": n,
        })
    return rows


# ------------------------------------------------------------------------ loading
def stac_modis_loader(items: list[pystac.Item], bbox: Bbox, resolution: float) -> xr.Dataset:
    """Load MOD13Q1 NDVI, EVI and pixel reliability for `items` with odc-stac."""
    import odc.stac

    # The Planetary Computer STAC metadata gives the pixel-reliability band dtype int8 but
    # nodata 255, which overflows int8 when odc-stac allocates the output. All three
    # bands fit in int16 (NDVI/EVI are int16 already), so load everything as int16.
    items = fresh_sign(items)  # avoid HTTP 403 from SAS tokens expiring mid-run
    ds = odc.stac.load(items, bands=list(BANDS), bbox=bbox, crs=CRS, resolution=resolution,
                       resampling="nearest", dtype="int16")
    if "longitude" in ds.dims:
        ds = ds.rename({"longitude": "x", "latitude": "y"})
    return ds


def _is_terra(item: pystac.Item) -> bool:
    """MOD13Q1 is the Terra product; the collection may also hold Aqua (MYD13Q1) items."""
    platform = str(item.properties.get("platform", "")).casefold()
    return platform == "terra" or item.id.upper().startswith("MOD13Q1")


def _year_windows(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[str, str]]:
    """Split [start, end] into calendar-year windows of ISO date strings."""
    out = []
    for year in range(start.year, end.year + 1):
        a = max(start, pd.Timestamp(year=year, month=1, day=1))
        b = min(end, pd.Timestamp(year=year, month=12, day=31))
        out.append((a.date().isoformat(), b.date().isoformat()))
    return out


def extract_modis(
    gdf: gpd.GeoDataFrame,
    mask: dict[str, xr.DataArray],
    start: str,
    end: str,
    *,
    cfg: Config | None = None,
    loader: ModisLoader | None = None,
    min_cropland_frac: float = 0.5,
) -> pd.DataFrame:
    """MOD13Q1 NDVI/EVI statistics per province and 16-day composite over cropland.

    Args:
        gdf: province polygons from `load_boundaries` (column `province`); only the
            province names are used here, since `mask` is already clipped to each polygon.
        mask: `{province: boolean (y, x) cropland mask in EPSG:4326}` from `cropland_mask`.
        start, end: ISO dates; composites whose end date falls in [start, end] are used,
            so no imagery after `end` enters the table.
        cfg: project config (collection, STAC URL, cache path); defaults to `configs/`.
        loader: function loading BANDS for items on a grid; defaults to odc-stac. Tests
            pass a synthetic loader.
        min_cropland_frac: share of a ~231 m cell that must be cropland for it to count.

    Returns:
        DataFrame with columns `province, date, index, mean, median, max, valid_frac,
        n_pixels` (`date` = composite end date), sorted by province, date and index. Failed chunks are missing.
    """
    cfg = cfg or load_config()
    loader = loader or stac_modis_loader
    src = cfg.pipeline.sources
    root = cache_dir(cfg, "modis")
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    missing = [p for p in gdf["province"] if p not in mask]
    if missing:
        raise ValueError(f"No cropland mask for province(s): {', '.join(missing)}")

    catalog = None
    frames: list[pd.DataFrame] = []
    failed = 0
    for province in gdf["province"]:
        frac = cropland_fraction(mask[province])
        cropland = frac >= min_cropland_frac
        bbox = _grid_bbox(frac)
        res = abs(float(frac["x"].values[1] - frac["x"].values[0])) \
            if frac.sizes["x"] > 1 else MODIS_RES_DEG
        log.info("MODIS %s: %d cropland cells", province, int(cropland.sum()))
        for w0, w1 in _year_windows(t0, t1):
            try:
                if catalog is None:
                    catalog = open_catalog(src.stac_url)
                items = search_items(src.modis_collection, bbox, w0, w1, catalog=catalog)
            except Exception as e:
                log.error("MODIS %s %s..%s: search failed, skipping (%s)", province, w0, w1, e)
                failed += 1
                continue
            by_date: dict[pd.Timestamp, list[pystac.Item]] = {}
            for it in items:
                d = item_end_date(it)
                if _is_terra(it) and pd.Timestamp(w0) <= d <= pd.Timestamp(w1):
                    by_date.setdefault(d, []).append(it)
            for d in sorted(by_date):
                path = chunk_path(root, province, d.date().isoformat())
                cached = read_chunk(path)
                if cached is not None:
                    frames.append(cached)
                    continue
                try:
                    ds = call_with_retry(lambda: loader(by_date[d], bbox, res),
                                         label=f"load MODIS {province} {d.date()}")
                    ds = ds.reindex_like(frac, method="nearest", tolerance=res)
                    rows = composite_stats(ds, cropland)
                except Exception as e:
                    log.error("MODIS %s %s: chunk failed, skipping (%s)", province, d.date(), e)
                    failed += 1
                    continue
                chunk = pd.DataFrame([{"province": province, "date": d, **r} for r in rows],
                                     columns=COLUMNS)
                write_chunk(chunk, path)
                frames.append(chunk)
                log.info("  %s %s done", province, d.date())
    if failed:
        log.warning("MODIS: %d chunk(s) failed; re-run to retry them", failed)
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.concat(frames, ignore_index=True)[COLUMNS]
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["province", "date", "index"], ignore_index=True)


def save_modis(df: pd.DataFrame, cfg: Config) -> Path:
    """Write the MODIS table to `<paths.interim>/modis_vi.parquet`."""
    out = cfg.pipeline.paths.interim / "modis_vi.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


# ------------------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> None:
    """Extract MODIS NDVI/EVI per province and write data/interim/modis_vi.parquet."""
    from cropyield.boundaries import load_cropland

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR),
                    help="folder with pipeline.yaml and provinces.yaml (default: repo configs/)")
    ap.add_argument("--start", help="first composite end date, YYYY-MM-DD (default: dates.modis_start)")
    ap.add_argument("--end", help="last composite end date, YYYY-MM-DD (default: dates.end)")
    ap.add_argument("--min-cropland-frac", type=float, default=0.5,
                    help="share of a 231 m cell that must be cropland (default: 0.5)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config_dir)
    start = args.start or cfg.pipeline.dates.modis_start.isoformat()
    end = args.end or cfg.pipeline.dates.end.isoformat()
    bpath = cfg.pipeline.paths.interim / "boundaries.parquet"
    if not bpath.exists():
        raise SystemExit(f"{bpath} not found; run `python -m cropyield.boundaries` first")
    gdf = gpd.read_parquet(bpath)
    masks = load_cropland(cfg, list(gdf["province"]))
    df = extract_modis(gdf, masks, start, end, cfg=cfg,
                       min_cropland_frac=args.min_cropland_frac)
    out = save_modis(df, cfg)
    print(f"wrote {out} ({len(df)} rows, {df['date'].nunique()} composites)")


if __name__ == "__main__":
    main()
