"""App data export (phase 3, Requirement 3; task 3).

`cropyield export` writes everything the static AniSight site reads:

- `app/data/predictions.json`  out-of-fold history per crop and unit (2012+)
- `app/data/forecast.json`     latest logged live forecast per unit and crop, with
                               conditions vs normal for that quarter
- `app/data/units.geojson`     simplified reporting-unit polygons (incl. combined units)
- `app/data/live_accuracy.json` live scoring rows + summary
- `app/data/findings.json`     selection table, satellite comparison, interval coverage,
                               rain shift, caveats, counts (all read from reports/ files)
- `app/data/meta.json`         generation time, commit, data vintage, models, sources
- `app/assets/figures/*.png`   copies of the Phase 2 report figures

Inputs: `data/processed/predictions.parquet`, `data/processed/features.parquet`,
`data/interim/boundaries.parquet`, `reports/{forecasts,selection,experiments}.csv`,
`reports/live_accuracy*.csv` (when `cropyield score` has run), `reports/phase2_results.md`
and `reports/figures/`.

Definitions used by the app (the site never derives numbers itself):

- `normal`: historical normal yield of a row = mean PSA yield of the same reporting
  unit, crop and quarter over years `start_year <= year' < Y` (expanding; `start_year`
  is the MODIS-era start of `modeling.eras`, the first year of the model's training
  rows), null with fewer than MIN_NORMAL_YEARS earlier years. A quarter is never
  compared with a mean that includes itself or later years.
- `pct`: predicted (B3) yield vs `normal`, in % (map colour).
- `ndvi_anom` / `evi_anom`: MODIS NDVI / EVI peak minus its expanding mean over earlier
  years (`select.expanding_anomalies`); `rain_anom` in mm vs the fixed 1981-2000 ERA5
  climatology and `rain_pct` = rain_anom / (rain_total - rain_anom) in %.

All JSON is written with `allow_nan=False`; missing values are null.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import mapping

from cropyield.boundaries import reporting_unit_geometries
from cropyield.config import DEFAULT_CONFIG_DIR, Config, load_config
from cropyield.models.baselines import GROUP
from cropyield.models.data import UNIT_KEYS, load_modeling_table
from cropyield.models.evaluate import EXPERIMENT_KEY, experiments_path
from cropyield.models.forecast import (FORECAST_COLUMNS, STATUS_AWAITING, STATUS_SCORED,
                                       _as_bool, first_seen_path, forecasts_path,
                                       git_commit, live_accuracy_path, live_summary_path,
                                       load_first_seen, new_first_seen, quarter_label,
                                       score_table, summarize_live)
from cropyield.models.report import figures_dir, report_path
from cropyield.models.select import (expanding_anomalies, interval_coverage,
                                     predictions_path, selection_path)

APP_DATA_FILES = ("predictions.json", "forecast.json", "units.geojson",
                  "live_accuracy.json", "findings.json", "meta.json")
# Requirement 3.3 / app steering: all of app/data/ under 1.5 MB (decimal, 1,500,000 B).
DATA_BUDGET_BYTES = 1_500_000
# design.md: units.geojson target <= 400 KB.
GEOJSON_BUDGET_BYTES = 400_000
SIMPLIFY_TOLERANCE_DEG = 0.002   # ~220 m at 7 deg N; province-scale map
COORD_DECIMALS = 4               # ~11 m
MIN_NORMAL_YEARS = 3
AREA_NOTE = "assumes the same harvested area as the same quarter last year"
CONDITIONS_NOTE = "context, not used in the forecast"
NDVI_COL, EVI_COL = "modis_ndvi_peak_anom", "modis_evi_peak_anom"

SOURCES: list[dict[str, str]] = [
    {"name": "Philippine Statistics Authority (PSA) OpenSTAT",
     "use": "palay and corn production, harvested area and yield by province and quarter",
     "url": "https://openstat.psa.gov.ph/",
     "licence": "PSA open data terms; free to use with attribution",
     "attribution": "Source: Philippine Statistics Authority (PSA), OpenSTAT"},
    {"name": "NASA MODIS MOD13Q1 v061 (Terra Vegetation Indices 16-Day 250 m)",
     "use": "NDVI/EVI conditions vs normal",
     "url": "https://doi.org/10.5067/MODIS/MOD13Q1.061",
     "licence": "NASA EOSDIS data: no restrictions on use",
     "attribution": "Didan, K. (2021). MODIS/Terra Vegetation Indices 16-Day L3 Global "
                    "250m SIN Grid V061. NASA EOSDIS Land Processes DAAC."},
    {"name": "Copernicus Sentinel-2 Level-2A",
     "use": "higher-resolution NDVI/EVI (Phase 2 comparison)",
     "url": "https://sentinels.copernicus.eu/",
     "licence": "Copernicus Sentinel data terms (free, full and open)",
     "attribution": "Contains modified Copernicus Sentinel data [2017-2026]"},
    {"name": "ESA WorldCover 10 m v200",
     "use": "cropland mask",
     "url": "https://esa-worldcover.org/",
     "licence": "CC BY 4.0",
     "attribution": "© ESA WorldCover project / Contains modified Copernicus Sentinel "
                    "data processed by the ESA WorldCover consortium"},
    {"name": "geoBoundaries (gbOpen PHL ADM2/ADM3)",
     "use": "province and Davao City boundaries",
     "url": "https://www.geoboundaries.org/",
     "licence": "CC BY 4.0 (gbOpen); see the boundary's metadata for its source licence",
     "attribution": "Runfola, D. et al. (2020) geoBoundaries: A global database of "
                    "political administrative boundaries. PLoS ONE 15(4): e0231866."},
    {"name": "Open-Meteo Historical Weather API (ERA5 reanalysis)",
     "use": "daily rainfall, conditions vs normal",
     "url": "https://open-meteo.com/",
     "licence": "CC BY 4.0 (non-commercial free tier); ERA5 under the Copernicus licence",
     "attribution": "Weather data by Open-Meteo.com; Hersbach, H. et al. (2023) ERA5 "
                    "hourly data on single levels, Copernicus Climate Change Service (C3S)"},
    {"name": "Microsoft Planetary Computer",
     "use": "STAC access to MODIS, Sentinel-2 and WorldCover",
     "url": "https://planetarycomputer.microsoft.com/",
     "licence": "access service; each dataset keeps its own licence",
     "attribution": "Data accessed through the Microsoft Planetary Computer"},
    {"name": "OpenStreetMap standard tiles",
     "use": "background map tiles",
     "url": "https://www.openstreetmap.org/copyright",
     "licence": "OpenStreetMap data: ODbL (https://www.openstreetmap.org/copyright)",
     "attribution": "© OpenStreetMap contributors",
     "note": "Tiles from tile.openstreetmap.org are subject to the OSMF Tile Usage "
             "Policy (https://operations.osmfoundation.org/policies/tiles/)"},
]


# --------------------------------------------------------------------------- helpers

def _num(x: Any, digits: int | None = None) -> float | int | None:
    """Finite number (optionally rounded) or None for NaN/Inf/None/NA."""
    if x is None or x is pd.NA:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, digits) if digits is not None else v


def jsonable(obj: Any) -> Any:
    """Recursively turn numpy/pandas values into JSON types; NaN/Inf -> None."""
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _num(obj)
    if obj is None or obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def write_json(path: Path, obj: Any) -> Path:
    """Compact JSON, NaN-free (`allow_nan=False` raises if one slipped through)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), allow_nan=False, ensure_ascii=False,
                               separators=(",", ":")), encoding="utf-8")
    return path


def _qi(year: Any, quarter: Any) -> int:
    return int(year) * 4 + int(quarter) - 1


def _qlabel(qi: int) -> str:
    return quarter_label(qi // 4, qi % 4 + 1)


def _stamp(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _flags(unit: str, *, b0_fallback: bool, area: Any, area_level: Any,
           outlier: bool = False) -> list[str]:
    out = []
    if "+" in unit:
        out.append("combined_unit")
    if b0_fallback:
        out.append("b0_fallback")
    if _num(area) is None:
        out.append("no_area_forecast")
    elif str(area_level) == "unit_quarter_mean":
        out.append("area_unit_mean")
    if outlier:
        out.append("outlier")
    return out


def rain_pct(rain_total: pd.Series, rain_anom: pd.Series) -> pd.Series:
    """Rainfall vs normal in %: anom / (total - anom); NaN where the normal is <= 0."""
    normal = rain_total.astype(float) - rain_anom.astype(float)
    return (100.0 * rain_anom.astype(float) / normal).where(normal > 0)


def historical_normal(history: pd.DataFrame, targets: pd.DataFrame, start_year: int,
                      min_years: int = MIN_NORMAL_YEARS) -> pd.DataFrame:
    """Expanding historical normal yield per target row (see module docstring).

    `history`: rows with `yield_t_ha` (PSA). `targets`: rows with UNIT_KEYS. Returns
    UNIT_KEYS + `normal` (mean yield over years start_year <= y' < year of the same
    unit/crop/quarter; NaN with fewer than `min_years` such years) + `normal_n_years`.
    """
    h = history.loc[(history["year"] >= start_year) & history["yield_t_ha"].notna(),
                    [*GROUP, "year", "yield_t_ha"]].rename(columns={"year": "h_year"})
    t = targets[UNIT_KEYS].drop_duplicates().copy()
    t["year"] = t["year"].astype(int)
    t["quarter"] = t["quarter"].astype(int)
    m = t.merge(h, on=GROUP, how="inner")
    m = m.loc[m["h_year"] < m["year"]]
    agg = (m.groupby(UNIT_KEYS).agg(normal=("yield_t_ha", "mean"),
                                    normal_n_years=("h_year", "nunique")).reset_index())
    out = t.merge(agg, on=UNIT_KEYS, how="left")
    out["normal_n_years"] = out["normal_n_years"].fillna(0).astype(int)
    out["normal"] = out["normal"].where(out["normal_n_years"] >= min_years)
    return out


def _pct(pred: Any, normal: Any) -> float | None:
    p, n = _num(pred), _num(normal)
    if p is None or n is None or n <= 0:
        return None
    return round(100.0 * (p / n - 1.0), 1)


# ------------------------------------------------------------------ predictions.json

def build_predictions_json(pred: pd.DataFrame, history: pd.DataFrame,
                           features: pd.DataFrame | None, start_year: int,
                           first_year: int) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """`{crop: {unit: [{y, q, actual, pred, lo, hi, b0, b2, normal, pct, rain_anom,
    rain_pct, ndvi_anom, evi_anom, flags[]}]}}` for years >= first_year."""
    p = pred.loc[pred["year"] >= first_year].copy()
    norm = historical_normal(history, p, start_year)
    p = p.merge(norm, on=UNIT_KEYS, how="left")
    if features is not None and "rain_total" in features:
        f = features[[*UNIT_KEYS, "rain_total"]].drop_duplicates(UNIT_KEYS)
        p = p.merge(f, on=UNIT_KEYS, how="left")
        p["rain_pct"] = rain_pct(p["rain_total"], p["rain_anom"])
    else:
        p["rain_pct"] = np.nan
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in p.sort_values(UNIT_KEYS).itertuples(index=False):
        out.setdefault(r.crop, {}).setdefault(r.reporting_unit, []).append({
            "y": int(r.year), "q": int(r.quarter),
            "actual": _num(r.actual_yield_t_ha, 3), "pred": _num(r.pred_yield_t_ha, 3),
            "lo": _num(r.pred_lo_t_ha, 3), "hi": _num(r.pred_hi_t_ha, 3),
            "b0": _num(r.b0_pred_yield_t_ha, 3), "b2": _num(r.b2_pred_yield_t_ha, 3),
            "normal": _num(r.normal, 3), "pct": _pct(r.pred_yield_t_ha, r.normal),
            "rain_anom": _num(r.rain_anom, 1), "rain_pct": _num(r.rain_pct, 1),
            "ndvi_anom": _num(getattr(r, NDVI_COL), 3),
            "evi_anom": _num(getattr(r, EVI_COL), 3),
            "flags": _flags(r.reporting_unit, b0_fallback=bool(r.b0_fallback),
                            area=r.predicted_area_ha, area_level=r.area_fallback_level,
                            outlier=bool(r.is_outlier))})
    return out


# --------------------------------------------------------------------- forecast.json

def read_forecasts(path: Path) -> pd.DataFrame:
    """forecasts.csv, or an empty frame with FORECAST_COLUMNS."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=list(FORECAST_COLUMNS))
    return pd.read_csv(path, dtype={"reporting_unit": str, "crop": str,
                                    "forecast_id": str, "backfill": str})


def latest_forecast(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Rows of the latest non-backfill target quarter, newest forecast per unit/crop."""
    if forecasts.empty:
        return forecasts
    f = forecasts.loc[~_as_bool(forecasts["backfill"])].copy()
    if f.empty:
        return f
    qi = f["year"].astype(int) * 4 + f["quarter"].astype(int) - 1
    f = f.loc[qi == qi.max()]
    f["_made"] = pd.to_datetime(f["made_at_utc"], utc=True)
    f = (f.sort_values(["_made", "forecast_id"])
         .groupby(["reporting_unit", "crop"], sort=False).tail(1))
    return f.drop(columns="_made").sort_values(["crop", "reporting_unit"])


def target_conditions(table: pd.DataFrame, features: pd.DataFrame | None,
                      targets: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """Conditions vs normal for target rows (UNIT_KEYS) from features.parquet.

    NDVI/EVI anomalies use the same expanding rule as predictions.parquet: history is
    the modeling-table rows with start_year <= year < target year (`expanding_anomalies`
    on those rows plus the target rows). Rows absent from `features` get NaN.
    """
    cols = [*UNIT_KEYS, NDVI_COL, EVI_COL, "rain_anom", "rain_pct", "window_start",
            "window_end"]
    t = targets[UNIT_KEYS].drop_duplicates().copy()
    if features is None or features.empty:
        return t.reindex(columns=cols)
    src = ["modis_ndvi_peak", "modis_evi_peak"]
    f = features.drop_duplicates(UNIT_KEYS)
    tf = t.merge(f, on=UNIT_KEYS, how="left")
    hist = table.loc[table["year"] >= start_year]
    known = pd.MultiIndex.from_frame(tf[UNIT_KEYS])
    hist = hist.loc[~pd.MultiIndex.from_frame(hist[UNIT_KEYS]).isin(known)]
    both = pd.concat([hist[[*UNIT_KEYS, *src]], tf[[*UNIT_KEYS, *src]]], ignore_index=True)
    an = expanding_anomalies(both, src).iloc[len(hist):].reset_index(drop=True)
    tf[NDVI_COL] = an[f"{src[0]}_anom"].to_numpy()
    tf[EVI_COL] = an[f"{src[1]}_anom"].to_numpy()
    for c in ("rain_anom", "rain_total", "window_start", "window_end"):
        if c not in tf:
            tf[c] = np.nan
    tf["rain_pct"] = rain_pct(tf["rain_total"], tf["rain_anom"])
    return tf[cols]


def _date_str(x: Any) -> str | None:
    if x is None or (np.isscalar(x) and pd.isna(x)) or x is pd.NaT:
        return None
    try:
        return pd.Timestamp(x).date().isoformat()
    except (TypeError, ValueError):
        return None


def build_forecast_json(forecasts: pd.DataFrame, table: pd.DataFrame,
                        features: pd.DataFrame | None, start_year: int,
                        data_end: Any = None) -> dict[str, Any]:
    """Latest live forecast per unit and crop with normal, % vs normal and conditions.

    `data_end` (pipeline `dates.end`): last date of downloaded imagery/rainfall. When a
    target window ends after it, conditions get `partial_window: true` (the NDVI/EVI
    peak then covers only part of the window, so read it with care).
    """
    end = _date_str(data_end)
    f = latest_forecast(forecasts)
    base: dict[str, Any] = {"target_quarter": None, "forecast_ids": [], "data_vintage": None,
                            "area_note": AREA_NOTE, "conditions_note": CONDITIONS_NOTE,
                            "forecasts": {}}
    if f.empty:
        return base
    f = f.copy()
    f["year"] = f["year"].astype(int)
    f["quarter"] = f["quarter"].astype(int)
    norm = historical_normal(table, f, start_year)
    cond = target_conditions(table, features, f, start_year)
    f = f.merge(norm, on=UNIT_KEYS, how="left").merge(cond, on=UNIT_KEYS, how="left")
    out: dict[str, dict[str, Any]] = {}
    for r in f.itertuples(index=False):
        out.setdefault(r.crop, {})[r.reporting_unit] = {
            "forecast_id": r.forecast_id, "made_at_utc": r.made_at_utc,
            "data_vintage": r.data_vintage, "model": r.model,
            "yield": _num(r.b3_yield, 3), "lo": _num(r.b3_lo80, 3),
            "hi": _num(r.b3_hi80, 3), "b0": _num(r.b0_yield, 3), "b2": _num(r.b2_yield, 3),
            "area_ha": _num(r.predicted_area_ha, 0),
            "area_source": (str(r.area_fallback_level)
                            if _num(r.predicted_area_ha) is not None else None),
            "production_mt": _num(r.pred_production_mt, 0),
            "production_lo_mt": _num(r.pred_production_lo_mt, 0),
            "production_hi_mt": _num(r.pred_production_hi_mt, 0),
            "normal": _num(r.normal, 3), "normal_n_years": int(r.normal_n_years),
            "pct": _pct(r.b3_yield, r.normal),
            "conditions": {"rain_anom": _num(r.rain_anom, 1),
                           "rain_pct": _num(r.rain_pct, 1),
                           "ndvi_anom": _num(getattr(r, NDVI_COL), 3),
                           "evi_anom": _num(getattr(r, EVI_COL), 3),
                           "window_start": _date_str(r.window_start),
                           "window_end": _date_str(r.window_end),
                           "data_through": end,
                           "partial_window": bool(
                               end is not None and _date_str(r.window_end) is not None
                               and _date_str(r.window_end) > end)},
            "flags": _flags(r.reporting_unit, b0_fallback=bool(_as_bool(
                pd.Series([r.b0_fallback])).iloc[0]), area=r.predicted_area_ha,
                area_level=r.area_fallback_level)}
    base.update(target_quarter=str(f["target_quarter"].iloc[0]),
                forecast_ids=sorted(f["forecast_id"].astype(str).unique()),
                data_vintage=max(f["data_vintage"].astype(str)), forecasts=out)
    return base


# ---------------------------------------------------------------------- units.geojson

def _polygonal(g: Any) -> Any:
    """Keep only the polygonal parts of `g` (after make_valid / set_precision)."""
    if g is None or g.is_empty:
        return g
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    parts = [p for p in getattr(g, "geoms", []) if p.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.union_all(parts) if parts else shapely.Polygon()


def _clean(g: Any, decimals: int) -> Any:
    g = shapely.set_precision(g, 10.0 ** -decimals)
    if not g.is_valid:
        g = shapely.make_valid(g)
    g = _polygonal(g)
    if hasattr(shapely, "orient_polygons"):
        g = shapely.orient_polygons(g)  # RFC 7946: exterior rings counter-clockwise
    return g


def simplify_provinces(gdf: gpd.GeoDataFrame, tolerance: float = SIMPLIFY_TOLERANCE_DEG,
                       decimals: int = COORD_DECIMALS) -> gpd.GeoDataFrame:
    """Simplify province polygons (EPSG:4326), snapping to a 10^-decimals grid.

    Uses `shapely.coverage_simplify` (shared edges simplified once, so neighbours keep
    no gaps or overlaps) when it returns valid polygons, else per-polygon
    topology-preserving simplify. (The real boundaries are not a strictly valid coverage:
    the Davao City cut-out leaves unmatched vertices along its edge with Davao del Sur,
    but they do not overlap and coverage_simplify still returns valid polygons.)
    """
    gdf = gdf.to_crs(4326)
    geoms = np.asarray(gdf.geometry.values, dtype=object)
    simp = None
    if hasattr(shapely, "coverage_simplify"):
        with np.errstate(all="ignore"):
            cand = shapely.coverage_simplify(geoms, tolerance)
        if all(shapely.is_valid(cand)) and not any(shapely.is_empty(cand)):
            simp = cand
    if simp is None:
        simp = shapely.simplify(geoms, tolerance, preserve_topology=True)
    out = gdf[["province"]].copy()
    out = gpd.GeoDataFrame(out, geometry=[_clean(g, decimals) for g in simp], crs=4326)
    return out


def _round_coords(obj: Any, decimals: int) -> Any:
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (int, float)):
            return [round(float(v), decimals) for v in obj]
        return [_round_coords(v, decimals) for v in obj]
    return obj


def build_units_geojson(boundaries: gpd.GeoDataFrame, cfg: Config,
                        spans: Mapping[str, tuple[int, int]],
                        tolerance: float = SIMPLIFY_TOLERANCE_DEG,
                        decimals: int = COORD_DECIMALS) -> dict[str, Any]:
    """FeatureCollection, one feature per unit in `spans` (unit -> (first qi, last qi)).

    Single provinces come from the simplified `boundaries`; combined units from
    `boundaries.reporting_unit_geometries` on the simplified polygons. Properties:
    {unit, members[], first, last} with quarters as "YYYYQn" (the quarters the unit
    appears in predictions.json / forecast.json). Raises if a geometry is invalid or
    empty after simplification, or a unit has no polygon.
    """
    simp = simplify_provinces(boundaries, tolerance, decimals)
    by_unit: dict[str, tuple[tuple[str, ...], Any]] = {
        p: ((p,), g) for p, g in zip(simp["province"], simp.geometry)}
    combined = reporting_unit_geometries(simp, cfg)
    for r in combined.itertuples(index=False):
        by_unit[r.reporting_unit] = (tuple(r.members), _clean(r.geometry, decimals))
    feats = []
    for unit in sorted(spans):
        if unit not in by_unit:
            raise ValueError(f"no polygon for reporting unit {unit!r}")
        members, g = by_unit[unit]
        if g is None or g.is_empty or not g.is_valid:
            raise ValueError(f"{unit}: geometry empty or invalid after simplification")
        geom = mapping(g)
        geom = {"type": geom["type"],
                "coordinates": _round_coords(geom["coordinates"], decimals)}
        first, last = spans[unit]
        feats.append({"type": "Feature",
                      "properties": {"unit": unit, "members": list(members),
                                     "first": _qlabel(first), "last": _qlabel(last)},
                      "geometry": geom})
    return {"type": "FeatureCollection", "features": feats}


def unit_spans(predictions: Mapping[str, Mapping[str, list[dict[str, Any]]]],
               forecast: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """unit -> (first, last) quarter index over predictions.json and forecast.json."""
    spans: dict[str, list[int]] = {}
    for units in predictions.values():
        for unit, rows in units.items():
            for r in rows:
                spans.setdefault(unit, []).append(_qi(r["y"], r["q"]))
    tq = forecast.get("target_quarter")
    if tq:
        qi = _qi(tq[:4], tq[-1])
        for units in forecast.get("forecasts", {}).values():
            for unit in units:
                spans.setdefault(unit, []).append(qi)
    return {u: (min(v), max(v)) for u, v in spans.items()}


# ------------------------------------------------------------------ live_accuracy.json

_PROD_COLS = ("production", "_prod_")


def _records(df: pd.DataFrame, cols: Sequence[str] | None = None,
             digits: int = 3, prod_digits: int = 0) -> list[dict[str, Any]]:
    """Rows as dicts; floats rounded (production columns to `prod_digits`)."""
    d = df[list(cols)] if cols is not None else df
    out = []
    for rec in d.to_dict(orient="records"):
        row = {}
        for k, v in rec.items():
            if isinstance(v, (float, np.floating)):
                nd = prod_digits if any(s in k for s in _PROD_COLS) else digits
                row[k] = _num(v, nd)
            else:
                row[k] = jsonable(v)
        out.append(row)
    return out


def build_live_accuracy_json(cfg: Config, forecasts: pd.DataFrame, table: pd.DataFrame,
                             now: datetime) -> dict[str, Any]:
    """Rows of live_accuracy.csv + per-crop summary.

    When `cropyield score` has not run (no live_accuracy.csv), the rows are computed in
    memory with the forecast module's scoring functions and the existing first-seen
    store (`load_first_seen` + `new_first_seen`); nothing is written, so the first-seen
    store is not stamped by export.
    """
    la, ls = live_accuracy_path(cfg), live_summary_path(cfg)
    if la.exists() and la.stat().st_size > 0 and ls.exists():
        scored = pd.read_csv(la, dtype={"reporting_unit": str, "crop": str})
        summary = pd.read_csv(ls)
        source = f"{la.parent.name}/{la.name}"
        scored_at = (str(summary["scored_at_utc"].iloc[0])
                     if len(summary) and "scored_at_utc" in summary else None)
    else:
        store = load_first_seen(first_seen_path(cfg))
        new = new_first_seen(table, store, now)
        store = pd.concat([store, new], ignore_index=True) if len(store) else new
        scored = score_table(forecasts, table, store, now)
        summary = summarize_live(scored, list(cfg.pipeline.crops),
                                 seed=int(cfg.pipeline.random_seed))
        source = ("computed in memory by cropyield export; cropyield score has not run, "
                  "so no PSA first-seen values are recorded yet")
        scored_at = None
    rows = scored.drop(columns=[c for c in ("scored_at_utc",) if c in scored])
    crop_sum = summary.loc[summary["level"] == "crop"] if len(summary) else summary
    keep = [c for c in ("basis", "crop", "n_quarters", "n", "status", "b3_mae", "b0_mae",
                        "b2_mae", "coverage80", "b3_vs_b2_mae_diff", "b3_vs_b2_ci_low",
                        "b3_vs_b2_ci_high", "b3_vs_b2_bootstrap", "b3_vs_b0_mae_diff",
                        "b3_vs_b0_ci_low", "b3_vs_b0_ci_high", "b3_vs_b0_bootstrap",
                        "n_production", "b3_prod_mae", "b0_prod_mae", "b2_prod_mae")
            if c in crop_sum]
    n_scored = (int(scored.loc[scored["counted"].astype(str).str.lower() == "true",
                               "target_quarter"].nunique()) if len(scored) else 0)
    return {"status": STATUS_SCORED if n_scored else STATUS_AWAITING,
            "source": source, "scored_at_utc": scored_at,
            "primary_metric": "yield MAE (t/ha) of B3 vs B2 and vs B0, against the first "
                              "PSA figure seen and the latest revised figure",
            "n_scored_quarters": n_scored, "rows": _records(rows),
            "summary": _records(crop_sum, keep, digits=4)}


# ---------------------------------------------------------------------- findings.json

def _plain(text: str) -> str:
    return " ".join(text.replace("**", "").split())


def md_bullets(text: str, heading: str) -> list[str]:
    """Top-level bullets under the `## heading...` section of a markdown report."""
    out, inside = [], False
    for ln in text.splitlines():
        if ln.startswith("## "):
            inside = ln[3:].strip().lower().startswith(heading.lower())
            continue
        if inside and ln.startswith("- "):
            out.append(_plain(ln[2:]))
    return out


def md_paragraph(text: str, label: str) -> str | None:
    """The paragraph that starts with `**label**`, without markdown bold."""
    for para in text.split("\n\n"):
        if para.strip().startswith(f"**{label}"):
            return _plain(para)
    return None


def latest_experiments(exp: pd.DataFrame) -> pd.DataFrame:
    """Latest experiments.csv row per (model, era, crop)."""
    e = exp.assign(_t=pd.to_datetime(exp["timestamp"], utc=True, errors="coerce"))
    e = e.sort_values("_t", kind="stable").drop_duplicates(EXPERIMENT_KEY, keep="last")
    return e.drop(columns="_t").sort_values(["crop", "era", "mae"]).reset_index(drop=True)


def per_unit_errors(pred: pd.DataFrame) -> pd.DataFrame:
    """MAE/RMSE/MAPE of the selected model per crop and unit, with B0/B2 MAE (same rows)."""
    rows = []
    for (crop, unit), g in pred.groupby(["crop", "reporting_unit"], sort=True):
        a = g["actual_yield_t_ha"].to_numpy(float)
        e = g["pred_yield_t_ha"].to_numpy(float) - a
        rows.append({"crop": crop, "model": str(g["model"].iloc[0]), "reporting_unit": unit,
                     "n": len(g), "mae": float(np.mean(np.abs(e))),
                     "rmse": float(np.sqrt(np.mean(e ** 2))),
                     "mape": float(np.mean(np.abs(e) / a) * 100),
                     "mae_b0": float(np.mean(np.abs(g["b0_pred_yield_t_ha"] - a))),
                     "mae_b2": float(np.mean(np.abs(g["b2_pred_yield_t_ha"] - a)))})
    return pd.DataFrame(rows)


def build_findings(selection: pd.DataFrame, experiments: pd.DataFrame,
                   pred: pd.DataFrame, table: pd.DataFrame, results_md: str,
                   modis_start: int, figures: Sequence[str] = ()) -> dict[str, Any]:
    """findings.json content; every number is read from the given report inputs.

    Sources: selection table <- reports/selection.csv; model table and satellite
    comparison <- reports/experiments.csv (latest row per model/era/crop); interval
    coverage, per-unit errors and counts <- predictions.parquet (the out-of-fold table
    behind phase2_results.md); rain shift <- modeling table; reasons, next steps,
    limitations and caveat paragraphs <- phase2_results.md.
    """
    exp = latest_experiments(experiments)
    sat = exp.loc[exp["model"].astype(str).str.contains(":")]
    n_sat = int(len(sat))
    n_better = int((sat["mae_diff_ci_high"] < 0).sum())
    n_worse = int((sat["mae_diff_ci_low"] > 0).sum())
    improved = n_better > 0
    statement = ("Satellite and rainfall features did not improve on the baselines."
                 if not improved else
                 "Some satellite/rainfall models beat the B2 baseline (see the table).")
    detail = (f"{n_better} of {n_sat} satellite/rainfall model runs have an MAE below B2 "
              f"with a 95% bootstrap CI that excludes zero; {n_worse} are significantly "
              "worse than B2.")
    best = (sat.sort_values("mae").groupby(["era", "crop"]).head(1)
            .sort_values(["era", "crop"]))

    sel = {}
    caveats = []
    for crop, g in selection.groupby("crop", sort=True):
        chosen = g.loc[g["selected"].astype(bool), "candidate"]
        label = str(chosen.iloc[0]) if len(chosen) else None
        recs = _records(g.drop(columns="crop"), digits=6)
        for r in recs:
            r["post_hoc"] = "post hoc" in str(r["role"])
        sel[crop] = {"selected": label, "rows": recs}
        row = g.loc[g["candidate"] == label]
        if len(row) and "post hoc" in str(row["role"].iloc[0]):
            caveats.append(f"{crop}: {label} was added as a candidate after seeing the "
                           "B0/B2 test-fold results, so its confidence interval vs B2 "
                           "is optimistic.")
    for label in ("Caveats.", "Deviation from Requirement 6.1."):
        p = md_paragraph(results_md, label)
        if p:
            caveats.append(p)

    m_era = table.loc[table["year"] >= modis_start]
    rain = {"mean_all_mm": _num(table["rain_anom"].mean(), 1),
            "years_all": [int(table["year"].min()), int(table["year"].max())],
            "mean_modis_era_mm": _num(m_era["rain_anom"].mean(), 1),
            "years_modis_era": [int(m_era["year"].min()), int(m_era["year"].max())]}
    if rain["mean_modis_era_mm"] is not None:
        caveats.append(f"Rainfall anomalies are shifted: recent growing windows average "
                       f"{rain['mean_modis_era_mm']:+.0f} mm vs the 1981-2000 ERA5 "
                       "climatology, so a positive rainfall anomaly is not unusual by "
                       "itself.")
    caveats.append(f"Predicted production {AREA_NOTE}; newly regrouped units have no "
                   "area forecast.")

    cov = interval_coverage(pred) if len(pred) else pd.DataFrame()
    counts = {"prediction_rows": int(len(pred)),
              "test_years": [int(pred["year"].min()), int(pred["year"].max())]
              if len(pred) else None,
              "n_test_years": int(pred["year"].nunique()),
              "reporting_units": int(pred["reporting_unit"].nunique()),
              "rows_with_interval": int(pred["pred_lo_t_ha"].notna().sum()),
              "b0_fallback_rows": int(pred["b0_fallback"].astype(bool).sum()),
              "no_area_forecast_rows": int(pred["predicted_area_ha"].isna().sum()),
              "outlier_rows": int(pred["is_outlier"].astype(bool).sum()),
              "satellite_runs": n_sat}
    model_cols = ["model", "era", "crop", "test_years", "n_test", "mae", "rmse", "mape",
                  "mae_no_outliers", "ref_model", "mae_diff", "mae_diff_ci_low",
                  "mae_diff_ci_high"]
    return {
        "sources": {"selection": "reports/selection.csv",
                    "models": "reports/experiments.csv",
                    "history": "data/processed/predictions.parquet",
                    "text": "reports/phase2_results.md"},
        "headline": {"satellite_improved": improved, "statement": statement,
                     "detail": detail, "n_satellite_runs": n_sat,
                     "n_better_than_b2": n_better, "n_worse_than_b2": n_worse},
        "selection": sel,
        "best_satellite": _records(best, ["crop", "era", "model", "mae", "mae_diff",
                                          "mae_diff_ci_low", "mae_diff_ci_high"], 6),
        "models": _records(exp, [c for c in model_cols if c in exp], 6),
        "interval_coverage": _records(cov, digits=4) if len(cov) else [],
        "per_unit": _records(per_unit_errors(pred), digits=4) if len(pred) else [],
        "rain_shift": rain,
        "caveats": caveats,
        "likely_reasons": md_bullets(results_md, "Why satellites"),
        "limitations": md_bullets(results_md, "Other limitations"),
        "next_steps": md_bullets(results_md, "Next steps"),
        "counts": counts,
        "figures": [{"file": f"assets/figures/{name}",
                     "title": Path(name).stem.replace("_", " ")} for name in figures],
    }


# -------------------------------------------------------------------------- export

def app_dir_default(cfg: Config) -> Path:
    """`<repo>/app` (the folder next to `paths.reports`)."""
    return cfg.pipeline.paths.reports.parent / "app"


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def export_app(cfg: Config, app_dir: Path | None = None, now: datetime | None = None,
               git: str | None = None, budget_bytes: int = DATA_BUDGET_BYTES
               ) -> dict[str, Path]:
    """Write app/data/* and app/assets/figures/*; return {file name: path}.

    Raises ValueError when the files in app/data/ together exceed `budget_bytes`.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    app_dir = Path(app_dir) if app_dir is not None else app_dir_default(cfg)
    data_dir = app_dir / "data"
    fig_out = app_dir / "assets" / "figures"
    paths = cfg.pipeline.paths
    eras = cfg.pipeline.eras
    start_year = int(eras["modis"].start_year)
    first_year = int(eras["modis"].first_test_year)

    pred = pd.read_parquet(predictions_path(cfg))
    feats_path = paths.processed / "features.parquet"
    features = pd.read_parquet(feats_path) if feats_path.exists() else None
    table = load_modeling_table(cfg)
    forecasts = read_forecasts(forecasts_path(cfg))
    selection = pd.read_csv(selection_path(cfg))
    experiments = pd.read_csv(experiments_path(cfg))
    rp = report_path(cfg)
    results_md = rp.read_text(encoding="utf-8") if rp.exists() else ""
    boundaries = gpd.read_parquet(paths.interim / "boundaries.parquet")

    figs = sorted(figures_dir(cfg).glob("*.png")) if figures_dir(cfg).exists() else []
    fig_out.mkdir(parents=True, exist_ok=True)
    for f in figs:
        shutil.copy2(f, fig_out / f.name)

    predictions = build_predictions_json(pred, table, features, start_year, first_year)
    forecast = build_forecast_json(forecasts, table, features, start_year,
                                   cfg.pipeline.dates.end)
    units = build_units_geojson(boundaries, cfg, unit_spans(predictions, forecast))
    live = build_live_accuracy_json(cfg, forecasts, table, now)
    findings = build_findings(selection, experiments, pred, table, results_md, start_year,
                              [f.name for f in figs])
    qi = table["year"].astype(int) * 4 + table["quarter"].astype(int) - 1
    vintage_by_crop = {c: _qlabel(int(v)) for c, v in qi.groupby(table["crop"]).max().items()}
    models = {str(c): str(g.loc[g["selected"].astype(bool), "candidate"].iloc[0])
              for c, g in selection.groupby("crop") if g["selected"].astype(bool).any()}
    meta = {
        "generated_at": _stamp(now), "git_commit": git if git is not None else git_commit(),
        "data_vintage": _qlabel(int(qi.max())), "data_vintage_by_crop": vintage_by_crop,
        "latest_forecast_quarter": forecast["target_quarter"],
        "latest_forecast_ids": forecast["forecast_ids"],
        "model": models,
        "recipe_version": (sorted(set(latest_forecast(forecasts)["recipe_version"]
                                      .astype(str))) if len(forecasts) else []),
        "definitions": {
            "normal": f"mean PSA yield of the same unit, crop and quarter over years "
                      f"{start_year}..Y-1 (null with fewer than {MIN_NORMAL_YEARS} years)",
            "pct": "predicted (B3) yield vs normal, %",
            "ndvi_anom": "MODIS NDVI peak minus its mean over earlier years (same unit, "
                         "crop, quarter)",
            "evi_anom": "MODIS EVI peak minus its mean over earlier years",
            "rain_anom": "growing-window rainfall minus the 1981-2000 ERA5 climatology, mm",
            "rain_pct": "rain_anom as % of the climatology for the same window",
            "flags": {"combined_unit": "PSA reported several provinces together",
                      "b0_fallback": "no same-quarter value last year; B0 used the "
                                     "unit-quarter mean",
                      "no_area_forecast": "newly regrouped unit without area history; no "
                                          "production forecast",
                      "area_unit_mean": "no area last year; unit's mean area used",
                      "outlier": "PSA value flagged as an outlier in Phase 1"},
            "area_note": AREA_NOTE, "conditions_note": CONDITIONS_NOTE},
        "sources": SOURCES,
    }
    out = {"predictions.json": write_json(data_dir / "predictions.json", predictions),
           "forecast.json": write_json(data_dir / "forecast.json", forecast),
           "units.geojson": write_json(data_dir / "units.geojson", units),
           "live_accuracy.json": write_json(data_dir / "live_accuracy.json", live),
           "findings.json": write_json(data_dir / "findings.json", findings),
           "meta.json": write_json(data_dir / "meta.json", meta)}
    total = dir_bytes(data_dir)
    if total > budget_bytes:
        raise ValueError(f"{data_dir} is {total:,} bytes, over the {budget_bytes:,}-byte "
                         "budget; simplify polygons further or trim fields")
    out.update({f"figures/{f.name}": fig_out / f.name for f in figs})
    return out


def main(argv: list[str] | None = None) -> None:
    """Write the static app's data files (app/data/*) and figures (app/assets/figures/)."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR),
                    help="folder with pipeline.yaml and provinces.yaml (default: repo configs/)")
    ap.add_argument("--app-dir", default=None, help="site folder (default: <repo>/app)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config_dir)
    out = export_app(cfg, Path(args.app_dir) if args.app_dir else None)
    for name, p in out.items():
        print(f"{p.stat().st_size:>10,} B  {p}")
    data_dir = next(iter(out.values())).parent
    print(f"app/data total: {dir_bytes(data_dir):,} B (budget {DATA_BUDGET_BYTES:,} B)")
    geo = out["units.geojson"].stat().st_size
    if geo > GEOJSON_BUDGET_BYTES:
        print(f"warning: units.geojson is {geo:,} B, over the {GEOJSON_BUDGET_BYTES:,} B "
              "target in design.md")


if __name__ == "__main__":
    main()
