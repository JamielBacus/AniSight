"""Model-ready feature table: PSA ground truth joined with seasonal vegetation-index features.

`build_features(gt, vi, cfg)` returns one row per province, crop, year and quarter of the
ground truth (`data/interim/ground_truth.parquet`) that has a PSA target, with seasonal
NDVI/EVI features from MODIS (`modis_vi.parquet`) and Sentinel-2 (`s2_vi.parquet`).
`save_features` writes `data/processed/features.parquet` and `write_quality_report` writes
`reports/data_quality.md`. `run(cfg)` does all three and is what the CLI calls.

Targets
- Rows flagged `not_reported` are dropped: PSA reported that unit inside its host
  province, so the row has no figure (its production and area sit in the host row, whose
  `reporting_unit` covers both). Every other row is kept with its `quality_flag`
  (`missing`, `zero`, `outlier`, ...) so Phase 2 decides what to train on.

Reporting units
- Satellite features are joined on `reporting_unit`, the footprint the PSA figure covers.
  A single-province unit uses that province's VI rows (cropland-masked by
  `boundaries.cropland_mask`).
- A combined unit (e.g. "Davao del Sur+Davao Occidental", members from
  `cfg.provinces.combined_reporting_units()`, the same members as
  `boundaries.reporting_unit_geometries`) is built from its members' VI rows, weighted by
  valid pixels. Member polygons do not overlap, so this equals extracting over the OR of
  the members' cropland masks for `mean`, `max`, `n_pixels` and `valid_frac`;
  `median` is approximated by the pixel-weighted mean of member medians. A combined
  composite is NaN unless every member has a valid (non-NaN) mean for that date, so a
  cloudy member never shrinks the footprint silently. Re-extracting imagery over merged
  masks was not chosen because the VI tables are already per province.

Leakage cutoff (Requirement 6.2)
- Each VI observation gets the date its imagery ends: MODIS `date` is the composite END
  date; Sentinel-2 `date` is the first day of the month, so its imagery ends on the last
  day of that month. Only observations ending on or before the quarter's last day (and on
  or after the growing-window start) are used. No PSA value enters the features.

Growing window
- `GROWING_WINDOW_MONTHS[(crop, quarter)]` months ending at quarter end; 6 for every crop
  and quarter (e.g. Q1 = 1 Oct of the previous year to 31 Mar). PSA counts production in
  the quarter it is harvested. In Davao Region palay and corn are planted year-round in
  staggered fashion, both with field cycles of about 100-120 days, so the crops harvested
  in a quarter were planted from about four months before the quarter starts until about
  one month before it ends: their green-up, peak and senescence fall in the six months
  ending at quarter end. With two cropping seasons a year, a longer window would also
  catch the previous season's peak. Change the table below to tune per crop or quarter.

Features per source (`modis`, `s2`) and index (`ndvi`, `evi`), over the window, from the
province/unit cropland `mean` of each observation:
- `<src>_<idx>_peak`       highest observation in the window
- `<src>_<idx>_integral`   sum over the window's days of the series, linearly interpolated
                           between observations and held flat to the window edges (index x days)
- `<src>_<idx>_peak_day`   days from window start to the peak observation
- `<src>_<idx>_greenup_day` first day (at or before the peak) the interpolated series reaches
                           halfway from its pre-peak minimum to the peak
- `<src>_<idx>_valid_frac` mean valid-pixel fraction of the window's observations
- `<src>_<idx>_n_obs`      observations with a valid mean in the window
The first four are NaN when `n_obs < MIN_OBS[src]`.

Rainfall (Open-Meteo ERA5, `data/interim/rainfall.parquet` from `cropyield.rainfall`),
joined on `reporting_unit` (its `province` column holds unit names, combined units
included). The window is the same growing window: the calendar months from
`window_start` to `window_end`. A monthly total covers its days up to the month's last
day, so the last month used ends exactly at quarter end and no later month enters.
- `rain_total`        sum of monthly rain (mm) over the window's months
- `rain_first_half`   sum over the window's first half (first 3 months of a 6-month window)
- `rain_second_half`  sum over the second half (last 3 months, ending at quarter end)
- `rain_anom`         `rain_total` minus the sum of the unit's month-of-year climatology
                      for the same months; the climatology is the mean monthly rain over
                      the fixed reference years `RAIN_CLIM_YEARS` (1981-2000), which end
                      before every modeling year, so no training or test year feeds it
Any month in a window with `rain_mm` NaN (or absent) makes that aggregate NaN; missing is
never treated as 0. A month-of-year climatology needs at least `RAIN_CLIM_MIN_YEARS`
valid reference years, else `rain_anom` is NaN. Without rainfall.parquet the four columns
are written as NaN (with a warning) so the table schema stays fixed.

Usage (needs data/interim/ground_truth.parquet and at least one VI table):
    python -m cropyield.features
    python -m cropyield.features --config-dir configs
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from cropyield.config import DEFAULT_CONFIG_DIR, Config, load_config

log = logging.getLogger(__name__)

KEYS = ["province", "crop", "year", "quarter"]
TARGETS = ["production_mt", "area_ha", "yield_t_ha"]
INDICES = ("NDVI", "EVI")
VI_FILES = {"modis": "modis_vi.parquet", "s2": "s2_vi.parquet"}
FEATURES = ("peak", "integral", "peak_day", "greenup_day", "valid_frac", "n_obs")

# Growing window length (months ending at quarter end) per (crop, quarter); see module doc.
GROWING_WINDOW_MONTHS: dict[tuple[str, int], int] = {
    (crop, q): 6 for crop in ("palay", "corn") for q in (1, 2, 3, 4)}
# Fewest valid observations needed in a window for peak/integral/timing features.
# A 6-month window holds about 11 MODIS 16-day composites and 6 Sentinel-2 months.
MIN_OBS = {"modis": 3, "s2": 2}

RAIN_FILE = "rainfall.parquet"
RAIN_COLUMNS = ("rain_total", "rain_first_half", "rain_second_half", "rain_anom")
# Fixed rainfall climatology reference years (inclusive); must end before every modeling
# year (phase 2 design.md), checked against the configured eras in `rain_features`.
RAIN_CLIM_YEARS = (1981, 2000)
# Fewest valid reference years per unit and month-of-year for a climatology value.
RAIN_CLIM_MIN_YEARS = 15


# --------------------------------------------------------------------- reporting units
def unit_vi(vi: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """VI table re-keyed by `reporting_unit`: every province plus every combined unit.

    Combined units aggregate their members' rows per date and index, weighted by
    `n_pixels` (see module doc). Returns the VI columns with `province` renamed to
    `reporting_unit`.
    """
    vi = vi.copy()
    # Cropland cells per province (constant per source): n_pixels / valid_frac.
    ratio = vi["n_pixels"] / vi["valid_frac"].where(vi["valid_frac"] > 0)
    vi["_n_crop"] = ratio.groupby(vi["province"]).transform("max").fillna(0)
    frames = [vi.rename(columns={"province": "reporting_unit"})]
    for name, members in cfg.provinces.combined_reporting_units().items():
        sub = vi[vi["province"].isin(members)]
        if sub.empty:
            continue
        sub = sub.assign(_ok=sub["mean"].notna(), _wm=sub["mean"] * sub["n_pixels"],
                         _wmed=sub["median"] * sub["n_pixels"])
        g = sub.groupby(["date", "index"], as_index=False).agg(
            _members=("province", "nunique"), _ok=("_ok", "sum"), n_pixels=("n_pixels", "sum"),
            _n_crop=("_n_crop", "sum"), _wm=("_wm", "sum"), _wmed=("_wmed", "sum"),
            max=("max", "max"))
        complete = (g["_members"] == len(members)) & (g["_ok"] == len(members))
        g["mean"] = (g["_wm"] / g["n_pixels"]).where(complete)
        g["median"] = (g["_wmed"] / g["n_pixels"]).where(complete)
        g["max"] = g["max"].where(complete)
        g["valid_frac"] = g["n_pixels"] / g["_n_crop"].where(g["_n_crop"] > 0)
        g["reporting_unit"] = name
        frames.append(g)
    cols = ["reporting_unit", "date", "index", "mean", "median", "max", "valid_frac",
            "n_pixels"]
    return pd.concat([f[cols] for f in frames], ignore_index=True)


# ------------------------------------------------------------------------ windows
def window_bounds(crop: str, year: int, quarter: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last day (inclusive) of the growing window for a crop-year-quarter."""
    q = pd.Period(year=year, quarter=quarter, freq="Q")
    months = GROWING_WINDOW_MONTHS[(crop, quarter)]
    start = q.start_time.normalize() - pd.DateOffset(months=months - 3)
    return start, q.end_time.normalize()


def observation_end(source: str, dates: pd.Series) -> pd.Series:
    """Last day covered by each VI observation (the date used for the cutoff)."""
    dates = pd.to_datetime(dates)
    if source == "modis":  # `date` is already the composite end date
        return dates
    if source == "s2":  # `date` is the first day of the month
        return dates + pd.offsets.MonthEnd(0)
    raise ValueError(f"unknown VI source {source!r}; expected one of {sorted(VI_FILES)}")


def season_features(days: np.ndarray, values: np.ndarray, valid_frac: np.ndarray,
                    n_days: int, min_obs: int) -> dict[str, float]:
    """Seasonal features from one window's observations.

    Args:
        days: day offset of each observation from the window start (0 = first day).
        values: cropland mean of each observation (NaN = no valid pixels).
        valid_frac: valid-pixel fraction of each observation.
        n_days: number of days in the window.
        min_obs: fewest valid observations needed for peak/integral/timing.
    """
    ok = np.isfinite(values)
    out = {"valid_frac": float(np.nanmean(valid_frac)) if np.isfinite(valid_frac).any()
           else np.nan, "n_obs": int(ok.sum())}
    out.update(peak=np.nan, integral=np.nan, peak_day=np.nan, greenup_day=np.nan)
    if ok.sum() < min_obs:
        return out
    d, v = days[ok], values[ok]
    order = np.argsort(d, kind="stable")
    d, v = d[order], v[order]
    daily = np.interp(np.arange(n_days), d, v)
    i_peak = int(np.argmax(v))
    peak_day = int(d[i_peak])
    base = daily[: peak_day + 1].min()
    half = base + 0.5 * (v[i_peak] - base)
    out.update(peak=float(v[i_peak]), integral=float(daily.sum()), peak_day=peak_day,
               greenup_day=int(np.argmax(daily[: peak_day + 1] >= half)))
    return out


# ------------------------------------------------------------------------ rainfall
def _monthly_rain(rain: pd.DataFrame) -> pd.Series:
    """`rain_mm` indexed by (unit, month start); raises on duplicate unit-months."""
    r = rain[["province", "date", "rain_mm"]].copy()
    r["date"] = pd.to_datetime(r["date"]).dt.to_period("M").dt.to_timestamp()
    dup = r.duplicated(["province", "date"])
    if dup.any():
        raise ValueError(f"rainfall table has {int(dup.sum())} duplicate (province, month) rows")
    return r.set_index(["province", "date"])["rain_mm"].astype(float)


def rain_climatology(rain: pd.DataFrame, years: tuple[int, int] = RAIN_CLIM_YEARS,
                     min_years: int = RAIN_CLIM_MIN_YEARS) -> pd.Series:
    """Mean monthly rain per unit and month-of-year over the reference `years` only.

    Args:
        rain: rainfall table (`province, date, rain_mm, ...`).
        years: first and last reference year, inclusive.
        min_years: fewest non-NaN reference years for a value; fewer gives NaN.

    Returns:
        Series indexed by (province, month 1-12). Rows outside `years` are ignored.
    """
    s = _monthly_rain(rain).reset_index()
    s = s[s["date"].dt.year.between(*years)]
    g = s.assign(month=s["date"].dt.month).groupby(["province", "month"])["rain_mm"]
    clim = g.mean().where(g.count() >= min_years)
    return clim.rename("rain_clim_mm")


def rain_features(rows: pd.DataFrame, rain: pd.DataFrame | None, cfg: Config) -> pd.DataFrame:
    """Window rainfall features for each row (see module doc), aligned to `rows.index`.

    Args:
        rows: needs `reporting_unit`, `window_start`, `window_end`.
        rain: rainfall table from `cropyield.rainfall`, or None (all features NaN).
        cfg: project config; the reference years must end before every era's start year.

    Raises:
        ValueError: if `RAIN_CLIM_YEARS` overlaps a modeling era.
    """
    first_year = min(e.start_year for e in cfg.pipeline.eras.values())
    if RAIN_CLIM_YEARS[1] >= first_year:
        raise ValueError(f"rainfall climatology {RAIN_CLIM_YEARS} must end before the first "
                         f"modeling year {first_year}")
    out = pd.DataFrame(np.nan, index=rows.index, columns=list(RAIN_COLUMNS))
    if rain is None:
        return out
    monthly = _monthly_rain(rain).to_dict()
    clim = rain_climatology(rain).to_dict()
    known = set(rain["province"])
    missing_units = sorted(set(rows["reporting_unit"]) - known)
    if missing_units:
        log.warning("rainfall table has no rows for reporting unit(s): %s; their rain "
                    "features are NaN", ", ".join(missing_units))
    for i, unit, start, end in zip(rows.index, rows["reporting_unit"], rows["window_start"],
                                   rows["window_end"]):
        months = pd.date_range(pd.Timestamp(start).to_period("M").to_timestamp(),
                               pd.Timestamp(end).to_period("M").to_timestamp(), freq="MS")
        v = np.array([monthly.get((unit, m), np.nan) for m in months], dtype=float)
        c = np.array([clim.get((unit, m.month), np.nan) for m in months], dtype=float)
        half = len(months) // 2
        # np.sum propagates NaN, so any missing month makes the aggregate NaN.
        total = float(np.sum(v)) if len(v) else np.nan
        out.loc[i] = [total, float(np.sum(v[:half])), float(np.sum(v[half:])),
                      total - float(np.sum(c))]
    return out


# ------------------------------------------------------------------------ features
def build_features(gt: pd.DataFrame, vi: Mapping[str, pd.DataFrame], cfg: Config,
                   rain: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join ground truth with seasonal VI features, one row per province-crop-year-quarter.

    Args:
        gt: ground truth from `ground_truth.load_psa` (incl. `quality_flag`,
            `reporting_unit`).
        vi: `{source: VI table}` with source "modis" and/or "s2" and columns
            `province, date, index, mean, median, max, valid_frac, n_pixels`.
        cfg: project config (reporting units).
        rain: rainfall table `province, date, rain_mm, valid_frac` from
            `cropyield.rainfall`, or None (rain columns all NaN).

    Returns:
        Columns `province, crop, year, quarter, reporting_unit, quality_flag,
        production_mt, area_ha, yield_t_ha, window_start, window_end`, then
        `<src>_<idx>_<feature>` for each source in `vi`, then `RAIN_COLUMNS` (see module
        doc). Rows flagged `not_reported` are excluded; only imagery and rainfall months
        ending on or before quarter end are used.
    """
    reported = ~gt["quality_flag"].str.split(";").map(lambda f: "not_reported" in f)
    out = gt.loc[reported, [*KEYS, "reporting_unit", "quality_flag", *TARGETS]]
    out = out.sort_values(KEYS).reset_index(drop=True)
    bounds = [window_bounds(c, y, q) for c, y, q in zip(out["crop"], out["year"], out["quarter"])]
    out["window_start"] = [b[0] for b in bounds]
    out["window_end"] = [b[1] for b in bounds]

    for source, table in vi.items():
        u = unit_vi(table, cfg)
        u["_end"] = observation_end(source, u["date"])
        u = u.sort_values("_end")
        series = {k: (g["_end"].to_numpy("datetime64[ns]"), g["mean"].to_numpy(float),
                      g["valid_frac"].to_numpy(float))
                  for k, g in u.groupby(["reporting_unit", "index"])}
        for index in INDICES:
            cols: dict[str, list] = {f: [] for f in FEATURES}
            for unit, (start, end) in zip(out["reporting_unit"], bounds):
                t, v, vf = series.get((unit, index), (np.array([], "datetime64[ns]"),
                                                      np.array([]), np.array([])))
                lo = np.searchsorted(t, np.datetime64(start), side="left")
                hi = np.searchsorted(t, np.datetime64(end), side="right")  # <= quarter end
                days = ((t[lo:hi] - np.datetime64(start)) // np.timedelta64(1, "D")).astype(int)
                feats = season_features(days, v[lo:hi], vf[lo:hi],
                                        (end - start).days + 1, MIN_OBS[source])
                for f in FEATURES:
                    cols[f].append(feats[f])
            for f in FEATURES:
                out[f"{source}_{index.lower()}_{f}"] = cols[f]
    rf = rain_features(out, rain, cfg)
    for col in RAIN_COLUMNS:
        out[col] = rf[col]
    return out


def save_features(df: pd.DataFrame, cfg: Config) -> Path:
    """Write the feature table to `<paths.processed>/features.parquet`."""
    out = cfg.pipeline.paths.processed / "features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


# ------------------------------------------------------------------ quality report
def quality_report(features: pd.DataFrame, gt: pd.DataFrame, sources: list[str],
                   has_rain: bool = False) -> str:
    """Markdown data-quality summary: row counts and missingness per column.

    `has_rain` says whether a rainfall table was joined (changes the rainfall note).
    """
    n_nr = int(len(gt) - len(features))
    flags = features["quality_flag"].str.split(";").explode()
    flags = flags.where(~flags.str.startswith("includes:"), "includes:<unit>")
    lines = [
        "# Data quality: Phase 1 feature table", "",
        "Generated by `python -m cropyield.features` from `data/interim/ground_truth.parquet` "
        f"and the VI tables for: {', '.join(sources) or 'none'}.", "",
        "## Row counts", "",
        "| item | rows |", "|---|---:|",
        f"| ground-truth rows | {len(gt)} |",
        f"| excluded `not_reported` rows (no PSA target) | {n_nr} |",
        f"| feature rows (`data/processed/features.parquet`) | {len(features)} |",
    ]
    for crop, n in features["crop"].value_counts().sort_index().items():
        lines.append(f"| feature rows, {crop} | {n} |")
    for src in sources:
        n = int(features[f"{src}_ndvi_peak"].notna().sum())
        lines.append(f"| rows with {src} NDVI features | {n} |")
    lines += ["", "## Quality flags (feature rows; a row can carry several)", "",
              "| flag | rows |", "|---|---:|"]
    lines += [f"| {f} | {n} |" for f, n in flags.value_counts().items()]
    lines += ["", "## Missingness per column", "", "| column | missing | % missing |",
              "|---|---:|---:|"]
    n = max(len(features), 1)
    for col in features.columns:
        m = int(features[col].isna().sum())
        lines.append(f"| {col} | {m} | {100 * m / n:.1f} |")
    lines += ["", "## Notes", "",
              "- Growing window: 6 months ending at quarter end for every crop and quarter "
              "(`GROWING_WINDOW_MONTHS` in `src/cropyield/features.py`). Only imagery ending "
              "on or before quarter end is used.",
              "- Combined PSA reporting units use their members' VI rows weighted by valid "
              "pixels; a date is missing unless every member is valid.",
              "- Satellite features are missing before each source starts (MODIS 2001, "
              "Sentinel-2 2017) and when a window has too few valid observations.",
              ("- Rainfall (Open-Meteo ERA5, daily totals at one cropland point per "
               "province, summed to months) columns `rain_*` use the calendar months of the "
               "same window; a missing month makes the aggregate missing. `rain_anom` is "
               f"relative to the fixed {RAIN_CLIM_YEARS[0]}-{RAIN_CLIM_YEARS[1]} climatology."
               if has_rain else
               "- Rainfall (Open-Meteo ERA5): `data/interim/rainfall.parquet` was not found, "
               "so the `rain_*` columns are all missing; run `cropyield rainfall` first."),
              ""]
    return "\n".join(lines)


def write_quality_report(features: pd.DataFrame, gt: pd.DataFrame, sources: list[str],
                         cfg: Config, has_rain: bool = False) -> Path:
    """Write `quality_report(...)` to `<paths.reports>/data_quality.md`."""
    out = cfg.pipeline.paths.reports / "data_quality.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(quality_report(features, gt, sources, has_rain), encoding="utf-8")
    return out


# ------------------------------------------------------------------------------ run
def run(cfg: Config) -> tuple[Path, Path]:
    """Read interim tables, build features, write the table and report.

    Missing VI tables are skipped with a warning. Returns (features path, report path).

    Raises:
        FileNotFoundError: if the ground truth or every VI table is missing.
    """
    interim = cfg.pipeline.paths.interim
    gt_path = interim / "ground_truth.parquet"
    if not gt_path.exists():
        raise FileNotFoundError(f"{gt_path} not found; run `python -m cropyield.ground_truth` first")
    gt = pd.read_parquet(gt_path)
    vi = {}
    for source, name in VI_FILES.items():
        path = interim / name
        if path.exists():
            vi[source] = pd.read_parquet(path)
        else:
            log.warning("%s not found; building features without %s", path, source)
    if not vi:
        raise FileNotFoundError(f"no VI tables ({', '.join(VI_FILES.values())}) in {interim}")
    rain_path = interim / RAIN_FILE
    rain = None
    if rain_path.exists():
        rain = pd.read_parquet(rain_path)
    else:
        log.warning("%s not found; rain columns are NaN (run `cropyield rainfall` first)",
                    rain_path)
    features = build_features(gt, vi, cfg, rain)
    return (save_features(features, cfg),
            write_quality_report(features, gt, list(vi), cfg, rain is not None))


def main(argv: list[str] | None = None) -> None:
    """Build data/processed/features.parquet and reports/data_quality.md."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR),
                    help="folder with pipeline.yaml and provinces.yaml (default: repo configs/)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    feats, report = run(load_config(args.config_dir))
    print(f"wrote {feats} ({len(pd.read_parquet(feats))} rows) and {report}")


if __name__ == "__main__":
    main()
