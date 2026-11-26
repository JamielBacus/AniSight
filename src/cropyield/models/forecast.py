"""Forecast logging (phase 3, Requirement 2.1-2.3; task 2.1).

`cropyield forecast --quarter YYYYQn [--backfill]` forecasts yield, 80% interval,
predicted area and predicted production for every current reporting unit and crop, and
appends the rows to `<paths.reports>/forecasts.csv` before PSA publishes the quarter.

- Recipes: the frozen B0/B2/selected-model recipes and the area rule of
  `<paths.models>/<crop>.joblib`, refitted by `select.forecast_production` on modeling-
  table rows with training_period[0] <= year < target year (no target-year PSA data).
- Target units ("current reporting units", design.md): per crop, the reporting units of
  the latest PSA quarter published before the target quarter. With the 2025 Q2 split this
  is Davao City and Davao del Sur separately (not "Davao del Sur+Davao City"). Feature
  columns of the target quarter are joined from features.parquet when present.
- Published quarters: a quarter is published when the modeling table (rows with a PSA
  yield) has it for any crop. Such quarters are refused unless `backfill=True`; backfill
  rows are logged with `backfill=True` and use the same year < Y history.
- The log is append-only: re-forecasting a quarter appends a new row set with its own
  `forecast_id`; earlier rows are never rewritten.

`cropyield score` (task 2.2, Requirement 2.4): `score_forecasts`, see "live scoring" below.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from cropyield.config import DEFAULT_CONFIG_DIR, Config, load_config
from cropyield.models.baselines import SameQuarterLastYear
from cropyield.models.data import UNIT_KEYS, load_modeling_table
from cropyield.models.select import forecast_production, models_dir

FORECAST_COLUMNS: dict[str, str] = {
    "forecast_id": "one id per `cropyield forecast` run: <target_quarter>-<made_at, UTC>",
    "made_at_utc": "time the forecast was made (UTC, ISO 8601)",
    "git_commit": "git HEAD of the repo ('+dirty' if tracked files changed), or 'unknown'",
    "code_hash": "sha256 (12 hex) of src/cropyield/models/*.py; works without git",
    "recipe_version": "recipe_version of the model bundle",
    "data_vintage": "latest PSA quarter (YYYYQn) of this crop in the modeling table",
    "target_quarter": "forecast quarter, YYYYQn",
    "year": "target year",
    "quarter": "target quarter 1-4",
    "reporting_unit": "PSA reporting unit (current unit set, see module docstring)",
    "crop": "palay or corn",
    "model": "selected model of the bundle (B3 for both crops after phase 2)",
    "b3_yield": "selected-model yield forecast (t/ha)",
    "b3_lo80": "80% interval low (t/ha), bundle out-of-fold residual quantiles",
    "b3_hi80": "80% interval high (t/ha)",
    "b0_yield": "B0 (same quarter last year) yield (t/ha)",
    "b2_yield": "B2 (unit-quarter mean + trend) yield (t/ha)",
    "b0_fallback": "B0 had no same-quarter yield for year-1 and used the unit-quarter mean",
    "predicted_area_ha": "last year's same-quarter PSA area (else the unit's mean); NaN "
                         "when the unit has no own area history (no area forecast)",
    "area_fallback_level": "source of the area: last_year, unit_quarter_mean, or (NaN "
                           "area) crop_quarter_mean / crop_mean / overall",
    "pred_production_mt": "b3_yield x predicted_area_ha (t)",
    "pred_production_lo_mt": "b3_lo80 x predicted_area_ha (t)",
    "pred_production_hi_mt": "b3_hi80 x predicted_area_ha (t)",
    "history_last_year": "latest history year the models were refitted on (target year-1)",
    "backfill": "True when the quarter was already published at forecast time",
}

_QUARTER_RE = re.compile(r"^(\d{4})-?Q([1-4])$")
_MODELS_SRC = Path(__file__).resolve().parent
_REPO = _MODELS_SRC.parents[2]


class PublishedQuarterError(ValueError):
    """The target quarter is already in the PSA data (use backfill)."""


def parse_quarter(text: str) -> tuple[int, int]:
    """'2026Q4' or '2026-Q4' -> (2026, 4)."""
    m = _QUARTER_RE.match(text.strip().upper())
    if not m:
        raise ValueError(f"quarter must look like 2026Q4, got {text!r}")
    return int(m.group(1)), int(m.group(2))


def quarter_label(year: int, quarter: int) -> str:
    return f"{int(year)}Q{int(quarter)}"


def forecasts_path(cfg: Config) -> Path:
    """`<paths.reports>/forecasts.csv`."""
    return cfg.pipeline.paths.reports / "forecasts.csv"


def code_hash(src_dir: Path = _MODELS_SRC) -> str:
    """sha256 (first 12 hex) over the names and bytes of `src_dir/*.py`, sorted."""
    h = hashlib.sha256()
    for p in sorted(src_dir.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def git_commit(repo: Path = _REPO) -> str:
    """HEAD commit, with '+dirty' if tracked files differ from it; 'unknown' without git."""
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10, check=True)
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return head.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


def _qindex(df: pd.DataFrame) -> pd.Series:
    return df["year"].astype(int) * 4 + df["quarter"].astype(int) - 1


def target_rows(table: pd.DataFrame, year: int, quarter: int, crop: str,
                features: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows to forecast for `crop` in (year, quarter).

    Units: those of the latest quarter of `crop` in `table` (published PSA rows) that is
    before the target quarter. Non-key columns come from the matching `features` rows
    (e.g. satellite features of the target quarter) when given; target values are
    dropped later by `forecast_production`.
    """
    t = table.loc[table["crop"] == crop]
    before = t.loc[_qindex(t) < year * 4 + quarter - 1]
    if before.empty:
        raise ValueError(f"no published {crop} quarter before {quarter_label(year, quarter)}")
    last = _qindex(before).max()
    units = sorted(before.loc[_qindex(before) == last, "reporting_unit"].unique())
    rows = pd.DataFrame({"reporting_unit": units, "crop": crop, "year": int(year),
                         "quarter": int(quarter)})
    if features is not None and len(features):
        f = features.loc[(features["crop"] == crop) & (features["year"] == year)
                         & (features["quarter"] == quarter)]
        rows = rows.merge(f.drop_duplicates(UNIT_KEYS), on=UNIT_KEYS, how="left")
    return rows


def forecast_quarter(table: pd.DataFrame, bundles: Mapping[str, Mapping[str, Any]],
                     year: int, quarter: int, *, backfill: bool = False,
                     features: pd.DataFrame | None = None,
                     made_at: datetime | None = None,
                     git: str | None = None, code: str | None = None) -> pd.DataFrame:
    """Forecast rows (FORECAST_COLUMNS) for every crop in `bundles` (in-memory core).

    `table` is the modeling table (published PSA rows only, `prepare_modeling_table`).
    Raises PublishedQuarterError if (year, quarter) is in `table` for any crop and
    `backfill` is False. History is limited to years before `year` by
    `forecast_production`, so backfill rows see the same information as a live forecast.
    """
    label = quarter_label(year, quarter)
    published = table.loc[(table["year"] == year) & (table["quarter"] == quarter)]
    if len(published) and not backfill:
        raise PublishedQuarterError(
            f"{label} is already published by PSA ({sorted(published['crop'].unique())}); "
            "use --backfill to log a backfill forecast")
    ts = (made_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    stamp = ts.isoformat(timespec="seconds").replace("+00:00", "Z")
    fid = f"{label}-{ts.strftime('%Y%m%dT%H%M%SZ')}" + ("-backfill" if backfill else "")
    git = git if git is not None else git_commit()
    code = code if code is not None else code_hash()
    parts = []
    for crop, bundle in sorted(bundles.items()):
        target = target_rows(table, year, quarter, crop, features)
        out = forecast_production(bundle, table, target)
        start = int(bundle["training_period"][0])
        hist = table.loc[(table["year"] >= start) & (table["year"] < year)]
        b0_fb = SameQuarterLastYear().fit(hist).fallback_mask(target).astype(bool)
        tc = table.loc[table["crop"] == crop]
        vq = int(_qindex(tc).max())
        p = pd.DataFrame({
            "forecast_id": fid, "made_at_utc": stamp, "git_commit": git,
            "code_hash": code, "recipe_version": str(bundle["recipe_version"]),
            "data_vintage": quarter_label(vq // 4, vq % 4 + 1), "target_quarter": label,
            "year": int(year), "quarter": int(quarter),
            "reporting_unit": target["reporting_unit"], "crop": crop,
            "model": str(bundle["model_name"]),
            "b3_yield": out["pred_yield_t_ha"], "b3_lo80": out["pred_lo_t_ha"],
            "b3_hi80": out["pred_hi_t_ha"], "b0_yield": out["b0_pred_yield_t_ha"],
            "b2_yield": out["b2_pred_yield_t_ha"], "b0_fallback": b0_fb,
            "predicted_area_ha": out["predicted_area_ha"],
            "area_fallback_level": out["area_fallback_level"].astype(str),
            "pred_production_mt": out["pred_production_mt"],
            "pred_production_lo_mt": out["pred_production_lo_mt"],
            "pred_production_hi_mt": out["pred_production_hi_mt"],
            "history_last_year": out["history_last_year"].astype(int),
            "backfill": bool(backfill)}, index=target.index)
        parts.append(p)
    return pd.concat(parts, ignore_index=True)[list(FORECAST_COLUMNS)]


def load_bundles(cfg: Config, crops: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """`<paths.models>/<crop>.joblib` for each crop (default: config crops)."""
    mdir = models_dir(cfg)
    out = {}
    for crop in crops or list(cfg.pipeline.crops):
        path = mdir / f"{crop}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `cropyield select` first")
        out[crop] = joblib.load(path)
    return out


def make_forecast(quarter: str, cfg: Config, backfill: bool = False,
                  made_at: datetime | None = None) -> pd.DataFrame:
    """Forecast `quarter` ('YYYYQn') from the modeling table and the saved bundles."""
    year, q = parse_quarter(quarter)
    table = load_modeling_table(cfg)
    feats = pd.read_parquet(cfg.pipeline.paths.processed / "features.parquet")
    bundles = load_bundles(cfg, sorted(table["crop"].unique()))
    return forecast_quarter(table, bundles, year, q, backfill=backfill,
                            features=feats, made_at=made_at)


def log_forecast(df: pd.DataFrame, path: Path) -> Path:
    """Append forecast rows to `path` (CSV); never rewrites existing rows.

    Raises if `df` lacks FORECAST_COLUMNS, if the existing file has a different header,
    or if one of `df`'s forecast_ids is already logged.
    """
    cols = list(FORECAST_COLUMNS)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"forecast rows are missing columns: {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        old = pd.read_csv(path, dtype=str)
        if list(old.columns) != cols:
            raise ValueError(f"{path} has a different header; not appending")
        dup = set(old["forecast_id"]) & set(df["forecast_id"].astype(str))
        if dup:
            raise ValueError(f"forecast_id already logged in {path}: {sorted(dup)}")
    df[cols].to_csv(path, mode="a", header=not exists, index=False)
    return path


# ------------------------------------------------------------------ live scoring (2.2)
# `cropyield score` (Requirement 2.4) joins logged forecasts to PSA actuals.
#
# - First-seen store `<paths.reports>/psa_first_seen.csv` (append-only): the first time a
#   PSA actual for (unit, crop, year, quarter) is in the modeling table, its yield, area and
#   production are recorded with the UTC time of that `score` run. Later PSA revisions
#   never overwrite it. Every published key is recorded, not only forecast ones.
# - Live forecast per (target quarter, unit, crop): the last non-backfill forecast made
#   before the actual was first seen. Non-backfill forecasts made at or after that time
#   are not counted (status "late"). Backfill forecasts are listed with status
#   "backfill" and errors, but never counted.
# - `<paths.reports>/live_accuracy.csv`: one row per (target quarter, unit, crop, kind),
#   errors against the first-seen and the latest actual. `counted` marks the rows that
#   enter the metrics.
# - `<paths.reports>/live_accuracy_summary.csv`: per basis (first / latest) and per crop
#   and crop x unit: n quarters, MAE/RMSE/MAPE of B3, B0, B2 in yield, production MAE
#   (rows with a predicted area), 80% interval coverage, and paired bootstrap MAE
#   differences B3 - B2 and B3 - B0 (resampling target quarters) once a group has at
#   least MIN_BOOTSTRAP_QUARTERS scored quarters; "insufficient" otherwise. With nothing
#   scored, each crop gets a row with status "awaiting PSA release".

FIRST_SEEN_COLUMNS = ["reporting_unit", "crop", "year", "quarter", "target_quarter",
                      "first_seen_utc", "data_vintage", "yield_t_ha", "area_ha",
                      "production_mt"]
# Design.md fixes no threshold; 8 quarters = two years of target-quarter groups.
MIN_BOOTSTRAP_QUARTERS = 8
_MODELS3 = ("b3", "b0", "b2")
_BASES = ("first", "latest")
STATUS_SCORED = "scored"
STATUS_AWAITING = "awaiting PSA release"
STATUS_NOT_REPORTED = "unit not reported by PSA"
STATUS_LATE = "late: made after PSA actual first seen"
STATUS_BACKFILL = "backfill: not a live forecast"

LIVE_COLUMNS = [
    "target_quarter", "year", "quarter", "reporting_unit", "crop", "kind", "status",
    "counted", "forecast_id", "made_at_utc", "data_vintage", "model", "b3_yield",
    "b3_lo80", "b3_hi80", "b0_yield", "b2_yield", "predicted_area_ha",
    "pred_production_mt", "pred_production_lo_mt", "pred_production_hi_mt",
    "first_seen_utc", "actual_first_yield_t_ha", "actual_first_production_mt",
    "actual_latest_yield_t_ha", "actual_latest_production_mt", "revised",
    *[f"{m}_err_{b}" for b in _BASES for m in _MODELS3],
    *[f"b3_in80_{b}" for b in _BASES],
    *[f"{m}_prod_err_{b}" for b in _BASES for m in _MODELS3],
    "scored_at_utc",
]


def first_seen_path(cfg: Config) -> Path:
    """`<paths.reports>/psa_first_seen.csv`."""
    return cfg.pipeline.paths.reports / "psa_first_seen.csv"


def live_accuracy_path(cfg: Config) -> Path:
    """`<paths.reports>/live_accuracy.csv`."""
    return cfg.pipeline.paths.reports / "live_accuracy.csv"


def live_summary_path(cfg: Config) -> Path:
    """`<paths.reports>/live_accuracy_summary.csv`."""
    return cfg.pipeline.paths.reports / "live_accuracy_summary.csv"


def _stamp(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["reporting_unit"] = df["reporting_unit"].astype(str)
    df["crop"] = df["crop"].astype(str)
    df["year"] = df["year"].astype(int)
    df["quarter"] = df["quarter"].astype(int)
    return df


def _empty_first_seen() -> pd.DataFrame:
    num = {"yield_t_ha": float, "area_ha": float, "production_mt": float,
           "year": int, "quarter": int}
    return pd.DataFrame({c: pd.Series(dtype=num.get(c, object)) for c in FIRST_SEEN_COLUMNS})


def load_first_seen(path: Path) -> pd.DataFrame:
    """The first-seen store at `path`, or an empty store."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return _empty_first_seen()
    df = pd.read_csv(path, dtype={"reporting_unit": str, "crop": str})
    if list(df.columns) != FIRST_SEEN_COLUMNS:
        raise ValueError(f"{path} has an unexpected header")
    return _keys(df)


def new_first_seen(actuals: pd.DataFrame, store: pd.DataFrame,
                   seen_at: datetime) -> pd.DataFrame:
    """Rows (FIRST_SEEN_COLUMNS) for published keys of `actuals` not yet in `store`.

    `actuals` is the modeling table (rows with a PSA yield). Keys already in `store` are
    skipped, so a revised actual never replaces its first-seen value.
    """
    a = _keys(actuals.loc[actuals["yield_t_ha"].notna()])
    for c in ("area_ha", "production_mt"):
        if c not in a:
            a[c] = np.nan
    if len(store):
        known = pd.MultiIndex.from_frame(_keys(store)[UNIT_KEYS])
        a = a.loc[~pd.MultiIndex.from_frame(a[UNIT_KEYS]).isin(known)]
    if a.empty:
        return _empty_first_seen()
    qi = _qindex(a)
    vintage = qi.groupby(a["crop"]).transform("max")
    a = a.assign(target_quarter=[quarter_label(y, q) for y, q in zip(a["year"], a["quarter"])],
                 first_seen_utc=_stamp(seen_at),
                 data_vintage=[quarter_label(v // 4, v % 4 + 1) for v in vintage])
    return a[FIRST_SEEN_COLUMNS].reset_index(drop=True)


def update_first_seen(actuals: pd.DataFrame, path: Path, seen_at: datetime) -> pd.DataFrame:
    """Append new first-seen rows to `path` (never rewrites rows); return the full store."""
    path = Path(path)
    store = load_first_seen(path)
    new = new_first_seen(actuals, store, seen_at)
    if len(new):
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists() and path.stat().st_size > 0
        new.to_csv(path, mode="a", header=not exists, index=False)
    return pd.concat([store, new], ignore_index=True) if len(store) else new


def _as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1"])


def score_table(forecasts: pd.DataFrame, actuals: pd.DataFrame, first_seen: pd.DataFrame,
                scored_at: datetime) -> pd.DataFrame:
    """Live-accuracy rows (LIVE_COLUMNS) from logged forecasts (in-memory core).

    `actuals`: current modeling table (latest PSA values). `first_seen`: the full
    first-seen store (already updated with `actuals`). See the section comment above for
    which forecast counts.
    """
    if forecasts.empty:
        return pd.DataFrame(columns=LIVE_COLUMNS)
    f = _keys(forecasts)
    f["backfill"] = _as_bool(f["backfill"])
    f["made_at"] = pd.to_datetime(f["made_at_utc"], utc=True)
    fs = _keys(first_seen)[[*UNIT_KEYS, "first_seen_utc", "yield_t_ha", "production_mt"]]
    fs = fs.rename(columns={"yield_t_ha": "actual_first_yield_t_ha",
                            "production_mt": "actual_first_production_mt"})
    lt = _keys(actuals.loc[actuals["yield_t_ha"].notna()])
    if "production_mt" not in lt:
        lt["production_mt"] = np.nan
    lt = lt[[*UNIT_KEYS, "yield_t_ha", "production_mt"]].rename(
        columns={"yield_t_ha": "actual_latest_yield_t_ha",
                 "production_mt": "actual_latest_production_mt"})
    f = f.merge(fs, on=UNIT_KEYS, how="left").merge(lt, on=UNIT_KEYS, how="left")
    seen = pd.to_datetime(f["first_seen_utc"], utc=True)
    f["eligible"] = seen.isna() | (f["made_at"] < seen)
    key = ["target_quarter", "reporting_unit", "crop", "backfill"]
    # eligible rows sort last, so tail(1) is the last eligible forecast when there is one
    f = (f.sort_values(["eligible", "made_at", "forecast_id"])
         .groupby(key, sort=False).tail(1))
    published = set(zip(fs["crop"], fs["year"], fs["quarter"]))
    has_actual = f["actual_first_yield_t_ha"].notna()
    pub = pd.Series([(c, y, q) in published
                     for c, y, q in zip(f["crop"], f["year"], f["quarter"])], index=f.index)
    status = np.select(
        [f["backfill"], ~f["eligible"], has_actual, pub],
        [STATUS_BACKFILL, STATUS_LATE, STATUS_SCORED, STATUS_NOT_REPORTED],
        default=STATUS_AWAITING)
    f["status"] = status
    f["kind"] = np.where(f["backfill"], "backfill", "live")
    f["counted"] = f["status"] == STATUS_SCORED
    first, latest = f["actual_first_yield_t_ha"], f["actual_latest_yield_t_ha"]
    f["revised"] = first.notna() & latest.notna() & ~np.isclose(
        first.fillna(0).to_numpy(float), latest.fillna(0).to_numpy(float))
    area = f["predicted_area_ha"].astype(float)
    for b in _BASES:
        act = f[f"actual_{b}_yield_t_ha"].astype(float)
        act_p = f[f"actual_{b}_production_mt"].astype(float)
        for m in _MODELS3:
            pred = f[f"{m}_yield"].astype(float)
            f[f"{m}_err_{b}"] = pred - act
            f[f"{m}_prod_err_{b}"] = pred * area - act_p  # NaN when no area forecast
        hit = (f["b3_lo80"].astype(float) <= act) & (act <= f["b3_hi80"].astype(float))
        f[f"b3_in80_{b}"] = hit.astype("boolean").mask(act.isna())
    f["scored_at_utc"] = _stamp(scored_at)
    out = f[LIVE_COLUMNS].sort_values(["year", "quarter", "crop", "reporting_unit", "kind"])
    return out.reset_index(drop=True)


def _err_stats(err: pd.Series, actual: pd.Series) -> dict[str, float]:
    e = err.to_numpy(float)
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mape": float(np.mean(np.abs(e) / actual.to_numpy(float)) * 100.0)}


def summarize_live(scored: pd.DataFrame, crops: list[str], seed: int = 42,
                   n_boot: int = 2000,
                   min_quarters: int = MIN_BOOTSTRAP_QUARTERS) -> pd.DataFrame:
    """Live-accuracy metrics per basis x (crop | crop x unit); see the section comment."""
    from cropyield.models.evaluate import bootstrap_mae_diff
    live = scored.loc[scored["kind"] == "live"] if len(scored) else scored
    rows = []
    for b in _BASES:
        act_col = f"actual_{b}_yield_t_ha"
        d = live.loc[live["counted"].astype(bool) & live[act_col].notna()] if len(live) \
            else live
        for crop in crops:
            groups = [("crop", "all", d.loc[d["crop"] == crop] if len(d) else d)]
            units = sorted(live.loc[live["crop"] == crop, "reporting_unit"].unique()) \
                if len(live) else []
            groups += [("crop_unit", u, d.loc[(d["crop"] == crop) & (d["reporting_unit"] == u)])
                       for u in units]
            for level, unit, g in groups:
                r: dict[str, Any] = {"basis": b, "level": level, "crop": crop,
                                     "reporting_unit": unit,
                                     "n_quarters": int(g["target_quarter"].nunique())
                                     if len(g) else 0, "n": int(len(g))}
                r["status"] = STATUS_SCORED if len(g) else STATUS_AWAITING
                for m in _MODELS3:
                    st = (_err_stats(g[f"{m}_err_{b}"], g[act_col]) if len(g)
                          else dict.fromkeys(("mae", "rmse", "mape"), np.nan))
                    r.update({f"{m}_{k}": v for k, v in st.items()})
                r["coverage80"] = (float(g[f"b3_in80_{b}"].astype(bool).mean()) if len(g)
                                   else np.nan)
                gp = g.loc[g[f"b3_prod_err_{b}"].notna()] if len(g) else g
                r["n_production"] = int(len(gp))
                for m in _MODELS3:
                    r[f"{m}_prod_mae"] = (float(gp[f"{m}_prod_err_{b}"].abs().mean())
                                          if len(gp) else np.nan)
                for ref in ("b2", "b0"):
                    pre = f"b3_vs_{ref}"
                    r[f"{pre}_mae_diff"] = r["b3_mae"] - r[f"{ref}_mae"]
                    r[f"{pre}_ci_low"] = r[f"{pre}_ci_high"] = np.nan
                    if r["n_quarters"] >= min_quarters:
                        base = g[[*UNIT_KEYS, "target_quarter"]].assign(
                            yield_t_ha=g[act_col].astype(float))
                        res = bootstrap_mae_diff(
                            base.assign(pred_yield_t_ha=g["b3_yield"].astype(float)),
                            base.assign(pred_yield_t_ha=g[f"{ref}_yield"].astype(float)),
                            n=n_boot, by="target_quarter", seed=seed)
                        r[f"{pre}_ci_low"], r[f"{pre}_ci_high"] = res.ci_low, res.ci_high
                        r[f"{pre}_bootstrap"] = "ok"
                    else:
                        r[f"{pre}_bootstrap"] = (f"insufficient: {r['n_quarters']} scored "
                                                 f"quarters < {min_quarters}")
                rows.append(r)
    return pd.DataFrame(rows)


def score_forecasts(cfg: Config, scored_at: datetime | None = None) -> pd.DataFrame:
    """Score logged forecasts against PSA actuals (Requirement 2.4).

    Updates the first-seen store, writes `live_accuracy.csv` and
    `live_accuracy_summary.csv` under `paths.reports`, and returns the live-accuracy rows.
    A missing or empty forecasts.csv gives header-only rows and "awaiting PSA release"
    summary rows.
    """
    now = scored_at or datetime.now(timezone.utc)
    fpath = forecasts_path(cfg)
    if fpath.exists() and fpath.stat().st_size > 0:
        forecasts = pd.read_csv(fpath, dtype={"reporting_unit": str, "crop": str,
                                              "forecast_id": str, "backfill": str})
    else:
        forecasts = pd.DataFrame(columns=list(FORECAST_COLUMNS))
    actuals = load_modeling_table(cfg)
    store = update_first_seen(actuals, first_seen_path(cfg), now)
    scored = score_table(forecasts, actuals, store, now)
    out = live_accuracy_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False)
    summary = summarize_live(scored, list(cfg.pipeline.crops),
                             seed=int(cfg.pipeline.random_seed))
    summary.insert(0, "scored_at_utc", _stamp(now))
    summary.to_csv(live_summary_path(cfg), index=False)
    return scored


def score_main(argv: list[str] | None = None) -> None:
    """Score logged forecasts against PSA actuals; write reports/live_accuracy.csv."""
    ap = argparse.ArgumentParser(description=score_main.__doc__)
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR),
                    help="folder with pipeline.yaml and provinces.yaml (default: repo configs/)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config_dir)
    scored = score_forecasts(cfg)
    counts = scored["status"].value_counts().to_dict() if len(scored) else {}
    print(f"{len(scored)} forecast rows: {counts or 'none logged'}")
    summary = pd.read_csv(live_summary_path(cfg))
    crop = summary.loc[summary["level"] == "crop"]
    if not (crop["n"] > 0).any():
        print(f"no forecasts scored yet: {STATUS_AWAITING}")
    else:
        cols = ["basis", "crop", "n_quarters", "b3_mae", "b0_mae", "b2_mae", "coverage80",
                "b3_vs_b2_bootstrap"]
        print(crop[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {live_accuracy_path(cfg)} and {live_summary_path(cfg)}")


def main(argv: list[str] | None = None) -> None:
    """Forecast a quarter for every current reporting unit and crop; append to
    reports/forecasts.csv."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--quarter", required=True, help="target quarter, e.g. 2026Q4")
    ap.add_argument("--backfill", action="store_true",
                    help="allow an already-published quarter; rows are marked backfill")
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR),
                    help="folder with pipeline.yaml and provinces.yaml (default: repo configs/)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config_dir)
    try:
        df = make_forecast(args.quarter, cfg, backfill=args.backfill)
    except PublishedQuarterError as e:
        raise SystemExit(str(e)) from e
    path = log_forecast(df, forecasts_path(cfg))
    show = ["crop", "reporting_unit", "b3_yield", "b3_lo80", "b3_hi80", "b0_yield",
            "b2_yield", "predicted_area_ha", "pred_production_mt"]
    print(f"forecast {df['forecast_id'].iloc[0]} (data vintage "
          f"{', '.join(sorted(set(df['data_vintage'])))}, backfill={args.backfill})")
    print(df[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"appended {len(df)} rows to {path}")


if __name__ == "__main__":
    main()
