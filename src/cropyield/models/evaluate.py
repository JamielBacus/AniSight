"""Rolling-origin evaluation harness (phase 2, Requirement 3).

Conventions (design.md, "Modeling conventions"):
- Models predict `log_yield`; every metric is computed on `yield_t_ha` after a plain
  back-transform `pred_yield_t_ha = exp(pred_log_yield)` (no smearing correction, so the
  prediction is the conditional median of yield under log-normal errors).
- MAE and RMSE are in t/ha; MAPE is in percent: mean(|pred - actual| / actual) * 100.
- Folds come from `cropyield.models.data.rolling_folds`: train on era years before Y,
  test on year Y only.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from cropyield.config import DEFAULT_CONFIG_DIR, DEFAULT_ERAS, Config, Era, load_config
from cropyield.models.baselines import MeanPlusTrend, SameQuarterLastYear, UnitQuarterMean
from cropyield.models.data import (UNIT_KEYS, era_rows, fold_summary, load_modeling_table,
                                   rolling_folds)
from cropyield.models.satellite import (FEATURE_SET_ERA, FEATURE_SETS, AnomalyRidge,
                                        BoostedModel)

log = logging.getLogger(__name__)

# Target columns removed from the test fold before `predict`, so no model can read them.
TARGET_COLS = ["log_yield", "yield_t_ha", "production_mt"]

OOF_COLUMNS = [*UNIT_KEYS, "is_outlier", "log_yield", "pred_log_yield", "yield_t_ha",
               "pred_yield_t_ha", "n_train_years", "train_first_year"]

EXPERIMENT_COLUMNS = [
    "timestamp", "model", "era", "crop", "params", "features",
    "test_years", "n_test",
    "mae", "rmse", "mape",
    "n_test_no_outliers", "mae_no_outliers", "rmse_no_outliers", "mape_no_outliers",
    "ref_model", "mae_diff", "mae_diff_ci_low", "mae_diff_ci_high",
]


class Model(Protocol):
    def fit(self, train_df: pd.DataFrame) -> Any: ...
    def predict(self, df: pd.DataFrame) -> pd.Series: ...


def cross_validate(model_factory: Callable[[], Model], df: pd.DataFrame, era: str,
                   eras: Mapping[str, Era] = DEFAULT_ERAS) -> pd.DataFrame:
    """Out-of-fold predictions for one model over the rolling-origin folds of `era`.

    `df` is the modeling table (`load_modeling_table`); it is restricted to `era` rows
    here. For each test year Y, a fresh model from `model_factory()` is fitted on the
    era rows with year < Y and predicts the rows of year Y, with the target columns
    (TARGET_COLS) dropped from what `predict` sees.

    Returns one row per test row (index = `df` index) with OOF_COLUMNS.
    """
    if era not in eras:
        raise ValueError(f"unknown era {era!r}; expected one of {sorted(eras)}")
    d = era_rows(df, era, eras)
    years = d["year"].to_numpy()
    parts = []
    for train_idx, test_idx, y in rolling_folds(d, eras[era].first_test_year):
        train = d.iloc[train_idx]
        test = d.iloc[test_idx]
        model = model_factory()
        model.fit(train)
        pred = model.predict(test.drop(columns=[c for c in TARGET_COLS if c in test]))
        pred = pd.Series(pred, index=test.index).astype(float)
        if pred.isna().any():
            raise ValueError(f"{type(model).__name__} returned NaN predictions for {y}")
        train_years = np.unique(years[train_idx])
        part = test[[*UNIT_KEYS, "log_yield", "yield_t_ha"]].copy()
        part["is_outlier"] = (test["is_outlier"].astype(bool) if "is_outlier" in test
                              else False)
        part["pred_log_yield"] = pred
        part["pred_yield_t_ha"] = np.exp(pred)
        part["n_train_years"] = len(train_years)
        part["train_first_year"] = int(train_years.min())
        parts.append(part[OOF_COLUMNS])
    if not parts:
        raise ValueError(f"era {era!r} has no test folds in this table")
    return pd.concat(parts)


def _errors(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - actual
    return {"n": int(len(err)),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mape": float(np.mean(np.abs(err) / actual) * 100.0)}


def metrics(oof: pd.DataFrame, by: str | Sequence[str] | None = None,
            exclude_outliers: bool = False) -> pd.DataFrame:
    """MAE, RMSE (t/ha) and MAPE (%) of `pred_yield_t_ha` vs `yield_t_ha`.

    `by=None` gives one overall row; otherwise one row per group (e.g. "crop",
    "reporting_unit", ["crop", "reporting_unit"]). `exclude_outliers=True` drops rows
    with `is_outlier` from the evaluated set (Requirement 3.4).
    Columns: <by...>, n, mae, rmse, mape.
    """
    d = oof.loc[~oof["is_outlier"].astype(bool)] if exclude_outliers else oof
    if (d["yield_t_ha"] <= 0).any():
        raise ValueError("yield_t_ha must be positive for MAPE")
    keys = [] if by is None else ([by] if isinstance(by, str) else list(by))
    cols = [*keys, "n", "mae", "rmse", "mape"]
    if not keys:
        if d.empty:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame([_errors(d["yield_t_ha"].to_numpy(float),
                                     d["pred_yield_t_ha"].to_numpy(float))])[cols]
    rows = []
    for k, g in d.groupby(keys, sort=True):
        k = k if isinstance(k, tuple) else (k,)
        rows.append({**dict(zip(keys, k)),
                     **_errors(g["yield_t_ha"].to_numpy(float),
                               g["pred_yield_t_ha"].to_numpy(float))})
    return pd.DataFrame(rows, columns=cols)


def metrics_report(oof: pd.DataFrame) -> pd.DataFrame:
    """Long table of `metrics` at every reporting level, with and without outliers.

    Levels: overall, crop, reporting_unit, crop x reporting_unit. Columns: subset
    ("all" | "no_outliers"), level, crop, reporting_unit, n, mae, rmse, mape
    (crop / reporting_unit are "all" where the level does not split by them).
    """
    levels = {"overall": [], "crop": ["crop"], "reporting_unit": ["reporting_unit"],
              "crop_unit": ["crop", "reporting_unit"]}
    parts = []
    for subset, excl in (("all", False), ("no_outliers", True)):
        for level, keys in levels.items():
            m = metrics(oof, by=keys or None, exclude_outliers=excl)
            for k in ("crop", "reporting_unit"):
                if k not in m:
                    m[k] = "all"
            m.insert(0, "level", level)
            m.insert(0, "subset", subset)
            parts.append(m)
    cols = ["subset", "level", "crop", "reporting_unit", "n", "mae", "rmse", "mape"]
    return pd.concat(parts, ignore_index=True)[cols]


@dataclass(frozen=True)
class BootstrapResult:
    """MAE(a) - MAE(b) in t/ha (negative = a is better) and its percentile CI."""
    diff: float
    ci_low: float
    ci_high: float
    n_boot: int
    n_groups: int
    level: float = 0.95


def bootstrap_mae_diff(oof_a: pd.DataFrame, oof_b: pd.DataFrame, n: int = 2000,
                       by: str = "year", seed: int = 42,
                       level: float = 0.95) -> BootstrapResult:
    """Paired bootstrap of MAE(a) - MAE(b), resampling whole `by` groups (test years).

    Both frames must cover exactly the same rows (matched on UNIT_KEYS) with the same
    actual yields. Each replicate draws as many groups as there are, with replacement,
    and computes the row-weighted MAE of each model over the drawn rows. The CI is the
    (1 - level)/2 and (1 + level)/2 percentiles of the replicates; the point estimate is
    the difference on the full set. Deterministic for a given `seed`.
    """
    a = oof_a.set_index(UNIT_KEYS)
    b = oof_b.set_index(UNIT_KEYS)
    if a.index.has_duplicates or b.index.has_duplicates:
        raise ValueError("out-of-fold tables have duplicate unit rows")
    if len(a) != len(b) or not a.index.sort_values().equals(b.index.sort_values()):
        raise ValueError("bootstrap_mae_diff needs both models scored on the same rows")
    b = b.loc[a.index]
    if not np.allclose(a["yield_t_ha"].to_numpy(float), b["yield_t_ha"].to_numpy(float)):
        raise ValueError("actual yields differ between the two out-of-fold tables")
    actual = a["yield_t_ha"].to_numpy(float)
    abs_a = np.abs(a["pred_yield_t_ha"].to_numpy(float) - actual)
    abs_b = np.abs(b["pred_yield_t_ha"].to_numpy(float) - actual)
    groups = a.index.get_level_values(by) if by in UNIT_KEYS else a[by]
    g = pd.DataFrame({"g": np.asarray(groups), "da": abs_a, "db": abs_b, "n": 1.0})
    g = g.groupby("g", sort=True).sum()
    sum_diff = (g["da"] - g["db"]).to_numpy()
    count = g["n"].to_numpy()
    point = float(sum_diff.sum() / count.sum())
    rng = np.random.default_rng(seed)
    k = len(g)
    draws = rng.integers(0, k, size=(n, k))
    reps = sum_diff[draws].sum(axis=1) / count[draws].sum(axis=1)
    lo, hi = np.quantile(reps, [(1 - level) / 2, (1 + level) / 2])
    return BootstrapResult(diff=point, ci_low=float(lo), ci_high=float(hi),
                           n_boot=n, n_groups=k, level=level)


def experiments_path(cfg: Config) -> Path:
    """`<paths.reports>/experiments.csv` (paths.reports comes from pipeline.yaml)."""
    return cfg.pipeline.paths.reports / "experiments.csv"


def _year_span(years: pd.Series) -> str:
    u = sorted(int(y) for y in years.unique())
    return f"{u[0]}-{u[-1]}" if u else ""


def log_experiment(cfg: Config, *, model: str, era: str, oof: pd.DataFrame,
                   params: Mapping[str, Any] | None = None,
                   features: Sequence[str] = (),
                   reference_oof: pd.DataFrame | None = None,
                   reference_name: str = "B2", seed: int | None = None,
                   timestamp: datetime | None = None) -> pd.DataFrame:
    """Append one row per crop in `oof` to `experiments_path(cfg)`; return those rows.

    Each row holds the run timestamp (UTC ISO 8601), model name, era, crop, params
    (JSON), feature list (JSON), tested years, metrics on all test rows and without
    outlier rows, and, when `reference_oof` is given, the paired bootstrap MAE
    difference vs that reference (default "B2") for the crop, resampled by test year
    with `seed` (default `cfg.pipeline.random_seed`).

    Never rewrites existing rows; raises ValueError if the existing file's header does
    not match EXPERIMENT_COLUMNS.
    """
    path = experiments_path(cfg)
    seed = cfg.pipeline.random_seed if seed is None else seed
    ts = (timestamp or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    rows = []
    for crop, g in oof.groupby("crop", sort=True):
        m = metrics(g).iloc[0]
        mo = metrics(g, exclude_outliers=True)
        mo = mo.iloc[0] if len(mo) else pd.Series(
            {"n": 0, "mae": np.nan, "rmse": np.nan, "mape": np.nan})
        row: dict[str, Any] = {
            "timestamp": ts, "model": model, "era": era, "crop": crop,
            "params": json.dumps(dict(params or {}), sort_keys=True, default=str),
            "features": json.dumps(list(features)),
            "test_years": _year_span(g["year"]), "n_test": int(m["n"]),
            "mae": m["mae"], "rmse": m["rmse"], "mape": m["mape"],
            "n_test_no_outliers": int(mo["n"]), "mae_no_outliers": mo["mae"],
            "rmse_no_outliers": mo["rmse"], "mape_no_outliers": mo["mape"],
            "ref_model": "", "mae_diff": np.nan, "mae_diff_ci_low": np.nan,
            "mae_diff_ci_high": np.nan,
        }
        if reference_oof is not None:
            ref = reference_oof.loc[reference_oof["crop"] == crop]
            bs = bootstrap_mae_diff(g, ref, seed=seed)
            row.update(ref_model=reference_name, mae_diff=bs.diff,
                       mae_diff_ci_low=bs.ci_low, mae_diff_ci_high=bs.ci_high)
        rows.append(row)
    out = pd.DataFrame(rows, columns=EXPERIMENT_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        header = list(pd.read_csv(path, nrows=0).columns)
        if header != EXPERIMENT_COLUMNS:
            raise ValueError(f"{path} has columns {header}, expected {EXPERIMENT_COLUMNS}")
    out.to_csv(path, mode="a", header=not exists, index=False)
    return out


# --- baseline evaluation run (`cropyield evaluate`) ----------------------------------

BASELINE_PARAMS: dict[str, dict[str, Any]] = {
    "B0": {"class": "SameQuarterLastYear", "fallback": "B1"},
    "B1": {"class": "UnitQuarterMean", "fallback": "crop-quarter > crop > global mean"},
    "B2": {"class": "MeanPlusTrend", "trend": "per-crop pooled within-group slope"},
}
MIN_BOOTSTRAP_YEARS = 8  # warn when a vs-B2 CI rests on fewer test years than this


class _CountingB0(SameQuarterLastYear):
    """B0 that remembers which test rows used the B1 fallback (`fallback_`)."""

    def predict(self, df: pd.DataFrame) -> pd.Series:
        self.fallback_ = self.fallback_mask(df)
        return super().predict(df)


@dataclass
class BaselineRun:
    era: str
    oof: dict[str, pd.DataFrame]   # model name -> out-of-fold predictions
    b0_fallback: pd.Series         # bool, aligned to oof["B0"].index
    folds: pd.DataFrame            # fold_summary of the era rows
    logged: pd.DataFrame           # rows appended to experiments.csv
    report: pd.DataFrame           # metrics_report per model (column "model")


def evaluate_baselines(cfg: Config, df: pd.DataFrame, era: str,
                       timestamp: datetime | None = None) -> BaselineRun:
    """Rolling-origin CV of B0, B1, B2 on `era`; log each to experiments.csv vs B2.

    `df` is the modeling table (`load_modeling_table`). Eras and the bootstrap seed
    come from `cfg.pipeline`. All three models share one timestamp.
    """
    eras = cfg.pipeline.eras
    ts = timestamp or datetime.now(timezone.utc)
    b0_models: list[_CountingB0] = []

    def b0_factory() -> _CountingB0:
        b0_models.append(_CountingB0())
        return b0_models[-1]

    factories: dict[str, Callable[[], Model]] = {
        "B0": b0_factory, "B1": UnitQuarterMean, "B2": MeanPlusTrend}
    oof = {name: cross_validate(f, df, era, eras=eras) for name, f in factories.items()}
    fallback = pd.concat([m.fallback_ for m in b0_models]).loc[oof["B0"].index].astype(bool)

    logged, reports = [], []
    for name in ("B2", "B1", "B0"):
        params = dict(BASELINE_PARAMS[name])
        if name == "B0":
            params["n_fallback"] = int(fallback.sum())
        ref = None if name == "B2" else oof["B2"]
        logged.append(log_experiment(cfg, model=name, era=era, oof=oof[name],
                                     params=params, features=[], reference_oof=ref,
                                     timestamp=ts))
        rep = metrics_report(oof[name])
        rep.insert(0, "era", era)
        rep.insert(0, "model", name)
        reports.append(rep)
    first_test = eras[era].first_test_year
    return BaselineRun(era=era, oof=oof, b0_fallback=fallback,
                       folds=fold_summary(era_rows(df, era, eras), first_test),
                       logged=pd.concat(logged, ignore_index=True),
                       report=pd.concat(reports, ignore_index=True))


def baseline_metrics_path(cfg: Config) -> Path:
    """`<paths.reports>/baseline_metrics.csv`: per-unit/crop metrics of the last run."""
    return cfg.pipeline.paths.reports / "baseline_metrics.csv"


def _summary(run: BaselineRun) -> str:
    """Plain-text summary of one era's run for the terminal."""
    lines = [f"== era {run.era} =="]
    f = run.folds
    lines.append(f"folds: {len(f)} test years {f['test_year'].min()}-{f['test_year'].max()}, "
                 f"training years per fold {f['n_train_years'].min()}-"
                 f"{f['n_train_years'].max()} (from {f['train_first_year'].min()})")
    lg = run.logged.copy()
    cols = ["model", "crop", "test_years", "n_test", "mae", "rmse", "mape",
            "n_test_no_outliers", "mae_no_outliers", "rmse_no_outliers",
            "mape_no_outliers", "mae_diff", "mae_diff_ci_low", "mae_diff_ci_high"]
    lg = lg[cols].sort_values(["crop", "model"])
    lines.append(lg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    fb = run.b0_fallback
    by_crop = fb.groupby(run.oof["B0"]["crop"]).sum()
    lines.append(f"B0 used the B1 fallback for {int(fb.sum())} of {len(fb)} test rows ("
                 + ", ".join(f"{c}: {int(n)}" for c, n in by_crop.items()) + ")")
    return "\n".join(lines)


# --- satellite model evaluation (`cropyield evaluate --models satellite|all`) ---------

SATELLITE_MODELS = ("AnomalyRidge", "BoostedModel")
RIDGE_ALPHA = 1.0


class _CountingRidge(AnomalyRidge):
    """AnomalyRidge that remembers which test rows got plain B2 (`fallback_`)."""

    def predict(self, df: pd.DataFrame) -> pd.Series:
        self.fallback_ = self.fallback_mask(df)
        return super().predict(df)


@dataclass
class SatelliteRun:
    feature_set: str
    era: str
    oof: dict[str, pd.DataFrame]     # model name -> out-of-fold predictions
    ridge_fallback: pd.Series        # bool, aligned to oof["AnomalyRidge"].index
    ridge_train_dropped: pd.DataFrame  # per fold: test_year, n_train, n_train_dropped
    logged: pd.DataFrame
    report: pd.DataFrame


def evaluate_satellite(cfg: Config, df: pd.DataFrame, feature_set: str,
                       b2_oof: pd.DataFrame,
                       timestamp: datetime | None = None,
                       era: str | None = None) -> SatelliteRun:
    """Rolling-origin CV of AnomalyRidge and BoostedModel on one feature set.

    The era is `era` if given, else FEATURE_SET_ERA[feature_set]; `b2_oof` must be B2's
    out-of-fold predictions on that era (same rows), used as the bootstrap reference.
    Each model is logged to experiments.csv as "<model>:<feature_set>" (the `era` column
    tells apart a set evaluated on two eras) with the feature list and, for the ridge,
    its B2-fallback and dropped-training-row counts.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; expected {sorted(FEATURE_SETS)}")
    era = era or FEATURE_SET_ERA[feature_set]
    if era not in cfg.pipeline.eras:
        raise ValueError(f"unknown era {era!r}; expected one of {sorted(cfg.pipeline.eras)}")
    feats = FEATURE_SETS[feature_set]
    eras = cfg.pipeline.eras
    ts = timestamp or datetime.now(timezone.utc)
    ridges: list[_CountingRidge] = []

    def ridge_factory() -> _CountingRidge:
        ridges.append(_CountingRidge(feats, alpha=RIDGE_ALPHA))
        return ridges[-1]

    factories: dict[str, Callable[[], Model]] = {
        "AnomalyRidge": ridge_factory, "BoostedModel": lambda: BoostedModel(feats)}
    oof = {name: cross_validate(f, df, era, eras=eras) for name, f in factories.items()}
    fallback = pd.concat([m.fallback_ for m in ridges]).loc[
        oof["AnomalyRidge"].index].astype(bool)
    test_years = sorted(oof["AnomalyRidge"]["year"].unique())
    dropped = pd.DataFrame({"test_year": test_years,
                            "n_train": [m.n_train_ for m in ridges],
                            "n_train_dropped": [m.n_train_dropped_ for m in ridges]})
    ridge_crop = oof["AnomalyRidge"]["crop"]
    params = {
        "AnomalyRidge": {"class": "AnomalyRidge", "alpha": RIDGE_ALPHA,
                         "feature_set": feature_set, "n_fallback": int(fallback.sum()),
                         "n_fallback_by_crop": {c: int(n) for c, n in
                                                fallback.groupby(ridge_crop).sum().items()},
                         "n_train_dropped_by_fold": dict(zip(
                             map(str, dropped["test_year"]),
                             map(int, dropped["n_train_dropped"])))},
        "BoostedModel": {"class": "BoostedModel",
                         "estimator": "HistGradientBoostingRegressor",
                         "early_stopping": False, "random_state": 42,
                         "feature_set": feature_set},
    }
    logged, reports = [], []
    for name in SATELLITE_MODELS:
        label = f"{name}:{feature_set}"
        logged.append(log_experiment(cfg, model=label, era=era, oof=oof[name],
                                     params=params[name], features=feats,
                                     reference_oof=b2_oof, timestamp=ts))
        rep = metrics_report(oof[name])
        rep.insert(0, "era", era)
        rep.insert(0, "model", label)
        reports.append(rep)
    return SatelliteRun(feature_set=feature_set, era=era, oof=oof,
                        ridge_fallback=fallback, ridge_train_dropped=dropped,
                        logged=pd.concat(logged, ignore_index=True),
                        report=pd.concat(reports, ignore_index=True))


def _satellite_summary(run: SatelliteRun) -> str:
    lines = [f"== feature set {run.feature_set} (era {run.era}) =="]
    cols = ["model", "crop", "test_years", "n_test", "mae", "rmse", "mape",
            "n_test_no_outliers", "mae_no_outliers", "mae_diff", "mae_diff_ci_low",
            "mae_diff_ci_high"]
    lg = run.logged[cols].sort_values(["crop", "model"])
    lines.append(lg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    fb = run.ridge_fallback
    by_crop = fb.groupby(run.oof["AnomalyRidge"]["crop"]).sum()
    lines.append(f"AnomalyRidge used the B2 fallback for {int(fb.sum())} of {len(fb)} test "
                 "rows (" + ", ".join(f"{c}: {int(n)}" for c, n in by_crop.items()) + ")")
    d = run.ridge_train_dropped
    lines.append("AnomalyRidge training rows dropped (missing features) per fold: "
                 + ", ".join(f"{y}: {k} of {n + k}" for y, n, k in zip(d["test_year"], d["n_train"],
                                                d["n_train_dropped"])))
    return "\n".join(lines)


def model_metrics_path(cfg: Config) -> Path:
    """`<paths.reports>/model_metrics.csv`: per-unit/crop metrics of satellite models."""
    return cfg.pipeline.paths.reports / "model_metrics.csv"


def parse_feature_sets(spec: str | None, eras: Sequence[str],
                       known_eras: Sequence[str]) -> list[tuple[str, str]]:
    """(feature_set, era) pairs to evaluate.

    `spec` is None (every FEATURE_SETS entry on its FEATURE_SET_ERA) or a
    comma-separated list of `name` or `name@era` (e.g. "rain,modis_rain,s2_rain,rain@s2").
    Only pairs whose era is in `eras` are kept; raises ValueError for unknown names or
    eras and when nothing is left.
    """
    if spec is None:
        pairs = [(fs, FEATURE_SET_ERA[fs]) for fs in FEATURE_SETS]
    else:
        pairs = []
        for tok in (t.strip() for t in spec.split(",")):
            if not tok:
                continue
            name, _, era = tok.partition("@")
            if name not in FEATURE_SETS:
                raise ValueError(f"unknown feature set {name!r}; expected {sorted(FEATURE_SETS)}")
            era = era or FEATURE_SET_ERA[name]
            if era not in known_eras:
                raise ValueError(f"unknown era {era!r} in {tok!r}; expected one of "
                                 f"{sorted(known_eras)}")
            if (name, era) not in pairs:
                pairs.append((name, era))
    pairs = [(fs, e) for fs, e in pairs if e in eras]
    if not pairs:
        raise ValueError(f"no feature sets to evaluate for era(s) {list(eras)}")
    return pairs


def main(argv: list[str] | None = None) -> None:
    """Rolling-origin CV of baselines and/or satellite models; log to experiments.csv."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR),
                    help="folder with pipeline.yaml and provinces.yaml (default: repo configs/)")
    ap.add_argument("--era", default="all",
                    help="era name from modeling.eras (modis, s2) or 'all' (default)")
    ap.add_argument("--models", default="baselines",
                    choices=["baselines", "satellite", "all"],
                    help="baselines (default): B0/B1/B2; satellite: AnomalyRidge and "
                         "BoostedModel per feature set, vs B2 (B2 recomputed, not "
                         "logged); all: both")
    ap.add_argument("--feature-sets", default=None,
                    help="satellite models only: comma-separated feature sets, each "
                         "'name' (its default era) or 'name@era', e.g. "
                         "'rain,modis_rain,s2_rain,rain@s2' (default: every set on its "
                         f"default era; sets: {', '.join(FEATURE_SETS)})")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config_dir)
    eras = list(cfg.pipeline.eras) if args.era == "all" else [args.era]
    unknown = [e for e in eras if e not in cfg.pipeline.eras]
    if unknown:
        ap.error(f"unknown era {unknown[0]!r}; expected one of "
                 f"{sorted(cfg.pipeline.eras)} or 'all'")
    if args.feature_sets is not None and args.models == "baselines":
        ap.error("--feature-sets needs --models satellite or all")
    pairs: list[tuple[str, str]] = []
    if args.models in ("satellite", "all"):
        try:
            pairs = parse_feature_sets(args.feature_sets, eras, list(cfg.pipeline.eras))
        except ValueError as e:
            ap.error(str(e))
    df = load_modeling_table(cfg)
    ts = datetime.now(timezone.utc)
    n_logged = 0
    written = []
    b2_oof: dict[str, pd.DataFrame] = {}
    if args.models in ("baselines", "all"):
        runs = [evaluate_baselines(cfg, df, e, timestamp=ts) for e in eras]
        for run in runs:
            print(_summary(run))
            n_years = len(run.folds)
            if n_years < MIN_BOOTSTRAP_YEARS:
                log.warning("era %s: only %d test years (%s); vs-B2 bootstrap CIs are "
                            "coarse", run.era, n_years, run.logged["test_years"].iloc[0])
            b2_oof[run.era] = run.oof["B2"]
        out = baseline_metrics_path(cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([r.report for r in runs], ignore_index=True).to_csv(out, index=False)
        n_logged += sum(len(r.logged) for r in runs)
        written.append(out)
    if args.models in ("satellite", "all"):
        sat_runs = []
        for fs, era in pairs:
            if era not in b2_oof:
                b2_oof[era] = cross_validate(MeanPlusTrend, df, era, eras=cfg.pipeline.eras)
            sat_runs.append(evaluate_satellite(cfg, df, fs, b2_oof[era], timestamp=ts,
                                               era=era))
            print(_satellite_summary(sat_runs[-1]))
        out = model_metrics_path(cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        new = pd.concat([r.report for r in sat_runs], ignore_index=True)
        if out.exists() and out.stat().st_size > 0:
            # keep other (model, era) rows from earlier runs; replace the re-run ones
            old = pd.read_csv(out)
            key = ["model", "era"]
            rerun = old.set_index(key).index.isin(new.set_index(key).index)
            new = pd.concat([old.loc[~rerun], new], ignore_index=True)
        new.to_csv(out, index=False)
        n_logged += sum(len(r.logged) for r in sat_runs)
        written.append(out)
    print(f"appended {n_logged} rows to {experiments_path(cfg)}; "
          f"wrote {', '.join(map(str, written))}")


if __name__ == "__main__":
    main()
