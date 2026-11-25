"""Model selection, final models and predictions (phase 2, Requirement 6; task 8).

Selection rule (the user's version, which overrides Requirement 6.1's "simplest model"):
- The MODIS era decides: candidates are scored on MODIS-era out-of-fold rows only
  (rolling origin, test years 2012-latest). Sentinel-2-era results are reported, not used.
- Candidates per crop: B0, B2, B3 (`B0B2Blend`, the row-wise mean of the B0 and B2
  log-predictions; added post hoc) and the best satellite/rainfall model, i.e. the
  lowest-MAE MODIS-era row of experiments.csv among AnomalyRidge / BoostedModel x
  {modis, rain, modis_rain} (as logged: ridge alpha 1.0, no further tuning).
- Per crop, pick the candidate with the lowest MAE among those whose MAE - MAE(B2) 95%
  paired bootstrap CI (resampled by test year, seed from config) lies below zero; if
  none, pick B2.

Final models: the selected model is fitted on all MODIS-era rows of both crops (every
model is pooled over crops, exactly as in cross-validation) and saved with its feature
list, training period and forecast interval to `<paths.models>/<crop>.joblib`.

Predictions (`<paths.processed>/predictions.parquet`): out-of-fold history of the
selected model per crop with an 80% interval and "conditions vs normal" columns; see
PREDICTION_COLUMNS for every column's definition.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from cropyield.config import DEFAULT_CONFIG_DIR, Config, load_config
from cropyield.models.baselines import (GROUP, B0B2Blend, MeanPlusTrend,
                                        SameQuarterLastYear, UnitQuarterMean)
from cropyield.models.data import (UNIT_KEYS, era_rows, fold_summary, load_modeling_table,
                                   rolling_folds)
from cropyield.models.evaluate import (RIDGE_ALPHA, Model, bootstrap_mae_diff,
                                       cross_validate, dedupe_experiments,
                                       experiments_path, log_experiment, metrics)
from cropyield.models.satellite import FEATURE_SETS, AnomalyRidge, BoostedModel

log = logging.getLogger(__name__)

SELECTION_ERA = "modis"
SATELLITE_CANDIDATE_MODELS = ("AnomalyRidge", "BoostedModel")
SATELLITE_CANDIDATE_SETS = ("modis", "rain", "modis_rain")
BASELINE_CANDIDATES = ("B0", "B2", "B3")
REFERENCE = "B2"
NAIVE = "B0"

INTERVAL_QUANTILES = (0.10, 0.90)  # 80% interval
MIN_INTERVAL_YEARS = 3  # earlier test years of residuals needed before a fold gets one
MIN_ANOMALY_YEARS = 3  # earlier years needed before an NDVI/EVI anomaly is defined

# "Conditions vs normal" columns for the app (not inputs of the selected models).
ANOMALY_SOURCE_FEATURES = ["modis_ndvi_peak", "modis_ndvi_integral",
                           "modis_evi_peak", "modis_evi_integral"]
CONDITIONS_COLUMNS: dict[str, str] = {
    **{f"{f}_anom": f"MODIS {f.split('_')[1].upper()} {f.split('_')[2]} vs normal"
       for f in ANOMALY_SOURCE_FEATURES},
    "rain_anom": "Rainfall vs normal (mm)",
}

PREDICTION_COLUMNS: dict[str, str] = {
    "reporting_unit": "PSA reporting unit (combined units as reported, e.g. 'A+B')",
    "crop": "palay or corn",
    "year": "calendar year",
    "quarter": "calendar quarter 1-4",
    "quality_flag": "Phase 1 quality flag of the PSA row",
    "is_outlier": "quality_flag contains 'outlier'",
    "model": "selected model for this crop (B0, B2, B3 or '<model>:<feature_set>')",
    "actual_yield_t_ha": "PSA yield (t/ha)",
    "pred_yield_t_ha": "out-of-fold prediction of the selected model, exp(pred log-yield)",
    "pred_lo_t_ha": "80% interval low: exp(pred log-yield + 10th pct of earlier-fold "
                    "log-residuals of this crop); NaN for the first folds",
    "pred_hi_t_ha": "80% interval high: same with the 90th percentile",
    "interval_n_years": "earlier test years whose residuals built the interval",
    "b2_pred_yield_t_ha": "out-of-fold B2 (unit-quarter mean + trend) prediction",
    "b0_pred_yield_t_ha": "out-of-fold B0 (same quarter last year) prediction",
    "b0_fallback": "B0 had no same-quarter value for year-1 and used B1 (unit-quarter "
                   "mean); happens at reporting-unit regroupings",
    "area_ha": "PSA harvested area (ha)",
    "actual_production_mt": "PSA production (t)",
    "pred_production_mt": "pred_yield_t_ha x area_ha (PSA area, known at release)",
    "n_train_years": "training years in this row's fold",
    **{f"{f}_anom": (f"{f} minus its mean over earlier years (< year, MODIS era from "
                     f"2002) of the same reporting unit, crop and quarter; NaN with fewer "
                     f"than {MIN_ANOMALY_YEARS} earlier values. Conditions vs normal; not "
                     "a model input")
       for f in ANOMALY_SOURCE_FEATURES},
    "rain_anom": ("growing-window rainfall (mm) minus the unit's fixed 1981-2000 ERA5 "
                  "month-of-year climatology (features.py). Conditions vs normal; not a "
                  "model input"),
}


# --- candidates --------------------------------------------------------------------

def model_factory(label: str) -> Callable[[], Model]:
    """Fresh-model factory for a model label as used in experiments.csv.

    Labels: B0, B1, B2, B3, or "<AnomalyRidge|BoostedModel>:<feature_set>" (ridge with
    the logged alpha RIDGE_ALPHA).
    """
    simple: dict[str, Callable[[], Model]] = {
        "B0": SameQuarterLastYear, "B1": UnitQuarterMean, "B2": MeanPlusTrend,
        "B3": B0B2Blend}
    if label in simple:
        return simple[label]
    name, _, fs = label.partition(":")
    if fs not in FEATURE_SETS:
        raise ValueError(f"unknown model label {label!r}")
    feats = FEATURE_SETS[fs]
    if name == "AnomalyRidge":
        return lambda: AnomalyRidge(feats, alpha=RIDGE_ALPHA)
    if name == "BoostedModel":
        return lambda: BoostedModel(feats)
    raise ValueError(f"unknown model label {label!r}")


def model_features(label: str) -> list[str]:
    """Input feature columns of a model label (baselines: none)."""
    _, _, fs = label.partition(":")
    return list(FEATURE_SETS[fs]) if fs else []


def best_satellite_model(experiments: pd.DataFrame, crop: str, era: str = SELECTION_ERA,
                         models: Sequence[str] = SATELLITE_CANDIDATE_MODELS,
                         feature_sets: Sequence[str] = SATELLITE_CANDIDATE_SETS) -> str:
    """Label of the lowest-MAE logged satellite/rainfall model for `crop` on `era`."""
    labels = {f"{m}:{fs}" for m in models for fs in feature_sets}
    e = experiments.loc[experiments["model"].isin(labels) & (experiments["era"] == era)
                        & (experiments["crop"] == crop)]
    if e.empty:
        raise ValueError(f"experiments.csv has no {sorted(labels)} rows for {crop}/{era}; "
                         "run `cropyield evaluate --models satellite` first")
    e = e.sort_values(["mae", "model"])  # ties broken by name, deterministic
    return str(e.iloc[0]["model"])


CANDIDATE_COLUMNS = ["crop", "candidate", "role", "n", "mae", "rmse", "mape",
                     "d_b2", "d_b2_lo", "d_b2_hi", "d_b0", "d_b0_lo", "d_b0_hi",
                     "qualifies", "selected"]


def candidate_table(oofs: Mapping[str, pd.DataFrame], crop: str,
                    roles: Mapping[str, str], seed: int = 42) -> pd.DataFrame:
    """MAE/RMSE/MAPE and paired-bootstrap MAE differences vs B2 and vs B0 for one crop.

    `oofs` maps candidate label -> out-of-fold predictions (must include B2 and B0, all
    on the same rows). Differences are MAE(candidate) - MAE(reference) in t/ha with 95%
    CIs resampled by test year (`bootstrap_mae_diff`, `seed`). `qualifies` is True when
    the vs-B2 CI upper bound is below zero. `selected` is filled by `select_candidate`.
    """
    sub = {k: v.loc[v["crop"] == crop] for k, v in oofs.items()}
    rows = []
    for name in roles:
        g = sub[name]
        m = metrics(g).iloc[0]
        row: dict[str, Any] = {"crop": crop, "candidate": name, "role": roles[name],
                               "n": int(m["n"]), "mae": m["mae"], "rmse": m["rmse"],
                               "mape": m["mape"]}
        for ref, key in ((REFERENCE, "d_b2"), (NAIVE, "d_b0")):
            if name == ref:
                row.update({key: 0.0, f"{key}_lo": np.nan, f"{key}_hi": np.nan})
            else:
                bs = bootstrap_mae_diff(g, sub[ref], seed=seed)
                row.update({key: bs.diff, f"{key}_lo": bs.ci_low, f"{key}_hi": bs.ci_high})
        row["qualifies"] = bool(name != REFERENCE and row["d_b2_hi"] < 0)
        row["selected"] = False
        rows.append(row)
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def select_candidate(table: pd.DataFrame) -> tuple[str, str]:
    """Apply the selection rule to one crop's `candidate_table`; return (label, reason)."""
    q = table.loc[table["qualifies"]].sort_values(["mae", "candidate"])
    if q.empty:
        return REFERENCE, (f"no candidate's MAE - MAE({REFERENCE}) 95% CI lies below zero, "
                           f"so the rule falls back to {REFERENCE}")
    r = q.iloc[0]
    others = [c for c in q["candidate"] if c != r["candidate"]]
    reason = (f"{r['candidate']} has the lowest MAE ({r['mae']:.4f} t/ha) among the "
              f"candidates whose MAE - MAE({REFERENCE}) 95% CI excludes zero "
              f"({r['d_b2']:+.4f} [{r['d_b2_lo']:+.4f}, {r['d_b2_hi']:+.4f}])")
    if others:
        reason += "; also qualifying: " + ", ".join(others)
    return str(r["candidate"]), reason


# --- intervals, fallbacks, anomalies -----------------------------------------------

def earlier_fold_intervals(oof: pd.DataFrame,
                           quantiles: tuple[float, float] = INTERVAL_QUANTILES,
                           min_years: int = MIN_INTERVAL_YEARS) -> pd.DataFrame:
    """Prediction interval per out-of-fold row from earlier folds' residuals only.

    For crop c and test year Y, the bounds are `pred_log_yield + quantile(r)` where r are
    the out-of-fold log-residuals (`log_yield - pred_log_yield`) of crop c in test years
    < Y. Folds with fewer than `min_years` earlier test years get NaN bounds.
    Returns (index = oof.index) pred_lo_log, pred_hi_log, interval_n_years.
    """
    resid = oof["log_yield"] - oof["pred_log_yield"]
    out = pd.DataFrame({"pred_lo_log": np.nan, "pred_hi_log": np.nan,
                        "interval_n_years": 0}, index=oof.index)
    for _, g in oof.groupby("crop", sort=True):
        for y in sorted(g["year"].unique()):
            prior = g.index[g["year"] < y]
            this = g.index[g["year"] == y]
            n_years = g.loc[prior, "year"].nunique()
            out.loc[this, "interval_n_years"] = n_years
            if n_years < min_years:
                continue
            lo, hi = np.quantile(resid.loc[prior].to_numpy(float), quantiles)
            out.loc[this, "pred_lo_log"] = oof.loc[this, "pred_log_yield"] + lo
            out.loc[this, "pred_hi_log"] = oof.loc[this, "pred_log_yield"] + hi
    return out


def b0_fallback_oof(d: pd.DataFrame, first_test_year: int) -> pd.Series:
    """Per test row of the rolling folds of `d`: did B0 fall back to B1?"""
    parts = []
    for train_idx, test_idx, _ in rolling_folds(d, first_test_year):
        b0 = SameQuarterLastYear().fit(d.iloc[train_idx])
        parts.append(b0.fallback_mask(d.iloc[test_idx]))
    return pd.concat(parts).astype(bool) if parts else pd.Series(dtype=bool)


def expanding_anomalies(df: pd.DataFrame, features: Sequence[str],
                        min_years: int = MIN_ANOMALY_YEARS) -> pd.DataFrame:
    """Each feature minus its mean over earlier years of the same unit, crop and quarter.

    For a row in year Y the reference is the mean of the non-NaN values of that
    reporting_unit / crop / quarter in rows of `df` with year < Y (an expanding mean up
    to Y-1), so no value from year Y or later is used. NaN when fewer than `min_years`
    earlier values exist. Returns `<feature>_anom` columns aligned to `df.index`.
    """
    if df.duplicated(UNIT_KEYS).any():
        raise ValueError("expanding_anomalies needs unique reporting_unit/crop/year/quarter")
    d = df.sort_values([*GROUP, "year"])
    keys = [d[k] for k in GROUP]
    out = pd.DataFrame(index=d.index)
    for f in features:
        x = d[f].astype(float)
        prev = x.groupby(keys).shift()
        prev_mean = prev.groupby(keys).transform(lambda s: s.expanding().mean())
        prev_n = prev.notna().groupby(keys).cumsum()
        out[f"{f}_anom"] = (x - prev_mean).where(prev_n >= min_years)
    return out.loc[df.index]


# --- run ---------------------------------------------------------------------------

@dataclass
class SelectionResult:
    era: str
    oofs: dict[str, pd.DataFrame]       # label -> MODIS-era out-of-fold predictions
    candidates: pd.DataFrame            # candidate_table rows of every crop
    selected: dict[str, tuple[str, str]]  # crop -> (label, reason)
    predictions: pd.DataFrame
    coverage: pd.DataFrame              # per crop: n_with_interval, coverage, years
    folds: pd.DataFrame
    model_paths: dict[str, Path]
    consistency: pd.DataFrame           # recomputed vs logged MAE per (label, crop)
    timestamp: datetime


def models_dir(cfg: Config) -> Path:
    """`paths.models` from pipeline.yaml (default `<repo>/models`)."""
    p = cfg.pipeline.paths
    return p.models if p.models is not None else p.reports.parent / "models"


def predictions_path(cfg: Config) -> Path:
    return cfg.pipeline.paths.processed / "predictions.parquet"


def selection_path(cfg: Config) -> Path:
    return cfg.pipeline.paths.reports / "selection.csv"


def build_predictions(df: pd.DataFrame, selected: Mapping[str, str],
                      oofs: Mapping[str, pd.DataFrame], b0_fallback: pd.Series,
                      era_df: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold history of the selected model per crop (PREDICTION_COLUMNS)."""
    anoms = expanding_anomalies(era_df, ANOMALY_SOURCE_FEATURES)
    parts = []
    for crop, label in sorted(selected.items()):
        o = oofs[label].loc[oofs[label]["crop"] == crop]
        iv = earlier_fold_intervals(o)
        src = df.loc[o.index]
        p = o[UNIT_KEYS].copy()
        p["quality_flag"] = src["quality_flag"].astype(str)
        p["is_outlier"] = o["is_outlier"].astype(bool)
        p["model"] = label
        p["actual_yield_t_ha"] = o["yield_t_ha"]
        p["pred_yield_t_ha"] = o["pred_yield_t_ha"]
        p["pred_lo_t_ha"] = np.exp(iv["pred_lo_log"])
        p["pred_hi_t_ha"] = np.exp(iv["pred_hi_log"])
        p["interval_n_years"] = iv["interval_n_years"].astype(int)
        p["b2_pred_yield_t_ha"] = oofs[REFERENCE].loc[o.index, "pred_yield_t_ha"]
        p["b0_pred_yield_t_ha"] = oofs[NAIVE].loc[o.index, "pred_yield_t_ha"]
        p["b0_fallback"] = b0_fallback.loc[o.index].astype(bool)
        p["area_ha"] = src["area_ha"].astype(float)
        p["actual_production_mt"] = src["production_mt"].astype(float)
        p["pred_production_mt"] = p["pred_yield_t_ha"] * p["area_ha"]
        p["n_train_years"] = o["n_train_years"].astype(int)
        for c in anoms.columns:
            p[c] = anoms.loc[o.index, c]
        p["rain_anom"] = src["rain_anom"].astype(float)
        parts.append(p)
    out = pd.concat(parts).sort_values(UNIT_KEYS).reset_index(drop=True)
    return out[list(PREDICTION_COLUMNS)]


def interval_coverage(pred: pd.DataFrame) -> pd.DataFrame:
    """Share of rows with an interval whose actual yield lies inside it, per crop."""
    rows = []
    for crop, g in pred.groupby("crop", sort=True):
        h = g.loc[g["pred_lo_t_ha"].notna()]
        inside = ((h["actual_yield_t_ha"] >= h["pred_lo_t_ha"])
                  & (h["actual_yield_t_ha"] <= h["pred_hi_t_ha"]))
        rows.append({"crop": crop, "model": g["model"].iloc[0], "n_rows": len(g),
                     "n_with_interval": len(h), "coverage": float(inside.mean()),
                     "below": float((h["actual_yield_t_ha"] < h["pred_lo_t_ha"]).mean()),
                     "above": float((h["actual_yield_t_ha"] > h["pred_hi_t_ha"]).mean()),
                     "years": f"{int(h['year'].min())}-{int(h['year'].max())}",
                     "mean_width_t_ha": float((h["pred_hi_t_ha"] - h["pred_lo_t_ha"]).mean())})
    return pd.DataFrame(rows)


def fit_final(label: str, crop: str, era_df: pd.DataFrame, oof: pd.DataFrame,
              era: str, timestamp: datetime) -> dict[str, Any]:
    """Fit `label` on all `era_df` rows (both crops) and bundle it for `<crop>.joblib`."""
    model = model_factory(label)()
    model.fit(era_df)
    o = oof.loc[oof["crop"] == crop]
    resid = (o["log_yield"] - o["pred_log_yield"]).to_numpy(float)
    lo, hi = np.quantile(resid, INTERVAL_QUANTILES)
    uses_b0 = label in ("B0", "B3")
    notes = ("Predicts log(yield_t_ha); yield = exp(prediction). Fitted on every "
             f"{era}-era row of both crops (models are pooled over crops, as in "
             "cross-validation).")
    if uses_b0:
        notes += (" Uses B0: forecasting year Y needs the same unit/crop/quarter yield of "
                  "Y-1 in the data passed to fit(); refit with the latest PSA year before "
                  "forecasting. Where last year's value is missing (reporting-unit "
                  "regroupings, e.g. a unit's first year after it is split off), B0 "
                  "falls back to B1 (unit-quarter mean).")
    return {
        "crop": crop, "model_name": label, "model": model,
        "features": model_features(label), "era": era,
        "training_period": (int(era_df["year"].min()), int(era_df["year"].max())),
        "n_train_rows": int(len(era_df)),
        "needs_last_year": uses_b0,
        "interval": {"level": INTERVAL_QUANTILES[1] - INTERVAL_QUANTILES[0],
                     "log_resid_lo": float(lo), "log_resid_hi": float(hi),
                     "quantiles": INTERVAL_QUANTILES, "n_resid": int(len(resid)),
                     "source": f"all out-of-fold residuals of {crop}, test years "
                               f"{int(o['year'].min())}-{int(o['year'].max())}"},
        "notes": notes,
        "created": timestamp.isoformat(timespec="seconds"),
    }


def run_selection(cfg: Config, df: pd.DataFrame, timestamp: datetime | None = None,
                  write: bool = True) -> SelectionResult:
    """Cross-validate the candidates, select per crop, fit final models, build predictions.

    With `write=True`: logs B3 to experiments.csv (vs B2; earlier B3 rows replaced),
    writes `<paths.models>/<crop>.joblib`, predictions.parquet and reports/selection.csv.
    """
    era = SELECTION_ERA
    eras = cfg.pipeline.eras
    seed = cfg.pipeline.random_seed
    ts = timestamp or datetime.now(timezone.utc)
    exp_path = experiments_path(cfg)
    experiments = pd.read_csv(exp_path) if exp_path.exists() else pd.DataFrame(
        columns=["model", "era", "crop", "mae"])
    crops = sorted(df["crop"].unique())
    best_sat = {c: best_satellite_model(experiments, c, era) for c in crops}
    labels = [*BASELINE_CANDIDATES, *sorted(set(best_sat.values()))]
    oofs = {lab: cross_validate(model_factory(lab), df, era, eras=eras) for lab in labels}

    # traceability: recomputed MAE must match the logged one
    cons = []
    for lab in labels:
        for c in crops:
            mae = float(metrics(oofs[lab].loc[oofs[lab]["crop"] == c]).iloc[0]["mae"])
            hit = experiments.loc[(experiments["model"] == lab) & (experiments["era"] == era)
                                  & (experiments["crop"] == c), "mae"]
            logged = float(hit.iloc[-1]) if len(hit) else np.nan
            cons.append({"model": lab, "crop": c, "mae_recomputed": mae,
                         "mae_logged": logged})
            if np.isfinite(logged) and abs(mae - logged) > 1e-9:
                log.warning("%s %s: recomputed MAE %.6f differs from logged %.6f",
                            lab, c, mae, logged)
    consistency = pd.DataFrame(cons)

    d = era_rows(df, era, eras)
    first_test = eras[era].first_test_year
    fallback = b0_fallback_oof(d, first_test)

    if write:
        fb_crop = fallback.groupby(df.loc[fallback.index, "crop"]).sum()
        log_experiment(cfg, model="B3", era=era, oof=oofs["B3"],
                       params={"class": "B0B2Blend", "components": ["B0", "B2"],
                               "combine": "mean of log-predictions", "post_hoc": True,
                               "n_fallback_by_crop": {k: int(v) for k, v in fb_crop.items()}},
                       features=[], reference_oof=oofs[REFERENCE], seed=seed, timestamp=ts)
        dedupe_experiments(exp_path, models=["B3"])

    tables, selected = [], {}
    for c in crops:
        roles = {"B0": "naive baseline", "B2": "reference baseline",
                 "B3": "post hoc blend of B0 and B2",
                 best_sat[c]: "best satellite/rainfall model (chosen post hoc)"}
        t = candidate_table(oofs, c, roles, seed=seed)
        lab, reason = select_candidate(t)
        t["selected"] = t["candidate"] == lab
        tables.append(t)
        selected[c] = (lab, reason)
    candidates = pd.concat(tables, ignore_index=True)

    predictions = build_predictions(df, {c: s[0] for c, s in selected.items()}, oofs,
                                    fallback, d)
    coverage = interval_coverage(predictions)
    model_paths: dict[str, Path] = {}
    if write:
        mdir = models_dir(cfg)
        mdir.mkdir(parents=True, exist_ok=True)
        for c, (lab, reason) in selected.items():
            bundle = fit_final(lab, c, d, oofs[lab], era, ts)
            bundle["selection_reason"] = reason
            model_paths[c] = mdir / f"{c}.joblib"
            joblib.dump(bundle, model_paths[c])
        out = predictions_path(cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(out, index=False)
        sp = selection_path(cfg)
        sp.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(sp, index=False)
    return SelectionResult(era=era, oofs=oofs, candidates=candidates, selected=selected,
                           predictions=predictions, coverage=coverage,
                           folds=fold_summary(d, first_test), model_paths=model_paths,
                           consistency=consistency, timestamp=ts)


def _summary(res: SelectionResult) -> str:
    lines = [f"== model selection (era {res.era}, test years "
             f"{res.folds['test_year'].min()}-{res.folds['test_year'].max()}) =="]
    cols = ["crop", "candidate", "n", "mae", "d_b2", "d_b2_lo", "d_b2_hi", "d_b0",
            "d_b0_lo", "d_b0_hi", "qualifies", "selected"]
    lines.append(res.candidates[cols].to_string(index=False,
                                                float_format=lambda v: f"{v:+.4f}"))
    for c, (lab, reason) in res.selected.items():
        lines.append(f"{c}: selected {lab}: {reason}")
    lines.append("80% interval coverage (earlier-fold residuals):")
    lines.append(res.coverage.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Select a model per crop, fit final models, write predictions and the report."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR),
                    help="folder with pipeline.yaml and provinces.yaml (default: repo configs/)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("cropyield.models.satellite").setLevel(logging.WARNING)
    cfg = load_config(args.config_dir)
    df = load_modeling_table(cfg)
    res = run_selection(cfg, df)
    print(_summary(res))
    from cropyield.models.report import write_report
    report = write_report(cfg, df, res)
    print("wrote " + ", ".join(map(str, [*res.model_paths.values(), predictions_path(cfg),
                                         selection_path(cfg), report])))


if __name__ == "__main__":
    main()
