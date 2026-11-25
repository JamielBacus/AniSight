import dataclasses

import joblib
import numpy as np
import pandas as pd
import pytest

from cropyield.config import Era, load_config
from cropyield.models.baselines import B0B2Blend, MeanPlusTrend, SameQuarterLastYear
from cropyield.models.data import prepare_modeling_table
from cropyield.models.evaluate import EXPERIMENT_COLUMNS, cross_validate, log_experiment
from cropyield.models.report import write_report
from cropyield.models.select import (CANDIDATE_COLUMNS, PREDICTION_COLUMNS,
                                     best_satellite_model, earlier_fold_intervals,
                                     expanding_anomalies, model_factory, run_selection,
                                     select_candidate)
from cropyield.models.satellite import FEATURE_SETS

UNITS = ["Davao City", "Davao Oriental"]
ERAS = {"modis": Era(start_year=2002, first_test_year=2008),
        "s2": Era(start_year=2005, first_test_year=2009)}


def _features(years=range(2002, 2013)) -> pd.DataFrame:
    """Feature-table-like rows with a trend, noise and every column select needs."""
    rng = np.random.default_rng(42)
    rows = []
    for u, unit in enumerate(UNITS):
        for crop in ["palay", "corn"]:
            for q in (1, 2, 3, 4):
                for y in years:
                    ly = 1.0 + 0.1 * u + 0.05 * q + 0.02 * (y - 2002) + 0.05 * rng.normal()
                    row = {"province": unit, "reporting_unit": unit, "crop": crop,
                           "year": y, "quarter": q, "quality_flag": "ok",
                           "yield_t_ha": float(np.exp(ly)), "area_ha": 1000.0,
                           "production_mt": 1000.0 * float(np.exp(ly))}
                    for f in {*FEATURE_SETS["modis"], *FEATURE_SETS["rain"]}:
                        row[f] = float(rng.normal())
                    rows.append(row)
    return pd.DataFrame(rows)


def _cfg(tmp_path):
    cfg = load_config()
    paths = dataclasses.replace(cfg.pipeline.paths, reports=tmp_path / "reports",
                                processed=tmp_path / "processed",
                                interim=tmp_path / "interim", models=tmp_path / "models")
    return dataclasses.replace(cfg, pipeline=dataclasses.replace(
        cfg.pipeline, paths=paths, eras=ERAS))


def _table(**over):
    base = {"crop": "corn", "role": "", "n": 10, "rmse": 0.1, "mape": 1.0,
            "d_b0": 0.0, "d_b0_lo": np.nan, "d_b0_hi": np.nan, "selected": False}
    rows = [{**base, **o} for o in over["rows"]]
    t = pd.DataFrame(rows)
    t["qualifies"] = (t["candidate"] != "B2") & (t["d_b2_hi"] < 0)
    return t[CANDIDATE_COLUMNS]


# --- selection rule ----------------------------------------------------------------

def test_select_picks_lowest_mae_among_ci_excluding_zero():
    t = _table(rows=[
        {"candidate": "B2", "mae": 0.30, "d_b2": 0.0, "d_b2_lo": np.nan, "d_b2_hi": np.nan},
        {"candidate": "B0", "mae": 0.25, "d_b2": -0.05, "d_b2_lo": -0.08, "d_b2_hi": -0.01},
        {"candidate": "B3", "mae": 0.26, "d_b2": -0.04, "d_b2_lo": -0.06, "d_b2_hi": -0.02},
        # lowest MAE overall, but its CI includes zero -> not eligible
        {"candidate": "AnomalyRidge:rain", "mae": 0.20, "d_b2": -0.10, "d_b2_lo": -0.3,
         "d_b2_hi": 0.05}])
    label, reason = select_candidate(t)
    assert label == "B0"
    assert "B3" in reason


def test_select_falls_back_to_b2_when_none_qualify():
    t = _table(rows=[
        {"candidate": "B2", "mae": 0.30, "d_b2": 0.0, "d_b2_lo": np.nan, "d_b2_hi": np.nan},
        {"candidate": "B0", "mae": 0.29, "d_b2": -0.01, "d_b2_lo": -0.03, "d_b2_hi": 0.01},
        {"candidate": "B3", "mae": 0.31, "d_b2": 0.01, "d_b2_lo": 0.001, "d_b2_hi": 0.02}])
    label, reason = select_candidate(t)
    assert label == "B2"
    assert "falls back to B2" in reason


def test_best_satellite_model_lowest_logged_mae():
    e = pd.DataFrame({
        "model": ["AnomalyRidge:rain", "BoostedModel:modis", "AnomalyRidge:s2", "B0",
                  "AnomalyRidge:modis"],
        "era": ["modis", "modis", "s2", "modis", "s2"],
        "crop": ["corn"] * 5, "mae": [0.20, 0.19, 0.01, 0.01, 0.01]})
    assert best_satellite_model(e, "corn") == "BoostedModel:modis"
    with pytest.raises(ValueError):
        best_satellite_model(e, "palay")


# --- B3 ----------------------------------------------------------------------------

def test_b3_is_mean_of_b0_and_b2_log_predictions():
    d = prepare_modeling_table(_features())
    train, test = d.loc[d["year"] < 2010], d.loc[d["year"] == 2010]
    b0 = SameQuarterLastYear().fit(train).predict(test)
    b2 = MeanPlusTrend().fit(train).predict(test)
    b3 = B0B2Blend().fit(train).predict(test)
    np.testing.assert_allclose(b3, (b0 + b2) / 2)
    oof = cross_validate(model_factory("B3"), d, "modis", eras=ERAS)
    o0 = cross_validate(model_factory("B0"), d, "modis", eras=ERAS)
    o2 = cross_validate(model_factory("B2"), d, "modis", eras=ERAS)
    np.testing.assert_allclose(oof["pred_log_yield"],
                               (o0["pred_log_yield"] + o2["pred_log_yield"]) / 2)


# --- intervals ---------------------------------------------------------------------

def _oof(years=range(2010, 2016), per_year=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        for i in range(per_year):
            rows.append({"crop": "corn", "year": y, "pred_log_yield": 1.0,
                         "log_yield": 1.0 + rng.normal(scale=0.1)})
    return pd.DataFrame(rows)


def test_interval_uses_earlier_folds_only():
    o = _oof()
    iv = earlier_fold_intervals(o, min_years=3)
    first = o["year"] < 2013
    assert iv.loc[first, "pred_lo_log"].isna().all()
    assert (iv.loc[o["year"] == 2014, "interval_n_years"] == 4).all()
    r = o["log_yield"] - o["pred_log_yield"]
    lo, hi = np.quantile(r[o["year"] < 2014], [0.1, 0.9])
    row = o.index[o["year"] == 2014][0]
    assert iv.loc[row, "pred_lo_log"] == pytest.approx(1.0 + lo)
    assert iv.loc[row, "pred_hi_log"] == pytest.approx(1.0 + hi)
    # perturbing year-Y and later residuals leaves year-Y intervals unchanged
    o2 = o.copy()
    o2.loc[o2["year"] >= 2014, "log_yield"] += 5.0
    iv2 = earlier_fold_intervals(o2, min_years=3)
    upto = o["year"] <= 2014
    pd.testing.assert_frame_equal(iv.loc[upto], iv2.loc[upto])


# --- anomalies ---------------------------------------------------------------------

def test_anomalies_use_only_earlier_years():
    d = pd.DataFrame({"reporting_unit": "A", "crop": "corn", "quarter": 1,
                      "year": range(2002, 2010),
                      "x": [1.0, 2.0, np.nan, 3.0, 10.0, 5.0, 6.0, 7.0]})
    a = expanding_anomalies(d, ["x"], min_years=3)["x_anom"]
    assert a.iloc[:4].isna().all()  # 2005 has only 2 earlier values (one NaN)
    assert a.iloc[4] == pytest.approx(10.0 - 2.0)  # 2006: mean(1, 2, 3)
    assert a.iloc[5] == pytest.approx(5.0 - 4.0)  # 2007: mean(1, 2, 3, 10)
    d2 = d.copy()
    d2.loc[d2["year"] >= 2008, "x"] = 100.0  # later years must not move 2002-2007
    a2 = expanding_anomalies(d2, ["x"], min_years=3)["x_anom"]
    pd.testing.assert_series_equal(a.iloc[:6], a2.iloc[:6])


def test_anomalies_are_per_group():
    d = pd.DataFrame({"reporting_unit": ["A"] * 4 + ["B"] * 4, "crop": "corn",
                      "quarter": 1, "year": list(range(2002, 2006)) * 2,
                      "x": [1.0, 1.0, 1.0, 2.0, 5.0, 5.0, 5.0, 9.0]})
    a = expanding_anomalies(d, ["x"], min_years=3)["x_anom"]
    assert a.iloc[3] == pytest.approx(1.0)
    assert a.iloc[7] == pytest.approx(4.0)


# --- end to end --------------------------------------------------------------------

def test_run_selection_writes_models_predictions_and_report(tmp_path):
    cfg = _cfg(tmp_path)
    df = prepare_modeling_table(_features())
    # a logged satellite run (the selection reads the best one from experiments.csv)
    for era in ("modis", "s2"):
        b2 = cross_validate(MeanPlusTrend, df, era, eras=ERAS)
        log_experiment(cfg, model="B2", era=era, oof=b2)
        ridge = cross_validate(model_factory("AnomalyRidge:rain"), df, era, eras=ERAS)
        log_experiment(cfg, model="AnomalyRidge:rain", era=era, oof=ridge, reference_oof=b2)
    res = run_selection(cfg, df)

    pred = pd.read_parquet(tmp_path / "processed" / "predictions.parquet")
    assert list(pred.columns) == list(PREDICTION_COLUMNS)
    assert set(pred["crop"]) == {"corn", "palay"}
    assert pred["year"].min() == 2008
    assert len(pred) == len(df.loc[df["year"] >= 2008])
    assert pred.loc[pred["year"] < 2011, "pred_lo_t_ha"].isna().all()
    has = pred["pred_lo_t_ha"].notna()
    assert (pred.loc[has, "pred_lo_t_ha"] <= pred.loc[has, "pred_yield_t_ha"]).all()
    assert (pred.loc[has, "pred_hi_t_ha"] >= pred.loc[has, "pred_yield_t_ha"]).all()
    np.testing.assert_allclose(pred["pred_production_mt"],
                               pred["pred_yield_t_ha"] * pred["area_ha"])
    for crop, (label, _) in res.selected.items():
        assert (pred.loc[pred["crop"] == crop, "model"] == label).all()
        bundle = joblib.load(tmp_path / "models" / f"{crop}.joblib")
        assert bundle["model_name"] == label
        assert bundle["training_period"] == (2002, 2012)
        assert hasattr(bundle["model"], "predict")

    cand = pd.read_csv(tmp_path / "reports" / "selection.csv")
    assert set(cand["candidate"]) == {"B0", "B2", "B3", "AnomalyRidge:rain"}
    assert cand.groupby("crop")["selected"].sum().eq(1).all()

    log = pd.read_csv(tmp_path / "reports" / "experiments.csv")
    assert list(log.columns) == EXPERIMENT_COLUMNS
    b3 = log.loc[log["model"] == "B3"]
    assert len(b3) == 2 and (b3["ref_model"] == "B2").all()
    run_selection(cfg, df)  # re-run replaces B3 rows
    log = pd.read_csv(tmp_path / "reports" / "experiments.csv")
    assert (log["model"] == "B3").sum() == 2

    sweep = pd.DataFrame({"feature_set": "rain", "era": "modis", "alpha": [1.0, 100.0] * 2,
                          "crop": ["corn", "corn", "palay", "palay"],
                          "d_b2": [0.01, 0.001, 0.02, 0.002], "lo": -0.01, "hi": 0.03})
    out = write_report(cfg, df, res, sweep=sweep)
    text = out.read_text()
    assert "did not improve on the baselines" in text
    assert "post hoc" in text
    for name in ("mae_by_model.png", "residuals_by_year.png",
                 "observed_vs_predicted_corn.png"):
        assert (tmp_path / "reports" / "figures" / name).exists()
