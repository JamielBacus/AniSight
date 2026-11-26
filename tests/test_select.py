import dataclasses

import joblib
import numpy as np
import pandas as pd
import pytest

from cropyield.config import Era, load_config
from cropyield.models.baselines import (B0B2Blend, MeanPlusTrend, SameQuarterLastYear,
                                        SameQuarterLastYearArea)
from cropyield.models.data import prepare_modeling_table
from cropyield.models.evaluate import EXPERIMENT_COLUMNS, cross_validate, log_experiment
from cropyield.models.report import production_errors, req61_selection, write_report
from cropyield.models.select import (CANDIDATE_COLUMNS, PREDICTION_COLUMNS,
                                     best_satellite_model, earlier_fold_intervals,
                                     expanding_anomalies, forecast_production,
                                     model_factory, predicted_area_oof, run_selection,
                                     select_candidate)
from cropyield.models.satellite import FEATURE_SETS

UNITS = ["Davao City", "Davao Oriental"]
ERAS = {"modis": Era(start_year=2002, first_test_year=2008),
        "s2": Era(start_year=2005, first_test_year=2009)}


def _area(u: int, q: int, y: int) -> float:
    """Synthetic harvested area (ha), distinct per unit, quarter and year."""
    return 1000.0 + 500.0 * u + 100.0 * q + 10.0 * (y - 2002)


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
                           "yield_t_ha": float(np.exp(ly)), "area_ha": _area(u, q, y),
                           "production_mt": _area(u, q, y) * float(np.exp(ly))}
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


# --- forecastable area -------------------------------------------------------------

def _area_rows(years, unit="A", quarter=1, crop="corn"):
    return pd.DataFrame({"reporting_unit": unit, "crop": crop, "quarter": quarter,
                         "year": list(years),
                         "area_ha": [100.0 + 10 * (y - 2000) for y in years]})


def test_predicted_area_uses_only_last_year():
    d = pd.concat([_area_rows(range(2000, 2008)), _area_rows(range(2000, 2008), unit="B")],
                  ignore_index=True)
    first = predicted_area_oof(d, first_test_year=2005)
    assert not first["area_fallback"].any()
    test = d.loc[first.index]
    np.testing.assert_allclose(first["predicted_area_ha"], 100.0 + 10 * (test["year"] - 2001))
    # perturbing year-Y areas does not change year-Y predicted areas
    pert = d.copy()
    is_2006 = pert["year"] == 2006
    pert.loc[is_2006, "area_ha"] *= 50
    again = predicted_area_oof(pert, first_test_year=2005)
    rows_2006 = d.index[is_2006]
    np.testing.assert_allclose(again.loc[rows_2006, "predicted_area_ha"],
                               first.loc[rows_2006, "predicted_area_ha"])


def test_predicted_area_falls_back_when_last_year_missing():
    # unit A has no 2004 row (e.g. a regrouping), so 2005 falls back to its mean area
    hist = _area_rows([2000, 2001, 2002, 2003])
    m = SameQuarterLastYearArea().fit(hist)
    target = pd.DataFrame({"reporting_unit": ["A", "A", "Z"], "crop": "corn",
                           "quarter": [1, 2, 1], "year": [2005, 2004, 2005]})
    pred = m.predict(target)
    # A Q1 2005: no 2004 -> unit/crop/quarter mean of 2000-2003
    assert pred.iloc[0] == pytest.approx(hist["area_ha"].mean())
    # A Q2 and unseen unit Z: crop-quarter / crop mean (only A Q1 exists -> same mean)
    assert pred.iloc[1] == pytest.approx(hist["area_ha"].mean())
    assert pred.iloc[2] == pytest.approx(hist["area_ha"].mean())
    assert m.fallback_mask(target).tolist() == [True, True, True]
    # A Q1 2004 has 2003 available: no fallback
    t2 = target.iloc[[0]].assign(year=2004)
    assert m.predict(t2).iloc[0] == pytest.approx(130.0)
    assert not m.fallback_mask(t2).iloc[0]


def test_area_fallback_level_names_the_level_used():
    hist = pd.concat([_area_rows([2000, 2001, 2002, 2003]),
                      _area_rows([2002, 2003], unit="B", quarter=2)], ignore_index=True)
    m = SameQuarterLastYearArea().fit(hist)
    target = pd.DataFrame({"reporting_unit": ["A", "A", "A", "Z", "Z"],
                           "crop": ["corn", "corn", "corn", "corn", "palay"],
                           "quarter": [1, 1, 2, 3, 1], "year": [2004, 2005, 2004, 2004, 2004]})
    assert m.fallback_level(target).tolist() == [
        "last_year", "unit_quarter_mean", "crop_quarter_mean", "crop_mean", "overall"]
    assert m.has_own_history(target).tolist() == [True, True, False, False, False]


def test_predicted_area_oof_is_nan_without_own_history():
    # unit N appears in 2006 (newly split off): its 2006 row has no own area history
    d = pd.concat([_area_rows(range(2000, 2008)), _area_rows([2006, 2007], unit="N")],
                  ignore_index=True)
    out = predicted_area_oof(d, first_test_year=2005)
    key = d.loc[out.index, ["reporting_unit", "year"]].apply(tuple, axis=1)
    lvl = dict(zip(key, out["area_fallback_level"]))
    area = dict(zip(key, out["predicted_area_ha"]))
    assert lvl[("N", 2006)] == "crop_quarter_mean" and np.isnan(area[("N", 2006)])
    assert lvl[("N", 2007)] == "last_year" and area[("N", 2007)] == pytest.approx(160.0)
    raw = predicted_area_oof(d, first_test_year=2005, mask_no_history=False)
    assert np.isfinite(raw.loc[out.index[[k == ("N", 2006) for k in key]],
                               "predicted_area_ha"]).all()


def test_forecast_production_nan_for_unit_without_history():
    df = prepare_modeling_table(_features())
    model = SameQuarterLastYear().fit(df.loc[df["year"] <= 2011])
    bundle = {"model": model, "interval": {"log_resid_lo": -0.1, "log_resid_hi": 0.1}}
    target = df.loc[df["year"] == 2012]
    unit = target["reporting_unit"].iloc[0]
    # history: this unit has no rows at all (new unit); another unit misses only Y-1
    other = sorted(set(target["reporting_unit"]) - {unit})[0]
    hist = df.loc[(df["reporting_unit"] != unit)
                  & ~((df["reporting_unit"] == other) & (df["year"] == 2011))]
    out = forecast_production(bundle, hist, target)
    new = target["reporting_unit"] == unit
    assert (out.loc[new, "area_fallback_level"] == "crop_quarter_mean").all()
    for c in ("predicted_area_ha", "pred_production_mt", "pred_production_lo_mt",
              "pred_production_hi_mt"):
        assert out.loc[new, c].isna().all()
    assert out.loc[new, "pred_yield_t_ha"].notna().all()
    # unit with history but no Y-1 row still gets a forecast (its own mean area)
    gap = target["reporting_unit"] == other
    assert (out.loc[gap, "area_fallback_level"] == "unit_quarter_mean").all()
    assert out.loc[gap, "pred_production_mt"].notna().all()
    own = hist.loc[(hist["reporting_unit"] == other) & (hist["year"] < 2012)].groupby(["crop", "quarter"])["area_ha"]
    expect = own.mean().loc[pd.MultiIndex.from_frame(target.loc[gap, ["crop", "quarter"]])]
    np.testing.assert_allclose(out.loc[gap, "predicted_area_ha"], expect.to_numpy())


def test_forecast_production_is_yield_times_last_year_area():
    df = prepare_modeling_table(_features())
    train = df.loc[df["year"] <= 2011]
    model = SameQuarterLastYear().fit(train)
    bundle = {"model": model, "interval": {"log_resid_lo": -0.1, "log_resid_hi": 0.1}}
    target = df.loc[df["year"] == 2012]
    perturbed = df.copy()
    perturbed.loc[perturbed["year"] == 2012, "area_ha"] *= 3  # target-year area ignored
    out = forecast_production(bundle, perturbed, target)
    last = df.loc[df["year"] == 2011].set_index(["reporting_unit", "crop", "quarter"])
    expect = last.loc[pd.MultiIndex.from_frame(target[["reporting_unit", "crop",
                                                       "quarter"]]), "area_ha"]
    np.testing.assert_allclose(out["predicted_area_ha"], expect.to_numpy())
    np.testing.assert_allclose(out["pred_production_mt"],
                               out["pred_yield_t_ha"] * out["predicted_area_ha"])
    np.testing.assert_allclose(out["pred_production_lo_mt"],
                               out["pred_lo_t_ha"] * out["predicted_area_ha"])
    assert (out["pred_production_lo_mt"] < out["pred_production_hi_mt"]).all()


def test_req61_selection_picks_simplest_qualifying_without_b3():
    c = pd.DataFrame({
        "crop": ["corn"] * 4 + ["palay"] * 3,
        "candidate": ["B0", "B2", "B3", "AnomalyRidge:rain", "B0", "B2", "B3"],
        "mae": [0.18, 0.20, 0.16, 0.17, 0.30, 0.30, 0.27],
        "qualifies": [True, False, True, True, False, False, True]})
    assert req61_selection(c) == {"corn": "B0", "palay": "B2"}


def test_production_errors_splits_fallback_and_skips_no_forecast():
    pred = pd.DataFrame({
        "crop": "palay", "model": "B3", "pred_yield_t_ha": 4.0, "b0_pred_yield_t_ha": 4.0,
        "b2_pred_yield_t_ha": 4.0, "actual_area_ha": 100.0,
        "predicted_area_ha": [100.0, 110.0, 150.0, np.nan],
        "area_fallback": [False, False, True, True],
        "actual_production_mt": 400.0})
    t = production_errors(pred)
    sel = t.loc[t["yield model"] == "B3 (selected)"].set_index("rows")
    assert sel.loc["all with area forecast", "n"] == 3
    assert sel.loc["last-year area only", "MAPE %, predicted area"] == "5.00"
    assert sel.loc["area fallback only", "MAPE %, predicted area"] == "50.00"
    assert sel.loc["all with area forecast", "median APE %, predicted area"] == "10.00"


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
                               pred["pred_yield_t_ha"] * pred["predicted_area_ha"])
    # no regroupings in the fixture: predicted area is exactly last year's PSA area
    assert not pred["area_fallback"].any()
    u = pred["reporting_unit"].map({n: i for i, n in enumerate(UNITS)})
    np.testing.assert_allclose(pred["predicted_area_ha"],
                               [_area(a, q, y - 1) for a, q, y in
                                zip(u, pred["quarter"], pred["year"])])
    np.testing.assert_allclose(pred["actual_area_ha"],
                               [_area(a, q, y) for a, q, y in
                                zip(u, pred["quarter"], pred["year"])])
    for crop, (label, _) in res.selected.items():
        assert (pred.loc[pred["crop"] == crop, "model"] == label).all()
        bundle = joblib.load(tmp_path / "models" / f"{crop}.joblib")
        assert bundle["model_name"] == label
        assert bundle["training_period"] == (2002, 2012)
        assert hasattr(bundle["model"], "predict")
        assert bundle["needs_last_year"]
        nxt = df.loc[(df["year"] == 2012) & (df["crop"] == crop)].assign(year=2013)
        area = bundle["area_model"].predict(nxt)
        np.testing.assert_allclose(area, nxt["area_ha"])  # 2012 area forecasts 2013

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
    assert "True out-of-sample test: the next PSA releases" in text
    assert "2013 Q1" in text  # fixture data end at 2012 Q4
    assert "## Predicted production" in text
    assert "Deviation from Requirement 6.1" in text
    assert "0 of" in text and "have no area forecast" in text
    assert (pred["area_fallback_level"] == "last_year").all()
    for name in ("mae_by_model.png", "residuals_by_year.png",
                 "observed_vs_predicted_corn.png"):
        assert (tmp_path / "reports" / "figures" / name).exists()
