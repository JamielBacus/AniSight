import numpy as np
import pandas as pd
import pytest

from cropyield.config import Era
from cropyield.models.baselines import MeanPlusTrend
from cropyield.models.data import rolling_folds
from cropyield.models.evaluate import cross_validate, metrics
from cropyield.models.satellite import (FEATURE_SET_ERA, FEATURE_SETS, AnomalyRidge,
                                        BoostedModel, Climatology)

UNITS = ["Davao City", "Davao del Sur+Davao Occidental", "Davao Oriental"]
FEATS = FEATURE_SETS["modis"]
SIGNAL = "modis_ndvi_peak"
BETA = 0.5  # log-yield change per unit NDVI-peak anomaly
ERAS = {"modis": Era(start_year=2002, first_test_year=2010)}


def _table(years=range(2002, 2018), trend: float = 0.02, noise: float = 0.01,
           seed: int = 42) -> pd.DataFrame:
    """log_yield = level + trend*(year-2002) + BETA * NDVI-peak anomaly + noise.

    Every feature = a fixed unit/crop/quarter level + an independent N(0, 0.1) anomaly;
    only the NDVI-peak anomaly drives yield.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u, unit in enumerate(UNITS):
        for crop in ["palay", "corn"]:
            for q in (1, 2, 3, 4):
                level = 1.0 + 0.1 * u + 0.05 * q + (0.3 if crop == "palay" else 0.0)
                feat_level = {f: 0.2 * u + 0.1 * q + 0.05 * k
                              for k, f in enumerate(FEATS)}
                for y in years:
                    anom = {f: 0.1 * rng.normal() for f in FEATS}
                    ly = (level + trend * (y - 2002) + BETA * anom[SIGNAL]
                          + noise * rng.normal())
                    rows.append({"reporting_unit": unit, "crop": crop, "quarter": q,
                                 "year": y, "log_yield": ly,
                                 **{f: feat_level[f] + anom[f] for f in FEATS}})
    df = pd.DataFrame(rows)
    df["yield_t_ha"] = np.exp(df["log_yield"])
    df["is_outlier"] = False
    return df


def _mae(oof: pd.DataFrame) -> float:
    return float(metrics(oof).iloc[0]["mae"])


def test_feature_sets_match_design():
    assert len(FEATURE_SETS["modis"]) == 10
    assert all(f.startswith("modis_") and not f.endswith("_n_obs")
               for f in FEATURE_SETS["modis"])
    assert FEATURE_SETS["modis_s2"] == FEATURE_SETS["modis"] + FEATURE_SETS["s2"]
    assert FEATURE_SET_ERA == {"modis": "modis", "s2": "s2", "modis_s2": "s2",
                               "rain": "modis", "modis_rain": "modis", "s2_rain": "s2"}


def test_climatology_is_training_group_mean_with_fallback():
    df = _table()
    clim = Climatology([SIGNAL]).fit(df)
    expected = df.groupby(["reporting_unit", "crop", "quarter"])[SIGNAL].transform("mean")
    assert np.allclose(clim.means(df)[SIGNAL], expected)
    new = pd.DataFrame({"reporting_unit": ["Nowhere", "Nowhere"], "crop": ["corn", "rye"],
                        "quarter": [2, 2], SIGNAL: [0.0, 0.0]})
    m = clim.means(new)[SIGNAL].to_numpy()
    assert m[0] == pytest.approx(df.query("crop == 'corn' and quarter == 2")[SIGNAL].mean())
    assert m[1] == pytest.approx(df[SIGNAL].mean())


def test_ridge_recovers_signal_and_beats_b2():
    df = _table()
    m = AnomalyRidge(FEATS, alpha=1.0).fit(df)
    coef = m.coef_ / m.scaler_.scale_  # back to per-unit-anomaly scale
    assert coef[SIGNAL] == pytest.approx(BETA, abs=0.05)
    assert coef.drop(SIGNAL).abs().max() < 0.05
    ridge = cross_validate(lambda: AnomalyRidge(FEATS), df, "modis", eras=ERAS)
    b2 = cross_validate(MeanPlusTrend, df, "modis", eras=ERAS)
    assert _mae(ridge) < 0.5 * _mae(b2)


def test_boosted_recovers_signal_and_beats_b2():
    # no trend: trees cannot extrapolate one, and that is not what this test checks
    df = _table(trend=0.0)
    boosted = cross_validate(lambda: BoostedModel(FEATS), df, "modis", eras=ERAS)
    b2 = cross_validate(MeanPlusTrend, df, "modis", eras=ERAS)
    assert _mae(boosted) < 0.8 * _mae(b2)


def _state(m: AnomalyRidge) -> list:
    return [m.b2_.slope_, m.b2_.b1_.levels_[0][1],
            *[t for _, t in m.climatology_.levels_], m.climatology_.global_,
            m.scaler_.mean_, m.scaler_.scale_, m.ridge_.coef_, m.ridge_.intercept_]


def _same(a, b) -> bool:
    if isinstance(a, (pd.Series, pd.DataFrame)):
        return a.equals(b)
    return np.array_equal(np.asarray(a), np.asarray(b))


def test_leakage_perturbing_test_years_changes_no_fitted_state():
    """design.md fold-safety test: perturb test-year (and later) targets and features;
    every training-derived quantity and every test prediction made from unperturbed
    test features stays identical."""
    df = _table()
    test_cols = ["log_yield", "yield_t_ha", *FEATS]
    for train_idx, test_idx, y in rolling_folds(df, 2010):
        pert = df.copy()
        later = pert["year"] >= y
        pert.loc[later, test_cols] = pert.loc[later, test_cols] * 3.0 + 7.0
        a_train, b_train = df.iloc[train_idx], pert.iloc[train_idx]
        test = df.iloc[test_idx]
        ra, rb = AnomalyRidge(FEATS).fit(a_train), AnomalyRidge(FEATS).fit(b_train)
        assert all(_same(x, z) for x, z in zip(_state(ra), _state(rb))), y
        assert np.array_equal(ra.predict(test), rb.predict(test))
        ba, bb = BoostedModel(FEATS).fit(a_train), BoostedModel(FEATS).fit(b_train)
        assert np.array_equal(ba.predict(test), bb.predict(test))
        # predictions never read test targets
        shifted = test.assign(log_yield=test["log_yield"] + 5.0, yield_t_ha=99.0)
        assert np.array_equal(ra.predict(test), ra.predict(shifted))
        assert np.array_equal(ba.predict(test), ba.predict(shifted))
        # each prediction depends only on its own row's features
        one = test.copy()
        one.loc[one.index[0], FEATS] += 1.0
        for m in (ra, ba):
            p0, p1 = m.predict(test).to_numpy(), m.predict(one).to_numpy()
            assert np.array_equal(p0[1:], p1[1:])


def test_ridge_missing_features_drop_in_training_and_fall_back_to_b2():
    df = _table()
    train, test = df[df["year"] < 2016].copy(), df[df["year"] == 2016].copy()
    train.loc[train.index[:5], "modis_evi_peak"] = np.nan
    test.loc[test.index[:3], "modis_ndvi_integral"] = np.nan
    m = AnomalyRidge(FEATS).fit(train)
    assert m.n_train_dropped_ == 5 and m.n_train_ == len(train) - 5
    pred = m.predict(test)
    fb = m.fallback_mask(test)
    assert fb.sum() == 3 and m.n_fallback_ == 3
    b2 = MeanPlusTrend().fit(train).predict(test)
    assert np.allclose(pred[fb], b2[fb])
    assert not np.allclose(pred[~fb], b2[~fb])


def test_all_missing_satellite_features_give_b2_and_boosted_still_predicts():
    # e.g. Sentinel-2 columns on pre-2017 rows: all NaN
    df = _table()
    s2 = FEATURE_SETS["s2"]
    df = df.assign(**{f: np.nan for f in s2})
    train, test = df[df["year"] < 2016], df[df["year"] == 2016]
    m = AnomalyRidge(s2).fit(train)
    assert m.ridge_ is None and m.fallback_mask(test).all()
    assert np.allclose(m.predict(test), MeanPlusTrend().fit(train).predict(test))
    assert (m.coef_ == 0).all()
    p = BoostedModel(s2).fit(train).predict(test)
    assert p.notna().all()


def test_boosted_handles_unseen_unit():
    df = _table(trend=0.0)
    m = BoostedModel(FEATS).fit(df[df["year"] < 2016])
    new = df[df["year"] == 2016].head(2).assign(reporting_unit="Nowhere")
    assert m.predict(new).notna().all()


def test_deterministic_with_seed():
    df = _table()
    train, test = df[df["year"] < 2015], df[df["year"] >= 2015]
    for factory in (lambda: AnomalyRidge(FEATS), lambda: BoostedModel(FEATS)):
        a = factory().fit(train).predict(test)
        b = factory().fit(train).predict(test)
        assert np.array_equal(a, b)
        assert a.name == "pred_log_yield" and a.index.equals(test.index)
    assert BoostedModel(FEATS).fit(train).model_.random_state == 42


def test_missing_columns_raise():
    df = _table().drop(columns=SIGNAL)
    with pytest.raises(ValueError, match="missing"):
        AnomalyRidge(FEATS).fit(df)
    with pytest.raises(ValueError, match="missing"):
        BoostedModel(FEATS).fit(df)
