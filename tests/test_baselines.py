import numpy as np
import pandas as pd
import pytest

from cropyield.models.baselines import MeanPlusTrend, SameQuarterLastYear, UnitQuarterMean
from cropyield.models.data import rolling_folds

UNITS = ["Davao City", "Davao del Sur+Davao Occidental"]
SLOPE = {"palay": 0.02, "corn": 0.01}


def _table(years=range(2002, 2016), noise: float = 0.0) -> pd.DataFrame:
    """log_yield = unit/crop/quarter level + crop slope * (year - 2002) + noise."""
    rng = np.random.default_rng(42)
    rows = []
    for u, unit in enumerate(UNITS):
        for crop in ["palay", "corn"]:
            for q in (1, 2, 3, 4):
                level = 1.0 + 0.1 * u + 0.05 * q + (0.3 if crop == "palay" else 0.0)
                for y in years:
                    rows.append({"reporting_unit": unit, "crop": crop, "quarter": q,
                                 "year": y,
                                 "log_yield": level + SLOPE[crop] * (y - 2002)
                                 + noise * rng.normal()})
    return pd.DataFrame(rows)


def test_b1_equals_unit_quarter_mean():
    df = _table(noise=0.05)
    pred = UnitQuarterMean().fit(df).predict(df)
    expected = df.groupby(["reporting_unit", "crop", "quarter"])["log_yield"].transform("mean")
    assert np.allclose(pred, expected)
    assert pred.index.equals(df.index)


def test_b1_unseen_unit_falls_back_to_crop_quarter_then_crop_then_global():
    df = _table()
    m = UnitQuarterMean().fit(df)
    new = pd.DataFrame({"reporting_unit": ["Nowhere", "Nowhere", "Nowhere"],
                        "crop": ["palay", "palay", "cassava"], "quarter": [1, 9, 1],
                        "year": [2016] * 3})
    pred = m.predict(new).to_numpy()
    assert pred[0] == pytest.approx(df.query("crop == 'palay' and quarter == 1")["log_yield"].mean())
    assert pred[1] == pytest.approx(df.query("crop == 'palay'")["log_yield"].mean())
    assert pred[2] == pytest.approx(df["log_yield"].mean())


def test_b2_recovers_per_crop_trend_and_extrapolates():
    train = _table(range(2002, 2016))
    test = _table(range(2016, 2018))
    m = MeanPlusTrend().fit(train)
    assert m.slope_["palay"] == pytest.approx(SLOPE["palay"])
    assert m.slope_["corn"] == pytest.approx(SLOPE["corn"])
    assert np.allclose(m.predict(test), test["log_yield"])
    # B1 cannot follow the trend
    assert not np.allclose(UnitQuarterMean().fit(train).predict(test), test["log_yield"])


def test_b2_trend_unbiased_with_unbalanced_groups():
    train = _table(range(2002, 2016))
    # drop early years from one group only: naive pooled regression would be biased
    drop = (train["reporting_unit"] == UNITS[0]) & (train["crop"] == "corn") & (train["year"] < 2010)
    m = MeanPlusTrend().fit(train.loc[~drop])
    assert m.slope_["corn"] == pytest.approx(SLOPE["corn"])


def test_b2_single_year_gives_zero_slope():
    m = MeanPlusTrend().fit(_table([2010]))
    assert (m.slope_ == 0).all()


def test_b0_returns_last_year_and_falls_back_to_b1():
    train = _table(range(2002, 2016), noise=0.05)
    test = _table([2016])
    gap = (train["reporting_unit"] == UNITS[1]) & (train["crop"] == "palay") & \
          (train["quarter"] == 3) & (train["year"] == 2015)
    train = train.loc[~gap]
    m = SameQuarterLastYear().fit(train)
    pred = m.predict(test)
    last = train[train["year"] == 2015].set_index(["reporting_unit", "crop", "quarter"])["log_yield"]
    fb = m.fallback_mask(test)
    assert fb.sum() == 1
    ok = test.loc[~fb]
    assert np.allclose(pred[~fb], last.reindex(pd.MultiIndex.from_frame(
        ok[["reporting_unit", "crop", "quarter"]])).to_numpy())
    b1 = UnitQuarterMean().fit(train).predict(test.loc[fb])
    assert np.allclose(pred[fb], b1)


def test_predictions_ignore_test_targets_and_fit_uses_training_rows_only():
    """Perturbing test-year targets (and later years) changes neither the fitted
    quantities nor the test-year predictions of any baseline."""
    df = _table(range(2002, 2018), noise=0.05)
    for train_idx, test_idx, y in rolling_folds(df, 2012):
        perturbed = df.copy()
        perturbed.loc[perturbed["year"] >= y, "log_yield"] += 5.0
        # same positional folds: perturbation touches only rows with year >= y
        a_train, b_train = df.iloc[train_idx], perturbed.iloc[train_idx]
        assert b_train.equals(a_train)
        for model_cls in (SameQuarterLastYear, UnitQuarterMean, MeanPlusTrend):
            a = model_cls().fit(a_train).predict(df.iloc[test_idx])
            b = model_cls().fit(b_train).predict(perturbed.iloc[test_idx])
            assert np.allclose(a, b), (model_cls.__name__, y)
        ma, mb = MeanPlusTrend().fit(a_train), MeanPlusTrend().fit(b_train)
        assert ma.slope_.equals(mb.slope_)
        assert ma.b1_.levels_[0][1].equals(mb.b1_.levels_[0][1])
    # B0 with a model fitted on all years still predicts year Y from Y-1 only
    m = SameQuarterLastYear().fit(df)
    test = df[df["year"] == 2016]
    shifted = test.assign(log_yield=test["log_yield"] + 5.0)
    assert np.allclose(m.predict(test), m.predict(shifted))


def test_fit_rejects_empty_and_missing_columns():
    with pytest.raises(ValueError):
        UnitQuarterMean().fit(_table().iloc[0:0])
    with pytest.raises(ValueError, match="missing"):
        MeanPlusTrend().fit(_table().drop(columns="log_yield"))
