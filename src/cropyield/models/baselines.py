"""Non-satellite baselines B0, B1, B2 (phase 2, Requirement 2).

All models follow a small sklearn-style API: `fit(train_df)` learns only from the rows it
is given (pass training-fold rows only), and `predict(df)` returns predicted `log_yield`
as a float Series aligned to `df.index`. `predict` never reads `log_yield` /
`yield_t_ha` from `df`, so test-year targets cannot leak into predictions.

Required columns: `reporting_unit, crop, quarter, year` (and `log_yield` for `fit`), as
produced by `cropyield.models.data.prepare_modeling_table`.

Unseen groups (deterministic fallback chain, used by every baseline):
    (reporting_unit, crop, quarter) mean
 -> (crop, quarter) mean over all training units
 -> crop mean
 -> overall training mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUP = ["reporting_unit", "crop", "quarter"]
PRED_NAME = "pred_log_yield"


def _check(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")


def _lookup(df: pd.DataFrame, table: pd.Series, keys: list[str]) -> np.ndarray:
    """Values of `table` (indexed by `keys`) for each row of `df`; NaN when absent."""
    idx = pd.MultiIndex.from_frame(df[keys]) if len(keys) > 1 else pd.Index(df[keys[0]])
    return table.reindex(idx).to_numpy(dtype=float, copy=True)


class UnitQuarterMean:
    """B1: mean log-yield of the reporting unit, crop and quarter over the training rows.

    Also stores the mean training year of each group (used by B2 to centre the trend)
    and the fallback-level means listed in the module docstring.
    """

    def fit(self, train_df: pd.DataFrame) -> UnitQuarterMean:
        _check(train_df, [*GROUP, "year", "log_yield"])
        if train_df.empty:
            raise ValueError("cannot fit on an empty training set")
        d = train_df[[*GROUP, "year", "log_yield"]].astype({"year": float})
        self.levels_: list[tuple[list[str], pd.Series, pd.Series]] = []
        for keys in (GROUP, ["crop", "quarter"], ["crop"]):
            g = d.groupby(keys, sort=True)
            self.levels_.append((keys, g["log_yield"].mean(), g["year"].mean()))
        self.global_mean_ = float(d["log_yield"].mean())
        self.global_year_ = float(d["year"].mean())
        return self

    def _means_and_years(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mean = np.full(len(df), np.nan)
        year = np.full(len(df), np.nan)
        for keys, m, y in self.levels_:
            todo = np.isnan(mean)
            if not todo.any():
                break
            mean[todo] = _lookup(df.loc[todo], m, keys)
            year[todo] = _lookup(df.loc[todo], y, keys)
        todo = np.isnan(mean)
        mean[todo] = self.global_mean_
        year[todo] = self.global_year_
        return mean, year

    def predict(self, df: pd.DataFrame) -> pd.Series:
        _check(df, GROUP)
        mean, _ = self._means_and_years(df)
        return pd.Series(mean, index=df.index, name=PRED_NAME)


class MeanPlusTrend:
    """B2: B1 plus a per-crop linear year trend in log-yield.

    The slope for crop c is the pooled within-group OLS slope: regress
    `log_yield - group_mean` on `year - group_mean_year` over all training groups of
    crop c. Prediction = group_mean + slope_c * (year - group_mean_year). Centring on
    each group's own mean year keeps unbalanced groups (missing years) unbiased.
    Crops with no within-group year variation get slope 0; unseen crops get slope 0.
    """

    def fit(self, train_df: pd.DataFrame) -> MeanPlusTrend:
        self.b1_ = UnitQuarterMean().fit(train_df)
        d = train_df[[*GROUP, "year", "log_yield"]].astype({"year": float})
        g = d.groupby(GROUP)
        dy = d["log_yield"] - g["log_yield"].transform("mean")
        dt = d["year"] - g["year"].transform("mean")
        num = (dy * dt).groupby(d["crop"]).sum()
        den = (dt * dt).groupby(d["crop"]).sum()
        slope = (num / den.where(den > 0)).fillna(0.0)
        self.slope_: pd.Series = slope.sort_index()
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        _check(df, [*GROUP, "year"])
        mean, year0 = self.b1_._means_and_years(df)
        slope = _lookup(df, self.slope_, ["crop"])
        slope = np.nan_to_num(slope, nan=0.0)
        pred = mean + slope * (df["year"].to_numpy(dtype=float) - year0)
        return pd.Series(pred, index=df.index, name=PRED_NAME)


class SameQuarterLastYear:
    """B0: log-yield of the same reporting unit, crop and quarter in year Y-1.

    Y-1 values come only from the rows passed to `fit` (in rolling-origin folds the
    year before the test year is always in training). When the Y-1 value is not in
    the training data, the prediction falls back to B1 (`UnitQuarterMean`, which
    itself falls back as described in the module docstring). `fallback_mask(df)` tells
    which rows used the fallback.
    """

    def fit(self, train_df: pd.DataFrame) -> SameQuarterLastYear:
        _check(train_df, [*GROUP, "year", "log_yield"])
        self.b1_ = UnitQuarterMean().fit(train_df)
        keys = [*GROUP, "year"]
        hist = train_df[[*keys, "log_yield"]].astype({"year": "int64"})
        self.history_: pd.Series = (hist.set_index(keys)["log_yield"]
                                    .astype(float).sort_index())
        return self

    def _last_year(self, df: pd.DataFrame) -> np.ndarray:
        _check(df, [*GROUP, "year"])
        prev = df[GROUP].copy()
        prev["year"] = df["year"].astype("int64") - 1
        return _lookup(prev, self.history_, [*GROUP, "year"])

    def fallback_mask(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(np.isnan(self._last_year(df)), index=df.index)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        last = self._last_year(df)
        miss = np.isnan(last)
        if miss.any():
            last[miss] = self.b1_.predict(df.loc[miss]).to_numpy()
        return pd.Series(last, index=df.index, name=PRED_NAME)
