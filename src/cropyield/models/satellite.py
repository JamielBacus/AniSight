"""Satellite models (phase 2, Requirement 4): anomaly ridge and gradient boosting.

Both follow the baseline API (`cropyield.models.baselines`): `fit(train_df)` learns only
from the rows it is given (training-fold rows), `predict(df)` returns predicted
`log_yield` as a float Series named `pred_log_yield` aligned to `df.index`, using only
state stored at fit time and the feature/key columns of `df`. Neither reads `log_yield`
or `yield_t_ha` in `predict`.

Feature sets (design.md, "Feature sets evaluated"): FEATURE_SETS maps a name to its
columns and FEATURE_SET_ERA to the era it is evaluated on. `<src>_<idx>_n_obs` is not a
model input; Phase 1 already sets peak/integral/timing to NaN when n_obs is too low.

AnomalyRidge (Requirement 4.1)
- Target: `log_yield - B2(train)`, with B2 (`MeanPlusTrend`) fitted on the training rows.
- Features: each satellite feature minus its training-fold climatology, the mean over
  training rows (non-NaN values) of the same reporting_unit, crop and quarter. Unseen
  groups use the baselines' fallback chain: crop-quarter -> crop -> overall training mean.
- Anomalies are standardised (StandardScaler) and fed to sklearn Ridge(alpha), one model
  pooled over crops. Climatology, scaler, ridge and B2 are all fitted on training rows.
- Missing features (design.md): training rows with any NaN anomaly are dropped
  (`n_train_dropped_`); test rows with any NaN anomaly get the B2 prediction
  (`fallback_mask(df)`, count logged and stored in `n_fallback_`). If fewer than
  MIN_RIDGE_ROWS complete training rows remain, the model predicts B2 everywhere.

BoostedModel (Requirement 4.2)
- sklearn HistGradientBoostingRegressor on `log_yield` with reporting_unit, crop and
  quarter as categorical inputs (categories = those seen in training; unseen values are
  treated as missing) plus the raw satellite features. NaN features use the estimator's
  native missing-value handling (learned split direction), so no imputation is needed.
  Features that are NaN in every training row are left out of that fit
  (`used_features_`); with none left, the model uses the categorical inputs only.
- Early stopping off, random_state=42, other hyperparameters at sklearn defaults.
- Trees cannot extrapolate the year trend that B2 captures; this model has no year input,
  as specified in Requirement 4.2.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from cropyield.models.baselines import GROUP, PRED_NAME, MeanPlusTrend, _check, _lookup

log = logging.getLogger(__name__)

SEED = 42
MIN_RIDGE_ROWS = 2
SEASON_FEATURES = ("peak", "integral", "peak_day", "greenup_day", "valid_frac")
INDICES = ("ndvi", "evi")


def source_features(source: str) -> list[str]:
    """`<source>_<idx>_<feature>` model inputs for one source ("modis" or "s2")."""
    return [f"{source}_{i}_{f}" for i in INDICES for f in SEASON_FEATURES]


# Rainfall columns from cropyield.features (Requirement 5.2). rain_anom is already an
# anomaly vs the fixed 1981-2000 climatology; AnomalyRidge still subtracts its
# training-fold unit/crop/quarter mean, which is fold-safe like every other feature.
RAIN_FEATURES = ["rain_total", "rain_first_half", "rain_second_half", "rain_anom"]

FEATURE_SETS: dict[str, list[str]] = {
    "modis": source_features("modis"),
    "s2": source_features("s2"),
    "modis_s2": source_features("modis") + source_features("s2"),
    "rain": list(RAIN_FEATURES),
    "modis_rain": source_features("modis") + RAIN_FEATURES,
    "s2_rain": source_features("s2") + RAIN_FEATURES,
}
# Default era per feature set (design.md); `cropyield evaluate --feature-sets name@era`
# evaluates a set on another era (e.g. rain@s2, to compare with s2_rain).
FEATURE_SET_ERA: dict[str, str] = {"modis": "modis", "s2": "s2", "modis_s2": "s2",
                                   "rain": "modis", "modis_rain": "modis", "s2_rain": "s2"}


class Climatology:
    """Training-fold mean of each feature per reporting_unit, crop and quarter.

    Means ignore NaN values. Lookup falls back crop-quarter -> crop -> overall training
    mean; a feature that is NaN in every training row has a NaN climatology.
    """

    def __init__(self, features: Sequence[str]):
        self.features = list(features)

    def fit(self, train_df: pd.DataFrame) -> Climatology:
        _check(train_df, [*GROUP, *self.features])
        d = train_df[[*GROUP, *self.features]].astype({f: float for f in self.features})
        self.levels_: list[tuple[list[str], pd.DataFrame]] = [
            (keys, d.groupby(keys, sort=True)[self.features].mean())
            for keys in (GROUP, ["crop", "quarter"], ["crop"])]
        self.global_: pd.Series = d[self.features].mean()
        return self

    def means(self, df: pd.DataFrame) -> pd.DataFrame:
        """Climatological mean of every feature for each row of `df`."""
        _check(df, GROUP)
        out = pd.DataFrame(np.nan, index=df.index, columns=self.features)
        for f in self.features:
            v = np.full(len(df), np.nan)
            for keys, table in self.levels_:
                todo = np.isnan(v)
                if not todo.any():
                    break
                v[todo] = _lookup(df.loc[todo], table[f].dropna(), keys)
            v[np.isnan(v)] = self.global_[f]
            out[f] = v
        return out

    def anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        _check(df, self.features)
        return df[self.features].astype(float) - self.means(df)


class AnomalyRidge:
    """Ridge on satellite-feature anomalies predicting the residual from B2."""

    def __init__(self, features: Sequence[str], alpha: float = 1.0):
        if not features:
            raise ValueError("AnomalyRidge needs at least one feature")
        self.features = list(features)
        self.alpha = alpha

    def fit(self, train_df: pd.DataFrame) -> AnomalyRidge:
        _check(train_df, [*GROUP, "year", "log_yield", *self.features])
        self.b2_ = MeanPlusTrend().fit(train_df)
        resid = train_df["log_yield"].to_numpy(float) - self.b2_.predict(train_df).to_numpy()
        self.climatology_ = Climatology(self.features).fit(train_df)
        x = self.climatology_.anomalies(train_df).to_numpy()
        ok = np.isfinite(x).all(axis=1)
        self.n_train_ = int(ok.sum())
        self.n_train_dropped_ = int((~ok).sum())
        if self.n_train_dropped_:
            log.info("AnomalyRidge: dropped %d of %d training rows with missing features",
                     self.n_train_dropped_, len(ok))
        self.scaler_: StandardScaler | None = None
        self.ridge_: Ridge | None = None
        if self.n_train_ < MIN_RIDGE_ROWS:
            log.warning("AnomalyRidge: %d complete training rows; predicting B2 only",
                        self.n_train_)
            return self
        self.scaler_ = StandardScaler().fit(x[ok])
        self.ridge_ = Ridge(alpha=self.alpha, random_state=SEED).fit(
            self.scaler_.transform(x[ok]), resid[ok])
        return self

    def _anomalies(self, df: pd.DataFrame) -> np.ndarray:
        return self.climatology_.anomalies(df).to_numpy()

    def fallback_mask(self, df: pd.DataFrame) -> pd.Series:
        """True where `predict` returns the plain B2 prediction (missing features)."""
        if self.ridge_ is None:
            return pd.Series(True, index=df.index)
        return pd.Series(~np.isfinite(self._anomalies(df)).all(axis=1), index=df.index)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        _check(df, [*GROUP, "year", *self.features])
        pred = self.b2_.predict(df).to_numpy(copy=True)
        fb = self.fallback_mask(df).to_numpy()
        self.n_fallback_ = int(fb.sum())
        if self.n_fallback_:
            log.info("AnomalyRidge: B2 fallback for %d of %d rows (missing features)",
                     self.n_fallback_, len(df))
        if self.ridge_ is not None and (~fb).any():
            x = self._anomalies(df)[~fb]
            pred[~fb] += self.ridge_.predict(self.scaler_.transform(x))
        return pd.Series(pred, index=df.index, name=PRED_NAME)

    @property
    def coef_(self) -> pd.Series:
        """Ridge coefficients per feature (on standardised anomalies)."""
        if self.ridge_ is None:
            return pd.Series(0.0, index=self.features)
        return pd.Series(self.ridge_.coef_, index=self.features)


class BoostedModel:
    """HistGradientBoostingRegressor on log-yield with categorical unit/crop/quarter."""

    def __init__(self, features: Sequence[str]):
        self.features = list(features)

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        x = pd.DataFrame(index=df.index)
        for c in GROUP:
            cats = self.categories_[c]
            x[c] = pd.Categorical(df[c].where(df[c].isin(cats)), categories=cats)
        for f in self.used_features_:
            x[f] = df[f].astype(float)
        return x

    def fit(self, train_df: pd.DataFrame) -> BoostedModel:
        _check(train_df, [*GROUP, "log_yield", *self.features])
        if train_df.empty:
            raise ValueError("cannot fit on an empty training set")
        self.categories_: dict[str, list] = {
            c: sorted(train_df[c].dropna().unique().tolist()) for c in GROUP}
        # Features with no value in any training row carry no information (and break
        # binning); they are left out of this fit.
        self.used_features_ = [f for f in self.features if train_df[f].notna().any()]
        dropped = sorted(set(self.features) - set(self.used_features_))
        if dropped:
            log.info("BoostedModel: features all-NaN in training, not used: %s", dropped)
        self.model_ = HistGradientBoostingRegressor(
            categorical_features="from_dtype", early_stopping=False, random_state=SEED)
        self.model_.fit(self._design(train_df), train_df["log_yield"].to_numpy(float))
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        _check(df, [*GROUP, *self.features])
        pred = self.model_.predict(self._design(df))
        return pd.Series(pred, index=df.index, name=PRED_NAME)
