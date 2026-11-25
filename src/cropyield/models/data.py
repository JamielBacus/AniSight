"""Modeling dataset, eras and rolling-origin folds (phase 2).

The unit of prediction is `reporting_unit x crop x quarter` for a year: combined PSA units
(e.g. "Davao del Sur+Davao Occidental") are predicted as reported, never split.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np
import pandas as pd

from cropyield.config import DEFAULT_ERAS, Config, Era

UNIT_KEYS = ["reporting_unit", "crop", "year", "quarter"]


def load_modeling_table(cfg: Config) -> pd.DataFrame:
    """Read `<paths.processed>/features.parquet` and return model-ready rows.

    Drops rows whose `yield_t_ha` is missing or non-positive. Keeps `outlier` and
    `includes:` rows; `quality_flag` stays as is and `is_outlier` (bool) is added for
    filtering. Adds `log_yield = log(yield_t_ha)`. Sorted by UNIT_KEYS, index reset.
    """
    df = pd.read_parquet(cfg.pipeline.paths.processed / "features.parquet")
    return prepare_modeling_table(df)


def prepare_modeling_table(df: pd.DataFrame) -> pd.DataFrame:
    """In-memory part of `load_modeling_table` (see there)."""
    missing = [c for c in [*UNIT_KEYS, "yield_t_ha", "quality_flag"] if c not in df.columns]
    if missing:
        raise ValueError(f"feature table is missing columns: {missing}")
    if df.duplicated(UNIT_KEYS).any():
        raise ValueError("feature table has duplicate reporting_unit/crop/year/quarter rows")
    y = pd.to_numeric(df["yield_t_ha"], errors="coerce")
    out = df.loc[y.notna() & (y > 0)].copy()
    out["log_yield"] = np.log(out["yield_t_ha"].astype(float))
    flags = out["quality_flag"].fillna("").astype(str).str.split(";")
    out["is_outlier"] = flags.map(lambda parts: "outlier" in parts).astype(bool)
    return out.sort_values(UNIT_KEYS).reset_index(drop=True)


def era_rows(df: pd.DataFrame, era: str,
             eras: Mapping[str, Era] = DEFAULT_ERAS) -> pd.DataFrame:
    """Rows of `era` ("modis": 2002+, "s2": 2018+ by default; see `modeling.eras`)."""
    if era not in eras:
        raise ValueError(f"unknown era {era!r}; expected one of {sorted(eras)}")
    return df.loc[df["year"] >= eras[era].start_year]


def rolling_folds(df: pd.DataFrame, first_test_year: int
                  ) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """Rolling-origin, expanding-window folds, one per test year.

    For each year Y >= `first_test_year` present in `df` (ascending), yields
    `(train_idx, test_idx, Y)` as positional indices into `df` (use `df.iloc`):
    train = every row with year < Y, test = rows with year == Y. Years after Y are never
    in training. Folds with an empty training set are skipped.
    """
    years = df["year"].to_numpy()
    for y in sorted(int(v) for v in np.unique(years) if v >= first_test_year):
        train_idx = np.flatnonzero(years < y)
        if train_idx.size == 0:
            continue
        yield train_idx, np.flatnonzero(years == y), y


def fold_summary(df: pd.DataFrame, first_test_year: int) -> pd.DataFrame:
    """One row per fold: test_year, n_train_years, train_first_year, n_train, n_test."""
    rows = []
    years = df["year"].to_numpy()
    for train_idx, test_idx, y in rolling_folds(df, first_test_year):
        train_years = np.unique(years[train_idx])
        rows.append({"test_year": y, "n_train_years": len(train_years),
                     "train_first_year": int(train_years.min()),
                     "n_train": len(train_idx), "n_test": len(test_idx)})
    return pd.DataFrame(rows, columns=["test_year", "n_train_years", "train_first_year",
                                       "n_train", "n_test"])
