import dataclasses

import numpy as np
import pandas as pd
import pytest

from cropyield.config import Era, load_config, load_pipeline
from cropyield.models.data import (era_rows, fold_summary, load_modeling_table,
                                   prepare_modeling_table, rolling_folds)

COMBINED = "Davao del Sur+Davao Occidental"


def _features(years=range(2000, 2024)) -> pd.DataFrame:
    rows = []
    for unit in ["Davao City", COMBINED]:
        for crop in ["palay", "corn"]:
            for y in years:
                for q in (1, 2, 3, 4):
                    rows.append({"province": unit.split("+")[0], "reporting_unit": unit,
                                 "crop": crop, "year": y, "quarter": q,
                                 "quality_flag": "ok", "yield_t_ha": 3.0 + 0.01 * (y - 2000)})
    return pd.DataFrame(rows)


def test_drops_missing_and_nonpositive_yield_keeps_outliers():
    df = _features(years=[2010])
    df.loc[0, ["yield_t_ha", "quality_flag"]] = [np.nan, "missing"]
    df.loc[1, ["yield_t_ha", "quality_flag"]] = [0.0, "zero"]
    df.loc[2, ["yield_t_ha", "quality_flag"]] = [-1.0, "ok"]
    df.loc[3, "quality_flag"] = "outlier;includes:Davao de Oro"
    df.loc[4, "quality_flag"] = "includes:Davao Occidental"
    out = prepare_modeling_table(df)
    assert len(out) == len(df) - 3
    assert (out["yield_t_ha"] > 0).all()
    assert out["quality_flag"].isin(["outlier;includes:Davao de Oro",
                                     "includes:Davao Occidental"]).sum() == 2
    assert out["is_outlier"].sum() == 1
    assert np.allclose(out["log_yield"], np.log(out["yield_t_ha"]))


def test_reporting_unit_is_kept_as_reported():
    out = prepare_modeling_table(_features(years=[2010]))
    assert set(out["reporting_unit"]) == {"Davao City", COMBINED}
    assert not out.duplicated(["reporting_unit", "crop", "year", "quarter"]).any()


def test_duplicate_unit_rows_rejected():
    df = _features(years=[2010])
    with pytest.raises(ValueError, match="duplicate"):
        prepare_modeling_table(pd.concat([df, df.iloc[[0]]]))


def test_load_modeling_table_reads_configured_path(tmp_path):
    cfg = load_config()
    paths = dataclasses.replace(cfg.pipeline.paths, processed=tmp_path)
    cfg = dataclasses.replace(cfg, pipeline=dataclasses.replace(cfg.pipeline, paths=paths))
    _features(years=[2010, 2011]).to_parquet(tmp_path / "features.parquet")
    out = load_modeling_table(cfg)
    assert len(out) == 32 and "log_yield" in out


def test_era_filters():
    df = prepare_modeling_table(_features())
    assert era_rows(df, "modis")["year"].min() == 2002
    assert era_rows(df, "s2")["year"].min() == 2018
    custom = {"x": Era(start_year=2020, first_test_year=2022)}
    assert era_rows(df, "x", custom)["year"].min() == 2020
    with pytest.raises(ValueError, match="unknown era"):
        era_rows(df, "landsat")


def test_config_eras_loaded():
    eras = load_config().pipeline.eras
    assert eras["modis"] == Era(2002, 2012) and eras["s2"] == Era(2018, 2021)


def test_config_eras_validated(tmp_path):
    src = load_config().pipeline
    text = (src.paths.raw.parent.parent / "configs" / "pipeline.yaml").read_text()
    bad = text.replace("first_test_year: 2012", "first_test_year: 2001")
    (tmp_path / "pipeline.yaml").write_text(bad)
    with pytest.raises(ValueError, match="first_test_year"):
        load_pipeline(tmp_path / "pipeline.yaml")


def test_rolling_folds_are_time_ordered_and_disjoint():
    df = era_rows(prepare_modeling_table(_features()), "modis")
    folds = list(rolling_folds(df, 2012))
    assert [y for _, _, y in folds] == list(range(2012, 2024))
    years = df["year"].to_numpy()
    for train_idx, test_idx, y in folds:
        assert (years[test_idx] == y).all()
        assert (years[train_idx] < y).all()  # no test year or later year in training
        assert set(years[train_idx]) == set(range(2002, y))  # expanding window
        assert not set(train_idx) & set(test_idx)
        assert len(test_idx) == 16  # 2 units x 2 crops x 4 quarters


def test_rolling_folds_skip_empty_training_and_missing_years():
    df = prepare_modeling_table(_features(years=[2018, 2019, 2021]))
    assert [y for _, _, y in rolling_folds(df, 2018)] == [2019, 2021]


def test_fold_summary_counts_training_years():
    df = era_rows(prepare_modeling_table(_features()), "s2")
    s = fold_summary(df, 2021)
    assert s["test_year"].tolist() == [2021, 2022, 2023]
    assert s["n_train_years"].tolist() == [3, 4, 5]
    assert (s["train_first_year"] == 2018).all()
