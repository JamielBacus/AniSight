import dataclasses
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from cropyield.config import Era, load_config
from cropyield.models.baselines import MeanPlusTrend, UnitQuarterMean
from cropyield.models.evaluate import (EXPERIMENT_COLUMNS, OOF_COLUMNS, bootstrap_mae_diff,
                                       cross_validate, experiments_path, log_experiment,
                                       metrics, metrics_report)

UNITS = ["Davao City", "Davao del Sur+Davao Occidental"]
ERAS = {"modis": Era(start_year=2002, first_test_year=2008)}


def _table(years=range(2000, 2012)) -> pd.DataFrame:
    """Modeling-table-like rows: log_yield = level + 0.02 * (year - 2002) + noise."""
    rng = np.random.default_rng(42)
    rows = []
    for u, unit in enumerate(UNITS):
        for crop in ["palay", "corn"]:
            for q in (1, 2, 3, 4):
                for y in years:
                    ly = 1.0 + 0.1 * u + 0.05 * q + 0.02 * (y - 2002) + 0.05 * rng.normal()
                    rows.append({"reporting_unit": unit, "crop": crop, "year": y,
                                 "quarter": q, "log_yield": ly, "yield_t_ha": np.exp(ly),
                                 "production_mt": 100.0, "is_outlier": False})
    return pd.DataFrame(rows)


def _oof(actual, pred, crop=None, unit=None, year=None, outlier=None) -> pd.DataFrame:
    k = len(actual)
    return pd.DataFrame({
        "reporting_unit": unit or ["A"] * k, "crop": crop or ["palay"] * k,
        "year": year or list(range(2010, 2010 + k)), "quarter": [1] * k,
        "is_outlier": outlier or [False] * k,
        "yield_t_ha": np.asarray(actual, float), "pred_yield_t_ha": np.asarray(pred, float),
    })


def _cfg(tmp_path):
    cfg = load_config()
    paths = dataclasses.replace(cfg.pipeline.paths, reports=tmp_path / "reports")
    return dataclasses.replace(cfg, pipeline=dataclasses.replace(cfg.pipeline, paths=paths))


# --- cross_validate -------------------------------------------------------------------

class _Spy:
    """Records what fit/predict saw; predicts the training mean."""
    seen: list[dict] = []

    def fit(self, train_df):
        self.train_years = set(train_df["year"])
        self.mean = train_df["log_yield"].mean()
        return self

    def predict(self, df):
        _Spy.seen.append({"train_years": self.train_years, "test_years": set(df["year"]),
                          "test_cols": set(df.columns)})
        return pd.Series(self.mean, index=df.index)


def test_cross_validate_folds_never_train_on_test_year():
    _Spy.seen = []
    oof = cross_validate(_Spy, _table(), "modis", eras=ERAS)
    assert list(oof.columns) == OOF_COLUMNS
    assert sorted(oof["year"].unique()) == list(range(2008, 2012))
    assert len(_Spy.seen) == 4
    for s in _Spy.seen:
        (y,) = s["test_years"]
        assert s["train_years"] == set(range(2002, y))  # era start, nothing >= y
        assert not {"log_yield", "yield_t_ha", "production_mt"} & s["test_cols"]
    first = oof.loc[oof["year"] == 2008].iloc[0]
    assert first["n_train_years"] == 6 and first["train_first_year"] == 2002


def test_cross_validate_perturbing_test_year_does_not_change_its_prediction():
    df = _table()
    base = cross_validate(MeanPlusTrend, df, "modis", eras=ERAS)
    bumped = df.copy()
    hit = bumped["year"] == 2011
    bumped.loc[hit, "log_yield"] += 5.0
    bumped.loc[hit, "yield_t_ha"] = np.exp(bumped.loc[hit, "log_yield"])
    pert = cross_validate(MeanPlusTrend, bumped, "modis", eras=ERAS)
    assert np.allclose(base["pred_log_yield"], pert["pred_log_yield"])
    assert np.allclose(pert["pred_yield_t_ha"], np.exp(pert["pred_log_yield"]))


def test_cross_validate_matches_manual_fit():
    df = _table()
    oof = cross_validate(UnitQuarterMean, df, "modis", eras=ERAS)
    d = df.loc[df["year"] >= 2002]
    manual = UnitQuarterMean().fit(d.loc[d["year"] < 2010]).predict(d.loc[d["year"] == 2010])
    assert np.allclose(oof.loc[manual.index, "pred_log_yield"], manual)


def test_cross_validate_unknown_era():
    with pytest.raises(ValueError, match="unknown era"):
        cross_validate(UnitQuarterMean, _table(), "landsat", eras=ERAS)


# --- metrics --------------------------------------------------------------------------

def test_metrics_hand_computed():
    # errors +1, -2, +1 on actuals 2, 4, 5
    m = metrics(_oof([2.0, 4.0, 5.0], [3.0, 2.0, 6.0])).iloc[0]
    assert m["n"] == 3
    assert m["mae"] == pytest.approx(4 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(6 / 3))
    assert m["mape"] == pytest.approx((0.5 + 0.5 + 0.2) / 3 * 100)


def test_metrics_by_crop_and_unit_and_outliers():
    oof = _oof([2.0, 4.0, 5.0, 1.0], [3.0, 2.0, 6.0, 1.5],
               crop=["palay", "palay", "corn", "corn"], unit=["A", "B", "A", "B"],
               outlier=[False, True, False, False])
    by_crop = metrics(oof, by="crop").set_index("crop")
    assert by_crop.loc["palay", "mae"] == pytest.approx(1.5)
    assert by_crop.loc["corn", "mape"] == pytest.approx((20 + 50) / 2)
    by_unit = metrics(oof, by="reporting_unit").set_index("reporting_unit")
    assert by_unit.loc["B", "rmse"] == pytest.approx(np.sqrt((4 + 0.25) / 2))
    clean = metrics(oof, exclude_outliers=True).iloc[0]
    assert clean["n"] == 3 and clean["mae"] == pytest.approx((1 + 1 + 0.5) / 3)
    rep = metrics_report(oof)
    assert set(rep["subset"]) == {"all", "no_outliers"}
    assert set(rep["level"]) == {"overall", "crop", "reporting_unit", "crop_unit"}
    row = rep.query("subset == 'all' and level == 'overall'").iloc[0]
    assert row["crop"] == "all" and row["n"] == 4


# --- bootstrap_mae_diff ---------------------------------------------------------------

def test_bootstrap_identical_models_ci_contains_zero_and_deterministic():
    oof = cross_validate(MeanPlusTrend, _table(), "modis", eras=ERAS)
    r = bootstrap_mae_diff(oof, oof.copy())
    assert r.diff == 0 and r.ci_low <= 0 <= r.ci_high
    a = cross_validate(UnitQuarterMean, _table(), "modis", eras=ERAS)
    r1 = bootstrap_mae_diff(a, oof)
    r2 = bootstrap_mae_diff(a.sample(frac=1, random_state=42), oof)  # row order irrelevant
    assert r1 == r2 and r1.n_groups == 4 and r1.n_boot == 2000
    assert r1.ci_low <= r1.diff <= r1.ci_high


def test_bootstrap_point_estimate_and_sign():
    b = _oof([2.0, 4.0, 5.0], [2.0, 4.0, 5.0])        # perfect
    a = _oof([2.0, 4.0, 5.0], [3.0, 5.0, 6.0])        # off by 1 everywhere
    r = bootstrap_mae_diff(a, b, n=500)
    assert r.diff == pytest.approx(1.0)
    assert r.ci_low == pytest.approx(1.0) and r.ci_high == pytest.approx(1.0)


def test_bootstrap_requires_same_rows():
    a = _oof([2.0, 4.0], [2.0, 4.0])
    with pytest.raises(ValueError, match="same rows"):
        bootstrap_mae_diff(a, a.iloc[:1])


# --- log_experiment -------------------------------------------------------------------

def test_log_experiment_appends_without_clobbering(tmp_path):
    cfg = _cfg(tmp_path)
    df = _table()
    b2 = cross_validate(MeanPlusTrend, df, "modis", eras=ERAS)
    b1 = cross_validate(UnitQuarterMean, df, "modis", eras=ERAS)
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    first = log_experiment(cfg, model="B2", era="modis", oof=b2, timestamp=ts)
    second = log_experiment(cfg, model="B1", era="modis", oof=b1, params={"x": 1},
                            features=["f1"], reference_oof=b2, timestamp=ts)
    path = experiments_path(cfg)
    assert path == tmp_path / "reports" / "experiments.csv"
    log = pd.read_csv(path)
    assert list(log.columns) == EXPERIMENT_COLUMNS
    assert len(first) == len(second) == 2 and len(log) == 4
    assert list(log["model"]) == ["B2", "B2", "B1", "B1"]
    assert list(log["crop"]) == ["corn", "palay", "corn", "palay"]
    assert log.loc[0, "timestamp"] == "2025-01-01T00:00:00+00:00"
    assert log.loc[0, "test_years"] == "2008-2011"
    assert log.loc[:1, "mae_diff"].isna().all()
    assert (log.loc[2:, "ref_model"] == "B2").all()
    assert log.loc[3, "params"] == '{"x": 1}' and log.loc[3, "features"] == '["f1"]'
    palay = metrics(b1.loc[b1["crop"] == "palay"]).iloc[0]
    assert log.loc[3, "mae"] == pytest.approx(palay["mae"])


def test_log_experiment_rejects_mismatched_header(tmp_path):
    cfg = _cfg(tmp_path)
    path = experiments_path(cfg)
    path.parent.mkdir(parents=True)
    path.write_text("a,b\n1,2\n")
    oof = cross_validate(MeanPlusTrend, _table(), "modis", eras=ERAS)
    with pytest.raises(ValueError, match="columns"):
        log_experiment(cfg, model="B2", era="modis", oof=oof)
    assert path.read_text() == "a,b\n1,2\n"


# --- evaluate_baselines / `cropyield evaluate` ------------------------------------------

def _tmp_config_dir(tmp_path):
    """Copy the repo configs into tmp_path/configs, so data/ and reports/ resolve to tmp."""
    import shutil

    from cropyield.config import DEFAULT_CONFIG_DIR
    cfg_dir = tmp_path / "configs"
    shutil.copytree(DEFAULT_CONFIG_DIR, cfg_dir)
    return cfg_dir


def _features(years=range(2001, 2024)) -> pd.DataFrame:
    """Feature-table-like rows (no satellite columns needed for baselines)."""
    t = _table(years).drop(columns=["log_yield", "is_outlier"])
    t["quality_flag"] = ""
    t.loc[(t["year"] == 2022) & (t["quarter"] == 2), "quality_flag"] = "outlier"
    t.loc[t.index[3], "yield_t_ha"] = np.nan  # dropped by load_modeling_table
    # remove 2014 for one unit/crop/quarter so B0 must fall back in 2015
    gap = ((t["reporting_unit"] == UNITS[0]) & (t["crop"] == "palay")
           & (t["quarter"] == 1) & (t["year"] == 2014))
    return t.loc[~gap].reset_index(drop=True)


def test_evaluate_cli_end_to_end(tmp_path, capsys):
    from cropyield.cli import app as cli_app
    from typer.testing import CliRunner

    cfg_dir = _tmp_config_dir(tmp_path)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    _features().to_parquet(processed / "features.parquet")

    res = CliRunner().invoke(cli_app, ["evaluate", "--config-dir", str(cfg_dir)])
    assert res.exit_code == 0, res.output
    assert "B0 used the B1 fallback for 1 of" in res.output

    log = pd.read_csv(tmp_path / "reports" / "experiments.csv")
    assert list(log.columns) == EXPERIMENT_COLUMNS
    # 3 models x 2 eras x 2 crops, one shared timestamp
    assert len(log) == 12 and log["timestamp"].nunique() == 1
    assert set(log["era"]) == {"modis", "s2"}
    assert set(log.loc[log["era"] == "modis", "test_years"]) == {"2012-2023"}
    assert set(log.loc[log["era"] == "s2", "test_years"]) == {"2021-2023"}
    b2 = log["model"] == "B2"
    assert log.loc[b2, "mae_diff"].isna().all()
    assert (log.loc[~b2, "ref_model"] == "B2").all()
    assert log.loc[~b2, "mae_diff_ci_low"].notna().all()
    b0 = log.loc[(log["model"] == "B0") & (log["era"] == "modis")]
    assert all('"n_fallback": 1' in p for p in b0["params"])
    # outlier rows exist in the test set, so the no-outlier subset is smaller
    assert (log["n_test_no_outliers"] < log["n_test"]).all()

    rep = pd.read_csv(tmp_path / "reports" / "baseline_metrics.csv")
    assert set(rep["model"]) == {"B0", "B1", "B2"}
    assert set(rep["level"]) == {"overall", "crop", "reporting_unit", "crop_unit"}
    assert set(rep["reporting_unit"]) >= set(UNITS)


def test_evaluate_baselines_matches_cross_validate(tmp_path):
    from cropyield.models.data import prepare_modeling_table
    from cropyield.models.evaluate import evaluate_baselines

    cfg = load_config(_tmp_config_dir(tmp_path))
    df = prepare_modeling_table(_features())
    run = evaluate_baselines(cfg, df, "modis")
    direct = cross_validate(MeanPlusTrend, df, "modis", eras=cfg.pipeline.eras)
    assert np.allclose(run.oof["B2"]["pred_log_yield"], direct["pred_log_yield"])
    assert int(run.b0_fallback.sum()) == 1
    assert list(run.folds["test_year"]) == list(range(2012, 2024))


def _features_with_satellite() -> pd.DataFrame:
    """`_features()` plus all satellite columns: random values, MODIS from 2001,
    Sentinel-2 from 2017, and one missing MODIS value in a 2015 test row."""
    from cropyield.models.satellite import FEATURE_SETS
    t = _features()
    rng = np.random.default_rng(42)
    for f in FEATURE_SETS["modis_s2"] + FEATURE_SETS["rain"]:
        t[f] = rng.normal(size=len(t))
    t.loc[t["year"] < 2017, FEATURE_SETS["s2"]] = np.nan
    hole = t.index[(t["year"] == 2015) & (t["crop"] == "corn")][0]
    t.loc[hole, "modis_ndvi_peak"] = np.nan
    return t


def test_evaluate_cli_satellite_models(tmp_path):
    from cropyield.cli import app as cli_app
    from typer.testing import CliRunner

    cfg_dir = _tmp_config_dir(tmp_path)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    _features_with_satellite().to_parquet(processed / "features.parquet")

    res = CliRunner().invoke(cli_app, ["evaluate", "--config-dir", str(cfg_dir),
                                       "--models", "satellite"])
    assert res.exit_code == 0, res.output
    assert "AnomalyRidge used the B2 fallback for 1 of" in res.output

    log = pd.read_csv(tmp_path / "reports" / "experiments.csv")
    # satellite only: 2 models x 6 feature sets x 2 crops; baselines not logged
    assert len(log) == 24 and log["timestamp"].nunique() == 1
    sets = ("modis", "s2", "modis_s2", "rain", "modis_rain", "s2_rain")
    assert set(log["model"]) == {f"{m}:{fs}" for m in ("AnomalyRidge", "BoostedModel")
                                 for fs in sets}
    assert (log["ref_model"] == "B2").all() and log["mae_diff_ci_low"].notna().all()
    modis_era = log["model"].str.split(":").str[1].isin(["modis", "rain", "modis_rain"])
    assert set(log.loc[modis_era, "era"]) == {"modis"}
    assert set(log.loc[~modis_era, "era"]) == {"s2"}
    assert set(log.loc[log["era"] == "s2", "test_years"]) == {"2021-2023"}
    ridge = log.loc[log["model"] == "AnomalyRidge:modis"]
    assert all('"n_fallback": 1' in p for p in ridge["params"])
    assert (tmp_path / "reports" / "model_metrics.csv").exists()
    assert not (tmp_path / "reports" / "baseline_metrics.csv").exists()

    # n_test per era/crop matches B2's on the same rows (compared in the bootstrap)
    res = CliRunner().invoke(cli_app, ["evaluate", "--config-dir", str(cfg_dir)])
    assert res.exit_code == 0, res.output
    log = pd.read_csv(tmp_path / "reports" / "experiments.csv")
    n = log.groupby(["era", "crop"])["n_test"].nunique()
    assert (n == 1).all()


def test_evaluate_satellite_reference_is_b2(tmp_path):
    from cropyield.models.data import prepare_modeling_table
    from cropyield.models.evaluate import evaluate_satellite

    cfg = load_config(_tmp_config_dir(tmp_path))
    df = prepare_modeling_table(_features_with_satellite())
    b2 = cross_validate(MeanPlusTrend, df, "s2", eras=cfg.pipeline.eras)
    run = evaluate_satellite(cfg, df, "modis_s2", b2)
    assert run.era == "s2"
    for oof in run.oof.values():
        assert oof.index.sort_values().equals(b2.index.sort_values())
    assert list(run.ridge_train_dropped["test_year"]) == [2021, 2022, 2023]
    with pytest.raises(ValueError, match="unknown feature set"):
        evaluate_satellite(cfg, df, "landsat", b2)


def test_rain_feature_sets():
    from cropyield.models.satellite import FEATURE_SET_ERA, FEATURE_SETS, RAIN_FEATURES
    assert FEATURE_SETS["rain"] == RAIN_FEATURES
    assert FEATURE_SETS["modis_rain"] == FEATURE_SETS["modis"] + RAIN_FEATURES
    assert FEATURE_SETS["s2_rain"] == FEATURE_SETS["s2"] + RAIN_FEATURES
    assert (FEATURE_SET_ERA["rain"], FEATURE_SET_ERA["modis_rain"],
            FEATURE_SET_ERA["s2_rain"]) == ("modis", "modis", "s2")


def test_parse_feature_sets():
    from cropyield.models.evaluate import parse_feature_sets
    both = ["modis", "s2"]
    assert parse_feature_sets("rain,modis_rain,s2_rain,rain@s2,rain", both, both) == [
        ("rain", "modis"), ("modis_rain", "modis"), ("s2_rain", "s2"), ("rain", "s2")]
    assert parse_feature_sets("rain,rain@s2", ["s2"], both) == [("rain", "s2")]
    assert len(parse_feature_sets(None, both, both)) == 6
    with pytest.raises(ValueError, match="unknown feature set"):
        parse_feature_sets("landsat", both, both)
    with pytest.raises(ValueError, match="unknown era"):
        parse_feature_sets("rain@landsat", both, both)
    with pytest.raises(ValueError, match="no feature sets"):
        parse_feature_sets("s2_rain", ["modis"], both)


def test_evaluate_cli_feature_sets_subset(tmp_path):
    from cropyield.cli import app as cli_app
    from typer.testing import CliRunner

    cfg_dir = _tmp_config_dir(tmp_path)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    _features_with_satellite().to_parquet(processed / "features.parquet")
    run = CliRunner()
    res = run.invoke(cli_app, ["evaluate", "--config-dir", str(cfg_dir), "--models",
                               "satellite", "--feature-sets", "modis"])
    assert res.exit_code == 0, res.output
    res = run.invoke(cli_app, ["evaluate", "--config-dir", str(cfg_dir), "--models",
                               "satellite", "--feature-sets", "rain,s2_rain,rain@s2"])
    assert res.exit_code == 0, res.output

    log = pd.read_csv(tmp_path / "reports" / "experiments.csv")
    second = log.loc[log["timestamp"] == log["timestamp"].iloc[-1]]
    # second run: only rainfall sets, 2 models x 3 (set, era) pairs x 2 crops, no baselines
    assert len(second) == 12 and not second["model"].str.startswith("B").any()
    pairs = set(zip(second["model"].str.split(":").str[1], second["era"]))
    assert pairs == {("rain", "modis"), ("s2_rain", "s2"), ("rain", "s2")}
    assert (second["ref_model"] == "B2").all()
    # rain on both eras is scored on the same rows as the era's other sets
    n = second.groupby(["era", "crop"])["n_test"].nunique()
    assert (n == 1).all()
    # model_metrics.csv keeps the first run's modis rows alongside the rainfall rows
    rep = pd.read_csv(tmp_path / "reports" / "model_metrics.csv")
    assert {("AnomalyRidge:modis", "modis"), ("AnomalyRidge:rain", "s2"),
            ("AnomalyRidge:rain", "modis")} <= set(zip(rep["model"], rep["era"]))
    assert not rep.duplicated(["model", "era", "subset", "level", "crop",
                               "reporting_unit"]).any()

    res = run.invoke(cli_app, ["evaluate", "--config-dir", str(cfg_dir),
                               "--feature-sets", "rain"])
    assert res.exit_code != 0  # needs --models satellite|all
