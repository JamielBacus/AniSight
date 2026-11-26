"""Forecast logging (phase 3, task 2.1): synthetic tables, tmp paths, no downloads."""
from __future__ import annotations

import dataclasses
import re
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import cropyield.models.forecast as fc
from cropyield.cli import app
from cropyield.config import load_config
from cropyield.models.baselines import B0B2Blend
from cropyield.models.data import prepare_modeling_table
from cropyield.models.forecast import (FORECAST_COLUMNS, PublishedQuarterError,
                                       forecast_quarter, log_forecast, make_forecast,
                                       parse_quarter, target_rows)
from cropyield.models.select import RECIPE_VERSION, model_recipe

COMBINED = "Sur+City"
T0 = datetime(2013, 1, 5, 3, 0, tzinfo=timezone.utc)


def _raw(last=(2012, 3)) -> pd.DataFrame:
    """PSA-like rows 2002..`last`. "Sur+City" splits into "Sur" and "City" from 2012 Q2,
    so Sur and City have no Q3/Q4 history before 2012 (like Davao City in 2025 Q2)."""
    rng = np.random.default_rng(42)
    rows = []
    for y in range(2002, last[0] + 1):
        for q in (1, 2, 3, 4):
            if (y, q) > last:
                continue
            units = ["Oriental", *(["Sur", "City"] if (y, q) >= (2012, 2) else [COMBINED])]
            for u, unit in enumerate(units):
                for crop in ("palay", "corn"):
                    ly = 1.0 + 0.1 * u + 0.05 * q + 0.02 * (y - 2002) + 0.05 * rng.normal()
                    area = 1000.0 + 300 * u + 50 * q + 10 * (y - 2002)
                    rows.append({"reporting_unit": unit, "crop": crop, "year": y,
                                 "quarter": q, "quality_flag": "ok",
                                 "yield_t_ha": float(np.exp(ly)), "area_ha": area,
                                 "production_mt": area * float(np.exp(ly))})
    return pd.DataFrame(rows)


def _bundle(crop: str) -> dict:
    return {"crop": crop, "model_name": "B3", "training_period": (2002, 2011),
            "recipes": {"model": model_recipe("B3"), "b0": model_recipe("B0"),
                        "b2": model_recipe("B2")},
            "recipe_version": RECIPE_VERSION,
            "interval": {"log_resid_lo": -0.1, "log_resid_hi": 0.1}}


BUNDLES = {c: _bundle(c) for c in ("palay", "corn")}


def _run(table, year=2012, quarter=4, **kw):
    kw.setdefault("made_at", T0)
    return forecast_quarter(table, BUNDLES, year, quarter, git="abc123", code="0" * 12, **kw)


def test_parse_quarter():
    assert parse_quarter("2026Q4") == (2026, 4)
    assert parse_quarter("2026-q1") == (2026, 1)
    with pytest.raises(ValueError):
        parse_quarter("2026Q5")


def test_target_units_are_latest_quarter_set():
    table = prepare_modeling_table(_raw())
    t = target_rows(table, 2012, 4, "corn")
    assert sorted(t["reporting_unit"]) == ["City", "Oriental", "Sur"]  # not the combined unit
    # a target before the split uses the unit set of its own latest earlier quarter
    t = target_rows(table, 2012, 1, "corn")
    assert sorted(t["reporting_unit"]) == ["Oriental", COMBINED]


def test_target_rows_join_features_of_target_quarter():
    table = prepare_modeling_table(_raw())
    feats = pd.DataFrame({"reporting_unit": ["Sur"], "crop": ["corn"], "year": [2012],
                          "quarter": [4], "rain_anom": [12.5]})
    t = target_rows(table, 2012, 4, "corn", feats).set_index("reporting_unit")
    assert t.loc["Sur", "rain_anom"] == 12.5
    assert np.isnan(t.loc["City", "rain_anom"])


def test_schema_and_values():
    table = prepare_modeling_table(_raw())
    out = _run(table)
    assert list(out.columns) == list(FORECAST_COLUMNS)
    assert len(out) == 6  # 3 units x 2 crops
    assert (out["target_quarter"] == "2012Q4").all()
    assert (out["data_vintage"] == "2012Q3").all()
    assert (out["history_last_year"] == 2011).all()
    assert (out["made_at_utc"] == "2013-01-05T03:00:00Z").all()
    assert out["forecast_id"].nunique() == 1
    assert (out["recipe_version"] == RECIPE_VERSION).all()
    assert (~out["backfill"]).all()
    assert (out["b3_lo80"] < out["b3_yield"]).all() and (out["b3_yield"] < out["b3_hi80"]).all()
    # B3 = mean of B0 and B2 log-yields, refitted on years < 2012
    np.testing.assert_allclose(np.log(out["b3_yield"]),
                               (np.log(out["b0_yield"]) + np.log(out["b2_yield"])) / 2)
    tgt = out[["reporting_unit", "crop", "year", "quarter"]]
    np.testing.assert_allclose(np.log(out["b3_yield"]),
                               B0B2Blend().fit(table.loc[table["year"] < 2012]).predict(tgt))


def test_real_code_hash_and_git_are_strings():
    assert re.fullmatch(r"[0-9a-f]{12}", fc.code_hash())
    assert isinstance(fc.git_commit(), str) and fc.git_commit()


def test_nan_area_for_unit_without_history():
    out = _run(prepare_modeling_table(_raw())).set_index(["reporting_unit", "crop"])
    for unit in ("Sur", "City"):  # split units: no own Q4 area history
        r = out.loc[unit]
        assert r["predicted_area_ha"].isna().all()
        assert r[["pred_production_mt", "pred_production_lo_mt",
                  "pred_production_hi_mt"]].isna().all().all()
        assert np.isfinite(r["b3_yield"]).all()
        assert r["b0_fallback"].all()
    r = out.loc["Oriental"]
    assert (r["area_fallback_level"] == "last_year").all()
    assert np.isfinite(r["pred_production_mt"]).all() and (~r["b0_fallback"]).all()


def test_refuses_published_quarter():
    table = prepare_modeling_table(_raw())
    with pytest.raises(PublishedQuarterError, match="2012Q3"):
        _run(table, 2012, 3)


def test_backfill_allowed_and_marked():
    table = prepare_modeling_table(_raw())
    out = _run(table, 2012, 3, backfill=True)
    assert out["backfill"].all()
    assert out["forecast_id"].str.endswith("-backfill").all()
    assert (out["history_last_year"] == 2011).all()
    assert sorted(out["reporting_unit"].unique()) == ["City", "Oriental", "Sur"]


@pytest.mark.parametrize("year,quarter,backfill", [(2012, 4, False), (2012, 3, True)])
def test_only_history_before_target_year(year, quarter, backfill):
    table = prepare_modeling_table(_raw())
    base = _run(table, year, quarter, backfill=backfill)
    raw = _raw()
    cur = raw["year"] == year
    raw.loc[cur, "yield_t_ha"] *= 5.0
    raw.loc[cur, "area_ha"] *= 3.0
    raw.loc[cur, "production_mt"] *= 15.0
    pert = _run(prepare_modeling_table(raw), year, quarter, backfill=backfill)
    num = [c for c in FORECAST_COLUMNS if pd.api.types.is_float_dtype(base[c])]
    assert "b3_yield" in num and "predicted_area_ha" in num
    pd.testing.assert_frame_equal(base[num], pert[num])


def test_log_appends_without_clobbering(tmp_path):
    table = prepare_modeling_table(_raw())
    path = tmp_path / "reports" / "forecasts.csv"
    first = _run(table)
    log_forecast(first, path)
    before = path.read_text()
    second = _run(table, made_at=datetime(2013, 1, 6, tzinfo=timezone.utc))
    log_forecast(second, path)
    assert path.read_text().startswith(before)
    logged = pd.read_csv(path)
    assert list(logged.columns) == list(FORECAST_COLUMNS)
    assert len(logged) == len(first) + len(second)
    assert logged["forecast_id"].nunique() == 2
    with pytest.raises(ValueError, match="already logged"):
        log_forecast(first, path)
    assert len(pd.read_csv(path)) == len(logged)


def test_log_refuses_other_header(tmp_path):
    path = tmp_path / "forecasts.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="different header"):
        log_forecast(_run(prepare_modeling_table(_raw())), path)
    assert path.read_text() == "a,b\n1,2\n"


def _cfg(tmp_path):
    cfg = load_config()
    paths = dataclasses.replace(cfg.pipeline.paths, reports=tmp_path / "reports",
                                processed=tmp_path / "processed",
                                interim=tmp_path / "interim", models=tmp_path / "models")
    return dataclasses.replace(cfg, pipeline=dataclasses.replace(cfg.pipeline, paths=paths))


def test_make_forecast_reads_config_paths(tmp_path):
    cfg = _cfg(tmp_path)
    raw = _raw()
    future = pd.DataFrame({"reporting_unit": ["Oriental"], "crop": ["corn"], "year": [2012],
                           "quarter": [4], "quality_flag": ["missing"],
                           "yield_t_ha": [np.nan], "area_ha": [np.nan],
                           "production_mt": [np.nan]})
    (tmp_path / "processed").mkdir()
    pd.concat([raw, future]).to_parquet(tmp_path / "processed" / "features.parquet")
    (tmp_path / "models").mkdir()
    for c, b in BUNDLES.items():
        joblib.dump(b, tmp_path / "models" / f"{c}.joblib")
    out = make_forecast("2012Q4", cfg, made_at=T0)
    assert len(out) == 6 and (out["data_vintage"] == "2012Q3").all()
    with pytest.raises(PublishedQuarterError):
        make_forecast("2012Q3", cfg)
    path = log_forecast(out, fc.forecasts_path(cfg))
    assert path == tmp_path / "reports" / "forecasts.csv"


def test_cli_forwards_options(monkeypatch):
    seen = {}
    monkeypatch.setattr(fc, "main", lambda argv=None: seen.setdefault("argv", argv))
    res = CliRunner().invoke(app, ["forecast", "--quarter", "2026Q4", "--backfill"])
    assert res.exit_code == 0, res.output
    assert seen["argv"] == ["--quarter", "2026Q4", "--backfill"]
    seen.clear()
    res = CliRunner().invoke(app, ["forecast", "--quarter", "2026Q4"])
    assert seen["argv"] == ["--quarter", "2026Q4"]
    assert CliRunner().invoke(app, ["forecast"]).exit_code != 0
