"""Tests for cropyield.features with tiny synthetic ground truth and VI tables (no network)."""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from cropyield import features as f
from cropyield.config import DEFAULT_CONFIG_DIR, load_config

VI_COLS = ["province", "date", "index", "mean", "median", "max", "valid_frac", "n_pixels"]


@pytest.fixture
def cfg():
    return load_config()


def _gt(cfg, rows: list[tuple[str, str, int, int]]) -> pd.DataFrame:
    """Ground-truth rows with flags and reporting units as `load_psa` would set them."""
    out = []
    for province, crop, year, quarter in rows:
        absorbed = cfg.provinces.host_of(province, year, quarter) is not None
        includes = cfg.provinces.absorbed_by(province, year, quarter)
        flag = "not_reported" if absorbed else (
            ";".join(f"includes:{u}" for u in includes) or "ok")
        value = np.nan if absorbed else 100.0
        out.append({"province": province, "crop": crop, "year": year, "quarter": quarter,
                    "production_mt": value, "area_ha": value / 25, "yield_t_ha": value / 4,
                    "quality_flag": flag,
                    "reporting_unit": cfg.provinces.reporting_unit(province, year, quarter)})
    return pd.DataFrame(out)


def _vi(rows: list[tuple]) -> pd.DataFrame:
    """VI rows from (province, date, index, mean, valid_frac, n_pixels)."""
    df = pd.DataFrame(rows, columns=["province", "date", "index", "mean", "valid_frac",
                                     "n_pixels"])
    df["date"] = pd.to_datetime(df["date"])
    df["median"], df["max"] = df["mean"], df["mean"] + 0.05
    return df[VI_COLS]


def _modis_series(province: str, start: str, end: str, value=0.5) -> list[tuple]:
    dates = pd.date_range(start, end, freq="16D")
    return [(province, d, i, value, 1.0, 100) for d in dates for i in ("NDVI", "EVI")]


def test_window_bounds_six_months_ending_at_quarter_end():
    assert f.window_bounds("palay", 2020, 1) == (pd.Timestamp("2019-10-01"),
                                                 pd.Timestamp("2020-03-31"))
    assert f.window_bounds("corn", 2020, 3) == (pd.Timestamp("2020-04-01"),
                                                pd.Timestamp("2020-09-30"))
    assert set(f.GROWING_WINDOW_MONTHS.values()) == {6}


def test_season_features_peak_integral_and_greenup():
    days = np.array([0, 30, 60, 90, 120])
    vals = np.array([0.0, 0.0, 0.5, 1.0, 0.4])
    out = f.season_features(days, vals, np.full(5, 0.8), n_days=121, min_obs=3)
    assert out["peak"] == 1.0 and out["peak_day"] == 90 and out["n_obs"] == 5
    # halfway from 0.0 to 1.0 is 0.5, first reached on day 60
    assert out["greenup_day"] == 60
    daily = np.interp(np.arange(121), days, vals)
    assert out["integral"] == pytest.approx(daily.sum())
    assert out["valid_frac"] == pytest.approx(0.8)


def test_season_features_too_few_observations_are_nan():
    out = f.season_features(np.array([0, 10]), np.array([0.3, np.nan]), np.array([1.0, 0.0]),
                            n_days=30, min_obs=2)
    assert out["n_obs"] == 1 and np.isnan(out["peak"]) and np.isnan(out["integral"])


def test_modis_cutoff_uses_only_composites_ending_by_quarter_end(cfg):
    gt = _gt(cfg, [("Davao Oriental", "palay", 2020, 1)])
    rows = _modis_series("Davao Oriental", "2019-10-01", "2020-03-20")
    rows += [("Davao Oriental", "2020-03-31", "NDVI", 0.8, 1.0, 100),   # ends on quarter end
             ("Davao Oriental", "2020-04-01", "NDVI", 0.99, 1.0, 100),  # one day later
             ("Davao Oriental", "2019-09-30", "NDVI", 0.95, 1.0, 100)]  # before the window
    out = f.build_features(gt, {"modis": _vi(rows)}, cfg)
    row = out.iloc[0]
    assert row["modis_ndvi_peak"] == pytest.approx(0.8)
    assert row["modis_ndvi_peak_day"] == (pd.Timestamp("2020-03-31")
                                          - pd.Timestamp("2019-10-01")).days
    assert row["window_end"] == pd.Timestamp("2020-03-31")


def test_s2_month_counts_until_its_last_day(cfg):
    gt = _gt(cfg, [("Davao Oriental", "corn", 2020, 1)])
    months = pd.date_range("2019-10-01", "2020-04-01", freq="MS")
    rows = [("Davao Oriental", m, "NDVI", 0.9 if m.month == 4 else 0.3 + 0.01 * k, 0.5, 50)
            for k, m in enumerate(months)]
    out = f.build_features(gt, {"s2": _vi(rows)}, cfg).iloc[0]
    assert out["s2_ndvi_n_obs"] == 6  # Oct..Mar; April imagery ends after quarter end
    assert out["s2_ndvi_peak"] == pytest.approx(0.35)
    assert out["s2_evi_n_obs"] == 0 and np.isnan(out["s2_evi_peak"])


def test_unit_vi_weights_members_by_valid_pixels(cfg):
    rows = [("Davao del Sur", "2010-03-06", "NDVI", 0.4, 1.0, 100),
            ("Davao Occidental", "2010-03-06", "NDVI", 0.8, 0.5, 300)]
    u = f.unit_vi(_vi(rows), cfg)
    c = u[u["reporting_unit"] == "Davao del Sur+Davao Occidental"].iloc[0]
    assert c["mean"] == pytest.approx(0.7)
    assert c["max"] == pytest.approx(0.85)
    assert c["n_pixels"] == 400
    assert c["valid_frac"] == pytest.approx(400 / (100 + 600))
    assert set(u["reporting_unit"]) >= {"Davao del Sur", "Davao Occidental"}


def test_unit_vi_combined_is_nan_when_a_member_is_invalid(cfg):
    rows = [("Davao del Sur", "2010-03-06", "NDVI", 0.4, 1.0, 100),
            ("Davao Occidental", "2010-03-06", "NDVI", np.nan, 0.0, 0),
            ("Davao Occidental", "2010-03-22", "NDVI", 0.6, 0.5, 300),
            ("Davao del Sur", "2010-04-07", "NDVI", 0.4, 1.0, 100)]
    u = f.unit_vi(_vi(rows), cfg)
    c = u[u["reporting_unit"] == "Davao del Sur+Davao Occidental"].set_index("date")
    assert c["mean"].isna().all()  # 03-06 member NaN; 03-22 and 04-07 a member is absent
    # cropland cells of Davao Occidental (600) are known from its other rows
    assert c.loc[pd.Timestamp("2010-03-06"), "valid_frac"] == pytest.approx(100 / 700)


def test_not_reported_rows_dropped_and_host_uses_combined_unit(cfg):
    gt = _gt(cfg, [("Davao del Sur", "palay", 2010, 1), ("Davao Occidental", "palay", 2010, 1),
                   ("Davao del Sur", "palay", 2019, 1)])
    rows = []
    for prov, val in (("Davao del Sur", 0.4), ("Davao Occidental", 0.8)):
        rows += [(p, d, i, val, 1.0, 100) for p, d, i, *_ in
                 _modis_series(prov, "2009-10-01", "2010-03-31")]
        rows += [(p, d, i, val, 1.0, 100) for p, d, i, *_ in
                 _modis_series(prov, "2018-10-01", "2019-03-31")]
    out = f.build_features(gt, {"modis": _vi(rows)}, cfg)
    assert "not_reported" not in set(out["quality_flag"])
    assert len(out) == 2 and set(out["province"]) == {"Davao del Sur"}
    host = out[out["year"] == 2010].iloc[0]
    assert host["reporting_unit"] == "Davao del Sur+Davao Occidental"
    assert host["modis_ndvi_peak"] == pytest.approx(0.6)  # equal pixel weights
    assert out[out["year"] == 2019].iloc[0]["modis_ndvi_peak"] == pytest.approx(0.4)


def test_unknown_source_raises(cfg):
    with pytest.raises(ValueError, match="unknown VI source"):
        f.build_features(_gt(cfg, [("Davao Oriental", "palay", 2020, 1)]),
                         {"landsat": _vi([])}, cfg)


def test_quality_report_lists_counts_and_missingness(cfg):
    gt = _gt(cfg, [("Davao del Sur", "palay", 2010, 1), ("Davao Occidental", "palay", 2010, 1)])
    feats = f.build_features(gt, {"modis": _vi([])}, cfg)
    text = f.quality_report(feats, gt, ["modis"])
    assert "| excluded `not_reported` rows (no PSA target) | 1 |" in text
    assert "| feature rows (`data/processed/features.parquet`) | 1 |" in text
    assert "| includes:<unit> | 1 |" in text
    assert "| modis_ndvi_peak | 1 | 100.0 |" in text
    assert "Rainfall (Open-Meteo ERA5)" in text and "CHIRPS" not in text
    rain_text = f.quality_report(feats, gt, ["modis"], has_rain=True)
    assert "Open-Meteo ERA5, daily totals" in rain_text and "CHIRPS" not in rain_text


def test_main_writes_features_and_report(tmp_path, capsys, caplog, cfg):
    conf = tmp_path / "configs"
    shutil.copytree(DEFAULT_CONFIG_DIR, conf)
    tcfg = load_config(conf)
    interim = tcfg.pipeline.paths.interim
    interim.mkdir(parents=True)
    _gt(cfg, [("Davao Oriental", "palay", 2020, 1)]).to_parquet(interim / "ground_truth.parquet")
    _vi(_modis_series("Davao Oriental", "2019-10-01", "2020-03-31")).to_parquet(
        interim / "modis_vi.parquet")
    with caplog.at_level("WARNING"):
        f.main(["--config-dir", str(conf)])
    feats = pd.read_parquet(tcfg.pipeline.paths.processed / "features.parquet")
    assert len(feats) == 1 and feats["modis_ndvi_n_obs"].iloc[0] > 0
    assert "s2_ndvi_peak" not in feats.columns  # s2_vi.parquet absent: skipped
    # rainfall.parquet absent: rain columns present but NaN, with a warning
    assert feats[list(f.RAIN_COLUMNS)].isna().all().all()
    assert "rainfall.parquet not found" in caplog.text
    assert (tcfg.pipeline.paths.reports / "data_quality.md").exists()
    assert "(1 rows)" in capsys.readouterr().out


def test_run_requires_ground_truth(tmp_path):
    conf = tmp_path / "configs"
    shutil.copytree(DEFAULT_CONFIG_DIR, conf)
    with pytest.raises(FileNotFoundError, match="ground_truth"):
        f.run(load_config(conf))


# ----------------------------------------------------------------------- rainfall
def _rain(province: str, start: str, end: str, fn=lambda m: float(m.month)) -> pd.DataFrame:
    """Monthly rainfall rows (`rainfall.parquet` schema) with rain_mm = fn(month start)."""
    months = pd.date_range(start, end, freq="MS")
    return pd.DataFrame({"province": province, "date": months,
                         "rain_mm": [fn(m) for m in months], "valid_frac": 1.0})


def _rows(unit: str, crop: str, year: int, quarter: int) -> pd.DataFrame:
    start, end = f.window_bounds(crop, year, quarter)
    return pd.DataFrame({"reporting_unit": [unit], "window_start": [start],
                         "window_end": [end]})


def test_rain_window_total_and_halves(cfg):
    # rain = month number; Q1 2020 window is Oct 2019 .. Mar 2020
    rain = _rain("Davao Oriental", "2019-01-01", "2020-12-01")
    r = f.rain_features(_rows("Davao Oriental", "palay", 2020, 1), rain, cfg).iloc[0]
    assert r["rain_first_half"] == 10 + 11 + 12
    assert r["rain_second_half"] == 1 + 2 + 3
    assert r["rain_total"] == 10 + 11 + 12 + 1 + 2 + 3
    assert np.isnan(r["rain_anom"])  # no 1981-2000 reference rows


def test_rain_ignores_months_after_quarter_end(cfg):
    rain = _rain("Davao Oriental", "2019-10-01", "2020-03-01", fn=lambda m: 1.0)
    rows = _rows("Davao Oriental", "corn", 2020, 1)
    base = f.rain_features(rows, rain, cfg).iloc[0]
    later = pd.concat([rain, _rain("Davao Oriental", "2020-04-01", "2020-06-01",
                                   fn=lambda m: 1000.0),
                       _rain("Davao Oriental", "2019-07-01", "2019-09-01",
                             fn=lambda m: 1000.0)])
    out = f.rain_features(rows, later, cfg).iloc[0]
    assert base["rain_total"] == out["rain_total"] == 6.0
    assert out["rain_second_half"] == 3.0


def test_rain_missing_month_gives_nan_not_zero(cfg):
    rain = _rain("Davao Oriental", "2019-10-01", "2020-03-01", fn=lambda m: 1.0)
    rows = _rows("Davao Oriental", "palay", 2020, 1)
    nan_month = rain.assign(rain_mm=np.where(rain["date"] == "2020-02-01", np.nan,
                                             rain["rain_mm"]))
    r = f.rain_features(rows, nan_month, cfg).iloc[0]
    assert np.isnan(r["rain_total"]) and np.isnan(r["rain_second_half"])
    assert np.isnan(r["rain_anom"])
    assert r["rain_first_half"] == 3.0  # Oct..Dec complete
    absent = f.rain_features(rows, rain[rain["date"] != "2019-11-01"], cfg).iloc[0]
    assert np.isnan(absent["rain_total"]) and np.isnan(absent["rain_first_half"])
    assert absent["rain_second_half"] == 3.0


def test_rain_anomaly_uses_fixed_reference_climatology_only(cfg):
    # reference 1981-2000: rain = 10 every month; later years: rain = 25
    rain = pd.concat([_rain("Davao Oriental", "1981-01-01", "2000-12-01", fn=lambda m: 10.0),
                      _rain("Davao Oriental", "2001-01-01", "2020-12-01", fn=lambda m: 25.0)])
    rows = _rows("Davao Oriental", "palay", 2020, 1)
    r = f.rain_features(rows, rain, cfg).iloc[0]
    assert r["rain_total"] == 150.0 and r["rain_anom"] == pytest.approx(150.0 - 60.0)
    clim = f.rain_climatology(rain)
    # perturbing years after the reference period leaves the climatology unchanged
    perturbed = rain.assign(rain_mm=np.where(rain["date"].dt.year > 2000, 999.0,
                                             rain["rain_mm"]))
    pd.testing.assert_series_equal(clim, f.rain_climatology(perturbed))
    assert (clim == 10.0).all() and len(clim) == 12
    # too few valid reference years -> NaN climatology
    sparse = rain[(rain["date"].dt.year > 1995) | (rain["date"].dt.month != 1)]
    assert np.isnan(f.rain_climatology(sparse).loc[("Davao Oriental", 1)])


def test_rain_climatology_must_end_before_modeling_eras(cfg, monkeypatch):
    monkeypatch.setattr(f, "RAIN_CLIM_YEARS", (1981, 2005))
    with pytest.raises(ValueError, match="climatology"):
        f.rain_features(_rows("Davao Oriental", "palay", 2020, 1), None, cfg)


def test_rain_joins_combined_reporting_unit(cfg, caplog):
    gt = _gt(cfg, [("Davao del Sur", "palay", 2010, 1), ("Davao Occidental", "palay", 2010, 1),
                   ("Davao Oriental", "palay", 2010, 1)])
    unit = "Davao del Sur+Davao Occidental"
    rain = pd.concat([_rain(unit, "2009-10-01", "2010-03-01", fn=lambda m: 7.0),
                      _rain("Davao del Sur", "2009-10-01", "2010-03-01", fn=lambda m: 1.0)])
    with caplog.at_level("WARNING"):
        out = f.build_features(gt, {"modis": _vi([])}, cfg, rain)
    host = out[out["reporting_unit"] == unit].iloc[0]
    assert host["rain_total"] == 42.0  # combined unit's own rows, not the member's
    assert np.isnan(out[out["reporting_unit"] == "Davao Oriental"].iloc[0]["rain_total"])
    assert "Davao Oriental" in caplog.text  # unmatched reporting unit is reported


def test_build_features_without_rain_has_nan_columns(cfg):
    out = f.build_features(_gt(cfg, [("Davao Oriental", "palay", 2020, 1)]),
                           {"modis": _vi([])}, cfg)
    assert list(out.columns[-4:]) == list(f.RAIN_COLUMNS)
    assert out[list(f.RAIN_COLUMNS)].isna().all().all()


def test_run_reads_rainfall_when_present(tmp_path, cfg):
    conf = tmp_path / "configs"
    shutil.copytree(DEFAULT_CONFIG_DIR, conf)
    tcfg = load_config(conf)
    interim = tcfg.pipeline.paths.interim
    interim.mkdir(parents=True)
    _gt(cfg, [("Davao Oriental", "palay", 2020, 1)]).to_parquet(interim / "ground_truth.parquet")
    _vi(_modis_series("Davao Oriental", "2019-10-01", "2020-03-31")).to_parquet(
        interim / "modis_vi.parquet")
    _rain("Davao Oriental", "2019-10-01", "2020-03-01", fn=lambda m: 2.0).to_parquet(
        interim / "rainfall.parquet")
    feats_path, report = f.run(tcfg)
    assert pd.read_parquet(feats_path)["rain_total"].iloc[0] == 12.0
    assert "1981-2000 climatology" in report.read_text()
