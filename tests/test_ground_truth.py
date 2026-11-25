"""Tests for cropyield.ground_truth with small hand-written PSA CSVs (no network)."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cropyield import ground_truth as gt
from cropyield.config import load_config

FIXTURES = Path(__file__).parent / "fixtures" / "psa"


@pytest.fixture
def cfg(tmp_path):
    base = load_config()
    paths = dataclasses.replace(base.pipeline.paths, raw=tmp_path / "raw",
                                interim=tmp_path / "interim")
    return dataclasses.replace(base, pipeline=dataclasses.replace(base.pipeline, paths=paths))


@pytest.fixture
def df(cfg):
    # The fixture CSVs use 2024, which is inside the configured Davao City reporting
    # window; these tests check the base cleaning rules, so they run without windows.
    no_windows = dataclasses.replace(cfg.provinces, reporting_units=())
    return gt.load_psa(FIXTURES, dataclasses.replace(cfg, provinces=no_windows))


def _row(df, province, crop, quarter, year=2024):
    r = df[(df.province == province) & (df.crop == crop) & (df.year == year)
           & (df.quarter == quarter)]
    assert len(r) == 1
    return r.iloc[0]


def _write(folder: Path, rows: list[tuple]) -> Path:
    """rows: (geolocation, croptype, year, period, production_mt, area_ha)."""
    folder.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows, columns=["geolocation", "croptype", "year", "period",
                                    "production_mt", "area_ha"])
    t.drop(columns="area_ha").to_csv(folder / gt.PRODUCTION_CSV, index=False)
    t.drop(columns="production_mt").to_csv(folder / gt.AREA_CSV, index=False)
    return folder


def test_columns_match_contract(df):
    assert list(df.columns) == gt.COLUMNS
    assert df[["year", "quarter"]].dtypes.tolist() == [np.int64, np.int64]


def test_province_names_harmonised_and_region_dropped(df):
    assert set(df.province) == {"Davao de Oro", "Davao City"}


def test_compostela_valley_alias(cfg, tmp_path):
    folder = _write(tmp_path / "psa", [("Compostela Valley", "Corn", 2010, "Quarter 1", 10, 5)])
    out = gt.load_psa(folder, cfg)
    assert set(out.province) == {"Davao de Oro"}


def test_province_lookup_is_case_insensitive(cfg):
    lookup = gt.province_lookup(cfg)
    assert lookup[gt._norm("city of davao")] == "Davao City"
    assert lookup[gt._norm("DAVAO DEL NORTE")] == "Davao del Norte"


def test_only_crop_totals_and_quarters_kept(df):
    assert set(df.crop) == {"palay", "corn"}
    assert set(df.quarter) == {1, 2, 3, 4}
    # Q1 palay is the "Palay" total, not Irrigated Palay, and no Annual row leaks in.
    assert _row(df, "Davao de Oro", "palay", 1).production_mt == 4000
    assert len(df) == 2 * 2 * 1 * 4  # provinces x crops x years x quarters


def test_yield_is_production_over_area(df):
    assert _row(df, "Davao de Oro", "palay", 4).yield_t_ha == pytest.approx(4.0)
    assert _row(df, "Davao de Oro", "corn", 1).yield_t_ha == pytest.approx(1.3)
    assert _row(df, "Davao City", "palay", 3).yield_t_ha == pytest.approx(4.0)
    assert _row(df, "Davao de Oro", "palay", 1).quality_flag == "ok"


def test_missing_value_is_flagged_not_filled(df):
    r = _row(df, "Davao de Oro", "corn", 4)
    assert np.isnan(r.production_mt) and r.area_ha == 900
    assert np.isnan(r.yield_t_ha)
    assert r.quality_flag == "missing"


def test_absent_quarters_are_added_and_flagged(df):
    r = _row(df, "Davao City", "palay", 4)
    assert np.isnan(r.production_mt) and np.isnan(r.area_ha)
    assert r.quality_flag == "missing"
    city_corn = df[(df.province == "Davao City") & (df.crop == "corn")]
    assert len(city_corn) == 4 and (city_corn.quality_flag == "missing").all()


def test_zero_rows_are_kept_and_flagged(df):
    r = _row(df, "Davao City", "palay", 1)
    assert r.production_mt == 0 and r.area_ha == 0 and np.isnan(r.yield_t_ha)
    assert r.quality_flag == "zero"


def test_zero_area_with_production_is_flagged(cfg, tmp_path):
    folder = _write(tmp_path / "psa", [("Davao Oriental", "Palay", 2020, "Quarter 2", 50, 0)])
    r = _row(gt.load_psa(folder, cfg), "Davao Oriental", "palay", 2, year=2020)
    assert r.quality_flag == "zero_area" and np.isnan(r.yield_t_ha)


def test_outlier_yield_is_flagged(cfg, tmp_path):
    rows = [("Davao del Norte", "Palay", y, "Quarter 1", 4000 + 10 * (y % 3), 1000)
            for y in range(2010, 2020)]
    rows.append(("Davao del Norte", "Palay", 2020, "Quarter 1", 40000, 1000))  # 40 t/ha
    out = gt.load_psa(_write(tmp_path / "psa", rows), cfg)
    q1 = out[out.quarter == 1].set_index("year")
    assert q1.loc[2020, "quality_flag"] == "outlier"
    assert (q1.loc[2010:2019, "quality_flag"] == "ok").all()
    assert q1.loc[2020, "production_mt"] == 40000  # kept as reported


def test_save_ground_truth_writes_parquet(df, cfg):
    out = gt.save_ground_truth(df, cfg)
    assert out == cfg.pipeline.paths.interim / "ground_truth.parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(out), df)


# ------------------------------------------------------------ reporting windows (provinces.yaml)

def test_reporting_window_bounds_are_inclusive(cfg, tmp_path):
    rows = [("City of Davao", "Palay", 2020, "Quarter 4", 900, 300),
            ("City of Davao", "Palay", 2021, "Quarter 1", 0, 0),
            ("City of Davao", "Palay", 2025, "Quarter 1", 0, 0),
            ("City of Davao", "Palay", 2025, "Quarter 2", 800, 200)]
    out = gt.load_psa(_write(tmp_path / "psa", rows), cfg)
    for year, quarter in [(2021, 1), (2025, 1)]:
        r = _row(out, "Davao City", "palay", quarter, year=year)
        assert r.quality_flag == "not_reported"
        assert np.isnan(r.production_mt) and np.isnan(r.area_ha) and np.isnan(r.yield_t_ha)
        assert r.reporting_unit == "Davao del Sur+Davao City"
    for year, quarter in [(2020, 4), (2025, 2)]:
        r = _row(out, "Davao City", "palay", quarter, year=year)
        assert r.quality_flag == "ok" and r.reporting_unit == "Davao City"
    # Quarters with no CSV row inside the window are not_reported, not missing.
    assert _row(out, "Davao City", "palay", 3, year=2022).quality_flag == "not_reported"


def test_host_rows_get_includes_and_combined_reporting_unit(cfg, tmp_path):
    rows = [("Davao del Sur", "Corn", 2018, "Quarter 4", 3000, 1500),
            ("Davao del Sur", "Corn", 2019, "Quarter 1", 2000, 1000),
            ("Davao del Sur", "Corn", 2021, "Quarter 1", 2600, 1300),
            ("Davao Del Norte", "Corn", 2001, "Quarter 4", 1000, 500),
            ("Davao Del Norte", "Corn", 2002, "Quarter 1", 1000, 500),
            ("Compostela Valley", "Corn", 2001, "Quarter 4", 0, 0),
            ("Davao Occidental", "Corn", 2018, "Quarter 4", 0, 0)]
    out = gt.load_psa(_write(tmp_path / "psa", rows), cfg)
    expect = {("Davao del Sur", 2018, 4): ("includes:Davao Occidental",
                                           "Davao del Sur+Davao Occidental"),
              ("Davao del Sur", 2019, 1): ("ok", "Davao del Sur"),
              ("Davao del Sur", 2021, 1): ("includes:Davao City", "Davao del Sur+Davao City"),
              ("Davao del Norte", 2001, 4): ("includes:Davao de Oro",
                                             "Davao del Norte+Davao de Oro"),
              ("Davao del Norte", 2002, 1): ("ok", "Davao del Norte"),
              ("Davao de Oro", 2001, 4): ("not_reported", "Davao del Norte+Davao de Oro"),
              ("Davao Occidental", 2018, 4): ("not_reported", "Davao del Sur+Davao Occidental")}
    for (province, year, quarter), (flag, unit) in expect.items():
        r = _row(out, province, "corn", quarter, year=year)
        assert (r.quality_flag, r.reporting_unit) == (flag, unit), (province, year, quarter)
    host = _row(out, "Davao del Sur", "corn", 1, year=2021)
    assert host.production_mt == 2600 and host.yield_t_ha == pytest.approx(2.0)


def test_isolated_zero_outside_windows_stays_zero(cfg, tmp_path):
    rows = [("City of Davao", "Corn", 1998, "Quarter 2", 0, 0),
            ("City of Davao", "Corn", 1998, "Quarter 3", 500, 400)]
    out = gt.load_psa(_write(tmp_path / "psa", rows), cfg)
    r = _row(out, "Davao City", "corn", 2, year=1998)
    assert r.quality_flag == "zero" and r.production_mt == 0 and r.area_ha == 0
    assert r.reporting_unit == "Davao City"


def test_outlier_reference_excludes_window_rows(cfg, tmp_path):
    # Davao del Norte hosts Davao de Oro up to 2001: 12 inflated host years at 8 t/ha,
    # then 6 own-footprint years at 4 t/ha. With the window rows in the reference the
    # median would be 8 and every 2002+ year an outlier.
    rows = [("Davao del Norte", "Palay", y, "Quarter 1", (8000 if y <= 2001 else 4000)
             + 10 * (y % 3), 1000) for y in range(1990, 2008)]
    out = gt.load_psa(_write(tmp_path / "psa", rows), cfg)
    q1 = out[(out.province == "Davao del Norte") & (out.quarter == 1)].set_index("year")
    assert (q1.loc[2002:2007, "quality_flag"] == "ok").all()
    # Window rows are still scored against the 2002+ reference.
    assert (q1.loc[1990:2001, "quality_flag"] == "outlier;includes:Davao de Oro").all()
