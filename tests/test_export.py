"""App data export (phase 3, task 3): synthetic inputs in tmp paths, no downloads."""
from __future__ import annotations

import dataclasses
import json
import re
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box, shape
from typer.testing import CliRunner

import cropyield.export as ex
from cropyield.cli import app
from cropyield.config import load_config
from cropyield.models.forecast import FORECAST_COLUMNS
from cropyield.models.select import PREDICTION_COLUMNS

NOW = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
COMBINED = "Davao del Sur+Davao City"
# 2002-2020: Norte, Sur, City separately; 2021-2022: Sur+City combined; 2023: split again.
YEARS = range(2002, 2024)


def _units(year: int) -> list[str]:
    return ["Davao del Norte", *([COMBINED] if year in (2021, 2022)
                                 else ["Davao del Sur", "Davao City"])]


def _cfg(tmp_path):
    cfg = load_config()
    paths = dataclasses.replace(cfg.pipeline.paths, reports=tmp_path / "reports",
                                processed=tmp_path / "processed",
                                interim=tmp_path / "interim", models=tmp_path / "models")
    return dataclasses.replace(cfg, pipeline=dataclasses.replace(cfg.pipeline, paths=paths))


def _features() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for y in YEARS:
        for q in (1, 2, 3, 4):
            for i, u in enumerate(_units(y)):
                for crop in ("palay", "corn"):
                    yl = 2.0 + 0.3 * i + 0.1 * q + 0.03 * (y - 2002) + 0.1 * rng.normal()
                    end = pd.Timestamp(y, 3 * q, 1) + pd.offsets.MonthEnd(0)
                    rows.append({"reporting_unit": u, "crop": crop, "year": y, "quarter": q,
                                 "quality_flag": "outlier" if (y, q, i) == (2015, 2, 0) else "ok",
                                 "yield_t_ha": yl, "area_ha": 1000.0 + 100 * i,
                                 "production_mt": yl * (1000.0 + 100 * i),
                                 "rain_total": 600.0 + 50 * rng.normal(),
                                 "rain_anom": 40.0 + 30 * rng.normal(),
                                 "modis_ndvi_peak": 0.7 + 0.05 * rng.normal(),
                                 "modis_evi_peak": 0.5 + 0.05 * rng.normal(),
                                 "window_start": end - pd.DateOffset(months=6) + pd.Timedelta(days=1),
                                 "window_end": end})
    f = pd.DataFrame(rows)
    # the target quarter 2024Q1: features but no PSA yield yet
    tgt = f.loc[(f["year"] == 2023) & (f["quarter"] == 1)].copy()
    tgt["year"] = 2024
    tgt[["yield_t_ha", "area_ha", "production_mt", "rain_anom"]] = np.nan
    tgt["window_end"] = pd.Timestamp("2024-03-31")
    return pd.concat([f, tgt], ignore_index=True)


def _predictions(feat: pd.DataFrame) -> pd.DataFrame:
    d = feat.loc[(feat["year"] >= 2012) & feat["yield_t_ha"].notna()].reset_index(drop=True)
    p = pd.DataFrame({c: np.nan for c in PREDICTION_COLUMNS}, index=d.index)
    for c in ("reporting_unit", "crop", "year", "quarter", "quality_flag"):
        p[c] = d[c]
    p["is_outlier"] = d["quality_flag"].eq("outlier")
    p["model"] = "B3"
    p["actual_yield_t_ha"] = d["yield_t_ha"]
    p["pred_yield_t_ha"] = d["yield_t_ha"] * 1.03
    later = d["year"] >= 2015
    p["pred_lo_t_ha"] = (p["pred_yield_t_ha"] * 0.9).where(later)
    p["pred_hi_t_ha"] = (p["pred_yield_t_ha"] * 1.1).where(later)
    p["interval_n_years"] = (d["year"] - 2012).clip(lower=0)
    p["b2_pred_yield_t_ha"] = d["yield_t_ha"] * 0.95
    p["b0_pred_yield_t_ha"] = d["yield_t_ha"] * 1.05
    split = d["reporting_unit"].eq(COMBINED) & d["year"].eq(2021)
    p["b0_fallback"] = split
    p["predicted_area_ha"] = d["area_ha"].where(~split)
    p["area_fallback"] = split
    p["area_fallback_level"] = np.where(split, "crop_quarter_mean", "last_year")
    p["pred_production_mt"] = p["pred_yield_t_ha"] * p["predicted_area_ha"]
    p["actual_area_ha"] = d["area_ha"]
    p["actual_production_mt"] = d["production_mt"]
    p["n_train_years"] = d["year"] - 2002
    p["modis_ndvi_peak_anom"] = 0.01
    p["modis_evi_peak_anom"] = -0.02
    p["rain_anom"] = d["rain_anom"]
    p["modis_ndvi_peak_anom"] = p["modis_ndvi_peak_anom"].where(d["year"] > 2012)
    return p


SELECTION = pd.DataFrame([
    {"crop": "corn", "candidate": "B0", "role": "naive baseline", "n": 50, "mae": 0.175,
     "rmse": 0.249, "mape": 11.08, "d_b2": -0.0245, "d_b2_lo": -0.045, "d_b2_hi": -0.003,
     "d_b0": 0.0, "d_b0_lo": np.nan, "d_b0_hi": np.nan, "qualifies": True, "selected": False},
    {"crop": "corn", "candidate": "B2", "role": "reference baseline", "n": 50, "mae": 0.2,
     "rmse": 0.266, "mape": 13.03, "d_b2": 0.0, "d_b2_lo": np.nan, "d_b2_hi": np.nan,
     "d_b0": 0.0245, "d_b0_lo": 0.003, "d_b0_hi": 0.045, "qualifies": False, "selected": False},
    {"crop": "corn", "candidate": "B3", "role": "post hoc blend of B0 and B2", "n": 50,
     "mae": 0.1628172, "rmse": 0.2267, "mape": 10.45, "d_b2": -0.0373, "d_b2_lo": -0.051,
     "d_b2_hi": -0.0248, "d_b0": -0.0129, "d_b0_lo": -0.024, "d_b0_hi": -0.0017,
     "qualifies": True, "selected": True},
    {"crop": "palay", "candidate": "B2", "role": "reference baseline", "n": 50, "mae": 0.297,
     "rmse": 0.418, "mape": 7.48, "d_b2": 0.0, "d_b2_lo": np.nan, "d_b2_hi": np.nan,
     "d_b0": 0.0017, "d_b0_lo": -0.02, "d_b0_hi": 0.025, "qualifies": False, "selected": True},
])

EXPERIMENTS = pd.DataFrame([
    {"timestamp": "2026-11-25T00:00:00+00:00", "model": "AnomalyRidge:rain", "era": "modis",
     "crop": "corn", "mae": 0.30, "mae_diff_ci_low": -0.1, "mae_diff_ci_high": -0.05},
    {"timestamp": "2026-11-26T00:00:00+00:00", "model": "AnomalyRidge:rain", "era": "modis",
     "crop": "corn", "mae": 0.1994, "mae_diff_ci_low": -0.0023, "mae_diff_ci_high": 0.0007},
    {"timestamp": "2026-11-26T00:00:00+00:00", "model": "BoostedModel:modis", "era": "modis",
     "crop": "palay", "mae": 0.36, "mae_diff_ci_low": 0.02, "mae_diff_ci_high": 0.1},
    {"timestamp": "2026-11-26T00:00:00+00:00", "model": "B2", "era": "modis",
     "crop": "palay", "mae": 0.297, "mae_diff_ci_low": np.nan, "mae_diff_ci_high": np.nan},
]).assign(params="{}", features="", test_years="2012-2023", n_test=50, rmse=0.3, mape=9.0,
          n_test_no_outliers=49, mae_no_outliers=0.2, rmse_no_outliers=0.3,
          mape_no_outliers=9.0, ref_model="B2", mae_diff=0.0)

RESULTS_MD = """# Phase 2 results

## Model selection

**Caveats.** B3 was chosen after seeing the results.

**Deviation from Requirement 6.1.** The rule was set after the results were seen.

## Why satellites and rainfall did not help (likely reasons)

- Cropland mask mixes **plantations**.
- PSA estimates are noisy.

## Other limitations

- One limitation.

## Next steps

- A rice-paddy mask.
"""


def _forecasts() -> pd.DataFrame:
    rows = []
    for fid, made, backfill, b3 in (("2024Q1-a", "2026-11-26T01:00:00Z", False, 2.5),
                                    ("2024Q1-b", "2026-11-26T02:00:00Z", False, 2.6),
                                    ("2023Q4-bf", "2026-11-26T03:00:00Z", True, 9.9)):
        for u in _units(2023):
            for crop in ("palay", "corn"):
                r = {c: np.nan for c in FORECAST_COLUMNS}
                tq = "2023Q4" if backfill else "2024Q1"
                r.update(forecast_id=fid, made_at_utc=made, git_commit="abc",
                         code_hash="h", recipe_version="phase3-v1", data_vintage="2023Q4",
                         target_quarter=tq, year=int(tq[:4]), quarter=int(tq[-1]),
                         reporting_unit=u, crop=crop, model="B3", b3_yield=b3,
                         b3_lo80=b3 * 0.9, b3_hi80=b3 * 1.1, b0_yield=b3 + 0.1,
                         b2_yield=b3 - 0.1, b0_fallback=False,
                         predicted_area_ha=np.nan if u == "Davao City" else 1000.0,
                         area_fallback_level=("crop_quarter_mean" if u == "Davao City"
                                              else "last_year"),
                         history_last_year=2023, backfill=backfill)
                r["pred_production_mt"] = r["b3_yield"] * r["predicted_area_ha"]
                r["pred_production_lo_mt"] = r["b3_lo80"] * r["predicted_area_ha"]
                r["pred_production_hi_mt"] = r["b3_hi80"] * r["predicted_area_ha"]
                rows.append(r)
    return pd.DataFrame(rows)[list(FORECAST_COLUMNS)]


def _boundaries() -> gpd.GeoDataFrame:
    """Adjacent boxes with many vertices on the edges, so simplification has work."""
    def dense(x0, y0, x1, y1, n=200):
        b = box(x0, y0, x1, y1)
        return b.segmentize((x1 - x0) / n)
    names = ["Davao de Oro", "Davao del Norte", "Davao del Sur", "Davao Oriental",
             "Davao Occidental", "Davao City"]
    geoms = [dense(125.0 + 0.5 * i, 7.0, 125.5 + 0.5 * i, 7.5) for i in range(len(names))]
    return gpd.GeoDataFrame({"province": names, "source_name": names}, geometry=geoms,
                            crs=4326)


@pytest.fixture
def setup(tmp_path):
    cfg = _cfg(tmp_path)
    for d in ("processed", "interim", "reports/figures"):
        (tmp_path / d).mkdir(parents=True)
    feat = _features()
    feat.to_parquet(tmp_path / "processed" / "features.parquet")
    _predictions(feat).to_parquet(tmp_path / "processed" / "predictions.parquet")
    _boundaries().to_parquet(tmp_path / "interim" / "boundaries.parquet")
    SELECTION.to_csv(tmp_path / "reports" / "selection.csv", index=False)
    EXPERIMENTS.to_csv(tmp_path / "reports" / "experiments.csv", index=False)
    _forecasts().to_csv(tmp_path / "reports" / "forecasts.csv", index=False)
    (tmp_path / "reports" / "phase2_results.md").write_text(RESULTS_MD)
    (tmp_path / "reports" / "figures" / "mae_by_model.png").write_bytes(b"\x89PNG fake")
    return cfg, tmp_path


def _export(setup, **kw):
    cfg, tmp = setup
    out = ex.export_app(cfg, app_dir=tmp / "app", now=NOW, git="abc123", **kw)
    return out, tmp / "app"


def _load(path):
    def bad(c):
        raise AssertionError(f"non-finite JSON constant {c}")
    return json.loads(path.read_text(), parse_constant=bad)


# ---------------------------------------------------------------------- unit helpers

def test_historical_normal_uses_only_earlier_years():
    hist = pd.DataFrame({"reporting_unit": "A", "crop": "corn", "quarter": 1,
                         "year": range(2000, 2010), "yield_t_ha": np.arange(10.0)})
    tgt = pd.DataFrame({"reporting_unit": "A", "crop": "corn", "year": [2002, 2003, 2006],
                        "quarter": 1})
    n = ex.historical_normal(hist, tgt, start_year=2000).set_index("year")
    assert np.isnan(n.loc[2002, "normal"])            # 2 earlier years < 3
    assert n.loc[2003, "normal"] == pytest.approx(1.0)  # mean of 0, 1, 2
    assert n.loc[2006, "normal_n_years"] == 6
    future = hist.assign(yield_t_ha=np.where(hist["year"] >= 2006, 99.0, hist["yield_t_ha"]))
    n2 = ex.historical_normal(future, tgt, start_year=2000).set_index("year")
    assert n2.loc[2006, "normal"] == n.loc[2006, "normal"] == pytest.approx(2.5)
    n3 = ex.historical_normal(hist, tgt, start_year=2002).set_index("year")
    assert n3.loc[2006, "normal"] == pytest.approx(3.5)  # years 2002-2005 only


def test_rain_pct_and_jsonable():
    pct = ex.rain_pct(pd.Series([150.0, 10.0]), pd.Series([50.0, 20.0]))
    assert pct.iloc[0] == pytest.approx(50.0) and np.isnan(pct.iloc[1])
    obj = ex.jsonable({"a": np.float64("nan"), "b": [np.int64(3), np.inf], "c": np.bool_(1)})
    assert obj == {"a": None, "b": [3, None], "c": True}


def test_md_helpers():
    assert ex.md_bullets(RESULTS_MD, "Why satellites") == [
        "Cropland mask mixes plantations.", "PSA estimates are noisy."]
    assert ex.md_bullets(RESULTS_MD, "Next steps") == ["A rice-paddy mask."]
    assert ex.md_paragraph(RESULTS_MD, "Caveats.").startswith("Caveats. B3 was chosen")
    assert ex.md_paragraph(RESULTS_MD, "Missing") is None


def test_latest_forecast_skips_backfill_and_takes_newest():
    f = ex.latest_forecast(_forecasts())
    assert set(f["forecast_id"]) == {"2024Q1-b"}
    assert len(f) == 6 and set(f["target_quarter"]) == {"2024Q1"}


# --------------------------------------------------------------------- full export

def test_export_writes_every_file(setup):
    out, app_dir = _export(setup)
    for name in ex.APP_DATA_FILES:
        assert (app_dir / "data" / name).exists()
    assert (app_dir / "assets" / "figures" / "mae_by_model.png").read_bytes() == b"\x89PNG fake"
    assert "figures/mae_by_model.png" in out


def test_json_is_nan_free(setup):
    _, app_dir = _export(setup)
    for name in ex.APP_DATA_FILES:
        text = (app_dir / "data" / name).read_text()
        assert not re.search(r"\bNaN\b|\bInfinity\b", text), name
        _load(app_dir / "data" / name)


def test_write_json_refuses_nan(tmp_path):
    with pytest.raises(ValueError):
        ex.json.dumps(float("nan"), allow_nan=False)
    p = ex.write_json(tmp_path / "x.json", {"v": float("nan")})
    assert _load(p) == {"v": None}


def test_predictions_schema_and_normal(setup):
    _, app_dir = _export(setup)
    p = _load(app_dir / "data" / "predictions.json")
    assert set(p) == {"palay", "corn"}
    rows = p["corn"]["Davao del Norte"]
    keys = {"y", "q", "actual", "pred", "lo", "hi", "b0", "b2", "normal", "pct", "rain_anom",
            "rain_pct", "ndvi_anom", "evi_anom", "flags"}
    assert all(set(r) == keys for r in rows)
    assert min(r["y"] for r in rows) == 2012
    # expanding normal from years 2002..Y-1 of the PSA rows in features.parquet
    feat = _features()
    h = feat.loc[(feat["reporting_unit"] == "Davao del Norte") & (feat["crop"] == "corn")
                 & (feat["quarter"] == 2) & feat["year"].between(2002, 2015)]
    r = next(r for r in rows if (r["y"], r["q"]) == (2016, 2))
    assert r["normal"] == pytest.approx(h["yield_t_ha"].mean(), abs=5e-4)
    assert r["pct"] == pytest.approx(100 * (r["pred"] / r["normal"] - 1), abs=0.1)
    out_row = next(r for r in rows if (r["y"], r["q"]) == (2015, 2))
    assert "outlier" in out_row["flags"]
    comb = p["corn"][COMBINED]
    first = next(r for r in comb if r["y"] == 2021)
    assert {"combined_unit", "b0_fallback", "no_area_forecast"} <= set(first["flags"])
    assert first["normal"] is None  # a new unit has no 3 earlier years
    assert next(r for r in rows if r["y"] == 2012)["lo"] is None


def test_forecast_json(setup):
    _, app_dir = _export(setup)
    f = _load(app_dir / "data" / "forecast.json")
    assert f["target_quarter"] == "2024Q1" and f["forecast_ids"] == ["2024Q1-b"]
    assert f["area_note"] == ex.AREA_NOTE
    e = f["forecasts"]["corn"]["Davao del Norte"]
    assert e["yield"] == 2.6 and e["lo"] == pytest.approx(2.34)
    assert e["production_mt"] == 2600 and e["area_source"] == "last_year"
    feat = _features()
    h = feat.loc[(feat["reporting_unit"] == "Davao del Norte") & (feat["crop"] == "corn")
                 & (feat["quarter"] == 1) & feat["year"].between(2002, 2023)]
    assert e["normal"] == pytest.approx(h["yield_t_ha"].mean(), abs=5e-4)  # years < 2024
    assert e["normal_n_years"] == 22
    c = e["conditions"]
    assert c["ndvi_anom"] is not None and c["rain_anom"] is None
    assert c["window_end"] == "2024-03-31" and c["partial_window"] is False
    city = f["forecasts"]["palay"]["Davao City"]
    assert city["production_mt"] is None and "no_area_forecast" in city["flags"]


def test_forecast_conditions_use_only_earlier_years(setup):
    cfg, tmp = setup
    feat = _features()
    table = ex.load_modeling_table(cfg)
    fc = ex.latest_forecast(_forecasts())
    a = ex.target_conditions(table, feat, fc, 2002)
    t2 = table.copy()
    t2.loc[t2["year"] == 2023, "modis_ndvi_peak"] = 5.0  # changes the history -> changes
    b = ex.target_conditions(t2, feat, fc, 2002)
    assert not np.allclose(a[ex.NDVI_COL], b[ex.NDVI_COL])
    expected = (feat.loc[(feat["year"] == 2024) & (feat["reporting_unit"] == "Davao del Norte")
                         & (feat["crop"] == "corn"), "modis_ndvi_peak"].iloc[0]
                - table.loc[(table["reporting_unit"] == "Davao del Norte")
                            & (table["crop"] == "corn") & (table["quarter"] == 1),
                            "modis_ndvi_peak"].mean())
    got = a.loc[(a["reporting_unit"] == "Davao del Norte") & (a["crop"] == "corn"), ex.NDVI_COL]
    assert got.iloc[0] == pytest.approx(expected)


def test_units_geojson(setup):
    _, app_dir = _export(setup)
    g = _load(app_dir / "data" / "units.geojson")
    props = {f["properties"]["unit"]: f["properties"] for f in g["features"]}
    assert set(props) == {"Davao del Norte", "Davao del Sur", "Davao City", COMBINED}
    assert props[COMBINED]["members"] == ["Davao del Sur", "Davao City"]
    assert (props[COMBINED]["first"], props[COMBINED]["last"]) == ("2021Q1", "2022Q4")
    assert props["Davao City"]["last"] == "2024Q1"  # forecast quarter
    geoms = {f["properties"]["unit"]: shape(f["geometry"]) for f in g["features"]}
    assert all(s.is_valid and not s.is_empty for s in geoms.values())
    assert geoms[COMBINED].area == pytest.approx(
        geoms["Davao del Sur"].area + geoms["Davao City"].area, rel=1e-3)
    coords = re.findall(r"-?\d+\.\d+", json.dumps(g["features"][0]["geometry"]))
    assert all(len(c.split(".")[1]) <= ex.COORD_DECIMALS for c in coords)
    n_in = sum(len(p.exterior.coords) for p in _boundaries().geometry)
    n_out = sum(len(json.dumps(f["geometry"])) for f in g["features"])
    assert n_out < n_in * 40  # simplified: far fewer vertices than the dense input


def test_simplify_keeps_shared_edges():
    s = ex.simplify_provinces(_boundaries())
    a, b = s.geometry.iloc[0], s.geometry.iloc[1]
    assert a.intersection(b).area == pytest.approx(0, abs=1e-12)
    assert a.union(b).area == pytest.approx(0.5, rel=1e-3)
    assert all(len(g.exterior.coords) <= 6 for g in s.geometry)


def test_size_budget(setup):
    _, app_dir = _export(setup)
    assert ex.dir_bytes(app_dir / "data") < ex.DATA_BUDGET_BYTES
    with pytest.raises(ValueError, match="budget"):
        _export(setup, budget_bytes=1000)


def test_findings_match_selection_csv(setup):
    cfg, tmp = setup
    _, app_dir = _export(setup)
    f = _load(app_dir / "data" / "findings.json")
    sel = pd.read_csv(tmp / "reports" / "selection.csv")
    for crop, g in sel.groupby("crop"):
        rows = {r["candidate"]: r for r in f["selection"][crop]["rows"]}
        assert f["selection"][crop]["selected"] == g.loc[g["selected"], "candidate"].iloc[0]
        for r in g.itertuples(index=False):
            for col in ("n", "mae", "rmse", "mape", "d_b2", "d_b2_lo", "d_b2_hi", "d_b0"):
                v = getattr(r, col)
                got = rows[r.candidate][col]
                if pd.isna(v):
                    assert got is None
                else:
                    assert got == pytest.approx(v, abs=1e-6)
    assert rows["B2"]["post_hoc"] is False
    assert {r["candidate"]: r["post_hoc"] for r in f["selection"]["corn"]["rows"]}["B3"]


def test_findings_text_and_counts(setup):
    _, app_dir = _export(setup)
    f = _load(app_dir / "data" / "findings.json")
    h = f["headline"]
    # latest AnomalyRidge:rain row (CI includes 0) replaces the older "better" one
    assert h["n_satellite_runs"] == 2 and h["n_better_than_b2"] == 0
    assert h["n_worse_than_b2"] == 1 and h["satellite_improved"] is False
    assert h["statement"].startswith("Satellite and rainfall features did not improve")
    assert any("corn: B3 was added as a candidate after seeing" in c for c in f["caveats"])
    assert not any(c.startswith("palay: B2") for c in f["caveats"])
    assert any(c.startswith("Deviation from Requirement 6.1.") for c in f["caveats"])
    assert f["likely_reasons"] == ["Cropland mask mixes plantations.", "PSA estimates are noisy."]
    assert f["next_steps"] == ["A rice-paddy mask."]
    pred = pd.read_parquet(setup[1] / "processed" / "predictions.parquet")
    assert f["counts"]["prediction_rows"] == len(pred)
    assert f["counts"]["no_area_forecast_rows"] == int(pred["predicted_area_ha"].isna().sum())
    cov = {r["crop"]: r for r in f["interval_coverage"]}
    assert cov["corn"]["coverage"] == pytest.approx(1.0)
    feat = _features()
    ok = feat["yield_t_ha"].notna()
    assert f["rain_shift"]["mean_all_mm"] == pytest.approx(feat.loc[ok, "rain_anom"].mean(),
                                                           abs=0.05)
    assert f["figures"] == [{"file": "assets/figures/mae_by_model.png",
                             "title": "mae by model"}]


def test_live_accuracy_awaiting_without_score(setup):
    cfg, tmp = setup
    _, app_dir = _export(setup)
    la = _load(app_dir / "data" / "live_accuracy.json")
    assert la["status"] == "awaiting PSA release" and la["n_scored_quarters"] == 0
    assert "cropyield score has not run" in la["source"]
    live = [r for r in la["rows"] if r["kind"] == "live"]
    assert live and all(r["status"] == "awaiting PSA release" for r in live)
    assert {r["forecast_id"] for r in live} == {"2024Q1-b"}
    assert {s["crop"] for s in la["summary"]} == {"palay", "corn"}
    # export must not stamp first-seen values or write scoring files
    assert not (tmp / "reports" / "psa_first_seen.csv").exists()
    assert not (tmp / "reports" / "live_accuracy.csv").exists()


def test_live_accuracy_reads_score_output(setup):
    cfg, tmp = setup
    from cropyield.models.forecast import score_forecasts
    score_forecasts(cfg, scored_at=NOW)  # tmp paths only
    _, app_dir = _export(setup)
    la = _load(app_dir / "data" / "live_accuracy.json")
    assert la["source"] == "reports/live_accuracy.csv"
    assert la["scored_at_utc"] == "2026-11-26T12:00:00Z"
    backfill = [r for r in la["rows"] if r["kind"] == "backfill"]
    assert backfill and backfill[0]["b3_err_latest"] is not None


def test_meta(setup):
    _, app_dir = _export(setup)
    m = _load(app_dir / "data" / "meta.json")
    assert m["generated_at"] == "2026-11-26T12:00:00Z" and m["git_commit"] == "abc123"
    assert m["data_vintage"] == "2023Q4" and m["latest_forecast_quarter"] == "2024Q1"
    assert m["model"] == {"corn": "B3", "palay": "B2"}
    names = " ".join(s["name"] for s in m["sources"])
    for src in ("PSA", "MODIS", "Sentinel-2", "WorldCover", "geoBoundaries", "Open-Meteo",
                "OpenStreetMap"):
        assert src in names


def test_cli_lists_export():
    res = CliRunner().invoke(app, ["export", "--help"])
    assert res.exit_code == 0 and "app/data" in res.output
