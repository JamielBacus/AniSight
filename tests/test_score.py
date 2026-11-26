"""Live scoring (phase 3, task 2.2): synthetic tables, tmp paths, no downloads."""
from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import cropyield.models.forecast as fc
from cropyield.cli import app
from cropyield.config import load_config
from cropyield.models.data import prepare_modeling_table
from cropyield.models.forecast import (FORECAST_COLUMNS, LIVE_COLUMNS, STATUS_AWAITING,
                                       STATUS_BACKFILL, STATUS_LATE, STATUS_NOT_REPORTED,
                                       STATUS_SCORED, log_forecast, new_first_seen,
                                       score_forecasts, score_table, summarize_live,
                                       update_first_seen)

T0 = datetime(2013, 1, 5, tzinfo=timezone.utc)   # forecasts of 2012Q4 are made here
T1 = datetime(2013, 2, 15, tzinfo=timezone.utc)  # PSA publishes 2012Q4
T2 = datetime(2013, 5, 15, tzinfo=timezone.utc)  # PSA revises 2012Q4


def _actuals(rows) -> pd.DataFrame:
    """rows: (unit, crop, year, quarter, yield, area)."""
    df = pd.DataFrame(rows, columns=["reporting_unit", "crop", "year", "quarter",
                                     "yield_t_ha", "area_ha"])
    df["production_mt"] = df["yield_t_ha"] * df["area_ha"]
    df["quality_flag"] = "ok"
    return prepare_modeling_table(df)


HIST = [("A", "corn", 2012, 3, 2.0, 100.0), ("B", "corn", 2012, 3, 3.0, 200.0)]


def _fc(unit="A", crop="corn", year=2012, quarter=4, made_at=T0, b3=2.0, b0=2.2, b2=1.9,
        area=100.0, backfill=False, fid=None) -> dict:
    tq = f"{year}Q{quarter}"
    return {"forecast_id": fid or f"{tq}-{made_at:%Y%m%dT%H%M%SZ}"
                                  + ("-backfill" if backfill else ""),
            "made_at_utc": made_at.isoformat().replace("+00:00", "Z"),
            "git_commit": "abc", "code_hash": "0" * 12, "recipe_version": "v",
            "data_vintage": "2012Q3", "target_quarter": tq, "year": year,
            "quarter": quarter, "reporting_unit": unit, "crop": crop, "model": "B3",
            "b3_yield": b3, "b3_lo80": b3 - 0.2, "b3_hi80": b3 + 0.2, "b0_yield": b0,
            "b2_yield": b2, "b0_fallback": False, "predicted_area_ha": area,
            "area_fallback_level": "last_year" if area == area else "crop_mean",
            "pred_production_mt": b3 * area, "pred_production_lo_mt": (b3 - 0.2) * area,
            "pred_production_hi_mt": (b3 + 0.2) * area, "history_last_year": year - 1,
            "backfill": backfill}


def _forecasts(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))[list(FORECAST_COLUMNS)]


def _score(forecasts, snapshots):
    """Run first-seen updates for each (time, actuals) snapshot, score at the last one."""
    store = new_first_seen(snapshots[0][1], fc._empty_first_seen(), snapshots[0][0])
    for t, a in snapshots[1:]:
        store = pd.concat([store, new_first_seen(a, store, t)], ignore_index=True)
    t, a = snapshots[-1]
    return score_table(forecasts, a, store, t), store


def test_first_seen_frozen_and_latest_uses_revision():
    first = _actuals([*HIST, ("A", "corn", 2012, 4, 2.1, 100.0)])
    revised = _actuals([*HIST, ("A", "corn", 2012, 4, 2.5, 110.0)])
    out, store = _score(_forecasts(_fc()), [(T0, _actuals(HIST)), (T1, first), (T2, revised)])
    s = store.set_index(["reporting_unit", "year", "quarter"]).loc[("A", 2012, 4)]
    assert s["yield_t_ha"] == pytest.approx(2.1)
    assert s["first_seen_utc"] == "2013-02-15T00:00:00Z"
    assert s["data_vintage"] == "2012Q4"
    r = out.iloc[0]
    assert r["status"] == STATUS_SCORED and r["counted"]
    assert r["actual_first_yield_t_ha"] == pytest.approx(2.1)
    assert r["actual_latest_yield_t_ha"] == pytest.approx(2.5)
    assert r["revised"]
    assert r["b3_err_first"] == pytest.approx(-0.1)
    assert r["b3_err_latest"] == pytest.approx(-0.5)
    assert r["b0_err_first"] == pytest.approx(0.1)
    assert r["b2_err_latest"] == pytest.approx(-0.6)
    assert r["b3_prod_err_first"] == pytest.approx(2.0 * 100 - 2.1 * 100)
    assert r["b3_prod_err_latest"] == pytest.approx(2.0 * 100 - 2.5 * 110)
    assert list(out.columns) == LIVE_COLUMNS


def test_update_first_seen_appends_only(tmp_path):
    path = tmp_path / "psa_first_seen.csv"
    update_first_seen(_actuals([*HIST, ("A", "corn", 2012, 4, 2.1, 100.0)]), path, T1)
    before = path.read_text()
    store = update_first_seen(_actuals([*HIST, ("A", "corn", 2012, 4, 2.5, 100.0)]), path, T2)
    assert path.read_text() == before  # the revision adds nothing
    assert len(store) == 3
    store = update_first_seen(_actuals([*HIST, ("A", "corn", 2012, 4, 2.5, 100.0),
                                        ("A", "corn", 2013, 1, 2.0, 90.0)]), path, T2)
    assert path.read_text().startswith(before) and len(store) == 4
    assert store.iloc[-1]["first_seen_utc"] == "2013-05-15T00:00:00Z"


def test_interval_hit():
    a = _actuals([*HIST, ("A", "corn", 2012, 4, 2.15, 100.0),
                  ("B", "corn", 2012, 4, 3.5, 200.0)])
    out, _ = _score(_forecasts(_fc("A"), _fc("B", b3=3.0)), [(T1, a)])
    hit = out.set_index("reporting_unit")["b3_in80_first"]
    assert bool(hit["A"]) and not bool(hit["B"])
    s = summarize_live(out, ["corn"], n_boot=50)
    crop = s.loc[(s["basis"] == "first") & (s["level"] == "crop")].iloc[0]
    assert crop["coverage80"] == pytest.approx(0.5)
    assert crop["n_quarters"] == 1 and crop["n"] == 2
    assert crop["b3_mae"] == pytest.approx((0.15 + 0.5) / 2)


def test_last_forecast_before_first_seen_counts_and_late_one_does_not():
    early = _fc(made_at=T0, b3=1.0)
    later = _fc(made_at=T0 + timedelta(days=10), b3=2.0)
    late = _fc(made_at=T1 + timedelta(days=1), b3=2.1)  # after PSA was first seen
    a = _actuals([*HIST, ("A", "corn", 2012, 4, 2.1, 100.0)])
    out, _ = _score(_forecasts(early, later, late), [(T0, _actuals(HIST)), (T1, a)])
    assert len(out) == 1
    assert out.iloc[0]["forecast_id"] == later["forecast_id"]
    # only a late forecast: listed, not counted
    out, _ = _score(_forecasts(late), [(T0, _actuals(HIST)), (T1, a)])
    assert out.iloc[0]["status"] == STATUS_LATE and not out.iloc[0]["counted"]
    s = summarize_live(out, ["corn"])
    assert (s.loc[s["level"] == "crop", "status"] == STATUS_AWAITING).all()


def test_backfill_listed_separately_not_counted():
    a = _actuals([*HIST, ("A", "corn", 2012, 4, 2.1, 100.0)])
    bf = _fc(made_at=T2, b3=2.1, backfill=True)
    out, _ = _score(_forecasts(bf, _fc(made_at=T0)), [(T0, _actuals(HIST)), (T1, a)])
    assert set(out["kind"]) == {"live", "backfill"}
    b = out.loc[out["kind"] == "backfill"].iloc[0]
    assert b["status"] == STATUS_BACKFILL and not b["counted"]
    assert b["b3_err_first"] == pytest.approx(0.0)  # scored for reference only
    s = summarize_live(out, ["corn"])
    crop = s.loc[(s["basis"] == "first") & (s["level"] == "crop")].iloc[0]
    assert crop["n"] == 1 and crop["b3_mae"] == pytest.approx(0.1)  # live row only


def test_production_skipped_when_area_nan():
    a = _actuals([*HIST, ("A", "corn", 2012, 4, 2.1, 100.0),
                  ("B", "corn", 2012, 4, 3.0, 200.0)])
    out, _ = _score(_forecasts(_fc("A"), _fc("B", b3=3.2, area=np.nan)), [(T1, a)])
    r = out.set_index("reporting_unit")
    assert np.isnan(r.loc["B", "b3_prod_err_first"])
    assert r.loc["B", "b3_err_first"] == pytest.approx(0.2)  # yield still scored
    s = summarize_live(out, ["corn"])
    crop = s.loc[(s["basis"] == "first") & (s["level"] == "crop")].iloc[0]
    assert crop["n"] == 2 and crop["n_production"] == 1
    assert crop["b3_prod_mae"] == pytest.approx(10.0)
    unit_b = s.loc[(s["level"] == "crop_unit") & (s["reporting_unit"] == "B")].iloc[0]
    assert unit_b["n_production"] == 0 and np.isnan(unit_b["b3_prod_mae"])


def test_awaiting_and_unit_not_reported():
    a = _actuals([*HIST, ("A", "corn", 2012, 4, 2.1, 100.0)])
    out, _ = _score(_forecasts(_fc("B"), _fc("A", year=2013, quarter=1)),
                    [(T0, _actuals(HIST)), (T1, a)])
    st = out.set_index("target_quarter")["status"]
    assert st["2012Q4"] == STATUS_NOT_REPORTED   # quarter published, unit B missing
    assert st["2013Q1"] == STATUS_AWAITING
    assert not out["counted"].any()


def test_bootstrap_only_with_enough_quarters():
    rng = np.random.default_rng(42)
    rows, fcs = list(HIST), []
    for i in range(9):
        y, q = 2012 + (i + 3) // 4, (i + 3) % 4 + 1
        act = 2.0 + 0.1 * rng.normal()
        rows.append(("A", "corn", y, q, act, 100.0))
        made = datetime(y, 3 * q - 2, 1, tzinfo=timezone.utc)  # inside the quarter
        fcs.append(_fc(year=y, quarter=q, made_at=made, b3=act + 0.05, b0=act + 0.3,
                       b2=act - 0.2))
    seen = datetime(2016, 1, 1, tzinfo=timezone.utc)
    out, _ = _score(_forecasts(*fcs), [(seen, _actuals(rows))])
    assert out["counted"].sum() == 9
    s = summarize_live(out, ["corn"], n_boot=200, min_quarters=8)
    crop = s.loc[(s["basis"] == "first") & (s["level"] == "crop")].iloc[0]
    assert crop["b3_vs_b2_bootstrap"] == "ok"
    assert crop["b3_vs_b2_mae_diff"] == pytest.approx(0.05 - 0.2)
    assert crop["b3_vs_b0_ci_high"] < 0
    s = summarize_live(out, ["corn"], n_boot=200, min_quarters=10)
    crop = s.loc[(s["basis"] == "first") & (s["level"] == "crop")].iloc[0]
    assert crop["b3_vs_b2_bootstrap"].startswith("insufficient")
    assert np.isnan(crop["b3_vs_b2_ci_low"])


def _cfg(tmp_path):
    cfg = load_config()
    paths = dataclasses.replace(cfg.pipeline.paths, reports=tmp_path / "reports",
                                processed=tmp_path / "processed",
                                interim=tmp_path / "interim", models=tmp_path / "models")
    return dataclasses.replace(cfg, pipeline=dataclasses.replace(cfg.pipeline, paths=paths))


def _write_features(tmp_path, rows):
    (tmp_path / "processed").mkdir(exist_ok=True)
    _actuals(rows).drop(columns=["log_yield", "is_outlier"]).to_parquet(
        tmp_path / "processed" / "features.parquet")


def test_score_forecasts_empty_is_awaiting(tmp_path):
    cfg = _cfg(tmp_path)
    _write_features(tmp_path, HIST)
    out = score_forecasts(cfg, scored_at=T0)  # no forecasts.csv at all
    assert out.empty and list(out.columns) == LIVE_COLUMNS
    live = pd.read_csv(tmp_path / "reports" / "live_accuracy.csv")
    assert list(live.columns) == LIVE_COLUMNS and live.empty
    s = pd.read_csv(tmp_path / "reports" / "live_accuracy_summary.csv")
    crop = s.loc[s["level"] == "crop"]
    assert set(crop["crop"]) == set(cfg.pipeline.crops)
    assert (crop["status"] == STATUS_AWAITING).all() and (crop["n"] == 0).all()
    assert (tmp_path / "reports" / "psa_first_seen.csv").exists()


def test_score_forecasts_roundtrip_through_csv(tmp_path):
    cfg = _cfg(tmp_path)
    _write_features(tmp_path, HIST)
    log_forecast(_forecasts(_fc("A"), _fc("B", b3=3.1, area=np.nan),
                            _fc("A", made_at=T0, backfill=True, fid="bf")),
                 fc.forecasts_path(cfg))
    out = score_forecasts(cfg, scored_at=T0)
    assert set(out.loc[out["kind"] == "live", "status"]) == {STATUS_AWAITING}
    _write_features(tmp_path, [*HIST, ("A", "corn", 2012, 4, 2.1, 100.0),
                               ("B", "corn", 2012, 4, 3.0, 200.0)])
    score_forecasts(cfg, scored_at=T1)
    _write_features(tmp_path, [*HIST, ("A", "corn", 2012, 4, 2.4, 100.0),
                               ("B", "corn", 2012, 4, 3.0, 200.0)])
    out = score_forecasts(cfg, scored_at=T2)
    live = out.loc[out["kind"] == "live"].set_index("reporting_unit")
    assert live["counted"].all()
    assert live.loc["A", "actual_first_yield_t_ha"] == pytest.approx(2.1)
    assert live.loc["A", "actual_latest_yield_t_ha"] == pytest.approx(2.4)
    assert out.loc[out["kind"] == "backfill", "status"].tolist() == [STATUS_BACKFILL]
    s = pd.read_csv(tmp_path / "reports" / "live_accuracy_summary.csv")
    corn = s.loc[(s["level"] == "crop") & (s["crop"] == "corn")].set_index("basis")
    assert corn.loc["first", "b3_mae"] == pytest.approx((0.1 + 0.1) / 2)
    assert corn.loc["latest", "b3_mae"] == pytest.approx((0.4 + 0.1) / 2)
    assert corn.loc["first", "n_production"] == 1
    assert corn.loc["first", "b3_vs_b2_bootstrap"].startswith("insufficient")


def test_cli_score_forwards_options(monkeypatch):
    seen = {}
    monkeypatch.setattr(fc, "score_main", lambda argv=None: seen.setdefault("argv", argv))
    res = CliRunner().invoke(app, ["score", "--config-dir", "x"])
    assert res.exit_code == 0, res.output
    assert seen["argv"] == ["--config-dir", "x"]
