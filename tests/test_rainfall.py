"""Tests for cropyield.rainfall (Open-Meteo ERA5) with synthetic masks and a fake HTTP session."""
from __future__ import annotations

import dataclasses
import json
from datetime import date, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import requests
from rasterio.transform import from_origin
from shapely.geometry import box

from cropyield import rainfall as r
from cropyield.boundaries import _slug
from cropyield.config import load_config, load_pipeline

RES = 0.01  # mask pixel size in degrees (real masks are ~10 m; small here for speed)
NO_SLEEP = lambda s: None  # noqa: E731


@pytest.fixture
def cfg(tmp_path):
    """Repo config with tmp paths, no retry delays and no reporting windows."""
    base = load_config()
    p = base.pipeline
    paths = dataclasses.replace(p.paths, raw=tmp_path / "raw", interim=tmp_path / "interim")
    rain = dataclasses.replace(p.rainfall, backoff_s=1.0, max_attempts=3)
    provinces = dataclasses.replace(base.provinces, reporting_units=())
    return dataclasses.replace(
        base, pipeline=dataclasses.replace(p, paths=paths, rainfall=rain), provinces=provinces)


def _free_budget() -> r.CallBudget:
    return r.CallBudget(1e9, 1e9, 1e9, clock=lambda: 0.0, sleep=NO_SLEEP)


def _write_mask(path, mask: np.ndarray, x0: float = 125.0, y1: float = 8.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", dtype="uint8", count=1,
                       height=mask.shape[0], width=mask.shape[1], crs="EPSG:4326",
                       transform=from_origin(x0, y1, RES, RES)) as dst:
        dst.write(mask.astype("uint8"), 1)
    return path


def _payload(start: str, end: str, value=lambda d: 1.0) -> dict:
    """Archive API JSON for daily precipitation_sum; value(day) may return None."""
    days = pd.date_range(start, end, freq="D")
    return {"latitude": 7.0, "longitude": 125.5, "timezone": "Asia/Manila",
            "daily_units": {"time": "iso8601", "precipitation_sum": "mm"},
            "daily": {"time": [d.strftime("%Y-%m-%d") for d in days],
                      "precipitation_sum": [value(d) for d in days]}}


def _response(status: int, body: dict | None = None, headers: dict | None = None):
    resp = requests.Response()
    resp.status_code = status
    resp._content = json.dumps(body or {"error": True, "reason": "boom"}).encode()
    resp.headers.update(headers or {})
    return resp


class FakeSession:
    """Answers each GET with `handler(params, n)`; records every call."""

    def __init__(self, handler=None):
        self.calls: list[dict] = []
        self.handler = handler or (lambda p, n: _response(
            200, _payload(p["start_date"], p["end_date"])))

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "timeout": timeout, **params})
        return self.handler(params, len(self.calls))


def _points(provinces=("Davao Oriental",), ha=(100.0,)) -> pd.DataFrame:
    return pd.DataFrame({"province": list(provinces), "lat": 7.0, "lon": 125.5,
                         "fallback": False, "centroid_lat": 7.0, "centroid_lon": 125.5,
                         "cropland_ha": list(ha)})


# ------------------------------------------------------------------------------ config
def test_repo_config_has_open_meteo_settings():
    p = load_config().pipeline
    assert p.sources.open_meteo_url == "https://archive-api.open-meteo.com/v1/archive"
    assert p.dates.rainfall_start == date(1981, 1, 1)
    rc = p.rainfall
    assert (rc.model, rc.timezone, rc.variable) == ("era5", "Asia/Manila", "precipitation_sum")
    assert (rc.per_minute, rc.per_hour, rc.per_day, rc.budget_fraction) == (600, 5000, 10000, 0.8)
    assert "cropyield" in rc.user_agent


def test_legacy_chirps_keys_still_load(tmp_path):
    from cropyield.config import DEFAULT_CONFIG_DIR
    text = (DEFAULT_CONFIG_DIR / "pipeline.yaml").read_text()
    text = text.replace('rainfall_start: "1981-01-01"', 'chirps_start: "1982-01-01"')
    text = text.replace('  open_meteo_url:', '  chirps_url: "https://x/{year}.{month:02d}.cog"\n'
                                             '  open_meteo_url:')
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "pipeline.yaml"
    path.write_text(text)
    with pytest.warns(DeprecationWarning):
        p = load_pipeline(path)
    assert p.dates.rainfall_start == date(1982, 1, 1)
    assert not hasattr(p.sources, "chirps_url")


# ------------------------------------------------------------------------------ points
def test_cropland_point_is_area_weighted_centroid(tmp_path):
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:20, 30:50] = True  # rows 10-19, cols 30-49
    path = _write_mask(tmp_path / "m.tif", mask)
    p = r.cropland_point(path, box(125.0, 7.6, 125.6, 8.0), block_rows=7)
    assert not p["fallback"]
    assert p["lon"] == pytest.approx(125.0 + 40 * RES)  # centre of cols 30-49
    # area weighting pulls the centroid a hair south (lower rows have bigger pixels)
    assert p["lat"] == pytest.approx(8.0 - 15 * RES, abs=1e-5) and p["lat"] < 8.0 - 15 * RES
    cell_ha = 111.32 ** 2 * RES ** 2 * np.cos(np.radians(7.85)) * 100
    assert p["cropland_ha"] == pytest.approx(200 * cell_ha, rel=0.01)


def test_cropland_point_falls_back_to_nearest_pixel_in_c_shape(tmp_path):
    # C-shaped province: the cropland centroid lies in the open middle, outside it
    mask = np.zeros((30, 30), dtype=bool)
    mask[0:5, 0:30] = mask[25:30, 0:30] = mask[0:30, 0:5] = True
    path = _write_mask(tmp_path / "c.tif", mask)
    c_shape = box(125.0, 7.95, 125.3, 8.0).union(box(125.0, 7.7, 125.3, 7.75)).union(
        box(125.0, 7.7, 125.05, 8.0))
    p = r.cropland_point(path, c_shape, block_rows=4)
    assert p["fallback"]
    assert not c_shape.covers(gpd.points_from_xy([p["centroid_lon"]], [p["centroid_lat"]])[0])
    # nearest cropland pixel centre to the centroid: on the C's spine, same latitude
    assert p["lat"] == pytest.approx(p["centroid_lat"], abs=RES / 2 + 1e-9)
    assert p["lon"] == pytest.approx(125.0 + 4.5 * RES)
    row, col = round((8.0 - p["lat"]) / RES - 0.5), round((p["lon"] - 125.0) / RES - 0.5)
    assert mask[row, col] and c_shape.covers(gpd.points_from_xy([p["lon"]], [p["lat"]])[0])


def test_compute_points_one_per_province(cfg):
    names = ["Davao Oriental", "Davao del Norte"]
    cfg = dataclasses.replace(cfg, provinces=dataclasses.replace(
        cfg.provinces, provinces=tuple(p for p in cfg.provinces.provinces if p.name in names)))
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    for i, name in enumerate(names):
        _write_mask(cfg.pipeline.paths.interim / "cropland" / f"{_slug(name)}.tif", mask,
                    x0=125.0 + i)
    gdf = gpd.GeoDataFrame({"province": names},
                           geometry=[box(125.0 + i, 7.8, 125.2 + i, 8.0) for i in range(2)],
                           crs="EPSG:4326")
    pts = r.compute_points(gdf, cfg)
    assert list(pts.columns) == r.POINT_COLUMNS
    assert set(pts["province"]) == set(names) and not pts["fallback"].any()
    lon = pts.set_index("province")["lon"]
    assert lon["Davao Oriental"] == pytest.approx(125.075)  # centre of cols 5-9
    assert lon["Davao del Norte"] == pytest.approx(126.075)
    saved = r.save_points(pts, cfg)
    assert saved.name == "rainfall_points.csv"
    assert pd.read_csv(saved).columns[:5].tolist() == ["unit", "lat", "lon", "fallback_used",
                                                       "cropland_area"]
    pd.testing.assert_frame_equal(r.load_points(cfg), pts)
    (cfg.pipeline.paths.interim / "cropland" / "davao_oriental.tif").unlink()
    with pytest.raises(FileNotFoundError, match="cropland"):
        r.compute_points(gdf, cfg)


# ------------------------------------------------------------------------------ chunks
def test_request_cost_follows_open_meteo_weighting():
    assert r.request_cost(14) == 1.0 and r.request_cost(3) == 1.0
    assert r.request_cost(28) == 2.0
    assert r.request_cost(365) == pytest.approx(26.07, abs=0.01)
    assert r.request_cost(14, n_variables=15) == 1.5  # pricing FAQ example


def test_resolve_range_caps_at_data_lag(cfg):
    today = date(2026, 11, 25)
    # dates.end in pipeline.yaml is 2026-09-30
    assert r.resolve_range(cfg, today=today) == (date(1981, 1, 1), date(2026, 9, 30))
    assert r.resolve_range(cfg, today=date(2026, 9, 10))[1] == date(2026, 9, 5)
    assert r.fetch_end(cfg.pipeline.rainfall, date(2026, 9, 30), today) == date(2026, 11, 20)
    assert r.fetch_end(cfg.pipeline.rainfall, date(2020, 3, 31), today) == date(2020, 12, 31)
    assert r.resolve_range(cfg, "2024-02", "2024-02", today) == (date(2024, 2, 1),
                                                                 date(2024, 2, 29))
    with pytest.raises(ValueError, match="empty"):
        r.resolve_range(cfg, "2026-12", "2026-12", today)


def test_year_chunks_params_and_cache_path(cfg):
    chunks = r.year_chunks(_points(), date(2024, 6, 1), date(2025, 3, 10))
    assert [(c.start, c.end) for c in chunks] == [(date(2024, 1, 1), date(2024, 12, 31)),
                                                  (date(2025, 1, 1), date(2025, 3, 10))]
    assert chunks[0].n_days == 366
    params = r.chunk_params(chunks[1], cfg.pipeline.rainfall)
    assert params == {"latitude": "7.0000", "longitude": "125.5000",
                      "start_date": "2025-01-01", "end_date": "2025-03-10",
                      "daily": "precipitation_sum", "timezone": "Asia/Manila",
                      "models": "era5"}
    assert r.chunk_path(cfg, chunks[1]) == (cfg.pipeline.paths.raw / "rainfall" / "era5"
                                            / "davao_oriental_7.0000_125.5000" / "2025.json")


def test_fetch_sends_user_agent_and_writes_raw_json(cfg):
    sess = FakeSession()
    chunk = r.year_chunks(_points(), date(2020, 1, 1), date(2020, 12, 31))[0]
    path = r.fetch_chunk(cfg, chunk, sess, _free_budget(), NO_SLEEP)
    call = sess.calls[0]
    assert call["url"] == "https://archive-api.open-meteo.com/v1/archive"
    assert call["headers"]["User-Agent"] == cfg.pipeline.rainfall.user_agent
    assert json.loads(path.read_text())["daily"]["time"][-1] == "2020-12-31"
    assert not list(path.parent.glob("*.part"))
    assert r.is_cached(path, chunk, "precipitation_sum")


def test_download_skips_cache_and_resumes_after_failure(cfg):
    pts = _points()
    first, last = date(2018, 1, 1), date(2020, 12, 31)

    def flaky(p, n):  # 2019 always fails with a server error in the first run
        if p["start_date"].startswith("2019"):
            return _response(503)
        return _response(200, _payload(p["start_date"], p["end_date"]))

    sess = FakeSession(flaky)
    s1 = r.download_rainfall(cfg, pts, first, last, session=sess, budget=_free_budget(),
                             sleep=NO_SLEEP)
    assert (s1.fetched, s1.cached, s1.pending) == (2, 0, 0)
    assert [(p, y) for p, y, _ in s1.failed] == [("Davao Oriental", 2019)]
    assert "after 3 attempts" in s1.failed[0][2] and "FAILED" in "\n".join(s1.lines())
    assert len(sess.calls) == 1 + 3 + 1

    sess2 = FakeSession()
    s2 = r.download_rainfall(cfg, pts, first, last, session=sess2, budget=_free_budget(),
                             sleep=NO_SLEEP)
    assert [c["start_date"] for c in sess2.calls] == ["2019-01-01"]  # only the missing year
    assert (s2.fetched, s2.cached, s2.failed) == (1, 2, [])


def test_cached_year_with_nulls_or_short_range_is_fetched_again(cfg):
    chunk = r.year_chunks(_points(), date(2021, 1, 1), date(2021, 12, 31))[0]
    path = r.chunk_path(cfg, chunk)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_payload("2021-01-01", "2021-12-31",
                                        lambda d: None if d.day == 31 else 1.0)))
    assert not r.is_cached(path, chunk, "precipitation_sum")
    path.write_text(json.dumps(_payload("2021-01-01", "2021-11-30")))  # older partial year
    assert not r.is_cached(path, chunk, "precipitation_sum")
    path.write_text("{not json")
    assert not r.is_cached(path, chunk, "precipitation_sum")


def test_429_respects_retry_after_then_succeeds(cfg):
    sleeps: list[float] = []

    def limited(p, n):
        if n == 1:
            return _response(429, headers={"Retry-After": "7"})
        if n == 2:
            return _response(429)  # no header: exponential backoff (backoff_s * 2**1)
        return _response(200, _payload(p["start_date"], p["end_date"]))

    chunk = r.year_chunks(_points(), date(2022, 1, 1), date(2022, 12, 31))[0]
    sess = FakeSession(limited)
    budget = _free_budget()
    path = r.fetch_chunk(cfg, chunk, sess, budget, sleeps.append)
    assert sleeps == [7.0, 2.0] and len(sess.calls) == 3
    assert len(budget.events) == 3  # every attempt, rate-limited ones too, is charged
    assert r.is_cached(path, chunk, "precipitation_sum")


def test_long_retry_after_stops_run_cleanly(cfg):
    sess = FakeSession(lambda p, n: _response(429, headers={"Retry-After": "86400"}))
    s = r.download_rainfall(cfg, _points(), date(2020, 1, 1), date(2021, 12, 31),
                            session=sess, budget=_free_budget(), sleep=NO_SLEEP)
    assert s.stopped and "Retry-After" in s.stopped and s.pending == 2 and s.fetched == 0
    assert len(sess.calls) == 1


def test_retry_after_http_date():
    from datetime import datetime, timezone
    resp = _response(429, headers={"Retry-After": "Wed, 25 Nov 2026 12:00:30 GMT"})
    now = datetime(2026, 11, 25, 12, 0, 0, tzinfo=timezone.utc)
    assert r.retry_after_s(resp, now) == 30.0
    assert r.retry_after_s(_response(429)) is None


def test_client_error_fails_without_retry(cfg):
    sess = FakeSession(lambda p, n: _response(400, {"error": True, "reason": "bad model"}))
    chunk = r.year_chunks(_points(), date(2022, 1, 1), date(2022, 12, 31))[0]
    with pytest.raises(r.ChunkFailed, match="HTTP 400: bad model"):
        r.fetch_chunk(cfg, chunk, sess, _free_budget(), NO_SLEEP)
    assert len(sess.calls) == 1 and not r.chunk_path(cfg, chunk).exists()


def test_mismatched_dates_are_retried_not_cached(cfg):
    sess = FakeSession(lambda p, n: _response(200, _payload(p["start_date"], "2022-06-30")))
    chunk = r.year_chunks(_points(), date(2022, 1, 1), date(2022, 12, 31))[0]
    with pytest.raises(r.ChunkFailed, match="bad response"):
        r.fetch_chunk(cfg, chunk, sess, _free_budget(), NO_SLEEP)
    assert len(sess.calls) == 3 and not r.chunk_path(cfg, chunk).exists()


class FakeClock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        assert s >= 0
        self.t += s


def _max_in_window(events, window):
    return max(sum(c for u, c in events if t - window < u <= t) for t, _ in events)


def test_throttle_never_exceeds_minute_and_hour_budgets(cfg):
    # scaled-down configured limits; the budget is 80% of each (60/min, 300/h)
    rc = dataclasses.replace(cfg.pipeline.rainfall, per_minute=75, per_hour=375, per_day=1e9)
    clock = FakeClock()
    b = r.CallBudget.from_config(rc, clock=clock, sleep=clock.sleep)
    assert (b.limits[60.0], b.limits[3600.0]) == (60.0, 300.0)
    for cost in [26.07, 26.14, 5.0, 26.07] * 10:
        b.acquire(cost)
        clock.t += 0.5  # request time
    assert _max_in_window(b.events, 60) <= 60 + 1e-9
    assert _max_in_window(b.events, 3600) <= 300 + 1e-9
    assert clock.t - 1_000_000 > 3600  # 834 calls at 300/h needs more than two hours


def test_budget_from_config_uses_80_percent_and_stops_before_daily_limit(cfg):
    rc = dataclasses.replace(cfg.pipeline.rainfall, per_day=100.0)
    clock = FakeClock()
    b = r.CallBudget.from_config(rc, clock=clock, sleep=clock.sleep)
    assert b.limits == {60.0: 480.0, 3600.0: 4000.0, 86400.0: 80.0}
    b.acquire(26.0), b.acquire(26.0), b.acquire(26.0)
    with pytest.raises(r.BudgetExhausted, match="daily budget"):
        b.acquire(26.0)
    with pytest.raises(ValueError, match="exceeds"):
        b.acquire(500.0)


def test_daily_stop_in_download_leaves_pending_chunks(cfg):
    clock = FakeClock()
    budget = r.CallBudget(1e9, 1e9, 60.0, clock=clock, sleep=clock.sleep)
    s = r.download_rainfall(cfg, _points(), date(2015, 1, 1), date(2019, 12, 31),
                            session=FakeSession(), budget=budget, sleep=NO_SLEEP)
    assert s.fetched == 2 and s.pending == 3 and "daily budget" in s.stopped


def test_daily_stop_persists_usage_and_next_day_run_resumes(cfg):
    rc = dataclasses.replace(cfg.pipeline.rainfall, per_day=100.0)  # 80 weighted calls/day
    ledger = cfg.pipeline.paths.raw / "rainfall" / r.LEDGER_FILE
    clock, pts = FakeClock(), _points()
    first, last = date(2015, 1, 1), date(2019, 12, 31)

    def run():
        sess = FakeSession()
        budget = r.CallBudget.from_config(rc, clock=clock, sleep=clock.sleep, ledger=ledger)
        return sess, r.download_rainfall(cfg, pts, first, last, session=sess, budget=budget,
                                         sleep=clock.sleep)

    sess, s1 = run()  # 3 years x ~26 calls fit in 80; the 4th would not
    assert (s1.fetched, s1.pending, s1.failed) == (3, 2, [])
    assert "daily budget" in s1.stopped and len(sess.calls) == 3
    text = "\n".join(s1.lines())
    assert "2 pending" in text and "resume" in text
    used = json.loads(ledger.read_text())
    assert sum(c for _, c in used) == pytest.approx(s1.cost) and s1.cost <= 80

    sess, s2 = run()  # same day: the ledger blocks any request
    assert (s2.fetched, s2.cached, s2.pending, len(sess.calls)) == (0, 3, 2, 0)

    clock.t += 86_401  # next day
    sess, s3 = run()
    assert [c["start_date"] for c in sess.calls] == ["2018-01-01", "2019-01-01"]
    assert (s3.fetched, s3.cached, s3.pending, s3.stopped) == (2, 3, 0, None)


def test_ledger_carries_usage_across_runs(tmp_path):
    clock, ledger = FakeClock(), tmp_path / "raw" / "rainfall" / "_usage.json"
    r.CallBudget(1e9, 1e9, 50.0, clock=clock, sleep=clock.sleep, ledger=ledger).acquire(30.0)
    again = r.CallBudget(1e9, 1e9, 50.0, clock=clock, sleep=clock.sleep, ledger=ledger)
    with pytest.raises(r.BudgetExhausted):
        again.acquire(30.0)
    clock.t += 86_401  # a day later the old calls have left the window
    again.acquire(30.0)


def test_estimate_download_counts_uncached_chunks(cfg):
    pts = _points(["Davao Oriental", "Davao City"], [1.0, 1.0])
    est = r.estimate_download(cfg, pts, date(2000, 1, 1), date(2009, 12, 31))
    assert est["requests"] == 20 and est["days"] == 1
    assert est["weighted_calls"] == pytest.approx(2 * 3653 / 14)
    # 522 weighted calls exceed one 480/minute window: one wait of about a minute
    assert 0.9 < est["throttle_minutes"] < 1.1


# --------------------------------------------------------------------------- aggregate
def test_monthly_sums_missing_day_and_partial_month_are_nan():
    days = pd.date_range("2024-01-01", "2024-03-20", freq="D")
    vals = pd.Series(2.0, index=days)
    vals[pd.Timestamp("2024-02-10")] = np.nan  # null from the API
    months = r.month_starts(date(2024, 1, 1), date(2024, 4, 30))
    out = r.monthly_from_daily(vals, months).set_index("date")
    assert out.loc["2024-01-01", "rain_mm"] == 62.0 and out.loc["2024-01-01", "valid_frac"] == 1
    assert np.isnan(out.loc["2024-02-01", "rain_mm"])
    assert out.loc["2024-02-01", "valid_frac"] == pytest.approx(28 / 29)
    assert np.isnan(out.loc["2024-03-01", "rain_mm"])  # latest month, only 20 days so far
    assert out.loc["2024-03-01", "valid_frac"] == pytest.approx(20 / 31)
    assert np.isnan(out.loc["2024-04-01", "rain_mm"]) and out.loc["2024-04-01", "valid_frac"] == 0
    zero = r.monthly_from_daily(pd.Series(0.0, index=pd.date_range("2024-05-01", "2024-05-31")),
                                [pd.Timestamp("2024-05-01")])
    assert zero["rain_mm"].tolist() == [0.0]  # a dry month is 0, not NaN


def test_combined_unit_is_cropland_weighted_and_nan_if_any_member_nan():
    months = pd.to_datetime(["2020-01-01", "2020-02-01"])
    base = pd.DataFrame({"province": ["A", "A", "B", "B"], "date": list(months) * 2,
                         "rain_mm": [100.0, 50.0, 200.0, np.nan],
                         "valid_frac": [1.0, 1.0, 1.0, 0.5]})
    out = r.combine_units(base, {"A+B": ("A", "B")}, {"A": 300.0, "B": 100.0})
    assert list(out.columns) == r.COLUMNS and out["province"].unique().tolist() == ["A+B"]
    assert out["rain_mm"].iloc[0] == pytest.approx(0.75 * 100 + 0.25 * 200)
    assert np.isnan(out["rain_mm"].iloc[1])
    assert out["valid_frac"].iloc[1] == pytest.approx(0.75 + 0.25 * 0.5)
    with pytest.raises(ValueError, match="C"):
        r.combine_units(base, {"A+C": ("A", "C")}, {"A": 1.0})


def _full_cfg(cfg):
    """`cfg` with the repo's reporting windows (combined units) restored."""
    return dataclasses.replace(cfg, provinces=load_config().provinces)


def test_aggregate_schema_covers_all_reporting_units(cfg):
    cfg = _full_cfg(cfg)
    names = cfg.provinces.names
    pts = _points(names, [float(i + 1) for i in range(len(names))])
    first, last = date(2015, 1, 1), date(2015, 12, 31)
    sess = FakeSession()
    r.download_rainfall(cfg, pts, first, last, session=sess, budget=_free_budget(),
                        sleep=NO_SLEEP)
    for i, name in enumerate(names):  # rewrite the cache with per-province values
        path = r.point_dir(cfg, name, 7.0, 125.5) / "2015.json"
        path.write_text(json.dumps(_payload("2015-01-01", "2015-12-31", lambda d, v=i + 1: v)))
    df = r.aggregate_rainfall(cfg, pts, first, last)
    assert list(df.columns) == r.COLUMNS
    assert pd.api.types.is_string_dtype(df["province"])
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["rain_mm"].dtype == df["valid_frac"].dtype == "float64"
    combined = cfg.provinces.combined_reporting_units()
    assert set(df["province"]) == set(names) | set(combined)
    assert len(df) == 12 * (len(names) + len(combined))
    jan = df[df["date"] == "2015-01-01"].set_index("province")["rain_mm"]
    w = dict(zip(names, range(1, len(names) + 1)))  # cropland_ha = rain per day here
    for unit, members in combined.items():
        expect = sum(w[m] * w[m] * 31 for m in members) / sum(w[m] for m in members)
        assert jan[unit] == pytest.approx(expect)
    assert jan["Davao Oriental"] == 31 * w["Davao Oriental"]
    # the output feeds features.rain_features directly, combined units included
    from cropyield import features
    units = ["Davao Oriental", *combined]
    rows = pd.DataFrame({"reporting_unit": units, "window_start": pd.Timestamp("2015-07-01"),
                         "window_end": pd.Timestamp("2015-12-31")})
    rf = features.rain_features(rows, df, cfg)
    totals = df[df["date"] >= "2015-07-01"].groupby("province")["rain_mm"].sum()
    assert rf["rain_total"].tolist() == pytest.approx(totals[units].tolist())
    assert rf["rain_anom"].isna().all()  # no 1981-2000 reference rows here


def test_main_download_then_aggregate_offline(cfg, monkeypatch, capsys):
    cfg = dataclasses.replace(cfg, provinces=dataclasses.replace(
        cfg.provinces, provinces=tuple(p for p in cfg.provinces.provinces
                                       if p.name == "Davao Oriental")))
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    _write_mask(cfg.pipeline.paths.interim / "cropland" / "davao_oriental.tif", mask)
    gpd.GeoDataFrame({"province": ["Davao Oriental"]}, geometry=[box(125.0, 7.8, 125.2, 8.0)],
                     crs="EPSG:4326").to_parquet(cfg.pipeline.paths.interim / "boundaries.parquet")
    monkeypatch.setattr(r, "load_config", lambda _: cfg)
    sess = FakeSession()
    monkeypatch.setattr(r.requests, "Session", lambda: sess)
    monkeypatch.setattr(r.time, "sleep", NO_SLEEP)

    r.main(["--start", "2024-01", "--end", "2024-02"])
    out = pd.read_parquet(cfg.pipeline.paths.interim / "rainfall.parquet")
    assert list(out.columns) == r.COLUMNS and out["province"].tolist() == ["Davao Oriental"] * 2
    assert out["rain_mm"].tolist() == [31.0, 29.0]
    assert len(sess.calls) == 1 and sess.calls[0]["end_date"] == "2024-12-31"
    pts = r.load_points(cfg)
    assert pts["lat"].iloc[0] == pytest.approx(7.925) and not pts["fallback"].iloc[0]
    assert "1 fetched" in capsys.readouterr().out

    r.main(["--step", "aggregate", "--start", "2024-01", "--end", "2024-02"])
    assert len(sess.calls) == 1  # aggregate makes no requests
    pd.testing.assert_frame_equal(
        pd.read_parquet(cfg.pipeline.paths.interim / "rainfall.parquet"), out)
