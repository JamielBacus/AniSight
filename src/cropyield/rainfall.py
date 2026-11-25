"""Monthly rainfall (mm/month) per province and PSA reporting unit from Open-Meteo ERA5.

Source: Open-Meteo Historical Weather API (`sources.open_meteo_url`, no API key), daily
`precipitation_sum` from the ERA5 reanalysis (`rainfall.model` in pipeline.yaml, default
`era5`, 0.25 degree; `era5_land`, 0.1 degree, is the configurable alternative). Days are
local calendar days (`rainfall.timezone`, Asia/Manila). ERA5 lags about 5 days
(`rainfall.data_lag_days`), so requests end at today minus the lag. Free tier:
non-commercial use, CC BY 4.0 attribution to Open-Meteo / ECMWF ERA5.

Steps:
1. `compute_points` puts ONE point per configured province at the area-weighted centroid
   of its cropland pixels (WorldCover masks `<paths.interim>/cropland/<slug>.tif` from
   `cropyield.boundaries`). If that centroid falls outside the province polygon (e.g. a
   C-shaped province), the cropland pixel nearest to the centroid is used instead
   (`fallback` True). Points and each province's cropland hectares are written to
   `<paths.interim>/rainfall_points.csv` (`unit, lat, lon, fallback_used, cropland_area`
   in ha, then the raw centroid `centroid_lat, centroid_lon`).
2. `download_rainfall` requests one calendar year per point and request (~26 weighted API
   calls each; the full 1981-present range at one point is ~1,200, which in one request
   would exceed the 600/minute limit). Raw JSON responses are cached under
   `<paths.raw>/rainfall/<model>/<slug>_<lat>_<lon>/<year>.json` (written as `.part`
   then renamed). A cached year is reused if it covers the year up to the data end and
   has no null values; otherwise it is fetched again, so re-runs resume. `CallBudget`
   keeps weighted calls below `budget_fraction` of the per-minute and per-hour limits
   (sliding windows, waiting when needed) and stops the run before the daily limit; its
   ledger (`_usage.json` in the cache folder) carries usage across runs. HTTP 429 waits
   for Retry-After (or backs off) and retries; 5xx and network errors retry with backoff;
   after `max_attempts` (or on another 4xx) the chunk is recorded as failed and the run
   continues. The run summary lists failed chunks.
3. `aggregate_rainfall` sums daily values to months. A month with any missing day, or
   not yet complete, gets `rain_mm` NaN, never a partial sum or 0; `valid_frac` is the
   share of the month's days with a value. Combined reporting units (e.g. "Davao del
   Sur+Davao Occidental") are the cropland-area-weighted mean of their members; any
   member NaN makes the combined month NaN.

Output `<paths.interim>/rainfall.parquet`: `province, date, rain_mm, valid_frac`, where
`province` is a configured province or a combined unit name and `date` is the month start.

Usage (download reads the network; needs data/interim/boundaries.parquet and cropland/):
    python -m cropyield.rainfall                      # download + aggregate, full range
    python -m cropyield.rainfall --step download --start 2024-01 --end 2024-12
    python -m cropyield.rainfall --step aggregate     # cached files only, no network
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Protocol

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.windows import Window
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from cropyield.boundaries import CRS, EARTH_RADIUS_M, _slug
from cropyield.config import DEFAULT_CONFIG_DIR, Config, RainfallConfig, load_config

log = logging.getLogger(__name__)

DEFAULT_START = date(1981, 1, 1)  # first month of the 1981-2000 climatology
COLUMNS = ["province", "date", "rain_mm", "valid_frac"]
POINT_COLUMNS = ["province", "lat", "lon", "fallback", "cropland_ha", "centroid_lat",
                 "centroid_lon"]
POINTS_FILE = "rainfall_points.csv"
# In-memory point column -> column name in POINTS_FILE.
POINTS_FILE_NAMES = {"province": "unit", "fallback": "fallback_used",
                     "cropland_ha": "cropland_area"}
LEDGER_FILE = "_usage.json"
BLOCK_ROWS = 512  # mask rows read per window (~4 MB of uint8 per block)
COORD_DECIMALS = 4  # ~11 m; keeps request coordinates and cache names stable


class BudgetExhausted(RuntimeError):
    """The next request would exceed the daily call budget; resume in a later run."""


class StopRun(RuntimeError):
    """The server asked to wait longer than `max_wait_s` (likely the daily limit)."""


class ChunkFailed(RuntimeError):
    """A chunk still failed after all attempts, or got a non-retryable HTTP error."""


# ------------------------------------------------------------------------------ points
def _row_lat(transform: rasterio.Affine, rows: np.ndarray) -> np.ndarray:
    return transform.f + (rows + 0.5) * transform.e


def _row_area_ha(transform: rasterio.Affine, rows: np.ndarray) -> np.ndarray:
    """Hectares of one pixel in each row of a geographic (degree) grid."""
    lat = np.radians(_row_lat(transform, rows))
    half = math.radians(abs(transform.e)) / 2
    return (EARTH_RADIUS_M ** 2 * math.radians(abs(transform.a))
            * np.abs(np.sin(lat + half) - np.sin(lat - half)) / 1e4)


def _blocks(src: rasterio.DatasetReader, block_rows: int):
    """Yield (first row, boolean cropland block) windows of band 1 (1 = cropland)."""
    for row0 in range(0, src.height, block_rows):
        h = min(block_rows, src.height - row0)
        yield row0, src.read(1, window=Window(0, row0, src.width, h)) == 1


def cropland_point(mask_path: Path, geom: BaseGeometry,
                   block_rows: int = BLOCK_ROWS) -> dict[str, Any]:
    """One rainfall point for a province from its cropland mask GeoTIFF.

    The point is the pixel-area-weighted centroid of cropland pixels. If it is not inside
    `geom`, the cropland pixel nearest to it (distance with longitude scaled by
    cos(latitude)) is used and `fallback` is True. The mask is read in row windows.

    Returns:
        dict with `lat, lon, fallback, centroid_lat, centroid_lon, cropland_ha`.

    Raises:
        ValueError: if the mask is not an unrotated EPSG:4326 grid or has no cropland.
    """
    with rasterio.open(mask_path) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError(f"{mask_path}: expected EPSG:4326, got {src.crs}")
        t = src.transform
        if t.b or t.d:
            raise ValueError(f"{mask_path}: rotated grids are not supported")
        x_centers = t.c + (np.arange(src.width) + 0.5) * t.a
        w_sum = x_sum = y_sum = 0.0
        for row0, block in _blocks(src, block_rows):
            counts = block.sum(axis=1)
            if not counts.any():
                continue
            rows = np.arange(row0, row0 + block.shape[0])
            area = _row_area_ha(t, rows)
            w_sum += float((counts * area).sum())
            y_sum += float((counts * area * _row_lat(t, rows)).sum())
            x_sum += float(area @ (block @ x_centers))
        if w_sum <= 0:
            raise ValueError(f"{mask_path}: no cropland pixels")
        clon, clat = x_sum / w_sum, y_sum / w_sum
        lon, lat, fallback = clon, clat, False
        if not geom.covers(Point(clon, clat)):
            coslat, best = math.cos(math.radians(clat)), math.inf
            for row0, block in _blocks(src, block_rows):
                r, c = np.nonzero(block)
                if r.size == 0:
                    continue
                ys, xs = _row_lat(t, r + row0), x_centers[c]
                d2 = ((xs - clon) * coslat) ** 2 + (ys - clat) ** 2
                i = int(np.argmin(d2))
                if d2[i] < best:
                    best, lon, lat = float(d2[i]), float(xs[i]), float(ys[i])
            fallback = True
    return {"lat": lat, "lon": lon, "fallback": fallback, "centroid_lat": clat,
            "centroid_lon": clon, "cropland_ha": w_sum}


def compute_points(gdf: gpd.GeoDataFrame, cfg: Config) -> pd.DataFrame:
    """One point per configured province (`POINT_COLUMNS`), in provinces.yaml order.

    Raises:
        FileNotFoundError: if a province's cropland mask is missing.
        ValueError: if a province has no polygon in `gdf`.
    """
    geoms = dict(zip(gdf["province"], gdf.to_crs(CRS).geometry))
    rows = []
    for province in cfg.provinces.names:
        if province not in geoms:
            raise ValueError(f"{province}: no polygon in boundaries")
        path = cfg.pipeline.paths.interim / "cropland" / f"{_slug(province)}.tif"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run "
                                    f"`cropyield boundaries --step cropland` first")
        p = cropland_point(path, geoms[province])
        p["lat"], p["lon"] = round(p["lat"], COORD_DECIMALS), round(p["lon"], COORD_DECIMALS)
        rows.append({"province": province, **p})
        log.info("point %s: %.4f, %.4f%s", province, p["lat"], p["lon"],
                 " (nearest cropland pixel; centroid outside polygon)" if p["fallback"] else "")
    return pd.DataFrame(rows, columns=POINT_COLUMNS)


def points_path(cfg: Config) -> Path:
    return cfg.pipeline.paths.interim / POINTS_FILE


def save_points(points: pd.DataFrame, cfg: Config) -> Path:
    """Write `points` (`POINT_COLUMNS`) to `rainfall_points.csv` with the file's column
    names (`POINTS_FILE_NAMES`)."""
    out = points_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    points[POINT_COLUMNS].rename(columns=POINTS_FILE_NAMES).to_csv(out, index=False)
    return out


def load_points(cfg: Config) -> pd.DataFrame:
    """Read `rainfall_points.csv` back into `POINT_COLUMNS` form."""
    df = pd.read_csv(points_path(cfg))
    df = df.rename(columns={v: k for k, v in POINTS_FILE_NAMES.items()})
    df["fallback"] = df["fallback"].astype(bool)
    return df[POINT_COLUMNS]


# ------------------------------------------------------------------------------ chunks
@dataclass(frozen=True)
class Chunk:
    """One request: a point and a date range inside one calendar year."""
    province: str
    lat: float
    lon: float
    start: date
    end: date

    @property
    def year(self) -> int:
        return self.start.year

    @property
    def n_days(self) -> int:
        return (self.end - self.start).days + 1


def request_cost(n_days: int, n_variables: int = 1) -> float:
    """Weighted API calls of one single-location request (Open-Meteo pricing FAQ):
    more than 14 days or more than 10 variables count fractionally as several calls."""
    return max(1.0, n_days / 14) * max(1.0, n_variables / 10)


def data_end(rc: RainfallConfig, today: date | None = None) -> date:
    """Latest day ERA5 is expected to have: today minus `data_lag_days`."""
    return (today or date.today()) - timedelta(days=rc.data_lag_days)


def fetch_end(rc: RainfallConfig, last: date, today: date | None = None) -> date:
    """Last day to request: the end of `last`'s year, capped at `data_end`. Requests
    always cover whole years so a narrow run never caches a shorter year."""
    return min(date(last.year, 12, 31), data_end(rc, today))


def resolve_range(cfg: Config, start: str | date | None = None, end: str | date | None = None,
                  today: date | None = None) -> tuple[date, date]:
    """(first day, last day) to cover: whole months, capped at `data_end`.

    Defaults: `dates.rainfall_start` (or 1981-01-01) to `dates.end`.
    """
    s = pd.Timestamp(start or cfg.pipeline.dates.rainfall_start or DEFAULT_START)
    e = pd.Timestamp(end or cfg.pipeline.dates.end)
    first = s.to_period("M").to_timestamp().date()
    last = min(e.to_period("M").to_timestamp(how="end").date(),
               data_end(cfg.pipeline.rainfall, today))
    if last < first:
        raise ValueError(f"empty rainfall range: {first} to {last} (data end "
                         f"{data_end(cfg.pipeline.rainfall, today)})")
    return first, last


def year_chunks(points: pd.DataFrame, first: date, last: date) -> list[Chunk]:
    """Whole calendar years (the last one ends at `last`) touching [first, last], per point.

    Chunks always start on 1 January, so a narrow run never overwrites a fuller cached year.
    """
    chunks = []
    for p in points.itertuples(index=False):
        for year in range(first.year, last.year + 1):
            chunks.append(Chunk(p.province, float(p.lat), float(p.lon), date(year, 1, 1),
                                min(date(year, 12, 31), last)))
    return chunks


def chunk_params(chunk: Chunk, rc: RainfallConfig) -> dict[str, str]:
    """Query parameters of the archive API request for `chunk`."""
    return {"latitude": f"{chunk.lat:.{COORD_DECIMALS}f}",
            "longitude": f"{chunk.lon:.{COORD_DECIMALS}f}",
            "start_date": chunk.start.isoformat(), "end_date": chunk.end.isoformat(),
            "daily": rc.variable, "timezone": rc.timezone, "models": rc.model}


def point_dir(cfg: Config, province: str, lat: float, lon: float) -> Path:
    name = f"{_slug(province)}_{lat:.{COORD_DECIMALS}f}_{lon:.{COORD_DECIMALS}f}"
    return cfg.pipeline.paths.raw / "rainfall" / cfg.pipeline.rainfall.model / name


def chunk_path(cfg: Config, chunk: Chunk) -> Path:
    return point_dir(cfg, chunk.province, chunk.lat, chunk.lon) / f"{chunk.year}.json"


def _daily(payload: Any, variable: str) -> tuple[list[str], list[Any]]:
    """(dates, values) from an archive API JSON response; ValueError if malformed."""
    try:
        times, values = payload["daily"]["time"], payload["daily"][variable]
    except (KeyError, TypeError) as e:
        raise ValueError(f"response has no daily.time / daily.{variable}") from e
    if len(times) != len(values):
        raise ValueError("daily.time and values differ in length")
    return times, values


def is_cached(path: Path, chunk: Chunk, variable: str) -> bool:
    """True if `path` covers `chunk` fully, with no null values in its range."""
    if not path.exists():
        return False
    try:
        times, values = _daily(json.loads(path.read_text(encoding="utf-8")), variable)
    except (ValueError, OSError):
        return False
    s, e = chunk.start.isoformat(), chunk.end.isoformat()
    if not times or times[0] > s or times[-1] < e:
        return False
    return all(v is not None for t, v in zip(times, values) if s <= t <= e)


# ---------------------------------------------------------------------------- throttle
class CallBudget:
    """Sliding-window budget of weighted API calls per minute, hour and day.

    `acquire(cost)` waits (via `sleep`) until the request fits in the minute and hour
    windows, and raises `BudgetExhausted` if it would exceed the day window. With a
    `ledger` path, recorded calls persist across runs (pruned to the last 24 hours).
    """
    MINUTE, HOUR, DAY = 60.0, 3600.0, 86400.0

    def __init__(self, per_minute: float, per_hour: float, per_day: float, *,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep,
                 ledger: Path | None = None) -> None:
        self.limits = {self.MINUTE: per_minute, self.HOUR: per_hour, self.DAY: per_day}
        self.clock, self.sleep, self.ledger = clock, sleep, ledger
        self.events: list[tuple[float, float]] = []
        if ledger is not None and ledger.exists():
            try:
                self.events = [(float(t), float(c))
                               for t, c in json.loads(ledger.read_text(encoding="utf-8"))]
            except (ValueError, TypeError):
                log.warning("%s unreadable; starting a fresh usage ledger", ledger)

    @classmethod
    def from_config(cls, rc: RainfallConfig, **kw: Any) -> "CallBudget":
        f = rc.budget_fraction
        return cls(rc.per_minute * f, rc.per_hour * f, rc.per_day * f, **kw)

    def used(self, window: float, now: float | None = None) -> float:
        now = self.clock() if now is None else now
        return sum(c for t, c in self.events if t > now - window)

    def _wait(self, window: float, cost: float, now: float) -> float:
        """Seconds until `cost` more fits in `window`."""
        limit = self.limits[window]
        inside = sorted((t, c) for t, c in self.events if t > now - window)
        used = sum(c for _, c in inside)
        if used + cost <= limit:
            return 0.0
        for t, c in inside:  # oldest first: wait until enough of them leave the window
            used -= c
            if used + cost <= limit:
                return t + window - now
        return 0.0  # not reached: acquire() checks cost <= limit

    def acquire(self, cost: float) -> None:
        if cost > min(self.limits.values()):
            raise ValueError(f"request cost {cost:.1f} exceeds a window budget "
                             f"{min(self.limits.values()):.1f}; use smaller chunks")
        while True:
            now = self.clock()
            self.events = [(t, c) for t, c in self.events if t > now - self.DAY]
            if self.used(self.DAY, now) + cost > self.limits[self.DAY]:
                raise BudgetExhausted(
                    f"daily budget of {self.limits[self.DAY]:.0f} weighted calls reached "
                    f"({self.used(self.DAY, now):.0f} used in the last 24 h)")
            wait = max(self._wait(self.MINUTE, cost, now), self._wait(self.HOUR, cost, now))
            if wait <= 0:
                break
            log.info("rate budget: waiting %.0f s", wait)
            self.sleep(wait + 0.01)
        self.events.append((now, cost))
        self._save()

    def _save(self) -> None:
        if self.ledger is None:
            return
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger.with_name(self.ledger.name + ".part")
        tmp.write_text(json.dumps(self.events), encoding="utf-8")
        os.replace(tmp, self.ledger)


# ---------------------------------------------------------------------------- download
class HttpSession(Protocol):
    def get(self, url: str, **kw: Any) -> requests.Response: ...


def retry_after_s(resp: requests.Response, now: datetime | None = None) -> float | None:
    """Seconds from a Retry-After header (delta seconds or HTTP date), else None."""
    value = resp.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    now = now or datetime.now(timezone.utc)
    return max(0.0, (when - now).total_seconds())


def _reason(resp: requests.Response) -> str:
    try:
        return str(resp.json().get("reason", resp.text[:200]))
    except (ValueError, AttributeError):
        return resp.text[:200]


def _write_atomic(dest: Path, text: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, dest)
    return dest


def _check_payload(payload: Any, params: dict[str, str], variable: str) -> None:
    """ValueError unless the response covers exactly the requested dates."""
    times, _ = _daily(payload, variable)
    if not times or times[0] != params["start_date"] or times[-1] != params["end_date"]:
        raise ValueError(f"dates {times[:1]}..{times[-1:]} do not match "
                         f"{params['start_date']}..{params['end_date']}")


def fetch_chunk(cfg: Config, chunk: Chunk, session: HttpSession, budget: CallBudget,
                sleep: Callable[[float], None] = time.sleep) -> Path:
    """Request `chunk` and cache the raw JSON at `chunk_path` (atomic `.part` rename).

    Each attempt is charged to `budget` first. HTTP 429 waits for Retry-After (else
    exponential backoff) and retries; 5xx, network errors and malformed responses retry
    with backoff; other 4xx fail at once.

    Raises:
        ChunkFailed: after `max_attempts` tries, or on a non-retryable HTTP error.
        StopRun: Retry-After longer than `max_wait_s`.
        BudgetExhausted: from `budget` (daily budget reached).
    """
    rc = cfg.pipeline.rainfall
    url, params = cfg.pipeline.sources.open_meteo_url, chunk_params(chunk, rc)
    cost, problem = request_cost(chunk.n_days), "no attempt"
    for attempt in range(rc.max_attempts):
        budget.acquire(cost)
        delay = rc.backoff_s * 2 ** attempt
        try:
            resp = session.get(url, params=params, headers={"User-Agent": rc.user_agent},
                               timeout=rc.timeout_s)
        except requests.RequestException as e:
            problem = f"network error: {e}"
        else:
            code = resp.status_code
            if code == 200:
                try:
                    _check_payload(resp.json(), params, rc.variable)
                except ValueError as e:
                    problem = f"bad response: {e}"
                else:
                    return _write_atomic(chunk_path(cfg, chunk), resp.text)
            elif code == 429:
                wait = retry_after_s(resp)
                if wait is not None:
                    if wait > rc.max_wait_s:
                        raise StopRun(f"HTTP 429 with Retry-After {wait:.0f} s "
                                      f"(> max_wait_s {rc.max_wait_s:.0f})")
                    delay = wait
                problem = "HTTP 429 (rate limited)"
            elif code >= 500:
                problem = f"HTTP {code}"
            else:
                raise ChunkFailed(f"HTTP {code}: {_reason(resp)}")
        if attempt < rc.max_attempts - 1:
            log.warning("%s %d: %s; retrying in %.0f s", chunk.province, chunk.year,
                        problem, delay)
            sleep(delay)
    raise ChunkFailed(f"{problem} after {rc.max_attempts} attempts")


@dataclass
class DownloadSummary:
    """Outcome of `download_rainfall`. `failed` holds (province, year, reason)."""
    cached: int = 0
    fetched: int = 0
    failed: list[tuple[str, int, str]] = field(default_factory=list)
    pending: int = 0  # chunks not attempted because the run stopped early
    stopped: str | None = None
    cost: float = 0.0  # weighted calls of successful fetches

    def lines(self) -> list[str]:
        out = [f"rainfall download: {self.fetched} fetched, {self.cached} already cached, "
               f"{len(self.failed)} failed, {self.pending} pending "
               f"({self.cost:.0f} weighted calls fetched)"]
        if self.stopped:
            out.append(f"stopped early: {self.stopped}. Re-run the same command later "
                       f"to resume.")
        out += [f"  FAILED {p} {y}: {why}" for p, y, why in self.failed]
        if self.failed:
            out.append("Re-run the same command to retry the failed chunks.")
        return out


def download_rainfall(cfg: Config, points: pd.DataFrame, first: date, last: date, *,
                      session: HttpSession | None = None, budget: CallBudget | None = None,
                      sleep: Callable[[float], None] = time.sleep) -> DownloadSummary:
    """Fetch every uncached yearly chunk for `points` over [first, last]; resumable.

    A failed chunk is logged and skipped; a daily-budget stop or an over-long
    Retry-After ends the run cleanly with the remaining chunks counted as pending.
    """
    rc = cfg.pipeline.rainfall
    session = session or requests.Session()
    ledger = cfg.pipeline.paths.raw / "rainfall" / LEDGER_FILE
    budget = budget or CallBudget.from_config(rc, ledger=ledger)
    summary = DownloadSummary()
    chunks = year_chunks(points, first, fetch_end(rc, last))
    for i, chunk in enumerate(chunks):
        if is_cached(chunk_path(cfg, chunk), chunk, rc.variable):
            summary.cached += 1
            continue
        try:
            fetch_chunk(cfg, chunk, session, budget, sleep)
        except (BudgetExhausted, StopRun) as e:
            summary.stopped = str(e)
            summary.pending = sum(not is_cached(chunk_path(cfg, c), c, rc.variable)
                                  for c in chunks[i:])
            break
        except ChunkFailed as e:
            log.error("%s %d failed: %s", chunk.province, chunk.year, e)
            summary.failed.append((chunk.province, chunk.year, str(e)))
            continue
        summary.fetched += 1
        summary.cost += request_cost(chunk.n_days)
        log.info("fetched %s %d", chunk.province, chunk.year)
    return summary


def estimate_download(cfg: Config, points: pd.DataFrame, first: date,
                      last: date) -> dict[str, float]:
    """Requests, weighted calls and throttle time for chunks not cached yet.

    Simulates `CallBudget` with a fake clock from an empty ledger; request latency is
    not included. `days` is the number of daily budgets needed.
    """
    rc = cfg.pipeline.rainfall
    todo = [c for c in year_chunks(points, first, fetch_end(rc, last))
            if not is_cached(chunk_path(cfg, c), c, rc.variable)]
    clock = [0.0]

    def advance(s: float) -> None:
        clock[0] += s

    sim = CallBudget.from_config(rc, clock=lambda: clock[0], sleep=advance)
    days = 1 if todo else 0
    for c in todo:
        try:
            sim.acquire(request_cost(c.n_days))
        except BudgetExhausted:
            days += 1
            sim.events.clear()
            sim.acquire(request_cost(c.n_days))
    return {"requests": len(todo),
            "weighted_calls": sum(request_cost(c.n_days) for c in todo),
            "throttle_minutes": clock[0] / 60, "days": days}


# --------------------------------------------------------------------------- aggregate
def read_daily(path: Path, variable: str) -> pd.Series:
    """Daily values (float, NaN for null) indexed by date from one cached response."""
    times, values = _daily(json.loads(path.read_text(encoding="utf-8")), variable)
    return pd.Series([np.nan if v is None else float(v) for v in values],
                     index=pd.DatetimeIndex(pd.to_datetime(times)), dtype=float)


def month_starts(first: date, last: date) -> list[pd.Timestamp]:
    """Month-start timestamps of every month touching [first, last]."""
    periods = pd.period_range(pd.Timestamp(first).to_period("M"),
                              pd.Timestamp(last).to_period("M"), freq="M")
    return [p.to_timestamp() for p in periods]


def monthly_from_daily(daily: pd.Series, months: list[pd.Timestamp]) -> pd.DataFrame:
    """Monthly totals for `months` (month starts): columns `date, rain_mm, valid_frac`.

    `rain_mm` is the sum of the daily values only if every day of the month has one; a
    month with a missing day (or not complete yet) is NaN, never a partial sum.
    `valid_frac` is the share of the month's days with a value.
    """
    daily = daily[~daily.index.duplicated(keep="last")].dropna()
    by_month = daily.groupby(daily.index.to_period("M"))
    idx = pd.PeriodIndex(pd.DatetimeIndex(months), freq="M")
    n = by_month.count().reindex(idx, fill_value=0).to_numpy()
    total = by_month.sum().reindex(idx).to_numpy(dtype=float)
    days = np.asarray(idx.days_in_month)
    return pd.DataFrame({"date": idx.to_timestamp(),
                         "rain_mm": np.where(n == days, total, np.nan),
                         "valid_frac": n / days})


def point_monthly(cfg: Config, points: pd.DataFrame, first: date, last: date) -> pd.DataFrame:
    """Monthly rows (`COLUMNS`) per point from cached responses only (no network)."""
    variable = cfg.pipeline.rainfall.variable
    months = month_starts(first, last)
    frames, n_missing = [], 0
    for p in points.itertuples(index=False):
        parts = []
        for year in range(first.year, last.year + 1):
            path = point_dir(cfg, p.province, float(p.lat), float(p.lon)) / f"{year}.json"
            if path.exists():
                parts.append(read_daily(path, variable))
            else:
                n_missing += 1
        daily = (pd.concat(parts) if parts
                 else pd.Series(dtype=float, index=pd.DatetimeIndex([])))
        frames.append(monthly_from_daily(daily, months).assign(province=p.province))
    if n_missing:
        log.warning("rainfall: %d point-year file(s) not cached; their months are NaN",
                    n_missing)
    return pd.concat(frames, ignore_index=True)[COLUMNS]


def combine_units(base: pd.DataFrame, members: dict[str, tuple[str, ...]],
                  weights: dict[str, float]) -> pd.DataFrame:
    """Rows for combined reporting units: cropland-area-weighted means of the members.

    `rain_mm` is NaN in any month where a member is NaN; `valid_frac` is the weighted
    mean of the members' `valid_frac`.

    Raises:
        ValueError: if a member has no rows or no positive weight.
    """
    rain = base.pivot(index="date", columns="province", values="rain_mm")
    vf = base.pivot(index="date", columns="province", values="valid_frac")
    frames = []
    for name, mem in members.items():
        bad = [m for m in mem if m not in rain.columns or not weights.get(m, 0) > 0]
        if bad:
            raise ValueError(f"{name}: no rainfall rows or cropland weight for "
                             f"{', '.join(bad)}")
        w = np.array([weights[m] for m in mem], dtype=float)
        w /= w.sum()
        frames.append(pd.DataFrame({
            "province": name, "date": rain.index,
            "rain_mm": rain[list(mem)].to_numpy() @ w,  # NaN in any member -> NaN
            "valid_frac": vf[list(mem)].to_numpy() @ w}))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)[COLUMNS]


def aggregate_rainfall(cfg: Config, points: pd.DataFrame, first: date,
                       last: date) -> pd.DataFrame:
    """Monthly rainfall per province and combined reporting unit, from the cache only.

    Returns a DataFrame with `COLUMNS`, sorted by province and date.
    """
    base = point_monthly(cfg, points, first, last)
    weights = dict(zip(points["province"], points["cropland_ha"].astype(float)))
    combined = combine_units(base, cfg.provinces.combined_reporting_units(), weights)
    out = pd.concat([base, combined], ignore_index=True) if len(combined) else base
    out["date"] = pd.to_datetime(out["date"])
    out["rain_mm"] = out["rain_mm"].astype(float)
    out["valid_frac"] = out["valid_frac"].astype(float)
    return out.sort_values(["province", "date"], ignore_index=True)


def save_rainfall(df: pd.DataFrame, cfg: Config) -> Path:
    """Write the rainfall table to `<paths.interim>/rainfall.parquet`."""
    out = cfg.pipeline.paths.interim / "rainfall.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


# ------------------------------------------------------------------------------ CLI
def load_or_compute_points(cfg: Config, recompute: bool = False) -> pd.DataFrame:
    """Points from `rainfall_points.csv`, or computed from the masks and saved."""
    if points_path(cfg).exists() and not recompute:
        return load_points(cfg)
    bpath = cfg.pipeline.paths.interim / "boundaries.parquet"
    if not bpath.exists():
        raise SystemExit(f"{bpath} not found; run `cropyield boundaries` first")
    points = compute_points(gpd.read_parquet(bpath), cfg)
    save_points(points, cfg)
    return points


def main(argv: list[str] | None = None) -> None:
    """Download Open-Meteo daily rainfall and/or write data/interim/rainfall.parquet."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR),
                    help="folder with pipeline.yaml and provinces.yaml (default: repo configs/)")
    ap.add_argument("--start", help="first month, YYYY-MM (default: dates.rainfall_start)")
    ap.add_argument("--end", help="last month, YYYY-MM (default: dates.end, capped at "
                                  "today minus rainfall.data_lag_days)")
    ap.add_argument("--step", choices=["download", "aggregate", "all"], default="all",
                    help="'aggregate' uses cached files only, no network (default: all)")
    ap.add_argument("--province", action="append",
                    help="download only this province (repeatable; for smoke tests)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config_dir)
    first, last = resolve_range(cfg, args.start, args.end)
    points = load_or_compute_points(cfg, recompute=args.step != "aggregate")
    print(points[["province", "lat", "lon", "fallback", "cropland_ha"]].to_string(index=False))

    if args.step in ("download", "all"):
        todo = points if not args.province else points[points["province"].isin(args.province)]
        if todo.empty:
            raise SystemExit(f"no configured province matches {args.province}")
        est = estimate_download(cfg, todo, first, last)
        print(f"to fetch: {est['requests']:.0f} requests, {est['weighted_calls']:.0f} "
              f"weighted calls, ~{est['throttle_minutes']:.0f} min of throttling "
              f"(+ request time), {est['days']:.0f} daily budget(s)")
        summary = download_rainfall(cfg, todo, first, last)
        print("\n".join(summary.lines()))
        if args.step == "download":
            return
    df = aggregate_rainfall(cfg, points, first, last)
    out = save_rainfall(df, cfg)
    n_missing = df.loc[df["rain_mm"].isna(), "date"].nunique()
    print(f"wrote {out} ({len(df)} rows, {df['date'].nunique()} months, "
          f"{df['province'].nunique()} units, {n_missing} month(s) with missing values)")
    print(f"points: {points_path(cfg)}")


if __name__ == "__main__":
    main()
