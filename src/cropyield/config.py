"""Load `configs/*.yaml` into typed dataclasses."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass(frozen=True)
class Province:
    name: str
    aliases: tuple[str, ...] = ()
    part_of: str | None = None  # parent unit to subtract this one from (e.g. Davao City)


@dataclass(frozen=True)
class ReportingWindow:
    """PSA reports `unit` inside `host` from `first` to `last` (inclusive (year, quarter))."""
    unit: str
    host: str
    first: tuple[int, int]
    last: tuple[int, int]

    def contains(self, year: int, quarter: int) -> bool:
        return self.first <= (year, quarter) <= self.last


@dataclass(frozen=True)
class ProvincesConfig:
    region: str
    provinces: tuple[Province, ...]
    reporting_units: tuple[ReportingWindow, ...] = ()

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.provinces]

    def host_of(self, unit: str, year: int, quarter: int) -> str | None:
        """Province whose PSA figure includes `unit` in that quarter, or None."""
        return next((w.host for w in self.reporting_units
                     if w.unit == unit and w.contains(year, quarter)), None)

    def absorbed_by(self, host: str, year: int, quarter: int) -> tuple[str, ...]:
        """Units reported inside `host` in that quarter, in `provinces` order."""
        units = {w.unit for w in self.reporting_units
                 if w.host == host and w.contains(year, quarter)}
        return tuple(n for n in self.names if n in units)

    def reporting_unit(self, province: str, year: int, quarter: int) -> str:
        """Footprint of the PSA figure for `province` in that quarter.

        The province itself, or inside a reporting window the host followed by every
        unit it absorbs then (in `provinces` order), joined by "+".
        """
        host = self.host_of(province, year, quarter) or province
        return "+".join((host, *self.absorbed_by(host, year, quarter)))

    def combined_reporting_units(self) -> dict[str, tuple[str, ...]]:
        """Every combined reporting unit name -> its member provinces (host first)."""
        out: dict[str, tuple[str, ...]] = {}
        for w in self.reporting_units:
            (y, q) = w.first
            while (y, q) <= w.last:
                members = (w.host, *self.absorbed_by(w.host, y, q))
                out.setdefault("+".join(members), members)
                y, q = (y + 1, 1) if q == 4 else (y, q + 1)
        return out


@dataclass(frozen=True)
class Dates:
    modis_start: date
    sentinel2_start: date
    end: date
    rainfall_start: date | None = None  # first rainfall day; None -> 1981-01-01


@dataclass(frozen=True)
class Sources:
    stac_url: str
    modis_collection: str
    sentinel2_collection: str
    worldcover_collection: str
    boundaries_url: str
    boundaries_adm3_url: str
    # Open-Meteo Historical Weather API endpoint (cropyield.rainfall).
    open_meteo_url: str = "https://archive-api.open-meteo.com/v1/archive"


@dataclass(frozen=True)
class RainfallConfig:
    """Open-Meteo request settings and free-tier limits (`rainfall:` in pipeline.yaml)."""
    model: str = "era5"  # "era5" (0.25 deg, 1940-) or "era5_land" (0.1 deg, 1950-)
    timezone: str = "Asia/Manila"
    variable: str = "precipitation_sum"
    data_lag_days: int = 5  # ERA5 is published daily with about 5 days delay
    per_minute: float = 600.0  # free-tier limits, in weighted API calls
    per_hour: float = 5000.0
    per_day: float = 10000.0
    budget_fraction: float = 0.8  # use at most this share of each limit
    max_attempts: int = 5  # tries per chunk (429s and transient errors)
    backoff_s: float = 10.0  # first retry delay without Retry-After; doubles per attempt
    max_wait_s: float = 3600.0  # a longer Retry-After stops the run (resume later)
    timeout_s: float = 60.0
    user_agent: str = ("cropyield-student-project/0.1 (Davao Region crop yield research, "
                       "non-commercial; python-requests)")


@dataclass(frozen=True)
class Paths:
    raw: Path
    interim: Path
    processed: Path
    reports: Path


@dataclass(frozen=True)
class Era:
    """Modeling era: rows from `start_year` on; rolling-origin test years from `first_test_year`."""
    start_year: int
    first_test_year: int


DEFAULT_ERAS: dict[str, Era] = {
    "modis": Era(start_year=2002, first_test_year=2012),
    "s2": Era(start_year=2018, first_test_year=2021),
}


@dataclass(frozen=True)
class PipelineConfig:
    crops: tuple[str, ...]
    dates: Dates
    sources: Sources
    cropland_class: int
    s2_cloud_scl_classes: tuple[int, ...]
    min_valid_fraction: float
    random_seed: int
    paths: Paths
    eras: dict[str, Era] = field(default_factory=lambda: dict(DEFAULT_ERAS))
    rainfall: RainfallConfig = field(default_factory=RainfallConfig)


@dataclass(frozen=True)
class Config:
    pipeline: PipelineConfig
    provinces: ProvincesConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return data


def _to_date(value: Any) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def load_pipeline(path: str | Path) -> PipelineConfig:
    """Load `pipeline.yaml`. Relative paths are resolved against the repo root."""
    path = Path(path)
    d = _read_yaml(path)
    root = path.resolve().parent.parent
    dates, sources = _migrate_chirps(dict(d["dates"]), dict(d["sources"]), path)
    return PipelineConfig(
        crops=tuple(d["crops"]),
        dates=Dates(**{k: _to_date(v) for k, v in dates.items()}),
        sources=Sources(**sources),
        rainfall=RainfallConfig(**(d.get("rainfall") or {})),
        cropland_class=int(d["cropland_class"]),
        s2_cloud_scl_classes=tuple(int(c) for c in d["s2_cloud_scl_classes"]),
        min_valid_fraction=float(d["min_valid_fraction"]),
        random_seed=int(d["random_seed"]),
        paths=Paths(**{k: root / v for k, v in d["paths"].items()}),
        eras=_eras(d.get("modeling", {}).get("eras"), path),
    )


def _migrate_chirps(dates: dict[str, Any], sources: dict[str, Any],
                    path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Accept pre-Open-Meteo configs: `dates.chirps_start` becomes `rainfall_start`;
    `sources.chirps_url` is dropped (rainfall now comes from Open-Meteo)."""
    if "chirps_start" in dates:
        dates.setdefault("rainfall_start", dates.pop("chirps_start"))
        warnings.warn(f"{path}: dates.chirps_start is deprecated; use dates.rainfall_start",
                      DeprecationWarning, stacklevel=3)
    if sources.pop("chirps_url", None) is not None:
        warnings.warn(f"{path}: sources.chirps_url is ignored; rainfall uses "
                      f"sources.open_meteo_url", DeprecationWarning, stacklevel=3)
    return dates, sources


def _eras(entries: dict[str, Any] | None, path: Path) -> dict[str, Era]:
    """Parse `modeling.eras`; missing section -> DEFAULT_ERAS."""
    if not entries:
        return dict(DEFAULT_ERAS)
    eras = {}
    for name, e in entries.items():
        era = Era(start_year=int(e["start_year"]), first_test_year=int(e["first_test_year"]))
        if era.first_test_year <= era.start_year:
            raise ValueError(f"{path}: modeling.eras.{name}: first_test_year must be after "
                             f"start_year (got {era.first_test_year} <= {era.start_year})")
        eras[name] = era
    return eras


def _to_year_quarter(value: Any, where: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d{4})-Q([1-4])\s*", str(value))
    if not m:
        raise ValueError(f"{where}: expected 'YYYY-Qn' (n = 1-4), got {value!r}")
    return int(m[1]), int(m[2])


def _reporting_windows(entries: list[dict[str, Any]], names: list[str],
                       path: Path) -> tuple[ReportingWindow, ...]:
    windows = []
    for i, e in enumerate(entries):
        where = f"{path}: reporting_units[{i}]"
        for key in ("unit", "host"):
            if e.get(key) not in names:
                raise ValueError(f"{where}: {key} {e.get(key)!r} is not a configured "
                                 f"province name ({', '.join(names)})")
        if e["unit"] == e["host"]:
            raise ValueError(f"{where}: unit and host are both {e['unit']!r}")
        w = ReportingWindow(unit=e["unit"], host=e["host"],
                            first=_to_year_quarter(e.get("first"), f"{where}.first"),
                            last=_to_year_quarter(e.get("last"), f"{where}.last"))
        if w.first > w.last:
            raise ValueError(f"{where}: first {e['first']} is after last {e['last']}")
        windows.append(w)
    return tuple(windows)


def load_provinces(path: str | Path) -> ProvincesConfig:
    """Load `provinces.yaml`, including the optional `reporting_units` table.

    Raises:
        ValueError: if a reporting window names an unknown unit or host, has unit ==
            host, a malformed 'YYYY-Qn' bound, or first after last.
    """
    path = Path(path)
    d = _read_yaml(path)
    provinces = tuple(
        Province(name=p["name"], aliases=tuple(p.get("aliases") or ()),
                 part_of=p.get("part_of"))
        for p in d["provinces"]
    )
    return ProvincesConfig(
        region=d["region"],
        provinces=provinces,
        reporting_units=_reporting_windows(d.get("reporting_units") or [],
                                           [p.name for p in provinces], path),
    )


def load_config(config_dir: str | Path = DEFAULT_CONFIG_DIR) -> Config:
    """Load `pipeline.yaml` and `provinces.yaml` from `config_dir`."""
    config_dir = Path(config_dir)
    return Config(
        pipeline=load_pipeline(config_dir / "pipeline.yaml"),
        provinces=load_provinces(config_dir / "provinces.yaml"),
    )
