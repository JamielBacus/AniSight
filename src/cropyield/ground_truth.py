"""PSA ground truth: quarterly palay and corn production, area and yield per province.

`load_psa` reads the two tidy CSVs written by `cropyield.download_psa`
(`palay_corn_production.csv` and `palay_corn_area.csv` in `data/raw/psa/`) and returns
one row per province, crop, year and quarter with columns
`province, crop, year, quarter, production_mt, area_ha, yield_t_ha, quality_flag,
reporting_unit`.
`save_ground_truth` writes it to `data/interim/ground_truth.parquet`.

Cleaning rules:
- Geolocations are mapped to the canonical names in `configs/provinces.yaml` through
  their aliases (case-insensitive). Rows that match no unit, such as the region total
  "REGION XI (DAVAO REGION)", are dropped; they are not provinces.
- Only the crop totals whose croptype equals a crop in `pipeline.yaml` ("Palay", "Corn")
  are kept. The sub-types (Irrigated/Rainfed Palay, White/Yellow Corn) add up to these
  totals, so they are not summed again.
- "Annual" rows are dropped; only Quarter 1-4 are kept.
- yield_t_ha = production_mt / area_ha, left empty (NaN) when area is 0 or missing.
- Reporting windows (`reporting_units` in `configs/provinces.yaml`): inside a window PSA
  reports the absorbed unit as 0 and counts it in the host province. The absorbed unit's
  rows get production, area and yield NaN and the flag `not_reported` only; the host's
  rows keep the reported values and get `includes:<unit>`.
- `reporting_unit` names the footprint the PSA figure covers: the province itself, or
  inside a window the host followed by every unit it absorbs in that quarter (in
  `provinces.yaml` order), joined by "+", e.g. "Davao del Sur+Davao Occidental". Absorbed
  rows carry the same combined name (their data sit in the host row).
  `boundaries.reporting_unit_geometries` returns polygons keyed by these names.
- Nothing is dropped or filled. Province-crop-year-quarters absent from the CSVs are
  added with empty values. `quality_flag` is "ok" or a ";"-joined list of:
    missing    production or area is empty or absent
    not_reported  the unit is reported inside its host this quarter (only flag; values NaN)
    zero       production and area are both reported as 0 outside a reporting window
    zero_area  area is 0 but production is not
    outlier    yield is far from the usual yield for that province, crop and quarter:
               modified z-score of log(yield) (median and MAD over all years of that
               group) above `outlier_z` (3.5). Advisory only; values are kept as reported.
               Host rows inside a reporting window are scored but left out of the
               median/MAD reference, so the reporting shift does not distort it.
    includes:<unit>  the host's figure also covers <unit> (one flag per absorbed unit)

Usage:
    python -m cropyield.ground_truth                 # data/raw/psa -> data/interim
    python -m cropyield.ground_truth --config-dir configs
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cropyield.config import DEFAULT_CONFIG_DIR, Config, load_config

log = logging.getLogger(__name__)

PRODUCTION_CSV = "palay_corn_production.csv"
AREA_CSV = "palay_corn_area.csv"
KEYS = ["province", "crop", "year", "quarter"]
COLUMNS = [*KEYS, "production_mt", "area_ha", "yield_t_ha", "quality_flag", "reporting_unit"]
OUTLIER_Z = 3.5  # Iglewicz-Hoaglin cut-off for the modified z-score
MIN_OUTLIER_GROUP = 5  # need this many positive yields to judge a group


def _norm(name: str) -> str:
    return " ".join(str(name).casefold().split())


def province_lookup(cfg: Config) -> dict[str, str]:
    """Map normalised names and aliases to canonical province names."""
    lookup: dict[str, str] = {}
    for p in cfg.provinces.provinces:
        for n in (p.name, *p.aliases):
            lookup[_norm(n)] = p.name
    return lookup


def _read(path: Path, value: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"geolocation", "croptype", "year", "period", value} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df[["geolocation", "croptype", "year", "period", value]]


def _outliers(df: pd.DataFrame, z: float, ref: pd.Series) -> pd.Series:
    """True where log(yield) has a robust z-score above `z` within province/crop/quarter.

    Median, MAD and group size come only from rows where `ref` is True; every row is scored.
    """
    logy = np.log(df["yield_t_ha"].where(df["yield_t_ha"] > 0))
    keys = [df["province"], df["crop"], df["quarter"]]
    ref_logy = logy.where(ref)
    med = ref_logy.groupby(keys).transform("median")
    dev = (logy - med).abs()
    mad = dev.where(ref).groupby(keys).transform("median")
    n = ref_logy.groupby(keys).transform("count")
    score = 0.6745 * dev / mad.where(mad > 0)
    return (score > z) & (n >= MIN_OUTLIER_GROUP)


def load_psa(path: str | Path, cfg: Config, outlier_z: float = OUTLIER_Z) -> pd.DataFrame:
    """Load the PSA production and area CSVs in folder `path` into the ground truth table."""
    path = Path(path)
    prod = _read(path / PRODUCTION_CSV, "production_mt")
    area = _read(path / AREA_CSV, "area_ha")
    df = prod.merge(area, on=["geolocation", "croptype", "year", "period"], how="outer")

    lookup = province_lookup(cfg)
    df["province"] = df["geolocation"].map(lambda g: lookup.get(_norm(g)))
    unknown = sorted(df.loc[df["province"].isna(), "geolocation"].unique())
    if unknown:
        log.info("dropping geolocations not in provinces.yaml: %s", unknown)
    crops = {_norm(c): c for c in cfg.pipeline.crops}
    df["crop"] = df["croptype"].map(lambda c: crops.get(_norm(c)))
    quarter = df["period"].str.extract(r"^\s*Quarter\s+([1-4])\s*$", expand=False)
    df["quarter"] = pd.to_numeric(quarter)
    df = df.dropna(subset=["province", "crop", "quarter"])
    df = df.astype({"year": int, "quarter": int})

    dup = df.duplicated(KEYS, keep=False)
    if dup.any():
        raise ValueError(f"several PSA rows map to the same key: {df.loc[dup, KEYS].head().to_dict('records')}")

    # Complete the province x crop x year x quarter grid so absent quarters show up flagged.
    provinces = [p for p in cfg.provinces.names if p in set(df["province"])]
    crops_found = [c for c in cfg.pipeline.crops if c in set(df["crop"])]
    years = range(df["year"].min(), df["year"].max() + 1) if len(df) else []
    grid = pd.MultiIndex.from_product([provinces, crops_found, years, [1, 2, 3, 4]], names=KEYS)
    df = df.set_index(KEYS)[["production_mt", "area_ha"]].reindex(grid).reset_index()

    # Reporting windows: absorbed units' values live in the host row, so blank them here.
    prov_cfg = cfg.provinces
    rows = list(zip(df["province"], df["year"], df["quarter"]))
    not_reported = pd.Series([prov_cfg.host_of(*r) is not None for r in rows], index=df.index)
    includes = pd.Series([prov_cfg.absorbed_by(*r) for r in rows], index=df.index)
    df["reporting_unit"] = [prov_cfg.reporting_unit(*r) for r in rows]
    reported = df.loc[not_reported, ["production_mt", "area_ha"]]
    if (reported.fillna(0) != 0).any(axis=None):
        log.warning("discarding non-zero values reported inside a reporting window: %s",
                    df.loc[reported.index[(reported.fillna(0) != 0).any(axis=1)], KEYS]
                    .to_dict("records"))
    df.loc[not_reported, ["production_mt", "area_ha"]] = np.nan

    p, a = df["production_mt"], df["area_ha"]
    df["yield_t_ha"] = p / a.where(a > 0)
    in_window = not_reported | (includes.map(len) > 0)
    flags = pd.DataFrame({
        "not_reported": not_reported,
        "missing": (p.isna() | a.isna()) & ~not_reported,
        "zero": (p == 0) & (a == 0),
        "zero_area": (a == 0) & (p != 0) & p.notna(),
        "outlier": _outliers(df, outlier_z, ref=~in_window),
    })
    df["quality_flag"] = [
        ";".join([*flags.columns[f], *(f"includes:{u}" for u in inc)]) or "ok"
        for f, inc in zip(flags.to_numpy(), includes)]
    return df[COLUMNS].sort_values(KEYS).reset_index(drop=True)


def save_ground_truth(df: pd.DataFrame, cfg: Config) -> Path:
    """Write the ground truth table to `<interim>/ground_truth.parquet`."""
    out = cfg.pipeline.paths.interim / "ground_truth.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def main(argv: list[str] | None = None) -> None:
    """Build data/interim/ground_truth.parquet from the PSA CSVs in data/raw/psa."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR),
                    help="folder with pipeline.yaml and provinces.yaml (default: repo configs/)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config_dir)
    df = load_psa(cfg.pipeline.paths.raw / "psa", cfg)
    out = save_ground_truth(df, cfg)
    counts = df["quality_flag"].value_counts().to_dict()
    log.info("%s rows -> %s; quality_flag counts: %s", f"{len(df):,}", out, counts)


if __name__ == "__main__":
    main()
