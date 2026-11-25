# Design: Phase 1 — Data Pipeline

## Overview
A set of independent, re-runnable steps orchestrated from `src/cropyield/cli.py`:

```
configs/*.yaml
      |
      v
boundaries ──> cropland mask ──┐
                               ├──> modis ──────┐
                               └──> sentinel2 ──┼──> features ──> data/processed/features.parquet
psa CSVs ──> ground_truth ──────────────────────┘
```

Steps 2 (boundaries), 3 (ground truth) and 4/5 (satellite) are independent once the
skeleton exists, so they can be built in parallel by the `satellite-data` and
`ground-truth` subagents.

## Data sources
| Source | Access | Notes |
|---|---|---|
| Province boundaries | geoBoundaries PHL ADM2 (GeoJSON download) | In geoBoundaries, PHL ADM2 = provinces. Verify Davao City handling. |
| Cropland | Planetary Computer `esa-worldcover` | Class 40 = cropland, 10 m |
| MODIS | Planetary Computer `modis-13Q1-061` | 250 m, 16-day composites, 2000– |
| Sentinel-2 | Planetary Computer `sentinel-2-l2a` | 10 m, 2017–, cloud mask via SCL |
| Rainfall (optional) | CHIRPS daily/pentad | Phase 2 covariate |
| Ground truth | PSA OpenSTAT, manual CSV export | Palay and corn, by province, quarterly |

## Components
- `config.py` — loads YAML into typed dataclasses.
- `boundaries.py` — `load_boundaries(cfg) -> GeoDataFrame`, `cropland_mask(gdf, cfg) -> dict`.
- `ground_truth.py` — `load_psa(path, cfg) -> DataFrame` with name harmonisation and flags.
- `satellite/stac.py` — shared STAC client, signing, search helpers.
- `satellite/modis.py` — `extract_modis(gdf, mask, start, end) -> DataFrame`.
- `satellite/sentinel2.py` — `extract_s2(gdf, mask, start, end) -> DataFrame`.
- `features.py` — `build_features(gt, vi, cfg) -> DataFrame` + quality report.
- `cli.py` — `cropyield boundaries|ground-truth|modis|sentinel2|features`.

## Data contracts
- Vegetation index tables: `province, date, index, mean, median, max, valid_frac, n_pixels`.
- Ground truth: `province, crop, year, quarter, production_mt, area_ha, yield_t_ha, quality_flag,
  reporting_unit`. `quality_flag` is `ok` or `;`-joined `not_reported`, `missing`, `zero`,
  `zero_area`, `outlier`, `includes:<unit>`. `reporting_unit` is the footprint the PSA figure
  covers: the province, or inside a `reporting_units` window in `provinces.yaml` the host plus
  its absorbed units joined by "+" (host first, then `provinces` order). Features must be
  aggregated over the `reporting_unit` footprint (`boundaries.reporting_unit_geometries`),
  and `not_reported` rows have no target.
- Features: one row per `province, crop, year, quarter`; satellite features are aggregated
  over that crop's growing window ending at quarter end.

## Error handling
- Network calls retry with backoff; failed chunks are logged and skipped, not fatal.
- Completed chunks are cached as Parquet under `data/interim/cache/` so re-runs resume.

## Testing strategy
- Unit tests with synthetic xarray rasters and small CSV fixtures; no network in tests.
- A `@pytest.mark.network` smoke test on a tiny bbox, skipped by default.
