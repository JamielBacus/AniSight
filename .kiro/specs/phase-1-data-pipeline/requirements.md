# Requirements: Phase 1 — Data Pipeline

## Introduction
Build a reproducible pipeline that produces a model-ready table joining satellite
vegetation indices with PSA crop statistics, per province, crop and quarter, for the
Davao Region provinces.

## Requirement 1: Project environment
**User story:** As the developer, I want a reproducible environment so anyone can run the
pipeline.

Acceptance criteria:
1. WHEN `conda env create -f environment.yml` is run THEN the system SHALL create an
   environment named `cropyield` with all pipeline dependencies.
2. WHEN `pip install -e .` is run THEN `import cropyield` SHALL succeed.
3. WHEN `pytest` is run on a fresh clone THEN all tests SHALL pass without network access.

## Requirement 2: Province boundaries and cropland mask
**User story:** As the developer, I want province polygons and a cropland mask so indices
are averaged only over farmland in each province.

Acceptance criteria:
1. WHEN the boundaries step runs THEN the system SHALL save province polygons for every
   province in `configs/provinces.yaml` to `data/interim/boundaries.parquet` in EPSG:4326.
2. WHEN a configured province name has no matching polygon THEN the system SHALL raise an
   error that names the province.
3. WHEN the cropland step runs THEN the system SHALL produce a per-province cropland mask
   from ESA WorldCover (class 40) and report cropland area in hectares per province.

## Requirement 3: PSA ground truth
**User story:** As the developer, I want clean official production statistics as the
prediction target.

Acceptance criteria:
1. WHEN `data/raw/psa/palay_corn_production.csv` and `palay_corn_area.csv` (written by
   `cropyield.download_psa`) exist THEN the system SHALL produce
   `data/interim/ground_truth.parquet` with columns province, crop, year, quarter,
   production_mt, area_ha, yield_t_ha (yield computed as production / area).
2. WHEN a province name is a known variant (e.g. Compostela Valley) THEN the system SHALL
   map it to the canonical name in `configs/provinces.yaml`.
3. WHEN a province-crop-quarter is missing or an outlier THEN the system SHALL keep it
   flagged in a `quality_flag` column rather than drop or fill it silently.

## Requirement 4: MODIS vegetation indices
**User story:** As the developer, I want long-history NDVI/EVI per province to train on
20+ years of data.

Acceptance criteria:
1. WHEN the MODIS step runs for a date range THEN the system SHALL query MOD13Q1 (v6.1) from
   the Planetary Computer STAC API and output `data/interim/modis_vi.parquet` with one row
   per province, composite date and index.
2. WHEN pixels are flagged unreliable by the quality band THEN the system SHALL exclude
   them and record the valid-pixel fraction.
3. WHEN values are written THEN NDVI and EVI SHALL be scaled to the -1..1 range.
4. WHEN a run is interrupted THEN re-running SHALL skip completed province-date chunks.

## Requirement 5: Sentinel-2 vegetation indices
**User story:** As the developer, I want 10 m Sentinel-2 indices as a higher-resolution
feature set for recent years.

Acceptance criteria:
1. WHEN the Sentinel-2 step runs THEN the system SHALL query sentinel-2-l2a, mask clouds
   using SCL classes 3, 8, 9, 10, and output monthly per-province NDVI/EVI aggregates to
   `data/interim/s2_vi.parquet`.
2. WHEN a month has under 10% valid cropland pixels THEN the system SHALL mark it as
   missing instead of reporting a biased mean.

## Requirement 6: Feature table
**User story:** As the developer, I want a single model-ready table.

Acceptance criteria:
1. WHEN the features step runs THEN the system SHALL join ground truth with satellite
   aggregates into `data/processed/features.parquet`, one row per province, crop, year and
   quarter.
2. WHEN building features for a quarter THEN the system SHALL use only imagery dated on or
   before the end of that quarter.
3. WHEN the table is written THEN the system SHALL also write a short data-quality summary
   (row counts, missingness per column) to `reports/data_quality.md`.
