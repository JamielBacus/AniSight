# Data

Nothing under `data/` is committed to git.

- `raw/psa/` — PSA OpenSTAT data, downloaded with `python -m cropyield.download_psa`
  (uses the PXWeb API, so no manual export is needed):
  - `palay_corn_production.csv` — geolocation, croptype, year, period, production_mt
  - `palay_corn_area.csv` — geolocation, croptype, year, period, area_ha
  Covers Region XI, its five provinces and City of Davao; all six croptypes; Q1–Q4 and
  Annual; 1987 to the latest release. Yield = production_mt / area_ha.
- `raw/boundaries/` — geoBoundaries PHL ADM2 GeoJSON (downloaded by the pipeline)
- `interim/` — cleaned and clipped intermediates, chunk cache
- `processed/` — model-ready feature tables
