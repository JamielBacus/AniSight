---
inclusion: always
---

# Tech stack and conventions

## Environment
- Python 3.11, conda env named `cropyield` (see `environment.yml`).
- Core: pandas, numpy, geopandas, rasterio, xarray, rioxarray, pystac-client,
  planetary-computer, odc-stac, scikit-learn, xgboost, matplotlib, pytest.
- Satellite data comes from the Microsoft Planetary Computer STAC API (no account needed
  for reading). Do not introduce Google Earth Engine unless asked.

## Code conventions
- Library code lives in `src/cropyield/`; notebooks only call library functions.
- Every pipeline step is a function with typed inputs/outputs plus a CLI entry in
  `src/cropyield/cli.py`.
- Paths and parameters come from `configs/*.yaml`, never hard-coded.
- Set random seeds (42) everywhere randomness is used.
- Write intermediate outputs as Parquet (tables) or GeoTIFF/Zarr (rasters) under `data/`.
- Add a pytest test for each new public function; use tiny synthetic fixtures, never real
  downloads, in tests.

## Working efficiently (credits are limited)
- Write code; do not run long downloads yourself (satellite, rainfall). Tell the user the
  command to run instead. Exception: `cropyield evaluate` and `cropyield select` run in
  seconds on the ~1,500-row feature table, so agents may run them.
- Never open or read files under `data/raw/` or large files under `data/` — inspect schemas
  with a small `head`-style script instead.
- Prefer small, focused edits over rewriting whole files.
- Ask before adding a new dependency or restructuring folders.
