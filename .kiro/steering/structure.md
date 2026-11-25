---
inclusion: always
---

# Project structure

```
configs/            YAML config: provinces, crops, date ranges, data sources
data/
  raw/              untouched downloads (git-ignored, never edit)
  interim/          cleaned / clipped intermediates
  processed/        model-ready feature tables
notebooks/          exploration only; import from src/
src/cropyield/
  boundaries.py     province boundaries + cropland mask
  ground_truth.py   PSA production/area loaders and cleaning
  satellite/
    modis.py        MOD13Q1 NDVI/EVI extraction
    sentinel2.py    Sentinel-2 L2A NDVI/EVI extraction with cloud masking
  features.py       joins satellite aggregates with ground truth
  models/           training, evaluation, baselines (phase 2)
  cli.py            command-line entry points
tests/              pytest tests with synthetic fixtures
app/                web demo (phase 3)
```

Specs live in `.kiro/specs/`. Each phase gets its own spec folder.
