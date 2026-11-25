# Design: Phase 2 — Modeling and Evaluation

## Overview
```
features.parquet ──> models/data.py (eras, folds) ──> models/evaluate.py (rolling-origin CV)
                                                         │   ├─ models/baselines.py  B0 B1 B2
                                                         │   └─ models/satellite.py  ridge, boosting
rainfall.py (ERA5) ────> features.py (+rain columns) ────┘
                                                         v
                     reports/experiments.csv, reports/phase2_results.md, reports/figures/
                                                         v
                     models/select.py ──> models/<crop>.joblib, data/processed/predictions.parquet
```
The table is small (about 1,500 rows), so a full rolling-origin evaluation runs in seconds.
Agents MAY run `cropyield evaluate` themselves; the only long step is the rainfall download.

## Modeling conventions
- **Unit of prediction:** `reporting_unit × crop × quarter` for year Y.
- **Target:** `log(yield_t_ha)`. Metrics are computed on `yield_t_ha` after `exp()`.
  Production is reported as predicted yield × PSA `area_ha` (area is known when PSA
  publishes; the app labels this clearly).
- **Eras:** MODIS era = 2002+ (first full year with complete windows); Sentinel-2 era =
  2018+. Test years: MODIS era 2012–latest; Sentinel-2 era 2021–latest, because its
  training history is short. Report how many training years each fold had.
- **Folds:** rolling origin, expanding window, one test year per fold.
- **Fold safety:** every statistic used to transform features or targets (unit-quarter
  means, trend coefficients, scalers, residual quantiles) is fitted inside the fold on
  training rows only. A unit test must prove this by perturbing a test-year value and
  showing training-derived quantities don't change.

## Components (`src/cropyield/models/`)
| Module | Contents |
|---|---|
| `data.py` | `load_modeling_table(cfg)`, `era_rows(df, era)`, `rolling_folds(df, first_test_year)` yielding `(train_idx, test_idx, year)` |
| `baselines.py` | `SameQuarterLastYear`, `UnitQuarterMean`, `MeanPlusTrend`: sklearn-style `fit(train_df)` / `predict(df)` returning log-yield |
| `satellite.py` | `AnomalyRidge(features, alpha)`, `BoostedModel(features)` (sklearn `HistGradientBoostingRegressor`, categorical unit/crop/quarter, early stopping off, fixed seed 42) |
| `evaluate.py` | `cross_validate(model_factory, df, era) -> oof DataFrame`, `metrics(oof, by=...)`, `bootstrap_mae_diff(oof_a, oof_b, n=2000, by="year")`, `log_experiment(...)` |
| `select.py` | selection rule (Requirement 6.1), final fit, `models/<crop>.joblib`, predictions with 80% intervals from out-of-fold residual quantiles (per crop) |
| `report.py` | writes `reports/phase2_results.md` and figures: observed vs predicted per unit, MAE by model, residuals by year |

`src/cropyield/rainfall.py` (satellite-data agent):
- Source: Open-Meteo Historical Weather API (`sources.open_meteo_url`,
  `https://archive-api.open-meteo.com/v1/archive`), daily `precipitation_sum` from the
  ERA5 reanalysis (`rainfall.model: era5`, 0.25°; `era5_land`, 0.1°, is configurable),
  timezone Asia/Manila so days are local calendar days. No API key; the free tier is
  non-commercial only and needs CC BY 4.0 attribution.
- Points: one point per province at the area-weighted centroid of its cropland pixels
  (WorldCover masks in `data/interim/cropland/`). If the centroid falls outside the
  province polygon, the nearest cropland pixel is used. Points and cropland hectares are
  saved to `data/interim/rainfall_points.csv`.
- Download: one calendar year per point per request. A weighted-call throttle keeps usage
  at 80% (`rainfall.budget_fraction`) of the free-tier limits (600/min, 5,000/h,
  10,000/day): it waits within the minute and hour windows and stops the run cleanly
  before the daily limit. Raw JSON responses are cached under `data/raw/rainfall/`, so
  re-running the same command resumes; failed years are retried with backoff and listed
  in the run summary.
- Output `data/interim/rainfall.parquet`: `province, date (month start), rain_mm, valid_frac`.
  `rain_mm` is the sum of the daily values; a month with any missing day, or not yet
  complete, is NaN. `valid_frac` is the share of the month's days with a value.
- Combined reporting units: cropland-area-weighted mean of member provinces; NaN in any
  month where a member is NaN.
- Limitations: ERA5 is a reanalysis on a ~0.25° (~28 km) grid, not gauge-corrected
  rainfall, and a single point per province is a coarse summary of rainfall over its
  cropland. The source was switched from CHIRPS because `data.chc.ucsb.edu` was
  unreachable from the user's network.
- `features.py` gains `rain_total`, `rain_first_half`, `rain_second_half`, `rain_anom`
  (window total minus the unit's month-of-year climatology over the fixed period
  1981–2000; the rainfall series starts in 1981 and that period ends before every
  training and test year, so the reference can't leak information from them).

CLI (`cli.py`): `cropyield rainfall`, `cropyield evaluate [--era modis|s2|all]`,
`cropyield select` (final fit + predictions + report).

## Feature sets evaluated
| Name | Era | Features |
|---|---|---|
| `modis` | MODIS | `modis_*` peak, integral, peak_day, greenup_day, valid_frac (NDVI, EVI) |
| `s2` | Sentinel-2 | `s2_*` same columns |
| `modis_s2` | Sentinel-2 | both |
| `rain` | MODIS | rainfall columns only |
| `modis_rain`, `s2_rain` | per era | combinations |

Rows with `<src>_<idx>_n_obs` below the Phase 1 minimum are already NaN; ridge models drop
rows with NaN features in training and fall back to B2 for prediction (log the count).

## Honest reporting
- Every table shows B0, B1, B2 next to the models.
- The results report states plainly if satellites add nothing over B2, and lists likely
  reasons (mixed cropland mask, survey noise, quarter-level aggregation) and next steps
  (rice-paddy mask, Sentinel-1 flooding signal).

## Testing strategy
- Synthetic tables where the true signal is known (e.g. yield = trend + 0.5 × NDVI anomaly
  + noise) to check models recover it and baselines don't.
- Leakage test for fold safety (see above).
- No network in tests; rainfall tests use mocked HTTP responses and tiny synthetic
  cropland masks.
