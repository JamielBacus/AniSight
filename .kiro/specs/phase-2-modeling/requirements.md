# Requirements: Phase 2 — Modeling and Evaluation

## Introduction
Phase 1 produced `data/processed/features.parquet`: 1,510 rows (reporting unit × crop ×
year × quarter) with PSA yield/production targets, MODIS features (2001–) and Sentinel-2
features (2017–).

A quick Phase 1 check found that, within each reporting unit, MODIS greenness anomalies
barely correlate with yield anomalies (|r| ≤ 0.07), Sentinel-2 correlates modestly
(r ≈ 0.2), and the strongest signal is the upward yield trend (r ≈ 0.3 with year). The
likely cause is that the cropland mask mixes rice/corn with plantations.

Phase 2 therefore measures, honestly, whether satellite (and rainfall) features improve on
strong non-satellite baselines, and ships whichever approach actually wins.

## Requirement 1: Modeling dataset and folds
**User story:** As the developer, I want one function that returns clean modeling data and
time-ordered folds, so every model is evaluated on exactly the same rows.

Acceptance criteria:
1. WHEN the dataset is built THEN the system SHALL drop rows with a missing or non-positive
   `yield_t_ha` and keep `outlier` and `includes:` rows, with `quality_flag` available for
   filtering.
2. WHEN an era is requested THEN the system SHALL return the MODIS era (years 2002+) or the
   Sentinel-2 era (years 2018+), so no row in an era lacks that era's features.
3. WHEN folds are generated THEN the system SHALL use rolling-origin splits by year: for
   each test year Y, train on all era years before Y and test on year Y only.
4. WHEN the unit of prediction is defined THEN the system SHALL use `reporting_unit`
   (not `province`) so combined PSA units are predicted as reported.

## Requirement 2: Baselines
**User story:** As the developer, I want strong non-satellite baselines, so any satellite
model has to earn its place.

Acceptance criteria:
1. The system SHALL implement B0 "same quarter last year": prediction = yield of the same
   reporting unit, crop and quarter in year Y−1 (fallback to B1 when missing).
2. The system SHALL implement B1 "unit-quarter mean": mean log-yield of the unit, crop and
   quarter over the training years.
3. The system SHALL implement B2 "mean + trend": B1 plus a per-crop linear year trend in
   log-yield fitted on the training years only.
4. WHEN any baseline is fitted THEN it SHALL use training-fold rows only.

## Requirement 3: Evaluation harness
**User story:** As the developer, I want one harness that scores any model the same way.

Acceptance criteria:
1. WHEN a model is evaluated THEN the system SHALL report MAE, RMSE and MAPE of
   `yield_t_ha` (back-transformed from log) overall, per crop, and per reporting unit.
2. WHEN a model is compared with B2 THEN the system SHALL report the MAE difference with a
   95% bootstrap confidence interval resampled by test year.
3. WHEN a model is evaluated THEN the system SHALL append one row per (model, era, crop) to
   `reports/experiments.csv` with the parameters, feature list, metrics and run timestamp.
4. WHEN outliers are evaluated THEN the system SHALL report metrics both with and without
   `outlier` rows in the test set.

## Requirement 4: Satellite models
**User story:** As the developer, I want to measure what satellite features add.

Acceptance criteria:
1. The system SHALL implement a ridge model on anomalies: target = log-yield minus the B2
   prediction; features = satellite features minus their training-fold unit-crop-quarter
   means.
2. The system SHALL implement a gradient-boosting model with reporting unit, crop and
   quarter as categorical inputs plus the satellite features.
3. WHEN anomalies or scalers are computed THEN they SHALL use training-fold rows only.
4. The system SHALL evaluate MODIS-only (MODIS era), Sentinel-2-only (Sentinel-2 era) and
   MODIS+Sentinel-2 (Sentinel-2 era) feature sets.

## Requirement 5: Rainfall features
**User story:** As the developer, I want rainfall as a non-greenness covariate.

Acceptance criteria:
1. WHEN the rainfall step runs THEN the system SHALL write monthly Open-Meteo ERA5 rainfall (mm) per
   province and reporting unit to `data/interim/rainfall.parquet`.
2. WHEN features are built THEN the system SHALL add rainfall totals for the growing window
   and its two halves, plus the window-total anomaly vs that unit's fixed 1981–2000
   month-of-year climatology, using only months ending on or before quarter end.
3. WHEN rainfall data is unavailable for a month THEN the feature SHALL be missing, not zero.

## Requirement 6: Model selection, final model and predictions
**User story:** As the developer, I want a defensible final model and a predictions file for
the Phase 3 app.

Acceptance criteria:
1. The system SHALL select, per crop, the simplest model whose MAE beats B2 with a 95%
   bootstrap CI that excludes zero; if none does, it SHALL select B2 and say so.
2. WHEN the final model is fitted THEN it SHALL be trained on all era rows and saved to
   `models/<crop>.joblib` with its feature list and training period.
3. The system SHALL write `data/processed/predictions.parquet` with reporting_unit, crop,
   year, quarter, actual yield and production, B2 prediction, selected-model prediction,
   and an 80% interval from out-of-fold residual quantiles. Historical rows SHALL use
   out-of-fold (rolling-origin) predictions, never in-sample fits.
4. The system SHALL write `reports/phase2_results.md` with the comparison tables, the
   selection decision and its reason, and figures in `reports/figures/`.
