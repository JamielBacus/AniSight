# Implementation Plan: Phase 2 — Modeling and Evaluation

- [x] 1. Modeling dataset and rolling-origin folds (subagent: modeler)
  - Implement `src/cropyield/models/data.py`: `load_modeling_table`, `era_rows`, `rolling_folds`
  - Tests: era filters, folds never put a test year in training, reporting_unit is the unit
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. Baselines B0, B1, B2 (subagent: modeler)
  - Implement `src/cropyield/models/baselines.py` with fit/predict on log-yield
  - Tests on a synthetic table with a known trend; B0 falls back to B1 when last year is missing
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 3. Evaluation harness (subagent: modeler)
  - [x] 3.1 Implement `cross_validate`, `metrics`, `bootstrap_mae_diff`, `log_experiment` in `evaluate.py`
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  - [x] 3.2 Add `cropyield evaluate --era modis|s2|all` and run it for the baselines only
    - Report the baseline table in chat (MAE per crop for B0, B1, B2)
    - _Requirements: 3.1, 3.3_

- [ ] 4. Satellite models (subagent: modeler)
  - [x] 4.1 Implement `AnomalyRidge` and `BoostedModel` in `satellite.py` with fold-safe anomalies
    - Include the leakage test from design.md and a synthetic-signal recovery test
    - _Requirements: 4.1, 4.2, 4.3_
  - [x] 4.2 Evaluate feature sets `modis`, `s2`, `modis_s2`; report MAE vs B2 with bootstrap CIs
    - _Requirements: 4.4, 3.2_

- [x] 5. Review checkpoint (subagent: critic)
  - Review tasks 1–4 for leakage, fold safety, metric correctness and honest reporting
  - Fix must-fix issues before continuing
  - _Requirements: 1.3, 2.4, 4.3_

- [x] 6. Rainfall (Open-Meteo ERA5) (subagent: satellite-data)
  - [x] 6.1 Implement `src/cropyield/rainfall.py` (download + cache + per-province monthly totals) and `cropyield rainfall`
    - Open-Meteo ERA5 daily rainfall at one cropland point per province, throttled and resumable; tests use mocked HTTP and synthetic masks, no network
    - Give the user the command to run the real download; do not run it yourself
    - _Requirements: 5.1, 5.3_
  - [x] 6.2 Add rainfall columns to `features.py` (window total, halves, anomaly) with tests
    - _Requirements: 5.2, 5.3_

- [-] 7. Evaluate rainfall feature sets (subagent: modeler)
  - After the user has run `cropyield rainfall` and `cropyield features`, evaluate `rain`, `modis_rain`, `s2_rain`
  - _Requirements: 4.4, 5.2_

- [ ] 8. Model selection, final models and predictions (subagent: modeler)
  - Implement `select.py` and `report.py`; add `cropyield select`
  - Write `models/<crop>.joblib`, `data/processed/predictions.parquet` (out-of-fold history + 80% intervals) and `reports/phase2_results.md` with figures
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 9. Final review (subagent: critic)
  - Review the whole phase; check the results report's claims match `reports/experiments.csv`
  - _Requirements: 3.2, 6.1, 6.4_
