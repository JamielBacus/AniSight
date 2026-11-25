---
inclusion: fileMatch
fileMatchPattern: ["src/cropyield/models/**", "src/cropyield/features.py", "notebooks/**"]
---

# Modeling rules

- Split by time, never randomly: train on earlier years, validate/test on the most recent
  years (e.g. leave-last-2-years-out, or rolling-origin CV by year).
- No leakage: features for a quarter may only use imagery dated within or before that
  quarter's harvest window. Never use the target quarter's PSA numbers or later imagery.
- Always report the naive baseline (same quarter, previous year) next to every model.
- Metrics: MAE, RMSE, and MAPE, broken down by province and crop.
- Start simple (ridge regression, random forest) before gradient boosting or deep learning.
- Log every experiment (params, features, metrics) to `reports/experiments.csv`.
