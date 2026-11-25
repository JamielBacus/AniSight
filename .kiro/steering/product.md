---
inclusion: always
---

# Product: Satellite Crop Yield Estimator

## Goal
Estimate province-level rice (palay) and corn production / yield in Mindanao from
satellite vegetation indices, ahead of official PSA releases.

## Scope (v1 defaults — change here if the plan changes)
- Region: Davao Region (Region XI) provinces first; expand to the rest of Mindanao later.
- Crops: palay (rice) and corn.
- Target: quarterly production (metric tons) and yield (t/ha) per province, from PSA OpenSTAT.
- Satellite inputs: MODIS MOD13Q1 NDVI/EVI (2000–present, long history) as the primary
  feature source; Sentinel-2 L2A (2017–present, 10 m) as a higher-resolution variant.
- Cropland mask: ESA WorldCover (cropland class) so indices are averaged over farmland only.
- Optional covariate: CHIRPS rainfall.
- Deliverable: a reproducible pipeline + model + simple web demo (map + per-province forecast).

## Non-goals (v1)
- Field/parcel-level prediction.
- CBMS or household data.
- Real-time alerts.

## Success criteria
- Beats a naive baseline (same quarter last year) on held-out recent years.
- Error reported per province and per crop, not just overall.
- Every number in the demo can be traced back to a script in `src/`.
