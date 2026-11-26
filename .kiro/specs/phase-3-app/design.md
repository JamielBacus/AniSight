# Design: Phase 3 — AniSight web app

## Overview
```
data/processed/predictions.parquet ─┐
models/<crop>.joblib ───────────────┼─> cropyield forecast ─> reports/forecasts.csv
data/interim/ground_truth.parquet ──┘                 │
                                                     └─> cropyield score ─> reports/live_accuracy.csv
reports/selection.csv, experiments.csv, figures/ ─┐
data/interim/boundaries.parquet ──────────────────┼─> cropyield export ─> app/data/*.json, app/assets/figures/
                                                  │
                                       app/ (static site) ── git push ──> Vercel (Root Directory = app)
```

## Timing (why it's an "early" estimate)
PSA publishes quarterly palay/corn figures about 40 days after the quarter ends. Rainfall
and satellite context are available within days. `cropyield forecast` is run after the
quarter ends and before PSA's release; its log entry proves the forecast came first.

## Python side (subagent: modeler)

### Carry-over fixes (`models/select.py`, `models/evaluate.py`, `models/report.py`)
- `forecast_production`: loop over target years; for year Y fit the area rule on history
  with year < Y.
- Bundle: add `b0` and `b2` recipes (fitted alongside B3) and a `recipe_version` string.
- Either make `forecast_production` use `bundle["area_model"]` (refitting it on history
  passed in) or remove it; fix the "Final models" wording in the report ("B0-style fallback").
- `evaluate.py`: per-crop ridge `n_fallback`; add `area_ha` to the columns dropped from test
  rows in `cross_validate`.

### `src/cropyield/models/forecast.py`
- `make_forecast(quarter, cfg, backfill=False) -> DataFrame`: loads ground truth, refuses
  quarters already published (unless backfill), refits B0/B2/B3 + area rule on all published
  rows, predicts for the **current** reporting units (the latest quarter's unit set), adds
  80% interval offsets from the bundle's out-of-fold residual quantiles.
- `log_forecast(df, path)`: append-only `reports/forecasts.csv` with `forecast_id`,
  `made_at_utc`, `git_commit` (or `"unknown"` if git is unavailable), `data_vintage`
  (latest PSA year-quarter in the input), `target_quarter`, `reporting_unit`, `crop`,
  `b3_yield`, `b3_lo80`, `b3_hi80`, `b0_yield`, `b2_yield`, `predicted_area_ha`,
  `pred_production_mt`, `backfill`.
- `score_forecasts(cfg) -> DataFrame`: joins non-backfill rows to PSA actuals; keeps
  `actual_first_seen` (the first value observed, stored in `reports/psa_first_seen.csv` the
  first time a quarter appears) and `actual_latest`; computes absolute errors for B3/B0/B2.
  Writes `reports/live_accuracy.csv`.
- CLI: `cropyield forecast --quarter 2026Q4 [--backfill]`, `cropyield score`.

### `src/cropyield/export.py`
| File | Contents |
|---|---|
| `predictions.json` | `{crop: {unit: [{y, q, actual, pred, lo, hi, b0, b2, rain_anom, ndvi_anom, evi_anom, flags[]}]}}` for 2012+, numbers rounded (yield 3 dp, anomalies 1–3 dp) |
| `forecast.json` | latest logged non-backfill forecast per unit and crop, with conditions vs normal for that quarter |
| `units.geojson` | one feature per reporting unit (including combined units) from `boundaries.reporting_unit_geometries`, simplified (target ≤ 400 KB), properties `{unit, members[], first, last}` |
| `live_accuracy.json` | rows from `live_accuracy.csv` + summary (n quarters, B3/B0/B2 MAE) |
| `findings.json` | selection table per crop, satellite/rain comparison summary, interval coverage, rain shift, caveat strings, counts — all read from `reports/*.csv` |
| `meta.json` | generated_at, git_commit, data_vintage, latest_forecast_quarter, model per crop, sources list |

"Historical normal" for map colouring = mean actual yield of the same unit, crop and quarter
over the years before the displayed year (expanding, ≥ 3 years), computed in export so the
map never compares a quarter with a mean that includes itself.

## Web side (subagent: web-dev) — `app/`
```
app/
  index.html            map + province panel (home)
  findings.html         findings and methods
  accuracy.html         live accuracy
  about.html            data sources, licences, attribution
  css/style.css
  js/main.js            page bootstrapping, routing between views via URL hash (#corn/2026Q4/Davao%20del%20Norte)
  js/map.js             Leaflet map, styling, legend
  js/panel.js           province detail + Chart.js history chart
  js/lib/format.js      pure helpers: formatting, colour scale, anomaly wording (tested)
  data/                 written by `cropyield export` (committed)
  assets/figures/       copied Phase 2 figures
  vendor/               leaflet, chart.js (pinned, minified, with LICENSE files)
  vercel.json           headers: short cache for /data/*, long cache for /vendor/*
  tests/format.test.mjs node --test
```
- **Map:** Leaflet with a light basemap (CARTO Positron or OpenStreetMap standard, with
  attribution). Provinces drawn from `units.geojson`; fill = diverging scale
  (e.g. ColorBrewer RdYlBu, clamped at ±25%); grey for missing. Works without tiles.
- **Chart:** Chart.js line chart: actual (dots), B3 (line), 80% band (filled area),
  optional B0/B2 (dashed). Mark forecast quarter distinctly.
- **Wording:** plain language, e.g. "Forecast 4.21 t/ha (likely 3.95–4.48). About 6% above
  this province's usual Q4." Never say "predicted by satellite".
- **Accessibility:** colour is never the only cue (values in tooltips/table); keyboard
  focusable province list as an alternative to clicking the map; sufficient contrast.
- **Performance:** total page weight under 1.5 MB excluding basemap tiles.

## Deployment (Vercel)
- Vercel project imported from GitHub `JamielBacus/AniSight`, Framework preset "Other",
  Root Directory `app`, no build command, output directory default.
- Every push to `main` deploys production; other branches get preview URLs.
- `app/data/` is committed (it is outside the git-ignored `data/` folder).

## Quarterly runbook (README)
1. `python -m cropyield.download_psa` (new PSA release) → `cropyield ground-truth`
2. `cropyield modis`, `cropyield sentinel2`, `cropyield rainfall` (resume; only new months)
3. `cropyield features` → `cropyield select`
4. `cropyield score` (scores earlier forecasts against the new release)
5. After the next quarter ends: `cropyield forecast --quarter <next>`
6. `cropyield export` → `git add . && git commit && git push` → Vercel deploys

## Testing
- Python: forecast refuses published quarters; backfill flag; per-year history in
  `forecast_production`; `build_predictions` leakage test; scoring with first-seen vs revised
  actuals; export JSON is valid, NaN-free, under the size budget, and its findings numbers
  equal `selection.csv`.
- JavaScript: `node --test app/tests` for `format.js`.
- Manual checklist in tasks: phone width, broken-tiles mode, keyboard navigation.
