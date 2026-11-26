# Implementation Plan: Phase 3 — AniSight web app

- [ ] 1. Phase 2 carry-over fixes (subagent: modeler)
  - Per-target-year history in `forecast_production`; B0/B2 recipes in the bundles; resolve the unused `area_model`; fix the report's "B0-style fallback" wording
  - Per-crop ridge `n_fallback`; drop `area_ha` from test rows in `cross_validate`
  - Add the `build_predictions` leakage test and the multi-year `forecast_production` test
  - Re-run `cropyield select`; confirm B3 is still selected and metrics are unchanged
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 2. Forecast logging and live scoring (subagent: modeler)
  - [ ] 2.1 Implement `models/forecast.py` (`make_forecast`, `log_forecast`) and `cropyield forecast --quarter [--backfill]`
    - Refuse already-published quarters unless `--backfill`
    - _Requirements: 2.1, 2.2, 2.3_
  - [ ] 2.2 Implement `score_forecasts` with first-seen vs latest actuals and `cropyield score`
    - _Requirements: 2.4_
  - [ ] 2.3 Run `cropyield forecast --quarter 2026Q4` for real and show me the forecast table
    - _Requirements: 2.1, 2.2_

- [ ] 3. App data export (subagent: modeler)
  - Implement `src/cropyield/export.py` and `cropyield export` writing `app/data/*` and `app/assets/figures/`
  - Expanding "historical normal" for map colouring; NaN → null; size budget; findings numbers read from reports
  - Tests for schema, NaN-free JSON, size budget and findings numbers matching `selection.csv`
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 4. App skeleton and map (subagent: web-dev)
  - `app/index.html`, `css/style.css`, `js/main.js`, `js/map.js`, vendored Leaflet and Chart.js (pinned, with licences)
  - Crop and quarter controls, URL hash state, diverging colour scale with legend, combined units, works without tiles
  - `js/lib/format.js` with `node --test` tests
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 7.2, 7.3_

- [ ] 5. Province detail panel (subagent: web-dev)
  - History chart with 80% band and B0/B2 toggles; latest forecast card with production note; conditions-vs-normal card; flags
  - Keyboard-accessible province list as an alternative to the map
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ] 6. Findings, accuracy and about pages (subagent: web-dev)
  - `findings.html` from `findings.json` and the copied figures; `accuracy.html` from `live_accuracy.json` with "awaiting PSA release" state; `about.html` with sources and licences
  - _Requirements: 6.1, 6.2, 6.3_

- [ ] 7. Polish and checks (subagent: web-dev)
  - 375 px phone layout, contrast, page weight < 1.5 MB excluding tiles
  - Manual checklist: open via a local static server (`python -m http.server -d app 8000`), test phone width, block tile requests, keyboard-only navigation
  - _Requirements: 7.1, 7.2_

- [ ] 8. Vercel config and runbook (subagent: web-dev)
  - `app/vercel.json` cache headers; README sections "Live site", "Quarterly update runbook", "Deploying on Vercel" (Root Directory = `app`, Framework "Other", no build command)
  - Give the user the dashboard steps; do not deploy yourself
  - _Requirements: 7.4_

- [ ] 9. Final review (subagent: critic)
  - Every number on the site traces to `app/data/*.json` and from there to `reports/*`
  - No forecast or production figure uses information unavailable at forecast time
  - Wording is honest (no "satellite-predicted"), attribution complete, accessibility basics met
  - _Requirements: 2.2, 3.5, 5.2, 6.1, 6.3_
