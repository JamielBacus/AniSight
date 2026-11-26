# Requirements: Phase 3 — AniSight web app

## Introduction
Phase 2 found that satellite and rainfall features did not improve on simple baselines.
The selected model for both crops is B3, the mean of B0 (same quarter last year) and B2
(unit-quarter mean + trend) in log-yield. Its 80% intervals are well calibrated
(84% corn, 82% palay over 2015–2026).

Phase 3 ships AniSight as an **early-warning monitor** for rice (palay) and corn in the
Davao Region:
- a B3 yield forecast per province with an 80% range, available before PSA publishes;
- "conditions vs normal" from rainfall and satellite greenness, shown as context, not as
  model inputs;
- an honest findings page and a live-accuracy tracker that scores forecasts against new
  PSA releases.

The app is a static website (HTML, CSS, JavaScript, pre-built JSON) deployed on Vercel.

## Requirement 1: Phase 2 carry-over fixes
**User story:** As the developer, I want the forecast code correct before the app depends on it.

Acceptance criteria:
1. WHEN `forecast_production` receives targets in more than one year THEN each target year
   SHALL use only history from earlier years (one area fit per target year).
2. WHEN a final bundle is saved THEN it SHALL include the B0 and B2 recipes so live scoring
   can compare B3 with both.
3. A `build_predictions` leakage test SHALL perturb year-Y targets and areas and assert that
   year-Y predicted columns do not change.
4. Ridge `n_fallback` SHALL be logged per crop, like B0.
5. `fit_final` SHALL either use its saved `area_model` in `forecast_production` or stop
   saving it, and the report text SHALL match.

## Requirement 2: Forecasts and live scoring
**User story:** As the developer, I want each quarter's forecast logged before PSA
publishes, so accuracy can be scored honestly later.

Acceptance criteria:
1. WHEN `cropyield forecast --quarter YYYYQn` runs THEN the system SHALL refit the frozen
   B0/B2/B3 recipes on all PSA data released before the forecast date and predict yield,
   80% interval, predicted area and predicted production for every current reporting unit
   and crop.
2. WHEN a forecast is made THEN the system SHALL append it to `reports/forecasts.csv` with
   a UTC timestamp, git commit hash, the latest PSA quarter in the input (data vintage), and
   the B3, B0 and B2 predictions.
3. WHEN a target quarter is already in the PSA data at forecast time THEN the command SHALL
   refuse, unless `--backfill` is given and the row is marked as a backfill.
4. WHEN `cropyield score` runs THEN the system SHALL join logged non-backfill forecasts to
   PSA actuals as they become available and write `reports/live_accuracy.csv`. The primary
   metric, fixed in advance, is yield MAE of B3 vs B2 and vs B0. It is reported against both
   the first PSA figure seen and the latest revised figure.

## Requirement 3: App data export
**User story:** As the developer, I want one command that writes everything the app needs.

Acceptance criteria:
1. WHEN `cropyield export` runs THEN the system SHALL write to `app/data/`:
   `predictions.json`, `forecast.json`, `units.geojson`, `live_accuracy.json`,
   `findings.json` and `meta.json`, and copy the Phase 2 figures to `app/assets/figures/`.
2. JSON SHALL contain no NaN or Infinity; missing values SHALL be `null`.
3. The files in `app/data/` together SHALL be under 1.5 MB, with polygons simplified enough
   to meet that.
4. `meta.json` SHALL record the generation time, data vintage, latest forecast quarter,
   selected model per crop, and the code's git commit.
5. Every number shown on the findings page SHALL come from `findings.json`, which is built
   from `reports/selection.csv` and `reports/experiments.csv`, never typed by hand.

## Requirement 4: Map view (home page)
**User story:** As a visitor, I want to see at a glance which provinces look above or below
normal this quarter.

Acceptance criteria:
1. The map SHALL show the reporting units for the selected quarter, coloured by the
   forecast's percentage difference from that unit's historical mean yield for the same
   quarter, using a colour-blind-safe diverging palette with a legend.
2. Controls SHALL select the crop (palay/corn) and the quarter (latest forecast by default,
   any historical quarter otherwise).
3. Combined PSA reporting units (e.g. "Davao del Sur+Davao Occidental") SHALL be drawn as
   their combined footprint for the quarters in which PSA reported them that way.
4. The map SHALL remain usable if basemap tiles fail to load (province shapes drawn on a
   plain background).

## Requirement 5: Province detail
**User story:** As a visitor, I want to see one province's history and this quarter's
conditions.

Acceptance criteria:
1. Selecting a unit SHALL show a chart of PSA actual yield and the B3 out-of-fold prediction
   since 2012, with the 80% band. B0 and B2 SHALL be available as toggles.
2. The panel SHALL show the latest forecast: yield, 80% range, and predicted production with
   the note "assumes the same harvested area as the same quarter last year".
3. A "conditions vs normal" card SHALL show `rain_anom` and the NDVI/EVI anomalies in plain
   language (e.g. "Rainfall 32% above normal"), labelled "context, not used in the forecast".
4. Rows with flags SHALL say so: no area forecast for a newly regrouped unit, B0 fallback,
   outlier, combined reporting unit.

## Requirement 6: Findings, methods and live accuracy pages
**User story:** As a reviewer, I want to know how good the forecast is and why satellites
didn't help.

Acceptance criteria:
1. A findings page SHALL state plainly that satellite and rainfall features did not beat the
   baselines, show the selection table, disclose that B3 and the selection rule were chosen
   after seeing results, and list the likely reasons and next steps from
   `reports/phase2_results.md`.
2. A live-accuracy page SHALL list each logged forecast against the PSA actual when
   available, show "awaiting PSA release" otherwise, and summarise the primary metric once
   at least one quarter is scored.
3. An about/data page SHALL credit every data source with its licence and required
   attribution (PSA, MODIS/NASA, Copernicus Sentinel-2, ESA WorldCover, geoBoundaries,
   Open-Meteo/ERA5, basemap provider).

## Requirement 7: Quality and deployment
**User story:** As the developer, I want a fast, accessible site that redeploys on every push.

Acceptance criteria:
1. The site SHALL work on a 375 px wide phone screen and a desktop browser.
2. The site SHALL have no build step: plain ES modules, with third-party libraries vendored
   in `app/vendor/` at pinned versions.
3. Pure JavaScript helpers (formatting, colour scale, anomaly wording) SHALL have tests run
   with `node --test 'app/tests/*.test.mjs'`.
4. Vercel SHALL serve the `app/` folder (Root Directory = `app`, no build command), and the
   README SHALL include the quarterly update runbook: pipeline → `select` → `forecast` →
   `export` → commit → push.
