---
inclusion: fileMatch
fileMatchPattern: ["app/**", "src/cropyield/export.py"]
---

# AniSight app conventions

- The site is static: `app/` is served as-is by Vercel (Root Directory = `app`).
- Data contract: the files in `app/data/` are produced only by `cropyield export`. If a
  page needs a new field, extend `export.py` and its tests; don't derive it in JavaScript.
- Product framing: AniSight is an early-warning monitor. The forecast is B3 (blend of
  "same quarter last year" and "average + trend"). Satellite and rainfall data are shown as
  conditions vs normal, labelled as context. Phase 2 found they did not improve accuracy;
  the findings page says so.
- Predicted production assumes last year's harvested area for the same quarter; say this
  wherever production is shown. Newly regrouped units have no area forecast.
- Keep the total of `app/data/` under 1.5 MB.
