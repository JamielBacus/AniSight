# Satellite Crop Yield Estimator — Mindanao

Province-level rice (palay) and corn yield estimation for Davao Region from MODIS and
Sentinel-2 vegetation indices, validated against PSA statistics.

## Setup
```bash
conda env create -f environment.yml
conda activate cropyield
pytest
```

## Data you download manually
PSA OpenSTAT exports for palay and corn (production, area harvested, by province and
quarter) go into `data/raw/psa/`. See `data/README.md`.

## Roadmap
1. **Phase 1, data pipeline** — `.kiro/specs/phase-1-data-pipeline/`
2. **Phase 2, modeling** — baselines, seasonal features, time-based validation
3. **Phase 3, demo** — map + per-province forecast web app

## Working with Kiro
- Steering: `.kiro/steering/` (product, tech, structure always load; ml-rules loads for model code)
- Subagents: `.kiro/agents/` — satellite-data, ground-truth, modeler, critic
- Open `tasks.md` in the Phase 1 spec and click **Start task** on each item.
