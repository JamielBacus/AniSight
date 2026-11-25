# Implementation Plan: Phase 1 — Data Pipeline

- [x] 1. Finish the project skeleton and config loader
  - Fill in `pyproject.toml` entry point for `cropyield` CLI
  - Implement `src/cropyield/config.py` to load `configs/*.yaml` into dataclasses
  - Add a test that loads the sample config
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 2. Province boundaries and cropland mask (subagent: satellite-data)
  - [x] 2.1 Implement `load_boundaries` using geoBoundaries PHL ADM2, filtered to configured provinces
    - Raise a clear error for unmatched province names
    - _Requirements: 2.1, 2.2_
  - [x] 2.2 Implement `cropland_mask` from ESA WorldCover class 40 and per-province cropland hectares
    - _Requirements: 2.3_

- [x] 3. PSA ground truth loader (subagent: ground-truth)
  - [x] 3.1 Write small CSV fixtures in the tidy format produced by `src/cropyield/download_psa.py`
    - The downloader already exists and is tested; do not rewrite it
    - _Requirements: 3.1_
  - [x] 3.2 Implement `load_psa` with province name harmonisation and quality flags
    - _Requirements: 3.1, 3.2, 3.3_

- [x] 4. MODIS extraction (subagent: satellite-data)
  - [x] 4.1 Implement shared STAC helpers in `satellite/stac.py`
    - _Requirements: 4.1_
  - [x] 4.2 Implement `extract_modis` with quality masking, scaling and resumable chunk cache
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 5. Sentinel-2 extraction (subagent: satellite-data)
  - Implement `extract_s2` with SCL cloud masking, monthly aggregation and low-coverage handling
  - _Requirements: 5.1, 5.2_

- [x] 6. Feature table and data-quality report (subagent: modeler)
  - Implement `build_features` respecting the quarter-end cutoff
  - Write `reports/data_quality.md`
  - _Requirements: 6.1, 6.2, 6.3_

- [x] 7. Wire up the CLI and review (subagent: critic)
  - Add `boundaries`, `ground-truth`, `modis`, `sentinel2`, `features` commands to `cli.py`
  - Run the critic agent over the whole phase and fix must-fix issues
  - _Requirements: 1.3, 6.2_
