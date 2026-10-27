# ADR-002: Parity first: reproduce the notebook exactly, including questionable choices

Status: Accepted (2026-10-27)

## Context
The brief asks for a refactor, not a new model. A refactor is only verifiable if the output does not change. The notebook contains several choices one might "fix": `zipcode` used as a number, hard-coded center coordinates, `cos(radians(47.6219))` in `center_distance` while the center latitude is 47.62774, `sqft_basement` recomputed instead of parsed, and missing `view`/`waterfront` filled with 0.

## Decision
Phases 1 to 3 reproduce the notebook exactly. All constants go into `config.py` as they are (`CENTER_LAT`, `CENTER_LONG`, `CENTER_COS_LAT`, `EARTH_RADIUS_KM`). Parity means: same rows, same `config.FEATURES` columns in the same order, values equal within `PARITY_RTOL = PARITY_ATOL = 1e-9` (the vectorized `water_distance` may differ from the loop by float rounding only). `sqft_price` uses the price, so it is EDA-only: it is never computed by the pipeline and is listed in `config.LEAKAGE_COLUMNS`. Improvements come only after parity, each with its own ADR.

## Consequences
`tests/data_gates/test_parity.py` compares the pipeline with `tests/fixtures/legacy_notebook.py` on the real CSV. Known weaknesses become documented limitations in the write-up, not silent changes.

## Amendment (2026-10-27, Phase 1 review)
`last_known_change` uses `yr_renovated > 0` (else `yr_built`), where the notebook uses "`yr_renovated` is 0 or NaN" (else `yr_renovated`). The two differ only for a negative `yr_renovated`, which the notebook would keep and we replace with `yr_built`. The real CSV has no negative values, so the output is identical (checked by the reviewer on all 21,596 rows). Accepted as a deliberate, harmless tightening rather than copying the edge case.
