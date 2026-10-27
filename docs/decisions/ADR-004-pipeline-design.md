# ADR-004: Cleaning as pandas functions, feature engineering as sklearn transformers

Status: Accepted (2026-10-27)

## Context
The brief asks for "a reusable pipeline" without naming a form. Options: (a) a chain of pandas functions, (b) a full sklearn `Pipeline`, (c) both. sklearn transformers cannot drop rows, but they can be fitted, cloned, saved with the model and reused at inference. `water_distance` needs state learned from data (ADR-005).

## Decision
`cleaning.py`: pure functions `load_raw(path) -> DataFrame` and `clean(df) -> DataFrame` (row filtering and column repair, no state). `features.py`: sklearn-compatible transformers (`CenterDistance`, `WaterDistance`, `SelectColumns`) that take and return DataFrames themselves; `last_known_change` is a stateless column repair inside `clean()`, not a transformer; the Pipeline does not use `set_output`. `pipeline.py`: `build_feature_pipeline()` returns the sklearn `Pipeline`; `run(raw_path)` does load, clean, fit_transform and writes `data/processed/features.parquet`. In Phase 4 the feature pipeline is placed in front of the notebook's model pipeline.

## Consequences
Transformers must never mutate their input and must handle any index (unit tests). `water_distance` is vectorized with numpy broadcasting instead of the notebook's double loop (21,596 x 146 pairs, fine in memory). The design matches what the API and a saved model need later.

## Amendment (2026-10-27)
Removed the `set_output(transform="pandas")` claim: `Pipeline.set_output(transform="pandas")` raises ValueError because the transformers lack `set_output`/`get_feature_names_out`. The transformers return DataFrames themselves, so the Pipeline outputs pandas without it. `SelectColumns.fit` stores `columns_` so the Pipeline counts as fitted. Closes Phase 2 review finding 4. Also corrected the Decision text: it listed a `LastKnownChange` transformer in `features.py` that never existed; `last_known_change` is computed in `cleaning.clean()`.
