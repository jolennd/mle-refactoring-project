# ADR-004: Cleaning as pandas functions, feature engineering as sklearn transformers

Status: Accepted (2026-10-27)

## Context
The brief asks for "a reusable pipeline" without naming a form. Options: (a) a chain of pandas functions, (b) a full sklearn `Pipeline`, (c) both. sklearn transformers cannot drop rows, but they can be fitted, cloned, saved with the model and reused at inference. `water_distance` needs state learned from data (ADR-005).

## Decision
`cleaning.py`: pure functions `load_raw(path) -> DataFrame` and `clean(df) -> DataFrame` (row filtering and column repair, no state). `features.py`: sklearn-compatible transformers (`LastKnownChange`, `CenterDistance`, `WaterDistance`, `SelectColumns`) that take and return DataFrames (`set_output(transform="pandas")`). `pipeline.py`: `build_feature_pipeline()` returns the sklearn `Pipeline`; `run(raw_path)` does load, clean, fit_transform and writes `data/processed/features.parquet`. In Phase 4 the feature pipeline is placed in front of the notebook's model pipeline.

## Consequences
Transformers must never mutate their input and must handle any index (unit tests). `water_distance` is vectorized with numpy broadcasting instead of the notebook's double loop (21,596 x 146 pairs, fine in memory). The design matches what the API and a saved model need later.
