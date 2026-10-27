# Experiment design

## Question
Can the notebook's cleaning and feature engineering run as a tested, reusable pipeline that gives the same features (and, as a stretch, the same model quality) as the notebook? The notebook's own scenario, estimating fair house prices in King County, stays the modeling question.

## Label
`price` in USD per sale, untransformed in the data. Inside the model it is standardized by `TransformedTargetRegressor`, as in the notebook.

## Unit of analysis and grain
One row per sale (not per house: `id` repeats, see docs/02_data_sources.md).

## Split
- Random split, not by time: `train_test_split(test_size=0.3, random_state=42)` after cleaning and feature engineering, exactly as in the notebook (ADR-006).
- Embargo / gap: none. Sales span only 12 months and the notebook's question is not a forecast.
- CV: `GridSearchCV(cv=5)` on the training rows (unshuffled KFold, as in the notebook); fold indices persisted to reports/results/cv_folds.json.

## Features (fixed set, in config.FEATURES)
| Step | Features | Source |
|---|---|---|
| Cleaned raw columns | bedrooms ... sqft_lot15 (16 columns, `sqft_basement` recomputed, `view`/`waterfront` filled with 0) | notebook cells 28-38 |
| Engineered | last_known_change (renovation year, else build year) | cells 41-43 |
| Engineered | center_distance (km to a fixed Seattle center point) | cell 60 |
| Engineered | water_distance (km to the nearest waterfront house) | cells 67-70 |
| Model | degree-2 polynomial expansion + StandardScaler inside the model pipeline | cell 117 |

Dropped before modeling: `id`, `date`, `price`, `sqft_price`, `delta_lat`, `delta_long`.

## Metrics and success criteria
- Phases 1 to 3 (required): feature parity with the notebook. Every feature column equals the frozen notebook code's output within config.PARITY_RTOL / PARITY_ATOL (ADR-002), and the row count matches.
- Phase 4 (stretch): primary metric is test R^2, the notebook's metric. No invented target: run the notebook's procedure once, lock its CV R^2 and test R^2 in reports/results/baseline_lock.json (committed alone), then the refactored training must match the locked test R^2 within config.R2_PARITY_ATOL (ADR-006). The notebook's conclusion says roughly 0.83; the locked number replaces that.
- Uncertainty: not required by the brief. The write-up reports a single test R^2 as the notebook does; a bootstrap CI is optional.

## Open decisions
- Parity tolerance values -> ADR-002 (Accepted: 1e-9)
- water_distance reference set: full data (parity) vs training data only -> ADR-005 (Accepted: report both variants)
- R^2 tolerance and lock procedure -> ADR-006 (Accepted: 0.001)
- API database and five fields -> ADR-008 (Accepted: bonus_solution ranges, GET / health, `api` dependency group)
