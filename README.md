# King County housing: refactoring project

This project moves the data cleaning and feature engineering of the King County house price notebook into a tested pipeline in `src/kc_housing/`. There are two notebooks (ADR-001): `King-County.ipynb` is the original, kept unchanged as the reference, and `notebooks/king_county_refactored.ipynb` is a copy that calls the pipeline and keeps the modeling cells. The pipeline output is checked against the original notebook, and the modeling step (stretch) reproduces the notebook's own test R^2.

The `bonus_solution/` folder is the course's reference for the API and Docker stretch goal. It is not this project's code.

## How to run

Requires [uv](https://docs.astral.sh/uv/) and the Python version in `.python-version`.

```bash
uv sync                      # install dependencies into .venv/
make data features pipeline  # Phases 1-3: clean, build features, run the pipeline and its parity gates
make train evaluate          # Phase 4 (stretch, done): grid search, saved model, metrics
make test-fast               # quick tests while working
make test                    # full suite, run before a gate
```

To re-run the refactored notebook top to bottom:

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/king_county_refactored.ipynb --output-dir /tmp
```

Writing the executed copy to `/tmp` keeps the committed notebook unchanged.

## Project layout

```text
King-County.ipynb                    original notebook (frozen reference)
notebooks/king_county_refactored.ipynb   refactored copy, thin, uses the pipeline
src/kc_housing/
  config.py                          paths, feature list, tolerances, seeds
  cleaning.py  features.py  pipeline.py      Phases 1-3
  splits.py  models.py  evaluate.py          Phase 4 (stretch)
tests/                               unit, integration, data_gates, model
docs/decisions/                      ADR-001 to ADR-009
reports/results/                     baseline_lock.json, metrics.json, cv_folds.json
PROJECT_WORKFLOW.md                  phases, gates, status
```

## Results

### Pipeline parity (Phases 1-3)

The pipeline reproduces the notebook's features on all 21,596 rows within 1e-9 (`config.PARITY_RTOL`), checked by `tests/data_gates/test_parity.py`. The row count is asserted as `EXPECTED_SALES` in `tests/data_gates/test_features_real.py`.

Rows where `bedrooms == config.IMPLAUSIBLE_BEDROOMS` are dropped by rule (ADR-003), instead of the notebook's hard-coded index. This is tested in `tests/unit/test_cleaning.py`.

### Modeling (Phase 4, stretch)

Split `train_test_split(test_size=0.3, random_state=42)`, ElasticNet with a 5-fold grid search, as in the notebook.

| Quantity | Value | Source |
|---|---|---|
| Locked notebook test R^2 | 0.8443 | `reports/results/baseline_lock.json` (`test_r2`) |
| Refactored, variant A test R^2 (reference fitted on all rows, as in the notebook) | 0.8443 | `reports/results/metrics.json` (`variant_a.test_r2`) |
| Variant B test R^2 (water_distance reference fitted on training rows only) | 0.8447 | `reports/results/metrics.json` (`variant_b.test_r2`) |
| A - B | -0.00042 | `reports/results/metrics.json` (`r2_a_minus_b`) |
| 95% paired bootstrap CI of A - B | [-0.00119, 0.00021] | `reports/results/metrics.json` (`r2_a_minus_b_ci`) |

Variant A matches the lock (ADR-006). The CI of A - B includes 0, so there is no significant difference between the two variants. The interval covers test-set sampling only (ADR-006).

The notebook's conclusion text says "roughly 83%". The locked reproduced value above replaces it (ADR-006).

## What steps did you take to complete the project?

1. Framed the scope before coding: the original notebook stays frozen as the reference (ADR-001), parity comes before any improvement (ADR-002), and each non-obvious choice got an ADR.
2. Phase 1, cleaning (`cleaning.py`): drop the implausible row by rule (ADR-003), recompute `sqft_basement`, fill `view` and `waterfront`, build `last_known_change`.
3. Phase 2, features (`features.py`): `center_distance` and `water_distance` as sklearn transformers, vectorized, with the water reference learned in `fit` (ADR-005), and a column selection to `config.FEATURES`.
4. Phase 3, pipeline (`pipeline.py`): one sklearn-compatible pipeline from raw data to features, a parity test against a copy of the notebook code in `tests/fixtures/legacy_notebook.py`, and the thin refactored notebook.
5. Phase 4, modeling (stretch): committed the notebook's own scores as a lock first (`reports/results/baseline_lock.json`), then trained and evaluated against it (ADR-006), with a train-only variant B and a bootstrap interval for the difference.
6. Worked test first: red tests committed on their own, then the simplest code to pass them, then refactoring with the tests green.
7. Wrote this README (Phase 7) before the API and Docker phases (ADR-009 amendment).

## What challenges did you face?

- Parity is hard to prove by reading. The vectorized `water_distance` differs from the notebook's loop in the last bits of floating point, so parity uses a tolerance (ADR-002) and is checked on the real data, not only on small fixtures.
- Notebook quirks had to be kept: `zipcode` as a number, hard-coded center coordinates, a center latitude that does not match the `cos` term in `center_distance` (ADR-002).
- The hard-coded row index in the notebook does not work on new data, so the drop became a rule (ADR-003). Sklearn transformers cannot remove rows, so the drop happens before the feature pipeline (ADR-004).
- The target for the model was unclear. The notebook text says "roughly 83%", but the notebook gives a higher value when run, so the notebook was run once and its scores were locked and committed before the refactored training existed (ADR-006).
- Leakage: fitting the water reference on all rows lets test rows shape a training feature (ADR-005). Both variants were run to measure the effect.
- Keeping the frozen notebook trustworthy: tests check that the refactored notebook has no cleaning code and that the reference cells are unchanged.

## What would you do differently if you had more time?

These limitations are documented in the ADRs and were not changed, to keep parity.

- Split by `id` instead of by row (ADR-006). Repeat sales of one house can land in both train and test. A group split by `id` would remove this optimism.
- Widen the grid. The best parameters (alpha 0.01, l1_ratio 0.2, in `baseline_lock.json` and `metrics.json`) sit at the corner of the grid, so a better value may lie outside it. This needs its own ADR.
- Fit the `water_distance` reference on training rows only in the main path (ADR-005). Variant B does this, but the notebook's full-data fit is kept as the parity variant. Also measure variability across other splits and seeds, which the bootstrap does not cover (ADR-006).
- Remove other notebook quirks listed in ADR-002 (numeric `zipcode`, hard-coded center, mismatched `cos` latitude), each with its own ADR and a new reference value.
- Finish Phases 5 and 6, the API and Docker (see below). A prediction endpoint is out of scope in ADR-008 because the model needs 19 features and the API stores 5.

## Stretch phases

### Phase 5: FastAPI CRUD API (planned)

Planned stretch work, not built yet. Scope (ADR-008): create, read, update and delete houses with 5 fields in a database, plus a health endpoint.

<!-- Add run instructions when Phase 5 closes -->

### Phase 6: Docker (planned)

Planned stretch work, not built yet. Scope (ADR-008): a Dockerfile for the API and a compose file with Postgres.

<!-- Add run instructions when Phase 6 closes -->
