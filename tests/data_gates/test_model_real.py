"""Phase 4 spec step 3 acceptance on real data: same best params and CV R^2 as the lock.

Runs the full grid search (9 settings x 5 folds) on the real training split, so it is
slow; the `data` and `model` markers keep it out of `make test-fast`. The fitted search
is shared through a module-scoped fixture so it runs once. Skips until
config.FEATURES_PATH exists (`make features`).
"""

import json

import pandas as pd
import pytest

from kc_housing import config
from kc_housing.models import train
from kc_housing.splits import split_data


@pytest.fixture(scope="module")
def baseline_lock() -> dict:
    """The committed baseline lock (ADR-006)."""
    return json.loads(config.BASELINE_LOCK_PATH.read_text())


@pytest.fixture(scope="module")
def fitted_search():
    """train() fitted once on the real variant A split (pipeline.run() output, ADR-005).

    Module-scoped, so it cannot use the function-scoped `processed_features`; it repeats
    that fixture's skip so the tests skip with the same reason until the data exists.
    """
    if not config.FEATURES_PATH.exists():
        pytest.skip(f"{config.FEATURES_PATH} not found: run `make features` first")
    features = pd.read_parquet(config.FEATURES_PATH)
    X_train, _, y_train, _ = split_data(features)
    return train(X_train, y_train)


@pytest.mark.data
@pytest.mark.model
def test_best_params_match_baseline_lock(fitted_search, baseline_lock):
    """train() on the real variant A split (pipeline.run() output, ADR-005) picks the
    best_params recorded in baseline_lock.json (spec stop condition if not)."""
    assert fitted_search.best_params_ == baseline_lock["best_params"]


@pytest.mark.data
@pytest.mark.model
def test_cv_r2_matches_locked_cv_r2(fitted_search, baseline_lock):
    """The mean 5-fold CV R^2 (best_score_) equals the lock's cv_r2 within
    config.R2_PARITY_ATOL (ADR-006, 2026-10-28 amendment): same code, folds and data."""
    assert fitted_search.best_score_ == pytest.approx(
        baseline_lock["cv_r2"], abs=config.R2_PARITY_ATOL
    )
