"""Phase 4 spec step 4 on real data: variant A test R^2 equals the lock (R11).

Variant A is the parity variant (ADR-005): the model trained on pipeline.run()'s
feature table (water_distance fitted on all clean rows, as the notebook does), scored
with evaluate.score() on the config split's test rows. It must match the lock's
test_r2 within config.R2_PARITY_ATOL (ADR-006). Runs the full grid search once
(module-scoped fixture); the `data` and `model` markers keep it out of
`make test-fast`. Skips until config.FEATURES_PATH exists (`make features`).
"""

import json

import pandas as pd
import pytest

from kc_housing import config
from kc_housing.evaluate import score
from kc_housing.models import train
from kc_housing.splits import split_data


@pytest.fixture(scope="module")
def variant_a_scores() -> dict:
    """score() of train()'s best estimator on the real variant A test rows.

    Module-scoped, so it repeats `processed_features`' skip with the same reason.
    """
    if not config.FEATURES_PATH.exists():
        pytest.skip(f"{config.FEATURES_PATH} not found: run `make features` first")
    X_train, X_test, y_train, y_test = split_data(pd.read_parquet(config.FEATURES_PATH))
    return score(train(X_train, y_train).best_estimator_, X_test, y_test)


@pytest.mark.data
@pytest.mark.model
def test_test_r2_matches_locked_notebook_r2(variant_a_scores):
    """Variant A test R^2 is within config.R2_PARITY_ATOL of baseline_lock.json
    test_r2 (spec stop condition if not)."""
    lock = json.loads(config.BASELINE_LOCK_PATH.read_text())
    assert variant_a_scores["test_r2"] == pytest.approx(
        lock["test_r2"], abs=config.R2_PARITY_ATOL
    )
