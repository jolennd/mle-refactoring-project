"""Phase 4 spec step 3 acceptance on real data: same best params as the lock (R10).

Runs the full grid search (9 settings x 5 folds) on the real training split, so it is
slow; the `data` and `model` markers keep it out of `make test-fast`. Skips until
config.FEATURES_PATH exists (`make features`).
"""

import json

import pytest

from kc_housing import config
from kc_housing.models import train
from kc_housing.splits import split_data


@pytest.mark.data
@pytest.mark.model
def test_best_params_match_baseline_lock(processed_features):
    """train() on the real variant A split (pipeline.run() output, ADR-005) picks the
    best_params recorded in baseline_lock.json (spec stop condition if not)."""
    lock = json.loads(config.BASELINE_LOCK_PATH.read_text())
    X_train, _, y_train, _ = split_data(processed_features)

    search = train(X_train, y_train)

    assert search.best_params_ == lock["best_params"]
