"""Phase 4 spec step 3: the saved model and the `make train` entry point (R10, R12).

Public API under test (kc_housing.models):
- ``save_model(estimator, path=config.MODEL_PATH) -> Path``: skops dump; creates
  missing parent dirs; returns the path.
- ``run(features_path=config.FEATURES_PATH, model_path=config.MODEL_PATH,
  folds_path=config.CV_FOLDS_PATH) -> Path``: reads the Phase 3 feature table
  (parquet: id, price, config.FEATURES), ``split_data``, ``save_cv_folds(X_train,
  folds_path)``, ``train(X_train, y_train)``, ``save_model(best_estimator_,
  model_path)``; returns the model path. `make train` runs it via
  ``python -m kc_housing.models``. It never reads or scores the test rows
  (evaluate.py does, after the lock; ADR-006).

Tests write to tmp_path only, never to config.MODEL_PATH.
"""

import inspect

import numpy as np
import pytest
import skops.io as sio
from sklearn.pipeline import Pipeline

from kc_housing import config
from kc_housing.models import run, save_model, train
from kc_housing.splits import save_cv_folds, split_data


@pytest.mark.integration
def test_saved_model_roundtrip(feature_table, tmp_path):
    """run() writes the folds and a skops model to the given paths (defaults from
    config); the model loads with skops and predicts on config.FEATURES exactly like
    the best estimator of train() on the same split."""
    features_path = tmp_path / "features.parquet"
    feature_table.to_parquet(features_path)
    model_path = tmp_path / "models" / "elastic_net.skops"
    folds_path = tmp_path / "results" / "cv_folds.json"

    written = run(
        features_path=features_path, model_path=model_path, folds_path=folds_path
    )

    defaults = inspect.signature(run).parameters
    assert defaults["features_path"].default == config.FEATURES_PATH
    assert defaults["model_path"].default == config.MODEL_PATH
    assert defaults["folds_path"].default == config.CV_FOLDS_PATH
    assert inspect.signature(save_model).parameters["path"].default == config.MODEL_PATH
    assert written == model_path and model_path.exists()

    X_train, X_test, y_train, _ = split_data(feature_table)
    expected_folds = save_cv_folds(X_train, path=tmp_path / "expected_folds.json")
    assert folds_path.read_bytes() == expected_folds.read_bytes()

    loaded = sio.load(model_path, trusted=sio.get_untrusted_types(file=model_path))
    assert isinstance(loaded, Pipeline)
    assert list(loaded.feature_names_in_) == config.FEATURES
    expected = train(X_train, y_train).best_estimator_.predict(X_test)
    np.testing.assert_array_equal(loaded.predict(X_test[config.FEATURES]), expected)
