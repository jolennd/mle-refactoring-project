"""Phase 4 spec step 3: models.py (R10).

The oracle is notebook cell 117 in tests/fixtures/legacy_notebook.py
(`notebook_elastic_cv()`, ADR-001).

Public API under test:
- ``build_model() -> Pipeline``: unfitted; steps ``polynomial``
  (PolynomialFeatures(config.POLY_DEGREE, include_bias=False)), ``scaler``
  (StandardScaler()), ``model`` (TransformedTargetRegressor(regressor=ElasticNet(
  max_iter=5000, tol=1e-4, precompute=True), transformer=StandardScaler())), so the
  config.PARAM_GRID keys (``model__regressor__alpha`` ...) address the ElasticNet.
- ``build_grid_search() -> GridSearchCV``: unfitted, over ``build_model()`` with
  ``param_grid=config.PARAM_GRID``, ``scoring=config.PRIMARY_METRIC``, refit on all
  training rows, and ``cv=splits.cv_splitter()``.
- ``splits.cv_splitter() -> KFold``: ``KFold(n_splits=config.CV_FOLDS)``, no shuffle;
  the same folds ``save_cv_folds`` writes and notebook ``cv=5`` gives a regressor.
- ``train(X_train, y_train) -> GridSearchCV``: ``build_grid_search()`` fitted on the
  given rows (X = config.FEATURES, as from ``splits.split_data``).
"""

import json

import numpy as np
import pytest
from fixtures.legacy_notebook import notebook_elastic_cv
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, ParameterGrid, check_cv
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from kc_housing import config
from kc_housing.models import build_grid_search, build_model, train
from kc_housing.splits import save_cv_folds, split_data

pytestmark = pytest.mark.model


def _spec(obj):
    """Estimator as (class name, params), recursively, for structural comparison."""
    if isinstance(obj, BaseEstimator):
        params = obj.get_params(deep=False)
        return type(obj).__name__, {key: _spec(value) for key, value in params.items()}
    if isinstance(obj, (list, tuple)):
        return [_spec(item) for item in obj]
    return obj


def _label_folds(cv, X):
    """Folds of a cv argument over X, as lists of X's index labels."""
    return [
        {"train": X.index[train].tolist(), "validation": X.index[val].tolist()}
        for train, val in check_cv(cv, classifier=False).split(X)
    ]


def test_model_pipeline_matches_notebook_steps():
    """build_model() is an unfitted Pipeline with the notebook's steps, names and
    parameters (cell 117), and every config.PARAM_GRID point is a valid setting."""
    model = build_model()
    notebook = notebook_elastic_cv().estimator

    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["polynomial", "scaler", "model"]
    assert _spec(model) == _spec(notebook)
    assert model.named_steps["polynomial"].degree == config.POLY_DEGREE
    with pytest.raises(NotFittedError):
        check_is_fitted(model)

    for point in ParameterGrid(config.PARAM_GRID):
        params = clone(model).set_params(**point).get_params()
        assert {key: params[key] for key in point} == point


def test_grid_search_uses_config_grid_folds_and_metric(feature_table, tmp_path):
    """build_grid_search() searches config.PARAM_GRID over build_model() with
    config.PRIMARY_METRIC, refits, and its cv folds equal cv_splitter(), the folds
    save_cv_folds writes, and the notebook's cv=5."""
    from kc_housing.splits import cv_splitter

    search = build_grid_search()
    X_train, _, _, _ = split_data(feature_table)
    saved = json.loads(save_cv_folds(X_train, path=tmp_path / "cv.json").read_text())

    assert isinstance(search, GridSearchCV)
    assert search.param_grid == config.PARAM_GRID
    assert search.scoring == config.PRIMARY_METRIC
    assert search.refit is True
    assert _spec(search.estimator) == _spec(build_model())
    folds = _label_folds(search.cv, X_train)
    assert folds == saved["folds"]
    assert folds == _label_folds(cv_splitter(), X_train)
    assert folds == _label_folds(notebook_elastic_cv().cv, X_train)


def test_training_is_reproducible(feature_table):
    """train() returns a fitted GridSearchCV on config.FEATURES; two runs on the same
    rows give identical best params, CV scores and test predictions."""
    X_train, X_test, y_train, _ = split_data(feature_table)

    first = train(X_train, y_train)
    second = train(X_train, y_train)

    assert isinstance(first, GridSearchCV)
    check_is_fitted(first.best_estimator_)
    assert list(first.best_estimator_.feature_names_in_) == config.FEATURES
    assert first.best_params_ == second.best_params_
    assert first.best_score_ == second.best_score_
    np.testing.assert_array_equal(
        first.cv_results_["mean_test_score"], second.cv_results_["mean_test_score"]
    )
    np.testing.assert_array_equal(first.predict(X_test), second.predict(X_test))
