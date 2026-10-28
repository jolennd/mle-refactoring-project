"""Phase 4 spec step 4: evaluate.py, variant B (train-only feature fit; ADR-005, R11).

Variant A (parity) scores the model trained on the Phase 3 feature table, whose
water_distance was fitted on all clean rows. Variant B puts build_feature_pipeline() in
front of the model, so the whole chain, WaterDistance included, is fitted on the
training rows only. B's input is therefore clean rows, not feature rows.

Public API under test (kc_housing.evaluate):
- ``split_variant_b(clean, features) -> (X_train, X_test, y_train, y_test)``: the rows
  of ``splits.split_data(features)`` (same index labels, same order), taken from the
  clean table ``clean`` (cleaning contract columns). X holds the clean columns minus
  config.TARGET, config.ID_COL and config.LEAKAGE_COLUMNS; y is config.TARGET. Both
  tables come from the same clean table, so their index labels match; raises
  ValueError if a label of ``features`` is missing from ``clean``.
- ``build_variant_b_search() -> GridSearchCV``: unfitted; estimator is a Pipeline with
  steps ``features`` (build_feature_pipeline()), then build_model()'s ``polynomial``,
  ``scaler``, ``model`` steps, so config.PARAM_GRID keys still apply; same grid,
  metric, refit and cv_splitter() folds as models.build_grid_search().
- ``train_variant_b(X_train, y_train) -> GridSearchCV``: build_variant_b_search()
  fitted on clean training rows.
"""

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, check_cv
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from kc_housing import config
from kc_housing.evaluate import build_variant_b_search, split_variant_b, train_variant_b
from kc_housing.models import build_model
from kc_housing.pipeline import build_feature_pipeline
from kc_housing.splits import cv_splitter, split_data


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


def test_variant_b_split_uses_variant_a_rows(model_clean_houses, model_features):
    """B's train and test rows are A's (same labels, same order) taken from the clean
    table; X has no target, id or leakage column; y equals A's y; a features label
    missing from the clean table raises ValueError."""
    a_train, a_test, a_y_train, a_y_test = split_data(model_features)

    X_train, X_test, y_train, y_test = split_variant_b(
        model_clean_houses, model_features
    )

    assert X_train.index.tolist() == a_train.index.tolist()
    assert X_test.index.tolist() == a_test.index.tolist()
    np.testing.assert_array_equal(y_train.to_numpy(), a_y_train.to_numpy())
    np.testing.assert_array_equal(y_test.to_numpy(), a_y_test.to_numpy())
    assert y_train.name == y_test.name == config.TARGET
    excluded = {config.TARGET, config.ID_COL, *config.LEAKAGE_COLUMNS}
    for X in (X_train, X_test):
        assert not set(X.columns) & excluded
        assert {"lat", "long", "waterfront"} <= set(X.columns)
        assert "water_distance" not in X.columns  # B computes it itself
        np.testing.assert_array_equal(
            X.to_numpy(), model_clean_houses.loc[X.index, X.columns].to_numpy()
        )

    with pytest.raises(ValueError):
        split_variant_b(model_clean_houses.drop(index=a_test.index[0]), model_features)


@pytest.mark.model
def test_variant_b_fits_water_reference_on_training_rows_only(
    model_clean_houses, model_features
):
    """ADR-005 variant B: the grid search is the notebook search with the feature
    pipeline in front; after fitting, WaterDistance's reference houses are exactly the
    training waterfront houses (no test row), and the model predicts on clean rows."""
    X_train, X_test, y_train, _ = split_variant_b(model_clean_houses, model_features)
    test_water = X_test[X_test["waterfront"] == 1]
    assert not test_water.empty  # fixture guard: otherwise A and B would agree

    unfitted = build_variant_b_search()
    assert isinstance(unfitted, GridSearchCV)
    assert list(unfitted.estimator.named_steps) == [
        "features",
        "polynomial",
        "scaler",
        "model",
    ]
    assert _spec(unfitted.estimator.named_steps["features"]) == _spec(
        build_feature_pipeline()
    )
    assert _spec(unfitted.estimator.steps[1:]) == _spec(build_model().steps)
    assert unfitted.param_grid == config.PARAM_GRID
    assert unfitted.scoring == config.PRIMARY_METRIC
    assert unfitted.refit is True
    assert _label_folds(unfitted.cv, X_train) == _label_folds(cv_splitter(), X_train)

    search = train_variant_b(X_train, y_train)

    best = search.best_estimator_
    assert isinstance(best, Pipeline)
    check_is_fitted(best)
    water = best.named_steps["features"].named_steps["water_distance"]
    train_water = X_train[X_train["waterfront"] == 1]
    np.testing.assert_array_equal(
        np.sort(water.reference_lat_), np.sort(train_water["lat"].to_numpy())
    )
    np.testing.assert_array_equal(
        np.sort(water.reference_long_), np.sort(train_water["long"].to_numpy())
    )
    assert not set(test_water["lat"]) & set(water.reference_lat_)
    assert np.isfinite(search.predict(X_test)).all()
