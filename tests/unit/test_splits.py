"""Phase 4 spec step 2: splits.py (R10).

Public API under test:
- ``split_data(df) -> (X_train, X_test, y_train, y_test)``: X is
  ``df[config.FEATURES]``, y is ``df[config.TARGET]``;
  ``train_test_split(test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)``,
  random and per row as in the notebook (ADR-006).
  Index labels are kept.
- ``save_cv_folds(X_train, path=config.CV_FOLDS_PATH) -> Path``: writes the GridSearchCV
  folds (``KFold(n_splits=config.CV_FOLDS)``, no shuffle, as ``cv=5`` gives a regressor)
  as JSON ``{"n_splits": int, "folds": [{"train": [labels], "validation": [labels]}]}``,
  labels being X_train index labels; creates missing parent dirs; returns the path.
"""

import inspect
import json

import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from kc_housing import config
from kc_housing.splits import save_cv_folds, split_data


def test_split_data_uses_config_test_size_and_seed(feature_table):
    """Rows match sklearn's train_test_split with config.TEST_SIZE / RANDOM_STATE, the
    split is identical across calls, and train and test share no row."""
    X_train, X_test, y_train, y_test = split_data(feature_table)
    again = split_data(feature_table)

    expected_train, expected_test = train_test_split(
        feature_table.index,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )
    pd.testing.assert_index_equal(X_train.index, expected_train)
    pd.testing.assert_index_equal(X_test.index, expected_test)
    pd.testing.assert_index_equal(y_train.index, expected_train)
    pd.testing.assert_index_equal(y_test.index, expected_test)
    for first, second in zip((X_train, X_test, y_train, y_test), again, strict=True):
        pd.testing.assert_index_equal(first.index, second.index)
    assert not set(X_train.index) & set(X_test.index)
    assert len(X_train) + len(X_test) == len(feature_table)


def test_split_data_x_is_config_features_y_is_target(feature_table):
    """R6: X holds exactly config.FEATURES in order, never LEAKAGE_COLUMNS or id; y is
    price. Repeat sales of one id may land on both sides (per-row split, ADR-006)."""
    X_train, X_test, y_train, y_test = split_data(feature_table)

    for X in (X_train, X_test):
        assert list(X.columns) == config.FEATURES
        assert not set(X.columns) & set(config.LEAKAGE_COLUMNS)
        assert config.ID_COL not in X.columns
    for y, X in ((y_train, X_train), (y_test, X_test)):
        assert y.name == config.TARGET
        pd.testing.assert_series_equal(y, feature_table.loc[X.index, config.TARGET])


def test_save_cv_folds_is_identical_every_run(feature_table, tmp_path):
    """Spec step 2 acceptance: cv_folds.json is byte-identical across runs, holds
    config.CV_FOLDS KFold folds over X_train labels, and defaults to CV_FOLDS_PATH."""
    X_train, _, _, _ = split_data(feature_table)
    first = save_cv_folds(X_train, path=tmp_path / "a" / "cv_folds.json")
    X_train_again = split_data(feature_table)[0]
    second = save_cv_folds(X_train_again, path=tmp_path / "cv_folds.json")

    assert first == tmp_path / "a" / "cv_folds.json"
    assert first.read_bytes() == second.read_bytes()
    assert (
        inspect.signature(save_cv_folds).parameters["path"].default
        == config.CV_FOLDS_PATH
    )
    written = json.loads(first.read_text())
    assert written["n_splits"] == config.CV_FOLDS
    expected = [
        {
            "train": X_train.index[train].tolist(),
            "validation": X_train.index[val].tolist(),
        }
        for train, val in KFold(n_splits=config.CV_FOLDS).split(X_train)
    ]
    assert written["folds"] == expected
