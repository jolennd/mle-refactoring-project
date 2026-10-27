"""Train/test split and GridSearchCV folds (notebook cells 82 and 117; ADR-006)."""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from kc_housing import config


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split `df` into X_train, X_test, y_train, y_test (notebook cell 82).

    X is config.FEATURES, y is config.TARGET; the split is random and per row, as in
    the notebook, so repeat sales of one id can land on both sides (ADR-006).
    """
    X = df[config.FEATURES]
    y = df[config.TARGET]
    return tuple(
        train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
    )


def save_cv_folds(
    X_train: pd.DataFrame, path: str | Path = config.CV_FOLDS_PATH
) -> Path:
    """Write the GridSearchCV folds over X_train index labels as JSON (cell 117).

    `cv=5` on a regressor means KFold(5) without shuffling, so the folds are fixed.
    """
    # Positions from KFold, stored as index labels so they survive a reload.
    folds = [
        {
            "train": X_train.index[train].tolist(),
            "validation": X_train.index[val].tolist(),
        }
        for train, val in KFold(n_splits=config.CV_FOLDS).split(X_train)
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n_splits": config.CV_FOLDS, "folds": folds}) + "\n")
    return path
