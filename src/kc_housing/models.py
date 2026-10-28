"""ElasticNet grid search on the feature table, saved with skops (cells 117, 119, 131).

Training only: run() never scores or predicts on the test rows; evaluate.py does that
after the baseline lock (ADR-006).
"""

import logging
from pathlib import Path

import pandas as pd
import skops.io as sio
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from kc_housing import config
from kc_housing.splits import cv_splitter, save_cv_folds, split_data

logger = logging.getLogger(__name__)


def build_model() -> Pipeline:
    """Return the notebook's unfitted ElasticNet pipeline (cell 117)."""
    return Pipeline(
        [
            ("polynomial", PolynomialFeatures(config.POLY_DEGREE, include_bias=False)),
            ("scaler", StandardScaler()),
            (
                "model",
                # The target is standardized too, then mapped back to USD.
                TransformedTargetRegressor(
                    # Fixed cell 117 solver constants, not tuned thresholds: kept here.
                    regressor=ElasticNet(max_iter=5000, tol=1e-4, precompute=True),
                    transformer=StandardScaler(),
                ),
            ),
        ]
    )


def build_grid_search(estimator: Pipeline | None = None) -> GridSearchCV:
    """Return the unfitted grid search over `estimator` or build_model() (cell 117).

    Same grid, metric and folds as the notebook; refit=True refits the best setting
    on all training rows (cell 119). Variant B passes its own pipeline so A and B
    share these settings.
    """
    return GridSearchCV(
        build_model() if estimator is None else estimator,
        config.PARAM_GRID,
        cv=cv_splitter(),
        scoring=config.PRIMARY_METRIC,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )


def train(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """Fit the grid search on the training rows and return it (cell 117)."""
    return build_grid_search().fit(X_train, y_train)


def save_model(estimator, path: str | Path = config.MODEL_PATH) -> Path:
    """Dump `estimator` with skops, creating parent dirs; return the path (cell 131)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(estimator, path)
    return path


def run(
    features_path: str | Path = config.FEATURES_PATH,
    model_path: str | Path = config.MODEL_PATH,
    folds_path: str | Path = config.CV_FOLDS_PATH,
) -> Path:
    """Split, write the CV folds, grid-search on the training rows, save the best model.

    The test rows are split off and left untouched (ADR-006).
    """
    X_train, _, y_train, _ = split_data(pd.read_parquet(features_path))
    save_cv_folds(X_train, folds_path)
    search = train(X_train, y_train)
    logger.info(
        "Best params %s, CV %s %.6f",
        search.best_params_,
        config.PRIMARY_METRIC,
        search.best_score_,
    )
    return save_model(search.best_estimator_, model_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    written = run()
    logger.info("Wrote %s", written)
