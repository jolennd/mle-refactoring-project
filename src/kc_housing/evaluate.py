"""Test-set scoring of both variants after the lock (cells 117-120; ADR-005, ADR-006).

Variant A (parity) scores the saved model, trained on the Phase 3 feature table whose
water_distance was fitted on all clean rows, as the notebook does. Variant B puts
build_feature_pipeline() in front of the model, so WaterDistance learns its reference
houses from the training rows only. Both use the same split rows, grid and folds.

This is the only code that scores the test rows, and it runs after the lock (ADR-006).
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from kc_housing import config
from kc_housing.models import build_grid_search, build_model
from kc_housing.pipeline import build_feature_pipeline
from kc_housing.splits import split_data

logger = logging.getLogger(__name__)

# Modules whose types the saved model may contain; anything else is refused.
TRUSTED_MODULES = ("sklearn.", "numpy.", "kc_housing.")

# Bootstrap resamples drawn per chunk, to bound memory (an implementation detail, not
# a threshold). Results are deterministic for a fixed seed AND chunk size.
BOOTSTRAP_CHUNK_SIZE = 1000


def split_variant_b(
    clean: pd.DataFrame, features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return A's train and test rows (same labels, same order) from the clean table.

    X is the clean columns minus the target, id and leakage columns; y is the target.
    """
    missing = features.index.difference(clean.index)
    if not missing.empty:
        raise ValueError(
            f"{len(missing)} feature-table labels missing from the clean table, "
            f"e.g. {missing[:5].tolist()}"
        )
    X_train_a, X_test_a, _, _ = split_data(features)
    excluded = {config.TARGET, config.ID_COL, *config.LEAKAGE_COLUMNS}
    x_columns = [col for col in clean.columns if col not in excluded]
    train, test = clean.loc[X_train_a.index], clean.loc[X_test_a.index]
    return (
        train[x_columns],
        test[x_columns],
        train[config.TARGET],
        test[config.TARGET],
    )


def build_variant_b_search() -> GridSearchCV:
    """Return the notebook grid search with the feature pipeline in front (ADR-005).

    Grid, metric, refit and folds are shared with variant A via
    models.build_grid_search. Step names after `features` are build_model()'s, so
    config.PARAM_GRID applies; the feature pipeline is refitted on each CV training
    fold.
    """
    return build_grid_search(
        Pipeline([("features", build_feature_pipeline()), *build_model().steps])
    )


def train_variant_b(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """Fit build_variant_b_search() on clean training rows and return it."""
    return build_variant_b_search().fit(X_train, y_train)


def score(estimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return test R^2 (cell 120, elastic.score) and RMSE in USD as plain floats."""
    pred = estimator.predict(X_test)
    return {
        "test_r2": float(r2_score(y_test, pred)),
        "test_rmse": float(root_mean_squared_error(y_test, pred)),
    }


def load_model(path: str | Path = config.MODEL_PATH):
    """Load a skops model, trusting only sklearn, numpy and kc_housing types."""
    untrusted = sio.get_untrusted_types(file=path)
    refused = [name for name in untrusted if not name.startswith(TRUSTED_MODULES)]
    if refused:
        raise ValueError(f"{path} contains types outside {TRUSTED_MODULES}: {refused}")
    return sio.load(path, trusted=untrusted)


def paired_bootstrap_r2_diff(
    y_true,
    pred_a,
    pred_b,
    n_resamples: int,
    confidence: float,
    seed: int,
    *,
    chunk_size: int = BOOTSTRAP_CHUNK_SIZE,
) -> dict:
    """Paired percentile bootstrap interval of test R^2 A - B (ADR-006).

    Each draw resamples the row positions with replacement once and scores A and B
    on that same draw. Draws come from one generator, chunk_size at a time.
    """
    y = np.asarray(y_true, dtype=float)
    n = y.size
    # Per-row vectors, computed once; y is centred to avoid cancellation in SS_tot.
    sq_err_a = (y - np.asarray(pred_a, dtype=float)) ** 2
    sq_err_b = (y - np.asarray(pred_b, dtype=float)) ** 2
    y_c = y - y.mean()
    y_c_sq = y_c**2

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for start in range(0, n_resamples, chunk_size):
        size = min(chunk_size, n_resamples - start)
        # Drawn as int32; np.take would convert it to intp on every call, so convert
        # once here (the int32 copy is freed) and reuse it for the four gathers.
        idx = rng.integers(0, n, size=(size, n), dtype=np.int32).astype(np.intp)
        sum_y_c = np.take(y_c, idx).sum(axis=1)
        ss_tot = np.take(y_c_sq, idx).sum(axis=1) - sum_y_c**2 / n
        sse_a = np.take(sq_err_a, idx).sum(axis=1)
        sse_b = np.take(sq_err_b, idx).sum(axis=1)
        # R^2_A - R^2_B: ss_tot is shared by A and B on each draw, so 1 - ... cancels.
        diffs[start : start + size] = (sse_b - sse_a) / ss_tot
        del idx  # free before the next draw, so two index chunks never coexist

    low, high = np.quantile(diffs, [(1 - confidence) / 2, (1 + confidence) / 2])
    return {
        "low": float(low),
        "high": float(high),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
    }


def _variant(scores: dict, lock_test_r2: float, best_params: dict) -> dict:
    """Scores plus the delta to the lock and the chosen grid point."""
    return {
        **scores,
        "delta_vs_lock": scores["test_r2"] - lock_test_r2,
        "best_params": best_params,
    }


def run(
    features_path: str | Path = config.FEATURES_PATH,
    clean_path: str | Path = config.CLEAN_PATH,
    model_path: str | Path = config.MODEL_PATH,
    lock_path: str | Path = config.BASELINE_LOCK_PATH,
    metrics_path: str | Path = config.METRICS_PATH,
) -> Path:
    """Score variant A (saved model) and B (train-only features); write metrics.json."""
    lock_test_r2 = float(json.loads(Path(lock_path).read_text())["test_r2"])
    features = pd.read_parquet(features_path)

    # A: the saved best_estimator_ Pipeline, so its grid point comes from get_params().
    _, X_test, _, y_test = split_data(features)
    model_a = load_model(model_path)
    params_a = model_a.get_params()
    pred_a = model_a.predict(X_test)
    variant_a = _variant(
        score(model_a, X_test, y_test),
        lock_test_r2,
        {key: params_a[key] for key in config.PARAM_GRID},
    )

    # B: same rows from the clean table; grid-searched here, not saved.
    Xb_train, Xb_test, yb_train, yb_test = split_variant_b(
        pd.read_parquet(clean_path), features
    )
    search_b = train_variant_b(Xb_train, yb_train)
    pred_b = search_b.predict(Xb_test)
    variant_b = _variant(
        score(search_b, Xb_test, yb_test), lock_test_r2, search_b.best_params_
    )

    metrics = {
        "lock_test_r2": lock_test_r2,
        "variant_a": variant_a,
        "variant_b": variant_b,
        "r2_a_minus_b": variant_a["test_r2"] - variant_b["test_r2"],
    }

    # The CI is paired: A and B must be scored on the same rows in the same order.
    if not (y_test.index.equals(yb_test.index) and y_test.equals(yb_test)):
        raise ValueError("Variant A and B test labels differ; the CI needs paired rows")
    ci = paired_bootstrap_r2_diff(
        y_test.to_numpy(),
        pred_a,
        pred_b,
        config.BOOTSTRAP_N_RESAMPLES,
        config.BOOTSTRAP_CONFIDENCE,
        config.RANDOM_STATE,
    )
    # The CI depends on the seed and the chunk size, so record the latter too.
    metrics["r2_a_minus_b_ci"] = {**ci, "chunk_size": BOOTSTRAP_CHUNK_SIZE}
    for name in ("variant_a", "variant_b"):
        logger.info(
            "%s: test R^2 %.6f, RMSE %.0f USD, delta vs lock %+.6f, %s",
            name,
            metrics[name]["test_r2"],
            metrics[name]["test_rmse"],
            metrics[name]["delta_vs_lock"],
            metrics[name]["best_params"],
        )
    logger.info(
        "R^2 A - B: %+.6f, %.0f%% paired bootstrap CI [%+.6f, %+.6f] (%d resamples)",
        metrics["r2_a_minus_b"],
        100 * ci["confidence"],
        ci["low"],
        ci["high"],
        ci["n_resamples"],
    )

    path = Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    written = run()
    logger.info("Wrote %s", written)
