"""Phase 4 spec step 4: metrics.json and the `make evaluate` entry point (R11).

Public API under test (kc_housing.evaluate):
- ``score(estimator, X_test, y_test) -> dict``: ``{"test_r2": r2_score,
  "test_rmse": root mean squared error in USD}``, plain Python floats.
- ``run(features_path=config.FEATURES_PATH, clean_path=config.CLEAN_PATH,
  model_path=config.MODEL_PATH, lock_path=config.BASELINE_LOCK_PATH,
  metrics_path=config.METRICS_PATH) -> Path``: `make evaluate` runs it via
  ``python -m kc_housing.evaluate``, after `make train`. It is the only code that
  scores the test rows (after the lock, ADR-006).
  Variant A: loads the skops model at ``model_path`` (written by models.run) and scores
  it on the test rows of ``splits.split_data(features)``.
  Variant B: ``split_variant_b(clean, features)`` (same rows), ``train_variant_b`` on
  the clean training rows, scored on the clean test rows (ADR-005).
  Writes JSON to ``metrics_path`` (creating parent dirs) and returns that path.

metrics.json schema (this test defines it; extra keys are allowed):
    lock_test_r2   float, test_r2 read from the lock at ``lock_path``
    variant_a      {"test_r2": float, "test_rmse": float (USD, >= 0),
                    "delta_vs_lock": test_r2 - lock_test_r2,
                    "best_params": {grid key: chosen value}}
    variant_b      same four keys as variant_a
    r2_a_minus_b   float, variant_a.test_r2 - variant_b.test_r2
    r2_a_minus_b_ci {"low": float, "high": float, "n_resamples": int,
                    "confidence": float}: evaluate.paired_bootstrap_r2_diff on the
                    test rows with config.BOOTSTRAP_N_RESAMPLES,
                    config.BOOTSTRAP_CONFIDENCE and seed config.RANDOM_STATE;
                    low <= high (r2_a_minus_b inside it is not guaranteed)
All numbers except best_params values are finite floats.

best_params (both variants) has exactly the keys of config.PARAM_GRID, and each value
is one of that key's grid values, so A - B can be told apart into the leak and a
different grid choice. Variant A's saved model is the fitted best_estimator_ Pipeline
(no best_params_), so A's best_params are the PARAM_GRID keys read from the saved
model's ``get_params()``; B's are ``train_variant_b(...).best_params_``.

Tests write to tmp_path only, never to the config paths.
"""

import inspect
import json
import math

import numpy as np
import pytest
import skops.io as sio
from sklearn.metrics import mean_squared_error, r2_score

from kc_housing import config, models
from kc_housing.evaluate import run, score, split_variant_b, train_variant_b
from kc_housing.splits import split_data

LOCK_TEST_R2 = 0.5  # synthetic lock value; only the delta arithmetic matters here
VARIANT_KEYS = {"test_r2", "test_rmse", "delta_vs_lock"}


@pytest.fixture
def evaluated(model_clean_houses, model_features, tmp_path):
    """Synthetic inputs on disk, models.run() then evaluate.run(); returns the paths."""
    paths = {
        "features_path": tmp_path / "features.parquet",
        "clean_path": tmp_path / "houses_clean.parquet",
        "model_path": tmp_path / "models" / "elastic_net.skops",
        "lock_path": tmp_path / "baseline_lock.json",
        "metrics_path": tmp_path / "results" / "metrics.json",
    }
    model_features.to_parquet(paths["features_path"])
    model_clean_houses.to_parquet(paths["clean_path"])
    paths["lock_path"].write_text(json.dumps({"test_r2": LOCK_TEST_R2}))
    models.run(
        features_path=paths["features_path"],
        model_path=paths["model_path"],
        folds_path=tmp_path / "cv_folds.json",
    )
    paths["written"] = run(
        **{key: value for key, value in paths.items() if key != "written"}
    )
    return paths


@pytest.mark.integration
def test_metrics_json_contract(evaluated):
    """run() writes metrics.json to the given path (defaults from config) with both
    variants' test_r2, test_rmse, delta_vs_lock and best_params (PARAM_GRID keys, grid
    values), the lock value, A - B and its paired bootstrap CI (config resamples and
    confidence, low <= high), all finite floats, and the deltas are the stated
    differences."""
    defaults = inspect.signature(run).parameters
    assert defaults["features_path"].default == config.FEATURES_PATH
    assert defaults["clean_path"].default == config.CLEAN_PATH
    assert defaults["model_path"].default == config.MODEL_PATH
    assert defaults["lock_path"].default == config.BASELINE_LOCK_PATH
    assert defaults["metrics_path"].default == config.METRICS_PATH
    assert evaluated["written"] == evaluated["metrics_path"]

    metrics = json.loads(evaluated["metrics_path"].read_text())

    numbers = [metrics["lock_test_r2"], metrics["r2_a_minus_b"]]
    for variant in ("variant_a", "variant_b"):
        assert VARIANT_KEYS <= set(metrics[variant]), variant
        numbers += [metrics[variant][key] for key in VARIANT_KEYS]
        assert metrics[variant]["test_rmse"] >= 0
        assert metrics[variant]["test_r2"] <= 1
        assert metrics[variant]["delta_vs_lock"] == pytest.approx(
            metrics[variant]["test_r2"] - LOCK_TEST_R2, abs=1e-12
        )
        best_params = metrics[variant]["best_params"]
        assert isinstance(best_params, dict), variant
        assert set(best_params) == set(config.PARAM_GRID), variant
        for key, grid_values in config.PARAM_GRID.items():
            assert best_params[key] in grid_values, (variant, key)
    for value in numbers:
        assert isinstance(value, float) and math.isfinite(value)
    assert metrics["lock_test_r2"] == LOCK_TEST_R2
    assert metrics["r2_a_minus_b"] == pytest.approx(
        metrics["variant_a"]["test_r2"] - metrics["variant_b"]["test_r2"], abs=1e-12
    )

    ci = metrics["r2_a_minus_b_ci"]
    assert {"low", "high", "n_resamples", "confidence"} <= set(ci)
    for key in ("low", "high", "confidence"):
        assert isinstance(ci[key], float) and math.isfinite(ci[key]), key
    assert ci["low"] <= ci["high"]
    assert ci["n_resamples"] == config.BOOTSTRAP_N_RESAMPLES
    assert ci["confidence"] == config.BOOTSTRAP_CONFIDENCE


@pytest.mark.integration
def test_metrics_values_match_rescored_variants(
    evaluated, model_clean_houses, model_features
):
    """score() is sklearn's R^2 and RMSE; metrics.json variant A equals score() of the
    saved model on split_data's test rows, and variant B equals score() of
    train_variant_b on the same rows from the clean table; best_params are the saved
    model's grid-key params (A) and train_variant_b's best_params_ (B)."""
    metrics = json.loads(evaluated["metrics_path"].read_text())

    _, X_test, _, y_test = split_data(model_features)
    path = evaluated["model_path"]
    model_a = sio.load(path, trusted=sio.get_untrusted_types(file=path))
    pred_a = model_a.predict(X_test)
    scored_a = score(model_a, X_test, y_test)
    assert scored_a["test_r2"] == pytest.approx(r2_score(y_test, pred_a), abs=1e-12)
    assert scored_a["test_rmse"] == pytest.approx(
        math.sqrt(mean_squared_error(y_test, pred_a)), rel=1e-12
    )

    Xb_train, Xb_test, yb_train, yb_test = split_variant_b(
        model_clean_houses, model_features
    )
    np.testing.assert_array_equal(yb_test.to_numpy(), y_test.to_numpy())
    search_b = train_variant_b(Xb_train, yb_train)
    scored_b = score(search_b, Xb_test, yb_test)

    for variant, scored in (("variant_a", scored_a), ("variant_b", scored_b)):
        for key in ("test_r2", "test_rmse"):
            assert metrics[variant][key] == pytest.approx(scored[key], rel=1e-12), (
                variant,
                key,
            )

    params_a = model_a.get_params()
    assert metrics["variant_a"]["best_params"] == {
        key: params_a[key] for key in config.PARAM_GRID
    }
    assert metrics["variant_b"]["best_params"] == search_b.best_params_
