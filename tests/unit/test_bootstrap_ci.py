"""Phase 4: paired bootstrap confidence interval for test R^2 A - B (R11, ADR-005/006).

Public API under test (kc_housing.evaluate), proposed; config values await the human:
- ``paired_bootstrap_r2_diff(y_true, pred_a, pred_b, n_resamples, confidence, seed)
  -> {"low": float, "high": float, "n_resamples": int, "confidence": float}``.
  Each of ``n_resamples`` draws resamples the test row positions with replacement
  ONCE (numpy Generator seeded with ``seed``) and computes
  ``r2_score(y[idx], pred_a[idx]) - r2_score(y[idx], pred_b[idx])`` on that same draw
  (paired). ``low`` and ``high`` are the percentile interval at ``confidence``
  (the (1 - confidence) / 2 and (1 + confidence) / 2 quantiles of the draws).
  ``n_resamples`` and ``confidence`` are echoed back.
- run() uses config.BOOTSTRAP_N_RESAMPLES (10000), config.BOOTSTRAP_CONFIDENCE
  (0.95) and seed config.RANDOM_STATE; see test_metrics_json_contract.

The function is looked up at call time, so a missing function fails each test with
AttributeError instead of breaking collection of the whole suite.
"""

import math

from sklearn.metrics import r2_score

from kc_housing import config, evaluate

N_RESAMPLES = 200  # small, to keep unit tests fast
CONFIDENCE = 0.95


def _ci(y_true, pred_a, pred_b, seed=config.RANDOM_STATE):
    return evaluate.paired_bootstrap_r2_diff(
        y_true, pred_a, pred_b, N_RESAMPLES, CONFIDENCE, seed
    )


def test_bootstrap_ci_is_deterministic_for_seed(bootstrap_predictions):
    """The same inputs and seed give the identical interval twice."""
    assert _ci(*bootstrap_predictions) == _ci(*bootstrap_predictions)


def test_bootstrap_ci_is_paired_zero_for_identical_predictions(bootstrap_predictions):
    """Paired resampling: if pred_a equals pred_b every draw's difference is 0, so the
    interval is exactly [0, 0] (unpaired draws would give a nonzero spread)."""
    y_true, pred_a, _ = bootstrap_predictions
    ci = _ci(y_true, pred_a, pred_a.copy())
    assert ci["low"] == 0.0
    assert ci["high"] == 0.0


def test_bootstrap_ci_contains_full_set_difference(bootstrap_predictions):
    """On the constructed example (A clearly better than B) the full-set point
    estimate r2(A) - r2(B) lies within [low, high], and the interval is above 0."""
    y_true, pred_a, pred_b = bootstrap_predictions
    point = r2_score(y_true, pred_a) - r2_score(y_true, pred_b)
    ci = _ci(y_true, pred_a, pred_b)
    assert ci["low"] <= point <= ci["high"]
    assert ci["low"] > 0


def test_bootstrap_ci_schema(bootstrap_predictions):
    """Exactly low, high, n_resamples, confidence; finite floats with low <= high;
    n_resamples and confidence echo the arguments."""
    ci = _ci(*bootstrap_predictions)
    assert set(ci) == {"low", "high", "n_resamples", "confidence"}
    for key in ("low", "high"):
        assert isinstance(ci[key], float) and math.isfinite(ci[key]), key
    assert ci["low"] <= ci["high"]
    assert ci["n_resamples"] == N_RESAMPLES and isinstance(ci["n_resamples"], int)
    assert ci["confidence"] == CONFIDENCE
    assert isinstance(ci["confidence"], float)
