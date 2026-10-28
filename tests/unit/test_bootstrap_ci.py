"""Phase 4: paired bootstrap confidence interval for test R^2 A - B (R11, ADR-005/006).

Public API under test (kc_housing.evaluate); config values approved by the human
(config.BOOTSTRAP_N_RESAMPLES 10000, config.BOOTSTRAP_CONFIDENCE 0.95, seed
config.RANDOM_STATE; ADR-006):
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

import numpy as np
import pytest
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


def test_bootstrap_chunks_match_r2_score_loop(bootstrap_predictions):
    """Characterization: the chunked, vectorised bootstrap equals a plain r2_score loop.

    Pins (1) the chunk loop, including a partial last chunk: chunk_size=7 and
    n_resamples=30 give chunks of 7, 7, 7, 7, 2 drawn from one generator in that
    order, and (2) the algebraic shortcut (SS_tot from centred sums, the SSE_B - SSE_A
    over SS_tot difference) equals sklearn's r2_score(A) - r2_score(B) on every draw.
    The reference below draws the same chunk shapes from the same seeded generator.
    """
    y_full, a_full, b_full = bootstrap_predictions
    y, a, b = y_full[:20], a_full[:20], b_full[:20]
    n, n_resamples, chunk_size, seed = y.size, 30, 7, config.RANDOM_STATE

    rng = np.random.default_rng(seed)
    expected_diffs = []
    for size in (7, 7, 7, 7, 2):
        idx = rng.integers(0, n, size=(size, n), dtype=np.int32)
        expected_diffs += [r2_score(y[i], a[i]) - r2_score(y[i], b[i]) for i in idx]
    assert len(expected_diffs) == n_resamples
    low, high = np.quantile(
        expected_diffs, [(1 - CONFIDENCE) / 2, (1 + CONFIDENCE) / 2]
    )

    ci = evaluate.paired_bootstrap_r2_diff(
        y, a, b, n_resamples, CONFIDENCE, seed, chunk_size=chunk_size
    )
    assert ci["low"] == pytest.approx(low, abs=1e-12)
    assert ci["high"] == pytest.approx(high, abs=1e-12)
