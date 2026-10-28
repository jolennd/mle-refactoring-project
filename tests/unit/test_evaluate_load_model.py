"""Phase 4: evaluate.load_model only loads skops files with trusted types (R11, R12).

Public API under test (kc_housing.evaluate):
- ``load_model(path=config.MODEL_PATH)``: reads ``skops.io.get_untrusted_types`` of the
  file; if any type is outside ``TRUSTED_MODULES`` (sklearn, numpy, kc_housing) it
  raises ValueError naming the refused types, otherwise it loads the model.

Tests write to tmp_path only.
"""

import numpy as np
import pytest
import skops.io as sio

from kc_housing.evaluate import load_model
from kc_housing.models import build_model


class ForeignPayload:
    """A type outside sklearn, numpy and kc_housing; skops lists it as untrusted."""

    def __init__(self):
        self.note = "not a model part"


def _fitted_model():
    """build_model() fitted on tiny synthetic data (fixed seed)."""
    rng = np.random.default_rng(11)
    X = rng.uniform(0, 10, size=(30, 3))
    y = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(0, 0.1, 30)
    return build_model().fit(X, y), X


def test_load_model_refuses_foreign_types_and_loads_real_model(tmp_path):
    """A skops file holding a type outside TRUSTED_MODULES raises ValueError naming
    that type; a fitted build_model() dumped with skops loads and predicts the same."""
    model, X = _fitted_model()

    foreign_path = tmp_path / "foreign.skops"
    sio.dump({"model": model, "extra": ForeignPayload()}, foreign_path)
    with pytest.raises(ValueError, match="ForeignPayload"):
        load_model(foreign_path)

    model_path = tmp_path / "elastic_net.skops"
    sio.dump(model, model_path)
    loaded = load_model(model_path)
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))
