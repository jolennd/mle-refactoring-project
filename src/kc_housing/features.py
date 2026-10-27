"""Feature engineering as sklearn transformers (notebook cell 60, ADR-004)."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from kc_housing import config


class CenterDistance(BaseEstimator, TransformerMixin):
    """Add `center_distance`, km from the wealth center (notebook cell 60)."""

    def fit(self, X: pd.DataFrame, y=None) -> "CenterDistance":
        """Stateless: nothing to learn."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `X` with `center_distance`; the input is not modified."""
        result = X.copy()
        delta_lat = np.absolute(config.CENTER_LAT - result["lat"])
        delta_long = np.absolute(config.CENTER_LONG - result["long"])
        # Same operation order as the notebook so floats match (ADR-002).
        result["center_distance"] = (
            (
                (delta_long * np.cos(np.radians(config.CENTER_COS_LAT))) ** 2
                + delta_lat**2
            )
            ** (1 / 2)
            * 2
            * np.pi
            * config.EARTH_RADIUS_KM
            / 360
        )
        return result
