"""Feature engineering as sklearn transformers (cells 60, 67-70; ADR-004, ADR-005)."""

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


class WaterDistance(BaseEstimator, TransformerMixin):
    """Add `water_distance`, km to the nearest waterfront house (cells 67-70)."""

    def fit(self, X: pd.DataFrame, y=None) -> "WaterDistance":
        """Store the coordinates of the waterfront houses seen in fit."""
        water = X[X["waterfront"] == 1]
        self.reference_long_ = water["long"].to_numpy()
        self.reference_lat_ = water["lat"].to_numpy()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `X` with `water_distance`; the input is not modified."""
        result = X.copy()
        long = result["long"].to_numpy()[:, np.newaxis]
        lat = result["lat"].to_numpy()[:, np.newaxis]
        ref_long = self.reference_long_[np.newaxis, :]
        ref_lat = self.reference_lat_[np.newaxis, :]
        # (n, m) km matrix; n*m floats (21,596 x ~146 ~ 3.2M ~ 25 MB), no chunking.
        # Same operation order as the notebook's dist() so floats match (ADR-002).
        delta_long = long - ref_long
        delta_lat = lat - ref_lat
        delta_long_corr = delta_long * np.cos(np.radians(ref_lat))
        distances = (
            ((delta_long_corr) ** 2 + (delta_lat) ** 2) ** (1 / 2)
            * 2
            * np.pi
            * config.EARTH_RADIUS_KM
            / 360
        )
        result["water_distance"] = distances.min(axis=1)
        return result
