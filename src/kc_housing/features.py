"""Feature engineering transformers (cells 60, 67-70, 78-80, 101; ADR-004, ADR-005)."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from kc_housing import config

logger = logging.getLogger(__name__)


def _km(delta_long, delta_lat, cos_lat_deg):
    """Equirectangular distance in km (notebook cells 60 and 67).

    ((delta_long * cos(radians(cos_lat_deg)))**2 + delta_lat**2) ** (1/2)
    * 2 * pi * EARTH_RADIUS_KM / 360, in the notebook's operation order (ADR-002).
    On full arrays numpy may round the last bit differently from the notebook's
    scalar loop, so water_distance matches to ~1e-14, well inside PARITY_ATOL.
    """
    return (
        ((delta_long * np.cos(np.radians(cos_lat_deg))) ** 2 + delta_lat**2) ** (1 / 2)
        * 2
        * np.pi
        * config.EARTH_RADIUS_KM
        / 360
    )


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
        result["center_distance"] = _km(delta_long, delta_lat, config.CENTER_COS_LAT)
        return result


class WaterDistance(BaseEstimator, TransformerMixin):
    """Add `water_distance`, km to the nearest waterfront house (cells 67-70)."""

    def fit(self, X: pd.DataFrame, y=None) -> "WaterDistance":
        """Store the coordinates of the waterfront houses seen in fit."""
        water = X[X["waterfront"] == 1]
        if water.empty:
            raise ValueError(
                "WaterDistance.fit needs at least one waterfront house "
                "(waterfront == 1); got none."
            )
        self.reference_long_ = water["long"].to_numpy()
        self.reference_lat_ = water["lat"].to_numpy()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `X` with `water_distance`; the input is not modified."""
        check_is_fitted(self)
        result = X.copy()
        long = result["long"].to_numpy()[:, np.newaxis]
        lat = result["lat"].to_numpy()[:, np.newaxis]
        ref_long = self.reference_long_[np.newaxis, :]
        ref_lat = self.reference_lat_[np.newaxis, :]
        # (n, m) km matrix; n*m floats (21,596 x ~146 ~ 3.2M ~ 25 MB), no chunking.
        distances = _km(long - ref_long, lat - ref_lat, ref_lat)
        result["water_distance"] = distances.min(axis=1)
        return result


class SelectColumns(BaseEstimator, TransformerMixin):
    """Keep `columns` in the given order (notebook cells 78-80 and 101)."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None) -> "SelectColumns":
        """Store the columns to keep; raise KeyError if any is missing from `X`."""
        missing = [col for col in self.columns if col not in X.columns]
        if missing:
            raise KeyError(f"SelectColumns.fit: columns missing from X: {missing}")
        self.columns_ = list(self.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame with only `columns_`, in order; the index is kept."""
        check_is_fitted(self)
        return X[self.columns_].copy()


def with_id_and_target(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Return `df`'s id and price columns, then `features`, aligned on the index."""
    return pd.concat([df[[config.ID_COL, config.TARGET]], features], axis=1)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return id, price, then config.FEATURES; the index is kept, `df` not modified.

    WaterDistance is fitted on `df` itself, the full clean table, as the notebook
    does (cells 67-70; ADR-005).
    """
    with_center = CenterDistance().fit_transform(df)
    with_water = WaterDistance().fit_transform(with_center)
    selected = SelectColumns(config.FEATURES).fit_transform(with_water)
    return with_id_and_target(df, selected)


def write_features(
    clean_path: str | Path = config.CLEAN_PATH,
    out_path: str | Path = config.FEATURES_PATH,
) -> Path:
    """Read the clean table, build the features and write them as parquet."""
    df = build_features(pd.read_parquet(clean_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return Path(out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    written = write_features()
    logger.info("Wrote %s (%d rows)", written, len(pd.read_parquet(written)))
