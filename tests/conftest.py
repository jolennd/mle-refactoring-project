import numpy as np
import pandas as pd
import pytest

from kc_housing import config
from kc_housing.cleaning import load_raw


@pytest.fixture
def raw_houses() -> pd.DataFrame:
    """Small synthetic raw table with the CSV's columns, fixed seed, non-default index.

    Includes the cases the cleaning rules handle: one 33-bedroom row, missing and zero
    `yr_renovated`, missing `view` and `waterfront`, a "?" in `sqft_basement`, and two
    waterfront houses.
    """
    rng = np.random.default_rng(0)
    n = 8
    sqft_living = rng.integers(800, 4000, n)
    sqft_above = sqft_living - rng.integers(0, 600, n)
    frame = pd.DataFrame(
        {
            # 101 sells twice, as repeat sales do in the real data
            "id": [101, 102, 103, 104, 105, 106, 107, 101],
            "date": ["10/13/2014"] * n,
            "price": rng.integers(200_000, 1_500_000, n).astype(float),
            "bedrooms": [3, 2, 33, 4, 3, 5, 1, 3],
            "bathrooms": rng.choice([1.0, 1.5, 2.25, 3.0], n),
            "sqft_living": sqft_living,
            "sqft_lot": rng.integers(2000, 20000, n),
            "floors": rng.choice([1.0, 2.0], n),
            "waterfront": [0.0, np.nan, 0.0, 1.0, 0.0, 1.0, np.nan, 0.0],
            "view": [0.0, 2.0, np.nan, 4.0, 0.0, 3.0, 0.0, np.nan],
            "condition": rng.integers(1, 6, n),
            "grade": rng.integers(5, 12, n),
            "sqft_above": sqft_above,
            "sqft_basement": ["0.0", "?", "400.0", "0.0", "?", "0.0", "0.0", "0.0"],
            "yr_built": [1955, 1951, 1933, 1965, 1987, 2001, 1995, 1955],
            "yr_renovated": [0.0, 1991.0, np.nan, 0.0, 2005.0, np.nan, 0.0, 0.0],
            "zipcode": rng.choice([98178, 98125, 98028], n),
            "lat": rng.uniform(47.2, 47.8, n),
            "long": rng.uniform(-122.5, -121.8, n),
            "sqft_living15": rng.integers(800, 4000, n),
            "sqft_lot15": rng.integers(2000, 20000, n),
        },
        index=range(100, 100 + n),
    )
    return frame


@pytest.fixture
def raw_csv() -> pd.DataFrame:
    """The committed raw CSV via the real load path; skip with a reason if missing."""
    if not config.RAW_DATA_PATH.exists():
        pytest.skip(f"{config.RAW_DATA_PATH} not found: restore the committed raw CSV")
    return load_raw(config.RAW_DATA_PATH)


@pytest.fixture
def processed_features() -> pd.DataFrame:
    """Real feature table for data-gate tests; skip with a reason until it exists."""
    if not config.FEATURES_PATH.exists():
        pytest.skip(f"{config.FEATURES_PATH} not found: run `make features` first")
    return pd.read_parquet(config.FEATURES_PATH)
