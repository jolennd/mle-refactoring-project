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


@pytest.fixture
def feature_table() -> pd.DataFrame:
    """Synthetic Phase 3 feature table (id, price, config.FEATURES) for split tests.

    20 rows, fixed seed, non-default index, one repeat sale (id 1000 twice) and an extra
    leakage column `sqft_price` that must never reach X.
    """
    rng = np.random.default_rng(4)
    n = 20
    frame = pd.DataFrame(
        {col: rng.uniform(0, 100, n) for col in config.FEATURES},
        index=range(500, 500 + n),
    )
    frame.insert(0, config.TARGET, rng.integers(200_000, 1_500_000, n).astype(float))
    ids = [*range(1000, 1000 + n - 1), 1000]  # last row resells house 1000
    frame.insert(0, config.ID_COL, ids)
    frame["sqft_price"] = frame[config.TARGET] / frame["sqft_living"]
    return frame


@pytest.fixture
def model_clean_houses() -> pd.DataFrame:
    """Synthetic clean table (the cleaning contract's columns) for variant B tests.

    40 rows, fixed seed, non-default index 300-339. Every third row is a waterfront
    house, so the config split puts waterfront houses on both sides and every
    GridSearchCV training fold of the training rows has one (WaterDistance.fit needs
    at least one). Price grows with sqft_living and grade so R^2 is meaningful.
    """
    rng = np.random.default_rng(7)
    n = 40
    sqft_living = rng.integers(800, 4000, n)
    sqft_basement = rng.integers(0, 600, n)
    grade = rng.integers(5, 12, n)
    frame = pd.DataFrame(
        {
            "id": np.arange(2000, 2000 + n),
            "date": pd.array(["10/13/2014"] * n, dtype="string"),
            "price": (
                150.0 * sqft_living + 40_000.0 * grade + rng.normal(0, 50_000, n)
            ),
            "bedrooms": rng.integers(1, 6, n),
            "bathrooms": rng.choice([1.0, 1.5, 2.25, 3.0], n),
            "sqft_living": sqft_living,
            "sqft_lot": rng.integers(2000, 20000, n),
            "floors": rng.choice([1.0, 2.0], n),
            "waterfront": [1.0 if i % 3 == 0 else 0.0 for i in range(n)],
            "view": rng.choice([0.0, 1.0, 2.0, 3.0, 4.0], n),
            "condition": rng.integers(1, 6, n),
            "grade": grade,
            "sqft_above": sqft_living - sqft_basement,
            "sqft_basement": sqft_basement,
            "zipcode": rng.choice([98178, 98125, 98028], n),
            "lat": rng.uniform(47.2, 47.8, n),
            "long": rng.uniform(-122.5, -121.8, n),
            "sqft_living15": rng.integers(800, 4000, n),
            "sqft_lot15": rng.integers(2000, 20000, n),
            "last_known_change": rng.integers(1900, 2015, n),
        },
        index=range(300, 300 + n),
    )
    return frame


@pytest.fixture
def model_features(model_clean_houses) -> pd.DataFrame:
    """Variant A feature table of model_clean_houses, as pipeline.run() builds it.

    id, price, then config.FEATURES; the feature pipeline is fitted on all 40 rows
    (ADR-005), and the index labels equal model_clean_houses' labels.
    """
    from kc_housing.features import with_id_and_target
    from kc_housing.pipeline import build_feature_pipeline

    features = build_feature_pipeline().fit_transform(model_clean_houses)
    return with_id_and_target(model_clean_houses, features)


@pytest.fixture
def bootstrap_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic (y_true, pred_a, pred_b) for the paired bootstrap, fixed seed.

    200 rows; A's errors are clearly smaller than B's, so r2(A) - r2(B) > 0 and no
    resample has a constant y_true.
    """
    rng = np.random.default_rng(21)
    n = 200
    y_true = rng.uniform(200_000, 1_500_000, n)
    pred_a = y_true + rng.normal(0, 50_000, n)
    pred_b = y_true + rng.normal(0, 150_000, n)
    return y_true, pred_a, pred_b
