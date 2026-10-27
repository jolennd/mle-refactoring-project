import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from kc_housing import config
from kc_housing.features import CenterDistance, SelectColumns, WaterDistance


@pytest.fixture
def located_houses() -> pd.DataFrame:
    """Three synthetic houses with a non-default index; the first sits at the center."""
    return pd.DataFrame(
        {
            "lat": [config.CENTER_LAT, 47.5112, 47.7210],
            "long": [config.CENTER_LONG, -122.2570, -122.3190],
            "bedrooms": [3, 2, 4],
        },
        index=[501, 307, 942],
    )


def _notebook_center_distance(lat: pd.Series, long: pd.Series) -> pd.Series:
    """Hand copy of notebook cell 60 with constants taken from config (ADR-002)."""
    delta_lat = np.absolute(config.CENTER_LAT - lat)
    delta_long = np.absolute(config.CENTER_LONG - long)
    return (
        ((delta_long * np.cos(np.radians(config.CENTER_COS_LAT))) ** 2 + delta_lat**2)
        ** (1 / 2)
        * 2
        * np.pi
        * config.EARTH_RADIUS_KM
        / 360
    )


def test_center_distance_is_zero_at_center(located_houses):
    """Mirrors notebook cell 60: a house at (CENTER_LAT, CENTER_LONG) is 0 km away."""
    result = CenterDistance().fit_transform(located_houses)

    assert result.loc[501, "center_distance"] == 0
    assert (result.loc[[307, 942], "center_distance"] > 0).all()


def test_center_distance_matches_notebook_formula(located_houses):
    """Mirrors notebook cell 60: km from the center, cos(CENTER_COS_LAT) correction."""
    result = CenterDistance().fit_transform(located_houses)

    expected = _notebook_center_distance(located_houses["lat"], located_houses["long"])
    np.testing.assert_allclose(
        result["center_distance"].to_numpy(),
        expected.to_numpy(),
        rtol=config.PARITY_RTOL,
        atol=config.PARITY_ATOL,
    )


def test_center_distance_does_not_mutate_input_and_keeps_index(located_houses):
    """ADR-004: returns a DataFrame, keeps the index, never mutates input.

    Also: fit returns self.
    """
    original = located_houses.copy()
    transformer = CenterDistance()

    assert transformer.fit(located_houses) is transformer
    result = transformer.transform(located_houses)

    pd.testing.assert_frame_equal(located_houses, original)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_index_equal(result.index, original.index)
    assert "center_distance" in result.columns
    assert "delta_lat" not in result.columns
    assert "delta_long" not in result.columns


@pytest.fixture
def waterfront_houses() -> pd.DataFrame:
    """Six synthetic houses, non-default index; 401, 117 and 865 are waterfront.

    waterfront is float, as after fillna(0) in cleaning.
    """
    return pd.DataFrame(
        {
            "lat": [47.6100, 47.5112, 47.7210, 47.4530, 47.6800, 47.3900],
            "long": [-122.4100, -122.2570, -122.3190, -122.1500, -122.0600, -122.3300],
            "waterfront": [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        },
        index=[401, 233, 117, 958, 612, 865],
    )


def _notebook_dist(long, lat, ref_long, ref_lat):
    """Hand copy of notebook cell 67 with EARTH_RADIUS_KM from config.

    The longitude correction uses cos(ref_lat), the reference house's latitude.
    """
    delta_long = long - ref_long
    delta_lat = lat - ref_lat
    delta_long_corr = delta_long * np.cos(np.radians(ref_lat))
    return (
        ((delta_long_corr) ** 2 + (delta_lat) ** 2) ** (1 / 2)
        * 2
        * np.pi
        * config.EARTH_RADIUS_KM
        / 360
    )


def _notebook_water_distance(
    houses: pd.DataFrame, reference: pd.DataFrame
) -> np.ndarray:
    """Hand copy of notebook cells 68-69: min km to any waterfront reference house."""
    water_list = reference.query("waterfront == 1")
    return np.array(
        [
            min(
                _notebook_dist(long, lat, ref_long, ref_lat)
                for ref_long, ref_lat in zip(
                    water_list["long"], water_list["lat"], strict=True
                )
            )
            for long, lat in zip(houses["long"], houses["lat"], strict=True)
        ]
    )


def test_water_distance_is_zero_for_waterfront_houses(waterfront_houses):
    """Mirrors notebook cells 68-69 (ADR-005): waterfront houses are 0 km from water.

    They are in the fitted reference set, so their water_distance is 0 km.
    """
    result = WaterDistance().fit_transform(waterfront_houses)

    assert (result.loc[[401, 117, 865], "water_distance"] == 0).all()
    assert (result.loc[[233, 958, 612], "water_distance"] > 0).all()


def test_water_distance_is_min_over_reference_houses(waterfront_houses):
    """Mirrors notebook cells 67-69: min over waterfront houses of dist().

    The longitude correction uses cos(ref_lat), as in the notebook.
    """
    result = WaterDistance().fit_transform(waterfront_houses)

    expected = _notebook_water_distance(waterfront_houses, waterfront_houses)
    np.testing.assert_allclose(
        result.loc[waterfront_houses.index, "water_distance"].to_numpy(),
        expected,
        rtol=config.PARITY_RTOL,
        atol=config.PARITY_ATOL,
    )


def test_water_distance_uses_reference_from_fit(waterfront_houses):
    """ADR-005: transform measures against the waterfront houses seen in fit only.

    Also ADR-004: fit returns self, output is a DataFrame with the input's index,
    input not mutated.
    """
    new_houses = pd.DataFrame(
        {
            # 72 is flagged waterfront here but was not in the fit data.
            "lat": [47.5600, 47.6400, 47.4000],
            "long": [-122.2000, -122.3800, -122.1000],
            "waterfront": [1.0, 0.0, 0.0],
        },
        index=[72, 15, 340],
    )
    original = new_houses.copy()
    transformer = WaterDistance()

    assert transformer.fit(waterfront_houses) is transformer
    result = transformer.transform(new_houses)

    pd.testing.assert_frame_equal(new_houses, original)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_index_equal(result.index, original.index)
    # The new waterfront house is not in the fitted reference, so it is not 0 km away.
    assert result.loc[72, "water_distance"] > 0
    expected = _notebook_water_distance(new_houses, waterfront_houses)
    np.testing.assert_allclose(
        result["water_distance"].to_numpy(),
        expected,
        rtol=config.PARITY_RTOL,
        atol=config.PARITY_ATOL,
    )


@pytest.fixture
def feature_frame() -> pd.DataFrame:
    """All config.FEATURES columns in shuffled order plus extras, non-default index.

    Extras are the id, the label, a leakage column, the raw date and an
    intermediate of cell 60 (delta_lat), none of which may reach the feature set.
    """
    rng = np.random.default_rng(0)
    n = 4
    columns = list(rng.permutation(config.FEATURES))
    frame = pd.DataFrame(
        {col: rng.uniform(0, 100, n) for col in columns},
        index=[810, 23, 457, 96],
    )
    frame.insert(3, "price", rng.uniform(200_000, 1_500_000, n))
    frame.insert(0, "id", [11, 12, 13, 14])
    frame["sqft_price"] = frame["price"] / 1000
    frame.insert(7, "date", ["10/13/2014"] * n)
    frame["delta_lat"] = rng.uniform(0, 1, n)
    assert list(frame.columns[: len(config.FEATURES)]) != config.FEATURES
    return frame


def test_select_columns_returns_config_features_in_order(feature_frame):
    """Mirrors notebook cells 78-80/101: the model sees exactly config.FEATURES.

    ADR-004: returns a DataFrame in config.FEATURES order, keeps the index, never
    mutates the input; extras (id, price, sqft_price, date, delta_lat) are dropped.
    """
    original = feature_frame.copy()

    result = SelectColumns(columns=config.FEATURES).fit_transform(feature_frame)

    pd.testing.assert_frame_equal(feature_frame, original)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == config.FEATURES
    pd.testing.assert_index_equal(result.index, original.index)
    pd.testing.assert_frame_equal(result, original[config.FEATURES])


def test_feature_set_excludes_leakage_columns(feature_frame):
    """R6: no leakage column (price, sqft_price) is a feature.

    Neither in config.FEATURES nor in SelectColumns output, even when the input
    carries them.
    """
    assert set(config.LEAKAGE_COLUMNS).issubset(feature_frame.columns)
    assert not set(config.LEAKAGE_COLUMNS) & set(config.FEATURES)

    result = SelectColumns(config.FEATURES).fit_transform(feature_frame)

    assert not set(config.LEAKAGE_COLUMNS) & set(result.columns)


@pytest.mark.parametrize(
    ("transformer", "data_fixture"),
    [
        (CenterDistance(), "waterfront_houses"),
        (WaterDistance(), "waterfront_houses"),
        (SelectColumns(config.FEATURES), "feature_frame"),
    ],
    ids=["center_distance", "water_distance", "select_columns"],
)
def test_transformers_are_clonable_and_picklable(transformer, data_fixture, request):
    """ADR-004 and spec guarantees: usable in sklearn pipelines and GridSearchCV.

    clone keeps get_params (SelectColumns keeps config.FEATURES); a fitted
    transformer (incl. WaterDistance with its fitted reference, ADR-005) survives
    pickle and gives an identical transform.
    """
    data = request.getfixturevalue(data_fixture)

    cloned = clone(transformer)

    assert type(cloned) is type(transformer)
    assert cloned.get_params() == transformer.get_params()
    if isinstance(transformer, SelectColumns):
        assert list(cloned.get_params()["columns"]) == config.FEATURES

    fitted = cloned.fit(data)
    restored = pickle.loads(pickle.dumps(fitted))

    pd.testing.assert_frame_equal(restored.transform(data), fitted.transform(data))


def test_water_distance_transform_before_fit_raises_not_fitted(waterfront_houses):
    """ADR-004/005: transform needs the reference from fit; sklearn's NotFittedError."""
    with pytest.raises(NotFittedError):
        WaterDistance().transform(waterfront_houses)


def test_water_distance_fit_without_waterfront_houses_raises(waterfront_houses):
    """ADR-005: an empty waterfront reference makes water_distance undefined.

    fit must fail loudly, naming waterfront, instead of failing later in transform.
    """
    no_water = waterfront_houses.assign(waterfront=0.0)

    with pytest.raises(ValueError, match="waterfront"):
        WaterDistance().fit(no_water)
