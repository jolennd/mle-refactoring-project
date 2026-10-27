import numpy as np
import pandas as pd
import pytest

from kc_housing import config
from kc_housing.features import CenterDistance


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

    expected = _notebook_center_distance(
        located_houses["lat"], located_houses["long"]
    )
    np.testing.assert_allclose(
        result["center_distance"].to_numpy(),
        expected.to_numpy(),
        rtol=config.PARITY_RTOL,
        atol=config.PARITY_ATOL,
    )


def test_center_distance_does_not_mutate_input_and_keeps_index(located_houses):
    """ADR-004: returns a DataFrame, keeps the index, never mutates input, fit returns self."""
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
