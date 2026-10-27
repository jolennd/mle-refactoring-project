import pandas as pd
import pytest

from kc_housing import config
from kc_housing.cleaning import clean


def test_drops_rows_with_implausible_bedrooms(raw_houses):
    """Mirrors notebook cell 28: drop the implausible-bedrooms row by rule (ADR-003)."""
    implausible_labels = raw_houses.index[
        raw_houses["bedrooms"] == config.IMPLAUSIBLE_BEDROOMS
    ]
    assert list(implausible_labels) == [102]  # fixture sanity check

    result = clean(raw_houses)

    assert 102 not in result.index
    assert not (result["bedrooms"] == config.IMPLAUSIBLE_BEDROOMS).any()


def test_sqft_basement_is_living_minus_above(raw_houses):
    """Mirrors notebook cell 30: sqft_basement = sqft_living - sqft_above."""
    result = clean(raw_houses)

    assert pd.api.types.is_numeric_dtype(result["sqft_basement"])
    expected = result["sqft_living"] - result["sqft_above"]
    pd.testing.assert_series_equal(
        result["sqft_basement"], expected, check_dtype=False, check_names=False
    )
    had_question_mark = raw_houses.index[raw_houses["sqft_basement"] == "?"]
    surviving = had_question_mark.intersection(result.index)
    assert len(surviving) > 0
    assert result.loc[surviving, "sqft_basement"].notna().all()


@pytest.mark.parametrize("column", config.FILL_ZERO_COLUMNS, ids=lambda c: c)
def test_view_and_waterfront_missing_become_zero(raw_houses, column):
    """Mirrors notebook cells 35/38: missing view and waterfront are filled with 0."""
    result = clean(raw_houses)

    assert result[column].notna().all()
    surviving = raw_houses.index.intersection(result.index)
    raw = raw_houses.loc[surviving, column]
    was_nan = raw.isna()
    assert was_nan.any()  # fixture sanity check
    assert (result.loc[raw.index[was_nan], column] == 0).all()
    pd.testing.assert_series_equal(
        result.loc[raw.index[~was_nan], column],
        raw[~was_nan],
        check_dtype=False,
    )
