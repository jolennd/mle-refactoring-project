import logging

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


@pytest.mark.parametrize(
    ("label", "expected_year"),
    [
        pytest.param(101, 1991, id="renovated-uses-yr_renovated"),
        pytest.param(100, 1955, id="never-renovated-zero-uses-yr_built"),
        pytest.param(105, 2001, id="missing-renovation-nan-uses-yr_built"),
    ],
)
def test_last_known_change_is_renovation_else_build_year(
    raw_houses, label, expected_year
):
    """Mirrors notebook cells 41-42: yr_renovated if not NaN/0, else yr_built (int)."""
    result = clean(raw_houses)

    assert label in result.index
    assert pd.api.types.is_integer_dtype(result["last_known_change"])
    assert result.loc[label, "last_known_change"] == expected_year


def test_drops_yr_built_and_yr_renovated(raw_houses):
    """Mirrors notebook cell 43: yr_built and yr_renovated are dropped."""
    result = clean(raw_houses)

    assert "yr_built" not in result.columns
    assert "yr_renovated" not in result.columns


def test_clean_does_not_mutate_input(raw_houses):
    """R3 (spec convention 2): clean copies its input and never mutates it."""
    original = raw_houses.copy(deep=True)

    clean(raw_houses)

    pd.testing.assert_frame_equal(original, raw_houses)


def test_clean_keeps_index(raw_houses):
    """R3: the original index is kept minus dropped rows, same order, no reset."""
    dropped = raw_houses["bedrooms"] == config.IMPLAUSIBLE_BEDROOMS
    expected_index = raw_houses.index[~dropped]

    result = clean(raw_houses)

    pd.testing.assert_index_equal(result.index, expected_index)


def test_clean_drops_only_rule_rows(raw_houses, caplog):
    """R2 (ADR-003): drop only rule rows and log the drop count once at INFO."""
    matches_rule = raw_houses["bedrooms"] == config.IMPLAUSIBLE_BEDROOMS
    n_dropped = int(matches_rule.sum())

    with caplog.at_level(logging.INFO, logger="kc_housing.cleaning"):
        result = clean(raw_houses)

    assert len(result) == len(raw_houses) - n_dropped
    assert set(result.index) == set(raw_houses.index[~matches_rule])
    count_records = [
        r
        for r in caplog.records
        if r.name == "kc_housing.cleaning"
        and r.levelno == logging.INFO
        and str(n_dropped) in r.getMessage()
    ]
    assert len(count_records) == 1
