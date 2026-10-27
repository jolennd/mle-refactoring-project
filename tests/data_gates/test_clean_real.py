import pytest

from kc_housing.cleaning import clean

# Documented in docs/02_data_sources.md: the single 33-bedroom row in the real CSV.
IMPLAUSIBLE_ROW_LABEL = 15856


@pytest.mark.data
def test_real_data_loses_exactly_one_row(raw_csv):
    """R2: on the real CSV, clean() drops exactly one row, label 15856."""
    cleaned = clean(raw_csv)

    assert len(cleaned) == len(raw_csv) - 1
    missing = set(raw_csv.index) - set(cleaned.index)
    assert missing == {IMPLAUSIBLE_ROW_LABEL}
