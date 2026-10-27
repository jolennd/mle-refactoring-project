import pandas as pd
import pytest

from kc_housing.cleaning import load_raw


@pytest.mark.integration
def test_load_raw_reads_csv_unchanged(raw_houses, tmp_path):
    """Mirrors notebook cell 7: pd.read_csv of the raw file, no changes."""
    path = tmp_path / "raw.csv"
    raw_houses.to_csv(path, index=False)

    loaded = load_raw(path)

    assert loaded.shape == raw_houses.shape
    assert list(loaded.columns) == list(raw_houses.columns)
    pd.testing.assert_frame_equal(
        loaded, raw_houses.reset_index(drop=True), check_dtype=False
    )
