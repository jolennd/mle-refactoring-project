import pandas as pd
import pytest

from kc_housing import cleaning
from kc_housing.cleaning import load_raw

EXPECTED_CLEAN_DTYPES = {
    "id": "int64",
    "date": "string",  # any string dtype, checked with is_string_dtype
    "price": "float64",
    "bedrooms": "int64",
    "bathrooms": "float64",
    "sqft_living": "int64",
    "sqft_lot": "int64",
    "floors": "float64",
    "waterfront": "float64",
    "view": "float64",
    "condition": "int64",
    "grade": "int64",
    "sqft_above": "int64",
    "sqft_basement": "int64",
    "zipcode": "int64",
    "lat": "float64",
    "long": "float64",
    "sqft_living15": "int64",
    "sqft_lot15": "int64",
    "last_known_change": "int64",
}


@pytest.fixture
def raw_csv_path(raw_houses, tmp_path):
    """Synthetic raw table as CSV without its index (RangeIndex 0..7 on read)."""
    path = tmp_path / "raw.csv"
    raw_houses.to_csv(path, index=False)
    return path


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


@pytest.mark.integration
def test_clean_table_schema(raw_csv_path, tmp_path):
    """R4: written clean table has exactly the spec's columns, dtypes and no NaN."""
    out_path = cleaning.write_clean(
        raw_path=raw_csv_path, out_path=tmp_path / "clean.parquet"
    )
    table = pd.read_parquet(out_path)

    assert set(table.columns) == set(EXPECTED_CLEAN_DTYPES)
    assert "yr_built" not in table.columns
    assert "yr_renovated" not in table.columns
    actual = {
        col: ("string" if col == "date" else str(table[col].dtype))
        for col in table.columns
    }
    assert actual == EXPECTED_CLEAN_DTYPES
    assert pd.api.types.is_string_dtype(table["date"])
    nan_counts = table.drop(columns="date").isna().sum()
    assert nan_counts[nan_counts > 0].empty


@pytest.mark.integration
def test_write_clean_roundtrip_matches_clean(raw_csv_path, tmp_path):
    """Spec step 5: parquet read back equals clean(load_raw(csv)), index included."""
    out_path = cleaning.write_clean(
        raw_path=raw_csv_path, out_path=tmp_path / "clean.parquet"
    )

    written = pd.read_parquet(out_path)
    expected = cleaning.clean(load_raw(raw_csv_path))

    pd.testing.assert_frame_equal(written, expected, check_dtype=True)


@pytest.mark.integration
def test_write_clean_creates_missing_parent_dir(raw_csv_path, tmp_path):
    """Spec step 5: write_clean creates missing parent dirs and returns out_path."""
    out_path = tmp_path / "a" / "b" / "clean.parquet"

    returned = cleaning.write_clean(raw_path=raw_csv_path, out_path=out_path)

    assert returned == out_path
    assert out_path.exists()
