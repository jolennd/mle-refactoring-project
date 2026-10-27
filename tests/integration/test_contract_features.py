import pandas as pd
import pytest

from kc_housing import cleaning, config, features

EXPECTED_COLUMNS = [config.ID_COL, config.TARGET, *config.FEATURES]


@pytest.fixture
def clean_parquet_path(raw_houses, tmp_path):
    """Synthetic clean table (Phase 1 output) written as parquet, index kept."""
    path = tmp_path / "houses_clean.parquet"
    cleaning.clean(raw_houses).to_parquet(path)
    return path


@pytest.mark.integration
def test_feature_table_columns_equal_config_features(clean_parquet_path, tmp_path):
    """R6 / handoff contract: written table is id, price, then config.FEATURES."""
    out_path = features.write_features(
        clean_path=clean_parquet_path, out_path=tmp_path / "features.parquet"
    )
    table = pd.read_parquet(out_path)

    assert list(table.columns) == EXPECTED_COLUMNS
    feature_part = list(table.columns)[2:]
    assert not set(feature_part) & set(config.LEAKAGE_COLUMNS)


@pytest.mark.integration
def test_feature_table_numeric_no_nan_one_row_per_sale(clean_parquet_path, tmp_path):
    """Handoff contract: numeric dtypes, no NaN, one row per cleaned sale."""
    clean_table = pd.read_parquet(clean_parquet_path)
    out_path = features.write_features(
        clean_path=clean_parquet_path, out_path=tmp_path / "features.parquet"
    )
    table = pd.read_parquet(out_path)

    non_numeric = [
        col for col in table.columns if not pd.api.types.is_numeric_dtype(table[col])
    ]
    assert non_numeric == []
    nan_counts = table.isna().sum()
    assert nan_counts[nan_counts > 0].empty
    pd.testing.assert_index_equal(table.index, clean_table.index)
    # Repeat sale: id 101 is two rows, so rows are sales, not houses.
    assert (table[config.ID_COL] == 101).sum() == 2


@pytest.mark.integration
def test_write_features_roundtrip_matches_build_features(raw_houses, tmp_path):
    """Spec step 5: parquet read back equals build_features(clean), index included;
    returns out_path and creates missing parent dirs."""
    clean_path = tmp_path / "houses_clean.parquet"
    cleaning.clean(raw_houses).to_parquet(clean_path)
    out_path = tmp_path / "a" / "b" / "features.parquet"

    returned = features.write_features(clean_path=clean_path, out_path=out_path)

    assert returned == out_path
    assert out_path.exists()
    written = pd.read_parquet(out_path)
    expected = features.build_features(pd.read_parquet(clean_path))
    pd.testing.assert_frame_equal(written, expected, check_dtype=True)
