import pandas as pd
import pytest

from kc_housing import config

# docs/02_data_sources.md: 21,597 raw sales minus the one 33-bedroom row.
EXPECTED_SALES = 21_596


@pytest.mark.data
def test_real_feature_table_contract(processed_features):
    """Handoff contract on real data: id, price, FEATURES; one row per clean sale."""
    expected = [config.ID_COL, config.TARGET, *config.FEATURES]

    assert list(processed_features.columns) == expected
    assert len(processed_features) == EXPECTED_SALES


@pytest.mark.data
def test_real_feature_table_numeric_no_nan(processed_features):
    """Real data: every column numeric, no NaN in any FEATURES column.

    Phase 3 parity uses assert_frame_equal, which treats matching NaN as equal,
    so a NaN the notebook also produces would slip through; the integration
    test checks this only on synthetic data.
    """
    columns = [config.ID_COL, config.TARGET, *config.FEATURES]
    non_numeric = [
        col
        for col in columns
        if not pd.api.types.is_numeric_dtype(processed_features[col])
    ]
    nan_counts = processed_features[config.FEATURES].isna().sum()
    with_nan = {col: int(n) for col, n in nan_counts.items() if n > 0}

    assert non_numeric == []
    assert with_nan == {}
