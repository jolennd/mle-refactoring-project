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
