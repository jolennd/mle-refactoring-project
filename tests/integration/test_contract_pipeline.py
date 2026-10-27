import pandas as pd
import pytest

from kc_housing import config
from kc_housing.cleaning import clean, load_raw
from kc_housing.pipeline import run

EXPECTED_COLUMNS = [config.ID_COL, config.TARGET, *config.FEATURES]


@pytest.mark.integration
def test_run_writes_feature_parquet_from_raw_csv(raw_houses, tmp_path):
    """Spec step 2: run(raw_path, out_path) writes id, price, FEATURES with the clean
    table's index, creates missing parent dirs and returns out_path."""
    raw_path = tmp_path / "raw.csv"
    raw_houses.to_csv(raw_path, index=False)
    out_path = tmp_path / "a" / "b" / "features.parquet"

    returned = run(raw_path=raw_path, out_path=out_path)

    assert returned == out_path
    table = pd.read_parquet(out_path)
    assert list(table.columns) == EXPECTED_COLUMNS
    pd.testing.assert_index_equal(table.index, clean(load_raw(raw_path)).index)
