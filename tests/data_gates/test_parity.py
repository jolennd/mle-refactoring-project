"""R8 parity gate (ADR-002): pipeline.run() on the real CSV equals the notebook.

The oracle is tests/fixtures/legacy_notebook.py (verbatim notebook cells, ADR-001).
Its cell 69 double loop takes about 30 s, so oracle and pipeline output are computed
once per module. The per-column max-abs-diff report is printed to stdout; run with
`-s` to see it (the Phase 3 status cell pastes it).
"""

import time

import numpy as np
import pandas as pd
import pytest
from fixtures.legacy_notebook import notebook_features

from kc_housing import config
from kc_housing.pipeline import run

COMPARED_COLUMNS = [config.ID_COL, config.TARGET, *config.FEATURES]


@pytest.fixture(scope="module")
def real_raw_path():
    """The committed raw CSV path; skip with a reason if missing."""
    if not config.RAW_DATA_PATH.exists():
        pytest.skip(f"{config.RAW_DATA_PATH} not found: restore the committed raw CSV")
    return config.RAW_DATA_PATH


@pytest.fixture(scope="module")
def oracle(real_raw_path) -> pd.DataFrame:
    """Notebook features from the verbatim cells, computed once (slow loop)."""
    start = time.perf_counter()
    frame = notebook_features(real_raw_path)
    print(f"\n[parity] oracle runtime: {time.perf_counter() - start:.1f} s")
    return frame


@pytest.fixture(scope="module")
def pipeline_output(real_raw_path, tmp_path_factory) -> pd.DataFrame:
    """kc_housing.pipeline.run() on the real CSV, written to a temp parquet."""
    out_path = tmp_path_factory.mktemp("parity") / "features.parquet"
    run(raw_path=real_raw_path, out_path=out_path)
    return pd.read_parquet(out_path)


def _max_abs_diff_report(ours: pd.DataFrame, theirs: pd.DataFrame) -> pd.DataFrame:
    """Per column: max |ours - theirs| and whether it is within PARITY_RTOL/ATOL."""
    rows = []
    for col in COMPARED_COLUMNS:
        a = ours[col].to_numpy(dtype=float)
        b = theirs[col].to_numpy(dtype=float)
        rows.append(
            {
                "column": col,
                "max_abs_diff": float(np.max(np.abs(a - b))) if len(a) else 0.0,
                "within_tol": bool(
                    np.allclose(a, b, rtol=config.PARITY_RTOL, atol=config.PARITY_ATOL)
                ),
            }
        )
    return pd.DataFrame(rows).set_index("column")


@pytest.mark.data
def test_pipeline_matches_notebook_features(pipeline_output, oracle):
    """R8: every column (id, price, FEATURES) equals the notebook within tolerance.

    Rows are aligned on the index; the ids must agree row by row as well.
    """
    assert list(oracle.columns) == COMPARED_COLUMNS, "oracle columns != config order"
    assert list(pipeline_output.columns) == COMPARED_COLUMNS
    common = oracle.index.intersection(pipeline_output.index)
    ours = pipeline_output.loc[common]
    theirs = oracle.loc[common]

    report = _max_abs_diff_report(ours, theirs)
    print(
        f"\n[parity] rtol={config.PARITY_RTOL} atol={config.PARITY_ATOL} "
        f"rows={len(common)}\n{report.to_string()}"
    )

    outside = report.index[~report["within_tol"]].tolist()
    assert outside == [], (
        f"columns outside tolerance: {outside}\nmax abs diff per column:\n"
        f"{report['max_abs_diff'].to_string()}"
    )


@pytest.mark.data
def test_pipeline_matches_notebook_row_count(pipeline_output, oracle):
    """R8: same number of rows, same index and the same id in every row."""
    assert len(pipeline_output) == len(oracle)
    pd.testing.assert_index_equal(
        pipeline_output.index, oracle.index, exact=False, check_names=False
    )
    np.testing.assert_array_equal(
        pipeline_output[config.ID_COL].to_numpy(), oracle[config.ID_COL].to_numpy()
    )
