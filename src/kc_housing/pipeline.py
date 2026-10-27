"""Clean-to-features Pipeline (cells 60, 67-70, 78-80, 101; ADR-004, ADR-005)."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from kc_housing import config
from kc_housing.cleaning import clean, load_raw
from kc_housing.features import CenterDistance, SelectColumns, WaterDistance

logger = logging.getLogger(__name__)


def build_feature_pipeline() -> Pipeline:
    """Return an unfitted Pipeline that maps clean rows to config.FEATURES."""
    return Pipeline(
        [
            ("center_distance", CenterDistance()),
            ("water_distance", WaterDistance()),
            ("select_columns", SelectColumns(config.FEATURES)),
        ]
    )


def run(
    raw_path: str | Path = config.RAW_DATA_PATH,
    out_path: str | Path = config.FEATURES_PATH,
) -> Path:
    """Load, clean, fit_transform and write id, price, then config.FEATURES as parquet.

    The pipeline is fitted on the full clean table, as the notebook does (ADR-005).
    """
    df = clean(load_raw(raw_path))
    features = build_feature_pipeline().fit_transform(df)
    out = pd.concat([df[[config.ID_COL, config.TARGET]], features], axis=1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    return Path(out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    written = run()
    logger.info("Wrote %s (%d rows)", written, len(pd.read_parquet(written)))
