"""Load the raw King County CSV and apply the cleaning rules (notebook cells 7-43)."""

import logging
from pathlib import Path

import pandas as pd

from kc_housing import config

logger = logging.getLogger(__name__)


def load_raw(path: str | Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV unchanged (notebook cell 7)."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of `df`; the input is not modified."""
    result = df[df["bedrooms"] != config.IMPLAUSIBLE_BEDROOMS].copy()
    result["sqft_basement"] = result["sqft_living"] - result["sqft_above"]
    for column in config.FILL_ZERO_COLUMNS:
        result[column] = result[column].fillna(0)
    result["last_known_change"] = (
        result["yr_renovated"]
        .where(result["yr_renovated"] > 0, result["yr_built"])
        .astype("int64")
    )
    result = result.drop(columns=["yr_built", "yr_renovated"])
    logger.info("Dropped %d rows with implausible bedrooms", len(df) - len(result))
    return result


def write_clean(
    raw_path: str | Path = config.RAW_DATA_PATH,
    out_path: str | Path = config.CLEAN_PATH,
) -> Path:
    """Load, clean and write the clean table as parquet, keeping the index."""
    df = clean(load_raw(raw_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return Path(out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    written = write_clean()
    logger.info("Wrote %s (%d rows)", written, len(pd.read_parquet(written)))
