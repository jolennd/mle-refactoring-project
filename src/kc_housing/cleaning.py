"""Load the raw King County CSV and apply the cleaning rules (notebook cells 7-43)."""

from pathlib import Path

import pandas as pd

from kc_housing import config


def load_raw(path: str | Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV unchanged (notebook cell 7)."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of `df`; the input is not modified."""
    result = df[df["bedrooms"] != config.IMPLAUSIBLE_BEDROOMS].copy()
    result["sqft_basement"] = result["sqft_living"] - result["sqft_above"]
    for column in config.FILL_ZERO_COLUMNS:
        result[column] = result[column].fillna(0)
    return result
