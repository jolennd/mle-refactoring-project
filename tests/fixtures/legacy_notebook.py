"""Parity oracle: King-County.ipynb's cleaning and feature cells, copied verbatim.

ADR-001/002: this file is the frozen reference in code form. It is never "cleaned
up"; the only changes from the notebook are listed below.

Source cells of King-County.ipynb, in execution order:
- cell 7: read_csv (the path is the `raw_path` argument instead of the relative path)
- cell 28: drop the implausible row by its hard-coded index 15856
- cell 30: recompute sqft_basement
- cells 35, 38: fill missing view and waterfront with 0
- cells 41-43: last_known_change loop; drop yr_renovated and yr_built
- cell 56: sqft_price (kept because cell 78 drops it by name)
- cell 60: delta_lat, delta_long, center_distance
- cells 67-70: dist(), water_list, the scalar double loop for water_distance
- cells 78-80: drop_lst, all_features, X
- cell 101: drop the id column (the notebook does it on X_train/X_test; here on X)

Display-only cells (head, info, describe, value_counts, plots, missing-value tables)
are left out; they do not change kc_data. Modeling cells (81 onward, except the id
drop of cell 101) are out of scope for Phase 3.

Only numpy and pandas are imported; nothing from kc_housing.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def notebook_features(raw_path: str | Path) -> pd.DataFrame:
    """Return id, price, then the notebook's model features (cell 101 column order).

    The index is kc_data's index after the notebook cells (the CSV row numbers
    minus 15856). Runtime is dominated by the cell 69 double loop.
    """
    # --- cell 7 ---
    kc_data = pd.read_csv(raw_path)

    # --- cell 28 ---
    # Drop the implausible row.
    kc_data.drop(15856, axis=0, inplace=True)

    # --- cell 30 ---
    # Recalculate `sqft_basement` as `sqft_living - sqft_above`.
    kc_data["sqft_basement"] = kc_data["sqft_living"] - kc_data["sqft_above"]

    # --- cell 35 ---
    # Replace missing values in `view` with the most frequent value (0).
    kc_data["view"] = kc_data["view"].fillna(0)

    # --- cell 38 ---
    # Replace missing values in `waterfront` with the most frequent value (0).
    kc_data["waterfront"] = kc_data["waterfront"].fillna(0)

    # --- cell 41 ---
    # Create an empty list to store the derived values.
    last_known_change = []

    # Inspect the `yr_renovated` value for each row.
    for idx, yr_re in kc_data["yr_renovated"].items():
        # If `yr_renovated` is 0 or missing, use the original build year instead.
        if str(yr_re) == "nan" or yr_re == 0.0:
            last_known_change.append(kc_data["yr_built"][idx])
        # Otherwise, keep the renovation year in the new list.
        else:
            last_known_change.append(int(yr_re))

    # --- cell 42 ---
    # Create a new column from the values collected above.
    kc_data["last_known_change"] = last_known_change

    # --- cell 43 ---
    # Drop the original `yr_renovated` and `yr_built` columns.
    kc_data.drop("yr_renovated", axis=1, inplace=True)
    kc_data.drop("yr_built", axis=1, inplace=True)

    # --- cell 56 ---
    # Create a variable for price per square foot of living space.
    kc_data["sqft_price"] = (
        kc_data.price / (kc_data.sqft_living + kc_data.sqft_lot)
    ).round(2)

    # --- cell 60 ---
    # Absolute difference in latitude between the center and the property.
    kc_data["delta_lat"] = np.absolute(47.62774 - kc_data["lat"])
    # Absolute difference in longitude between the center and the property.
    kc_data["delta_long"] = np.absolute(-122.24194 - kc_data["long"])
    # Distance between the center and the property.
    kc_data["center_distance"] = (
        (
            (kc_data["delta_long"] * np.cos(np.radians(47.6219))) ** 2
            + kc_data["delta_lat"] ** 2
        )
        ** (1 / 2)
        * 2
        * np.pi
        * 6378
        / 360
    )

    # --- cell 67 ---
    # This helper function calculates the distance between one house and a
    # reference location.
    def dist(long, lat, ref_long, ref_lat):
        """dist computes the distance in km to a reference location.
        Input: long and lat of the location of interest and ref_long and ref_lat
        as the long and lat of the reference location"""
        delta_long = long - ref_long
        delta_lat = lat - ref_lat
        delta_long_corr = delta_long * np.cos(np.radians(ref_lat))
        return (
            ((delta_long_corr) ** 2 + (delta_lat) ** 2) ** (1 / 2)
            * 2
            * np.pi
            * 6378
            / 360
        )

    # --- cell 68 ---
    # Add all waterfront houses to the reference list.
    water_list = kc_data.query("waterfront == 1")

    # --- cell 69 ---
    water_distance = []
    # For each row, calculate the distance to the nearest waterfront house.
    for idx in kc_data.index:
        ref_list = []
        for x, y in zip(list(water_list["long"]), list(water_list["lat"])):  # noqa: B905 (verbatim)
            ref_list.append(dist(kc_data["long"][idx], kc_data["lat"][idx], x, y).min())
        water_distance.append(min(ref_list))

    # --- cell 70 ---
    # Create a new column from the previously computed list.
    kc_data["water_distance"] = water_distance

    # --- cell 78 ---
    # Remove columns that would cause data leakage or add no predictive value.
    drop_lst = [
        "price",
        "sqft_price",
        "date",
        "delta_lat",
        "delta_long",
    ]

    # --- cell 79 ---
    # Keep all remaining variables as candidate features.
    all_features = [x for x in kc_data.columns if x not in drop_lst]

    # --- cell 80 ---
    # `X` contains the descriptive variables defined above.
    X = kc_data[all_features]

    # --- cell 101 (applied to X instead of X_train / X_test) ---
    # Drop the `id` column.
    X_poly = X.copy()
    X_poly = X_poly.drop(columns=["id"])

    # Oracle output: id and price next to the model features, same index.
    return pd.concat([kc_data[["id", "price"]], X_poly], axis=1)
