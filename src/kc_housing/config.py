"""Single source of truth: paths, constants, the fixed feature set and thresholds.

Nothing else may redefine these. Only the human edits thresholds and the fixed scope.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "King_County_House_prices_dataset.csv"  # committed, ADR-007
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_PATH = PROCESSED_DIR / "houses_clean.parquet"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
REFERENCE_NOTEBOOK = ROOT / "King-County.ipynb"  # frozen, ADR-001
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "elastic_net.skops"  # ADR-007
RESULTS_DIR = ROOT / "reports" / "results"
BASELINE_LOCK_PATH = RESULTS_DIR / "baseline_lock.json"
CV_FOLDS_PATH = RESULTS_DIR / "cv_folds.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"

# --- Columns -------------------------------------------------------------------
ID_COL = "id"
TARGET = "price"

# --- Cleaning rules (notebook cells 26-43) -------------------------------------
# Row rule, not the notebook's hard-coded index 15856 (ADR-003).
IMPLAUSIBLE_BEDROOMS = 33
FILL_ZERO_COLUMNS = ["view", "waterfront"]

# --- Feature constants (notebook cells 60-69), kept exactly for parity (ADR-002) --
CENTER_LAT = 47.62774
CENTER_LONG = -122.24194
# The notebook corrects longitude with cos(47.6219), not cos(CENTER_LAT).
CENTER_COS_LAT = 47.6219
EARTH_RADIUS_KM = 6378

# --- Fixed feature set: model input, in this order (notebook cell 101) ---------
FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "last_known_change",
    "center_distance",
    "water_distance",
]
# Never model inputs: the target and anything computed from it (ADR-002).
LEAKAGE_COLUMNS = ["price", "sqft_price"]

# --- Parity with the notebook (Phase 3) -----------------------------------------
PARITY_RTOL = 1e-9  # ADR-002
PARITY_ATOL = 1e-9  # ADR-002

# --- Modeling (Phase 4, stretch; notebook cells 82 and 117) ----------------------
TEST_SIZE = 0.3
RANDOM_STATE = 42
CV_FOLDS = 5
POLY_DEGREE = 2
PARAM_GRID = {
    "model__regressor__alpha": [0.01, 0.1, 1],
    "model__regressor__l1_ratio": [0.2, 0.5, 0.8],
}
PRIMARY_METRIC = "r2"
# Allowed gap between the refactored test R^2 and the notebook's own, locked value.
R2_PARITY_ATOL = 0.001  # ADR-006

# --- API (Phase 5, stretch) --------------------------------------------------------
API_FEATURES = ["bedrooms", "bathrooms", "sqft_living", "grade", "zipcode"]  # ADR-008
DATABASE_URL_ENV = "DATABASE_URL"
