"""Load and preprocess the credit-card fraud dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
TARGET_COLUMN = "Class"

# V1..V28 are already PCA components (roughly standardized); only the two raw
# columns, Time and Amount, need scaling to match their range.
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
SCALE_FEATURES = ["Time", "Amount"]
FEATURE_COLUMNS = SCALE_FEATURES + PCA_FEATURES


def load_clean_data(
    data_path: Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, int]:
    """Load the raw CSV and remove exact duplicate rows."""
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find the dataset at {data_path}")

    data = pd.read_csv(data_path)
    expected = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = sorted(expected - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    duplicate_count = int(data.duplicated().sum())
    clean_data = data.drop_duplicates().reset_index(drop=True)
    return clean_data, duplicate_count


def split_features_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the transaction features X and binary fraud target y."""
    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Scale Time and Amount; pass the PCA components through unchanged."""
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), SCALE_FEATURES),
        ],
        remainder="passthrough",
    )
