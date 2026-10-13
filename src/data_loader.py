"""
FraudShield AI - Data Loader
Validates and loads the European Credit Card Fraud dataset.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATASET_PATH,
    DEMO_DATA_DIR,
    DEMO_DATASET_PATH,
    TARGET_COLUMN,
    TIME_COLUMN,
    AMOUNT_COLUMN,
    PCA_FEATURES,
    FEATURE_COLUMNS,
    ALL_COLUMNS,
    SAMPLE_SIZE,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Public helpers                                                              #
# --------------------------------------------------------------------------- #

def dataset_available() -> bool:
    """Return True when the primary real creditcard.csv exists and is non-empty."""
    return DATASET_PATH.exists() and DATASET_PATH.stat().st_size > 0


def demo_dataset_available() -> bool:
    """Return True when the synthetic demonstration CSV exists and is non-empty."""
    return DEMO_DATASET_PATH.exists() and DEMO_DATASET_PATH.stat().st_size > 0


def generate_demo_dataset(n_samples: int = 5000, n_fraud: int = 100) -> Path:
    """
    Generate an isolated SYNTHETIC DEMONSTRATION DATASET.
    Clearly isolated in data/demo_synthetic/ and labelled as synthetic demo data.
    NOT mixed with real data. NOT for real-world benchmarking.
    """
    DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)
    data = {col: rng.standard_normal(n_samples) for col in FEATURE_COLUMNS}
    fraud_idx = rng.choice(n_samples, n_fraud, replace=False)
    for col in ["V1", "V4", "V14"]:
        data[col][fraud_idx] += 2.5
    data[TARGET_COLUMN] = np.zeros(n_samples, dtype=int)
    data[TARGET_COLUMN][fraud_idx] = 1
    data[AMOUNT_COLUMN] = np.abs(data[AMOUNT_COLUMN]) * 88.0 + 1.0
    data[TIME_COLUMN] = np.sort(np.abs(data[TIME_COLUMN]) * 10000.0)
    df = pd.DataFrame(data)
    df.to_csv(DEMO_DATASET_PATH, index=False)
    logger.info("Generated synthetic demo dataset with %d rows (%d fraud) at %s", n_samples, n_fraud, DEMO_DATASET_PATH)
    return DEMO_DATASET_PATH


def load_demo_dataset() -> pd.DataFrame:
    """Load the synthetic demonstration dataset, generating it first if needed."""
    if not demo_dataset_available():
        generate_demo_dataset()
    return pd.read_csv(DEMO_DATASET_PATH)


def load_dataset(sample_size: int | None = None) -> pd.DataFrame:
    """
    Load and validate the Credit Card Fraud dataset.

    Parameters
    ----------
    sample_size : int or None
        If given, return a stratified random sample of this many rows.

    Returns
    -------
    pd.DataFrame
        Validated dataframe.

    Raises
    ------
    FileNotFoundError
        If the CSV is missing.
    ValueError
        If required columns are absent or the target column has unexpected values.
    """
    if not dataset_available():
        raise FileNotFoundError(
            f"Dataset not found at '{DATASET_PATH}'. "
            "Please download 'creditcard.csv' from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in the 'data/' directory."
        )

    logger.info("Loading dataset from %s", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)

    # ── Column validation ────────────────────────────────────────────────────
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Expected columns: {ALL_COLUMNS}"
        )

    # ── Target validation ────────────────────────────────────────────────────
    unique_targets = set(df[TARGET_COLUMN].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' contains unexpected values: "
            f"{unique_targets}. Expected only {{0, 1}}."
        )

    # ── Optional stratified sample ───────────────────────────────────────────
    actual_sample = sample_size or SAMPLE_SIZE
    if actual_sample and actual_sample < len(df):
        logger.info(
            "Using stratified sample of %d rows (full size: %d).",
            actual_sample,
            len(df),
        )
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(
            df,
            train_size=actual_sample,
            stratify=df[TARGET_COLUMN],
            random_state=RANDOM_STATE,
        )
        df = df.reset_index(drop=True)

    logger.info(
        "Dataset loaded: %d rows, %d fraud, %d legitimate.",
        len(df),
        df[TARGET_COLUMN].sum(),
        (df[TARGET_COLUMN] == 0).sum(),
    )
    return df


def get_dataset_summary(df: pd.DataFrame, sampled: bool = False, is_demo: bool = False) -> dict:
    """
    Return a JSON-serialisable summary dict describing the loaded dataset.
    """
    fraud_count = int(df[TARGET_COLUMN].sum())
    legit_count = int((df[TARGET_COLUMN] == 0).sum())
    total       = len(df)
    fraud_pct   = round(fraud_count / total * 100, 4) if total else 0.0

    source = (
        "SYNTHETIC DEMONSTRATION DATA (FOR TESTING & UI PREVIEWS ONLY - NOT REAL BENCHMARK)"
        if is_demo
        else "European Credit Card Fraud Dataset (Kaggle / Zenodo – real anonymised transactions)"
    )
    path_name = str(DEMO_DATASET_PATH.name if is_demo else DATASET_PATH.name)

    return {
        "total_transactions": total,
        "fraud_count":        fraud_count,
        "legitimate_count":   legit_count,
        "fraud_percentage":   fraud_pct,
        "feature_count":      len(FEATURE_COLUMNS),
        "features":           FEATURE_COLUMNS,
        "target_column":      TARGET_COLUMN,
        "missing_values":     int(df.isnull().sum().sum()),
        "duplicate_rows":     int(df.duplicated().sum()),
        "amount_stats": {
            "min":    round(float(df[AMOUNT_COLUMN].min()), 4),
            "max":    round(float(df[AMOUNT_COLUMN].max()), 4),
            "mean":   round(float(df[AMOUNT_COLUMN].mean()), 4),
            "median": round(float(df[AMOUNT_COLUMN].median()), 4),
            "std":    round(float(df[AMOUNT_COLUMN].std()), 4),
        },
        "time_stats": {
            "min":    round(float(df[TIME_COLUMN].min()), 4),
            "max":    round(float(df[TIME_COLUMN].max()), 4),
            "mean":   round(float(df[TIME_COLUMN].mean()), 4),
        },
        "is_sampled":         sampled,
        "is_synthetic_demo":  is_demo,
        "dataset_source":     source,
        "dataset_path":       path_name,
    }
