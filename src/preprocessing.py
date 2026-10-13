"""
FraudShield AI - Preprocessing Pipeline
Handles scaling, train/test split, and SMOTE (training only).
"""
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    TARGET_COLUMN,
    TIME_COLUMN,
    AMOUNT_COLUMN,
    PCA_FEATURES,
    FEATURE_COLUMNS,
    TEST_SIZE,
    RANDOM_STATE,
    SMOTE_PARAMS,
    SCALER_PATH,
    FEATURE_NAMES_PATH,
    PREPROCESSOR_PATH,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Core preprocessing                                                          #
# --------------------------------------------------------------------------- #

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into X (features) and y (target)."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train-test split."""
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def build_scaler() -> StandardScaler:
    """Return a new StandardScaler."""
    return StandardScaler()


def fit_scaler(scaler: StandardScaler, X_train: pd.DataFrame) -> StandardScaler:
    """Fit the scaler on training data ONLY."""
    scaler.fit(X_train)
    return scaler


def transform_data(scaler: StandardScaler, X: pd.DataFrame) -> np.ndarray:
    """Apply pre-fitted scaler transform."""
    return scaler.transform(X)


def apply_smote(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to the TRAINING partition only.

    Returns
    -------
    X_resampled, y_resampled – numpy arrays.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError("imbalanced-learn is required for SMOTE. "
                          "Install it with: pip install imbalanced-learn")

    smote = SMOTE(**SMOTE_PARAMS)
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train)
    logger.info(
        "SMOTE applied: %d → %d training samples "
        "(fraud: %d → %d).",
        len(y_train), len(y_res),
        int(y_train.sum()), int(y_res.sum()),
    )
    return X_res, y_res


def run_preprocessing(df: pd.DataFrame) -> dict:
    """
    Full preprocessing pipeline.

    Returns a dict containing all splits, the fitted scaler, and metadata.
    The test set is NEVER touched by SMOTE or scaler fitting.
    """
    logger.info("Starting preprocessing pipeline...")

    # 1. Separate features and target
    X, y = prepare_features(df)

    # 2. Train / test split (stratified)
    X_train, X_test, y_train, y_test = split_data(X, y)
    logger.info(
        "Train: %d rows (%d fraud), Test: %d rows (%d fraud).",
        len(y_train), int(y_train.sum()),
        len(y_test),  int(y_test.sum()),
    )

    # 3. Fit scaler on TRAINING data only
    scaler = build_scaler()
    fit_scaler(scaler, X_train)

    # 4. Transform both sets
    X_train_scaled = transform_data(scaler, X_train)
    X_test_scaled  = transform_data(scaler, X_test)

    # 5. SMOTE on training set only
    X_train_res, y_train_res = apply_smote(X_train_scaled, y_train)

    # 6. Meta
    meta = {
        "feature_names":      FEATURE_COLUMNS,
        "n_features":         len(FEATURE_COLUMNS),
        "train_size":         len(y_train),
        "test_size":          len(y_test),
        "train_fraud":        int(y_train.sum()),
        "test_fraud":         int(y_test.sum()),
        "train_size_smote":   len(y_train_res),
        "train_fraud_smote":  int(y_train_res.sum()),
        "smote_params":       SMOTE_PARAMS,
        "random_state":       RANDOM_STATE,
    }

    return {
        "X_train":        X_train_res,
        "X_train_scaled": X_train_scaled,
        "X_test":         X_test_scaled,
        "y_train":        y_train_res,
        "y_test":         y_test,
        "X_train_raw":    X_train,   # for Isolation Forest (unfitted SMOTE)
        "y_train_raw":    y_train,
        "scaler":         scaler,
        "feature_names":  FEATURE_COLUMNS,
        "meta":           meta,
    }


# --------------------------------------------------------------------------- #
#  Persistence                                                                 #
# --------------------------------------------------------------------------- #

def save_preprocessor(scaler: StandardScaler, feature_names: list[str]) -> None:
    """Persist the scaler and feature name list to disk."""
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f)
    logger.info("Preprocessor saved to %s", SCALER_PATH)


def load_preprocessor() -> tuple[StandardScaler, list[str]]:
    """Load persisted scaler and feature name list."""
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at '{SCALER_PATH}'. "
            "Please run the training pipeline first."
        )
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_NAMES_PATH) as f:
        feature_names = json.load(f)
    return scaler, feature_names


def transform_single(scaler: StandardScaler,
                     feature_names: list[str],
                     raw_values: dict) -> np.ndarray:
    """
    Transform a single transaction dict into a scaled feature vector.

    Parameters
    ----------
    raw_values : dict
        Keys must match feature_names.

    Returns
    -------
    np.ndarray of shape (1, n_features).
    """
    row = pd.DataFrame([raw_values], columns=feature_names)
    return scaler.transform(row)
