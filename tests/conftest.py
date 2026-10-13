# FraudShield AI Tests - conftest.py
# Shared fixtures for all test modules.

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Small synthetic fixture (for unit tests only — NOT used for benchmarking) ─
# Clearly labelled as TEST FIXTURE, not real data.
N_LEGIT  = 1000
N_FRAUD  = 20
SEED     = 42

@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def fake_df(rng):
    """
    Tiny labelled fixture for fast unit tests.
    This is SYNTHETIC TEST DATA — not used for any reported metric.
    """
    from src.config import FEATURE_COLUMNS, TARGET_COLUMN

    n = N_LEGIT + N_FRAUD
    data = {col: rng.standard_normal(n) for col in FEATURE_COLUMNS}
    # Make fraud slightly separable so classifiers can learn
    fraud_idx = rng.choice(n, N_FRAUD, replace=False)
    for col in ["V1", "V4", "V14"]:
        data[col][fraud_idx] += 3.0
    data[TARGET_COLUMN] = np.zeros(n, dtype=int)
    data[TARGET_COLUMN][fraud_idx] = 1
    # Amount and Time must be positive
    data["Amount"] = np.abs(data["Amount"]) * 100
    data["Time"]   = np.abs(data["Time"])   * 50_000
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def preprocessed(fake_df):
    """Run the preprocessing pipeline on the synthetic fixture."""
    from src.preprocessing import run_preprocessing
    return run_preprocessing(fake_df)


@pytest.fixture(scope="session")
def trained_models(preprocessed):
    """Train all models on the synthetic fixture."""
    import tempfile, os
    from src.train_models import train_supervised, train_isolation_forest
    from src.config import MODELS_DIR

    results = {}
    for name in ["logistic_regression", "decision_tree", "random_forest", "xgboost"]:
        model, elapsed = train_supervised(
            name, preprocessed["X_train"], preprocessed["y_train"]
        )
        results[name] = model

    if_model, _ = train_isolation_forest(preprocessed["X_train_raw"])
    results["isolation_forest"] = if_model
    return results


@pytest.fixture(scope="session")
def real_dataset_path():
    """Path to the real creditcard.csv (may not exist in CI)."""
    from src.config import DATASET_PATH
    return DATASET_PATH
