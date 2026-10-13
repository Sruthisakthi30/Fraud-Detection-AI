"""
FraudShield AI - Backend Services
Loads and caches dataset, models, and evaluation artifacts.
"""
import json
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded singletons
_dataset = None
_scaler  = None
_feature_names = None
_eda_cache = None
_best_model = None
_best_model_name = None
_if_model  = None
_X_test_cache = None
_y_test_cache = None


def _root():
    return Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

def get_dataset_status() -> dict:
    """Return status information without loading the full dataset."""
    sys.path.insert(0, str(_root()))
    from src.data_loader import dataset_available, demo_dataset_available, DATASET_PATH, DEMO_DATASET_PATH
    from src.train_models import load_comparison_report, load_best_model_meta
    from src.config import MODEL_COMPARISON_PATH, BEST_MODEL_META_PATH

    report = load_comparison_report()
    best   = load_best_model_meta()
    real_avail = dataset_available()
    demo_avail = demo_dataset_available()

    return {
        "dataset_available":  real_avail or demo_avail,
        "is_real_dataset":    real_avail,
        "is_synthetic_demo":  not real_avail and demo_avail,
        "dataset_path":       str(DATASET_PATH if real_avail else (DEMO_DATASET_PATH if demo_avail else DATASET_PATH)),
        "models_trained":     report is not None,
        "best_model":         best.get("display_name") if best else None,
        "training_timestamp": report.get("timestamp") if report else None,
        "is_sampled":         best.get("is_sampled") if best else None,
    }


def get_dataset(force_reload: bool = False) -> Any:
    """Return the cached pandas DataFrame (loads on first call)."""
    global _dataset
    if _dataset is None or force_reload:
        import sys
        sys.path.insert(0, str(_root()))
        from src.data_loader import dataset_available, demo_dataset_available, load_dataset, load_demo_dataset
        if dataset_available():
            _dataset = load_dataset()
        elif demo_dataset_available():
            _dataset = load_demo_dataset()
        else:
            raise FileNotFoundError("Dataset not found. Please place creditcard.csv in data/ or run with --demo.")
    return _dataset


def get_dataset_summary_cached() -> dict:
    """Return cached dataset summary from artifact or recompute."""
    from src.train_models import load_dataset_meta
    meta = load_dataset_meta()
    if meta:
        return meta

    # Fall back to computing from loaded dataset
    from src.data_loader import get_dataset_summary, dataset_available, demo_dataset_available
    if dataset_available() or demo_dataset_available():
        df = get_dataset()
        is_demo = not dataset_available() and demo_dataset_available()
        return get_dataset_summary(df, is_demo=is_demo)
    return {}


# --------------------------------------------------------------------------- #
#  Preprocessor                                                                #
# --------------------------------------------------------------------------- #

def get_scaler_and_features():
    """Return (scaler, feature_names) loading from disk on first call."""
    global _scaler, _feature_names
    if _scaler is None:
        import sys
        sys.path.insert(0, str(_root()))
        from src.preprocessing import load_preprocessor
        _scaler, _feature_names = load_preprocessor()
    return _scaler, _feature_names


# --------------------------------------------------------------------------- #
#  Models                                                                      #
# --------------------------------------------------------------------------- #

def get_best_model():
    """Return (model, model_name) for the best supervised classifier."""
    global _best_model, _best_model_name
    if _best_model is None:
        import sys
        sys.path.insert(0, str(_root()))
        from src.train_models import load_best_model_meta, load_model
        meta = load_best_model_meta()
        if meta is None:
            raise FileNotFoundError("Best model metadata not found. Run training first.")
        _best_model_name = meta["model_name"]
        _best_model = load_model(_best_model_name)
    return _best_model, _best_model_name


def get_isolation_forest():
    """Return the Isolation Forest model."""
    global _if_model
    if _if_model is None:
        import sys
        sys.path.insert(0, str(_root()))
        from src.train_models import load_model
        _if_model = load_model("isolation_forest")
    return _if_model


# --------------------------------------------------------------------------- #
#  Test set                                                                    #
# --------------------------------------------------------------------------- #

def get_test_arrays():
    """Return (X_test, y_test) numpy arrays persisted during training."""
    global _X_test_cache, _y_test_cache
    if _X_test_cache is None:
        models_dir = _root() / "models"
        x_path = models_dir / "X_test.npy"
        y_path = models_dir / "y_test.npy"
        if not x_path.exists():
            raise FileNotFoundError("Test arrays not found. Run training first.")
        _X_test_cache = np.load(x_path)
        _y_test_cache = np.load(y_path)
    return _X_test_cache, _y_test_cache


# --------------------------------------------------------------------------- #
#  EDA cache                                                                   #
# --------------------------------------------------------------------------- #

def get_eda_cached() -> dict:
    """Return cached EDA data, computing from dataset if needed."""
    global _eda_cache
    if _eda_cache is not None:
        return _eda_cache

    import sys
    sys.path.insert(0, str(_root()))
    eda_path = _root() / "results" / "reports" / "eda_cache.json"
    if eda_path.exists():
        with open(eda_path) as f:
            _eda_cache = json.load(f)
        return _eda_cache

    from src.data_loader import dataset_available
    if not dataset_available():
        return {"available": False}

    from src.eda import get_full_eda
    df = get_dataset()
    _eda_cache = get_full_eda(df)
    _eda_cache["available"] = True
    return _eda_cache


# --------------------------------------------------------------------------- #
#  Transactions (paginated)                                                    #
# --------------------------------------------------------------------------- #

def get_transactions(page: int = 1, per_page: int = 50,
                     filter_class: int | None = None) -> dict:
    """Return a paginated slice of transactions with fraud labels."""
    import sys
    sys.path.insert(0, str(_root()))
    from src.data_loader import dataset_available
    if not dataset_available():
        return {"available": False, "transactions": [], "total": 0}

    df = get_dataset()

    if filter_class is not None:
        df = df[df["Class"] == filter_class]

    total = len(df)
    start = (page - 1) * per_page
    end   = start + per_page
    slice_df = df.iloc[start:end]

    records = []
    for _, row in slice_df.iterrows():
        record = {
            "Time":   round(float(row["Time"]), 2),
            "Amount": round(float(row["Amount"]), 2),
            "Class":  int(row["Class"]),
            "label":  "Fraud" if row["Class"] == 1 else "Legitimate",
        }
        # Add V features
        for i in range(1, 29):
            record[f"V{i}"] = round(float(row[f"V{i}"]), 6)
        records.append(record)

    return {
        "available":    True,
        "transactions": records,
        "total":        total,
        "page":         page,
        "per_page":     per_page,
        "total_pages":  (total + per_page - 1) // per_page,
    }


import sys
