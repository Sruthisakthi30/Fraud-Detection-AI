"""
FraudShield AI - Model Training
Trains supervised classifiers and Isolation Forest.
"""
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config import (
    LR_PARAMS, DT_PARAMS, RF_PARAMS, XGB_PARAMS, IF_PARAMS,
    RANDOM_STATE, BEST_MODEL_METRIC,
    MODEL_COMPARISON_PATH, BEST_MODEL_META_PATH, DATASET_META_PATH,
    MODELS_DIR,
)
from src.config import model_path

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Model factory                                                               #
# --------------------------------------------------------------------------- #

SUPERVISED_MODELS = {
    "logistic_regression": None,
    "decision_tree":       None,
    "random_forest":       None,
    "xgboost":             None,
}

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree":       "Decision Tree",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
    "isolation_forest":    "Isolation Forest",
}


def _build_lr():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(**LR_PARAMS)


def _build_dt():
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(**DT_PARAMS)


def _build_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**RF_PARAMS)


def _build_xgb():
    from xgboost import XGBClassifier
    params = {k: v for k, v in XGB_PARAMS.items()
              if k != "use_label_encoder"}   # removed in newer xgb
    return XGBClassifier(**params)


def _build_if():
    from sklearn.ensemble import IsolationForest
    return IsolationForest(**IF_PARAMS)


_BUILDERS = {
    "logistic_regression": _build_lr,
    "decision_tree":       _build_dt,
    "random_forest":       _build_rf,
    "xgboost":             _build_xgb,
    "isolation_forest":    _build_if,
}

# --------------------------------------------------------------------------- #
#  Training                                                                    #
# --------------------------------------------------------------------------- #

def train_supervised(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    """Train a single supervised model and persist it."""
    logger.info("Training %s...", MODEL_DISPLAY_NAMES[name])
    t0 = time.time()
    model = _BUILDERS[name]()
    model.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 2)
    logger.info("%s trained in %.2fs.", MODEL_DISPLAY_NAMES[name], elapsed)
    _save_model(name, model)
    return model, elapsed


def train_isolation_forest(
    X_train: np.ndarray,
) -> Any:
    """
    Train Isolation Forest (unsupervised).
    Uses the unsmoted, scaled training features.
    """
    logger.info("Training Isolation Forest (unsupervised)...")
    t0 = time.time()
    model = _build_if()
    model.fit(np.asarray(X_train))
    elapsed = round(time.time() - t0, 2)
    logger.info("Isolation Forest trained in %.2fs.", elapsed)
    _save_model("isolation_forest", model)
    return model, elapsed


def _save_model(name: str, model: Any) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path(name))
    logger.info("Saved %s → %s", name, model_path(name))


def load_model(name: str) -> Any:
    p = model_path(name)
    if not p.exists():
        raise FileNotFoundError(
            f"Model artifact not found: '{p}'. "
            "Please run the training pipeline."
        )
    return joblib.load(p)


def model_available(name: str) -> bool:
    return model_path(name).exists()


# --------------------------------------------------------------------------- #
#  Persistence helpers for reports and metadata                                #
# --------------------------------------------------------------------------- #

def save_comparison_report(report: dict) -> None:
    MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_COMPARISON_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Comparison report saved → %s", MODEL_COMPARISON_PATH)


def load_comparison_report() -> dict | None:
    if not MODEL_COMPARISON_PATH.exists():
        return None
    with open(MODEL_COMPARISON_PATH) as f:
        return json.load(f)


def save_best_model_meta(meta: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BEST_MODEL_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def load_best_model_meta() -> dict | None:
    if not BEST_MODEL_META_PATH.exists():
        return None
    with open(BEST_MODEL_META_PATH) as f:
        return json.load(f)


def save_dataset_meta(meta: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATASET_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def load_dataset_meta() -> dict | None:
    if not DATASET_META_PATH.exists():
        return None
    with open(DATASET_META_PATH) as f:
        return json.load(f)
