"""
FraudShield AI - Central Configuration
All tuneable settings are defined here.
"""
import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR   = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
REPORTS_DIR = RESULTS_DIR / "reports"
FIGURES_DIR = RESULTS_DIR / "figures"

# Primary dataset – European Credit Card Fraud dataset (Kaggle / Zenodo)
DATASET_PATH = DATA_DIR / "creditcard.csv"

# Optional Synthetic Demonstration Dataset (Isolated location for demo/smoke testing only)
DEMO_DATA_DIR = DATA_DIR / "demo_synthetic"
DEMO_DATASET_PATH = DEMO_DATA_DIR / "synthetic_demo_creditcard.csv"

# ── Dataset schema ────────────────────────────────────────────────────────────
TARGET_COLUMN   = "Class"
TIME_COLUMN     = "Time"
AMOUNT_COLUMN   = "Amount"
PCA_FEATURES    = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = [TIME_COLUMN, AMOUNT_COLUMN] + PCA_FEATURES
ALL_COLUMNS     = FEATURE_COLUMNS + [TARGET_COLUMN]

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Train / test split ────────────────────────────────────────────────────────
TEST_SIZE = 0.20     # 80/20 stratified split

# ── Optional sample for faster experimentation ────────────────────────────────
# Set to None to use the full dataset; set an integer (e.g. 50_000) to sample.
SAMPLE_SIZE: int | None = None

# ── Model hyperparameters ─────────────────────────────────────────────────────
LR_PARAMS = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "solver": "lbfgs",
    "C": 1.0,
}

DT_PARAMS = {
    "max_depth": 10,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 12,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "tree_method": "hist",
}

IF_PARAMS = {
    "n_estimators": 100,
    "contamination": "auto",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ── SMOTE ────────────────────────────────────────────────────────────────────
SMOTE_PARAMS = {
    "sampling_strategy": 0.1,   # minority class becomes 10% of majority
    "random_state": RANDOM_STATE,
    "k_neighbors": 5,
}

# ── Best-model selection criterion ───────────────────────────────────────────
BEST_MODEL_METRIC = "f1"   # used to rank supervised classifiers

# ── Artifact file names ───────────────────────────────────────────────────────
PREPROCESSOR_PATH     = MODELS_DIR / "preprocessor.joblib"
SCALER_PATH           = MODELS_DIR / "scaler.joblib"
FEATURE_NAMES_PATH    = MODELS_DIR / "feature_names.json"
DATASET_META_PATH     = MODELS_DIR / "dataset_metadata.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.json"
BEST_MODEL_META_PATH  = MODELS_DIR / "best_model_meta.json"

def model_path(name: str) -> Path:
    """Return the joblib path for a named model."""
    return MODELS_DIR / f"{name}.joblib"

# ── SHAP ─────────────────────────────────────────────────────────────────────
SHAP_MAX_SAMPLES = 500   # rows used for global SHAP summary (speed)
