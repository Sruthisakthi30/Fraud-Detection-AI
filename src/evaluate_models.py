"""
FraudShield AI - Model Evaluation
Computes classification metrics and builds the comparison report.
"""
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    average_precision_score,
)

from src.config import BEST_MODEL_METRIC
from src.train_models import MODEL_DISPLAY_NAMES

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Supervised evaluation                                                       #
# --------------------------------------------------------------------------- #

def evaluate_supervised(
    name: str,
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    training_time: float = 0.0,
) -> dict:
    """
    Compute classification metrics for a fitted supervised model.

    Returns a JSON-serialisable dict.
    """
    y_pred = model.predict(X_test)

    # Probabilities (if supported)
    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    acc  = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec  = float(recall_score(y_test, y_pred, zero_division=0))
    f1   = float(f1_score(y_test, y_pred, zero_division=0))
    cm   = confusion_matrix(y_test, y_pred).tolist()

    roc_auc = None
    pr_auc  = None
    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_test, y_prob))
            pr_auc  = float(average_precision_score(y_test, y_prob))
        except Exception:
            pass

    # ROC curve points (sampled to ≤200 for API efficiency)
    roc_points = _compute_roc_points(y_test, y_prob)

    tn, fp, fn, tp = np.array(cm).ravel()

    return {
        "model_name":      name,
        "display_name":    MODEL_DISPLAY_NAMES.get(name, name),
        "model_type":      "supervised_classifier",
        "accuracy":        round(acc,  6),
        "precision":       round(prec, 6),
        "recall":          round(rec,  6),
        "f1":              round(f1,   6),
        "roc_auc":         round(roc_auc, 6) if roc_auc is not None else None,
        "pr_auc":          round(pr_auc,  6) if pr_auc  is not None else None,
        "confusion_matrix": cm,
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
        "roc_curve":       roc_points,
        "has_proba":       has_proba,
        "training_time_s": training_time,
    }


def _compute_roc_points(
    y_test: np.ndarray,
    y_prob: np.ndarray | None,
    n_points: int = 100,
) -> dict | None:
    """Return FPR / TPR pairs for the ROC curve (downsampled)."""
    if y_prob is None:
        return None
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        idx = np.linspace(0, len(fpr) - 1, min(n_points, len(fpr)), dtype=int)
        return {
            "fpr": [round(float(v), 6) for v in fpr[idx]],
            "tpr": [round(float(v), 6) for v in tpr[idx]],
        }
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  Anomaly detection evaluation                                                #
# --------------------------------------------------------------------------- #

def evaluate_isolation_forest(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    training_time: float = 0.0,
) -> dict:
    """
    Evaluate Isolation Forest by converting its decision scores to binary
    predictions at the contamination threshold.

    NOTE: Isolation Forest is unsupervised; its labels are inferred, not
    learned from ground-truth supervision. Metrics here reflect *agreement*
    with ground-truth labels, not training performance.
    """
    # IF returns +1 (normal) / -1 (anomaly)
    raw_pred = model.predict(X_test)          # +1 / -1
    y_pred   = (raw_pred == -1).astype(int)   # 1=anomaly, 0=normal

    scores = model.decision_function(X_test)  # higher = more normal

    acc  = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec  = float(recall_score(y_test, y_pred, zero_division=0))
    f1   = float(f1_score(y_test, y_pred, zero_division=0))
    cm   = confusion_matrix(y_test, y_pred).tolist()

    # ROC using decision scores (negate: lower score = more anomalous)
    roc_auc = None
    roc_points = None
    try:
        roc_auc = float(roc_auc_score(y_test, -scores))
        roc_points = _compute_roc_points(y_test, -scores)
    except Exception:
        pass

    tn, fp, fn, tp = np.array(cm).ravel()

    return {
        "model_name":      "isolation_forest",
        "display_name":    "Isolation Forest",
        "model_type":      "unsupervised_anomaly_detector",
        "accuracy":        round(acc,  6),
        "precision":       round(prec, 6),
        "recall":          round(rec,  6),
        "f1":              round(f1,   6),
        "roc_auc":         round(roc_auc, 6) if roc_auc is not None else None,
        "pr_auc":          None,
        "confusion_matrix": cm,
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
        "roc_curve":       roc_points,
        "has_proba":       False,
        "training_time_s": training_time,
        "note": (
            "Isolation Forest is unsupervised. Metrics reflect agreement with "
            "ground-truth labels at the inferred contamination threshold and "
            "are not directly comparable with supervised classifier metrics."
        ),
    }


# --------------------------------------------------------------------------- #
#  Comparison report                                                           #
# --------------------------------------------------------------------------- #

def build_comparison_report(
    results: list[dict],
    dataset_meta: dict,
) -> dict:
    """
    Build the full comparison report persisted to results/reports/.

    Selects the best supervised classifier by BEST_MODEL_METRIC (F1).
    """
    supervised = [r for r in results if r["model_type"] == "supervised_classifier"]
    unsupervised = [r for r in results if r["model_type"] != "supervised_classifier"]

    # Select best supervised model
    best = max(supervised, key=lambda r: r.get(BEST_MODEL_METRIC, 0.0))

    report = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "selection_metric": BEST_MODEL_METRIC,
        "best_model":       best["model_name"],
        "best_display":     best["display_name"],
        "best_metrics": {
            "f1":       best["f1"],
            "precision": best["precision"],
            "recall":   best["recall"],
            "roc_auc":  best["roc_auc"],
            "accuracy": best["accuracy"],
        },
        "supervised_results":   supervised,
        "unsupervised_results": unsupervised,
        "dataset_meta":         dataset_meta,
    }
    return report


# --------------------------------------------------------------------------- #
#  Threshold analysis                                                          #
# --------------------------------------------------------------------------- #

def threshold_analysis(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[dict]:
    """
    Compute precision/recall/F1 at multiple probability thresholds.
    Only valid for models with predict_proba.
    """
    if not hasattr(model, "predict_proba"):
        return []

    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.1, 1.0, 0.05)]

    y_prob = model.predict_proba(X_test)[:, 1]
    rows = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "threshold": round(thresh, 3),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "fp":        int(fp),
            "fn":        int(fn),
            "tp":        int(tp),
            "tn":        int(tn),
        })
    return rows
