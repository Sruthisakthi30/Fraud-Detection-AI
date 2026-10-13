"""
FraudShield AI - Explainability (SHAP)
Generates global feature importance and local explanations.
"""
import logging
from typing import Any

import numpy as np

from src.config import FEATURE_COLUMNS, SHAP_MAX_SAMPLES

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Explainer builder                                                           #
# --------------------------------------------------------------------------- #

def build_explainer(model: Any, X_background: np.ndarray, model_name: str):
    """
    Return a SHAP explainer appropriate for the given model type.

    - Tree-based models (RF, XGB, DT): TreeExplainer (fast, exact)
    - Logistic Regression:              LinearExplainer
    - Fallback:                         KernelExplainer (slow, model-agnostic)
    """
    import shap

    tree_models = ("random_forest", "xgboost", "decision_tree")
    linear_models = ("logistic_regression",)

    if model_name in tree_models:
        explainer = shap.TreeExplainer(model)
    elif model_name in linear_models:
        # Use the mean of background data as a single summary
        background = shap.maskers.Independent(X_background[:200])
        explainer = shap.LinearExplainer(model, background)
    else:
        background_summary = shap.kmeans(X_background, 10)
        explainer = shap.KernelExplainer(model.predict_proba, background_summary)

    return explainer


# --------------------------------------------------------------------------- #
#  Global feature importance                                                   #
# --------------------------------------------------------------------------- #

def global_feature_importance(
    explainer,
    X_test: np.ndarray,
    feature_names: list[str],
    max_samples: int = SHAP_MAX_SAMPLES,
) -> dict:
    """
    Compute mean |SHAP| values for global feature importance.

    Returns a JSON-serialisable dict.
    """
    import shap

    # Subsample for speed
    n = min(max_samples, len(X_test))
    idx = np.random.default_rng(42).choice(len(X_test), n, replace=False)
    X_sample = X_test[idx]

    try:
        shap_vals = explainer(X_sample)
        # Handle multi-output (binary classification usually gives (n, features, 2))
        if hasattr(shap_vals, "values"):
            vals = shap_vals.values
        else:
            vals = shap_vals

        if vals.ndim == 3:
            # Take class 1 (fraud) SHAP values
            vals = vals[:, :, 1]

        mean_abs = np.abs(vals).mean(axis=0)

    except Exception as e:
        logger.warning("TreeExplainer failed (%s); retrying with interventional...", e)
        try:
            import shap as shap2
            explainer2 = shap2.TreeExplainer(
                explainer.model,
                feature_perturbation="interventional",
                data=X_sample[:100],
            )
            vals = explainer2.shap_values(X_sample)
            if isinstance(vals, list):
                vals = vals[1]
            mean_abs = np.abs(vals).mean(axis=0)
        except Exception as e2:
            logger.error("SHAP global explanation failed: %s", e2)
            return {"error": str(e2), "available": False}

    importance = [
        {"feature": fn, "importance": round(float(v), 6)}
        for fn, v in zip(feature_names, mean_abs)
    ]
    importance.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "available":         True,
        "n_samples_used":    n,
        "feature_importance": importance,
    }


# --------------------------------------------------------------------------- #
#  Local explanation for a single instance                                     #
# --------------------------------------------------------------------------- #

def local_explanation(
    explainer,
    x_instance: np.ndarray,
    feature_names: list[str],
    predicted_class: int,
    predicted_prob: float | None,
) -> dict:
    """
    Compute SHAP values for a single row.

    Returns a JSON-serialisable dict.
    """
    import shap

    x_2d = x_instance.reshape(1, -1)

    try:
        shap_vals = explainer(x_2d)

        if hasattr(shap_vals, "values"):
            vals = shap_vals.values
        else:
            vals = shap_vals

        if vals.ndim == 3:
            vals = vals[:, :, 1]   # fraud class

        row_vals = vals[0]         # shape (n_features,)

        # Base value
        if hasattr(shap_vals, "base_values"):
            base = shap_vals.base_values
            if hasattr(base, "__len__"):
                base = float(base[0][1]) if base[0].ndim > 0 else float(base[0])
            else:
                base = float(base)
        else:
            base = None

    except Exception as e:
        logger.warning("Local explanation failed (%s); retrying with shap_values...", e)
        try:
            row_vals_raw = explainer.shap_values(x_2d)
            if isinstance(row_vals_raw, list):
                row_vals_raw = row_vals_raw[1]
            row_vals = row_vals_raw[0]
            base     = None
        except Exception as e2:
            logger.error("Local SHAP failed: %s", e2)
            return {"available": False, "error": str(e2)}

    contributions = [
        {
            "feature":      fn,
            "shap_value":   round(float(v), 6),
            "direction":    "increases_fraud_risk" if v > 0 else "decreases_fraud_risk",
        }
        for fn, v in zip(feature_names, row_vals)
    ]
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    top_positive = [c for c in contributions if c["shap_value"] > 0][:5]
    top_negative = [c for c in contributions if c["shap_value"] < 0][:5]

    return {
        "available":          True,
        "predicted_class":    predicted_class,
        "predicted_prob":     predicted_prob,
        "base_value":         base,
        "contributions":      contributions,
        "top_positive":       top_positive,
        "top_negative":       top_negative,
        "interpretation": (
            f"SHAP values show each feature's contribution to the model output. "
            f"Positive values push the prediction toward fraud (Class 1); "
            f"negative values push it toward legitimate (Class 0). "
            f"The predicted class is {predicted_class} "
            + (f"with probability {predicted_prob:.4f}." if predicted_prob is not None
               else ".")
        ),
    }
