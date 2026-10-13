"""
FraudShield AI - Flask Routes
All REST API endpoints.
"""
import json
import logging
import sys
from pathlib import Path

from flask import Blueprint, jsonify, request, Response

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FEATURE_COLUMNS
from src.utils import risk_category, utc_now_iso, safe_json_dumps

logger = logging.getLogger(__name__)
api = Blueprint("api", __name__, url_prefix="/api")


def _json_response(data: dict, status: int = 200):
    """Return a Flask Response with safe JSON serialisation."""
    return Response(
        safe_json_dumps(data),
        status=status,
        mimetype="application/json",
    )


# ============================================================ HEALTH =========

@api.route("/health", methods=["GET"])
def health():
    from backend.services import get_dataset_status
    try:
        status = get_dataset_status()
    except Exception as e:
        status = {"error": str(e)}
    return _json_response({
        "status": "ok",
        "service": "FraudShield AI",
        "version": "1.0.0",
        "timestamp": utc_now_iso(),
        "dataset": status,
    })


# ============================================================ DATASET ========

@api.route("/dataset/summary", methods=["GET"])
def dataset_summary():
    from backend.services import get_dataset_summary_cached
    from src.data_loader import dataset_available, demo_dataset_available
    if not (dataset_available() or demo_dataset_available()):
        return _json_response({
            "available": False,
            "message": (
                "Dataset not found. Please download creditcard.csv from "
                "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
                "and place it in the data/ directory, or run python train_pipeline.py --demo."
            ),
        }, 404)
    try:
        summary = get_dataset_summary_cached()
        summary["available"] = True
        return _json_response(summary)
    except Exception as e:
        logger.exception("dataset/summary error")
        return _json_response({"available": False, "error": str(e)}, 500)


# ============================================================ EDA ============

@api.route("/eda", methods=["GET"])
def eda():
    from backend.services import get_eda_cached
    from src.data_loader import dataset_available, demo_dataset_available
    if not (dataset_available() or demo_dataset_available()):
        return _json_response({"available": False,
                               "message": "Dataset not loaded."}, 404)
    try:
        data = get_eda_cached()
        data["available"] = True
        return _json_response(data)
    except Exception as e:
        logger.exception("eda error")
        return _json_response({"available": False, "error": str(e)}, 500)


@api.route("/eda/feature", methods=["GET"])
def eda_feature():
    feature = request.args.get("name", "Amount")
    from backend.services import get_dataset
    from src.data_loader import dataset_available, demo_dataset_available
    from src.eda import feature_distribution_by_class
    if not (dataset_available() or demo_dataset_available()):
        return _json_response({"available": False}, 404)
    try:
        df   = get_dataset()
        data = feature_distribution_by_class(df, feature)
        data["available"] = True
        return _json_response(data)
    except Exception as e:
        logger.exception("eda/feature error")
        return _json_response({"available": False, "error": str(e)}, 400)


# ============================================================ MODELS =========

@api.route("/models/status", methods=["GET"])
def models_status():
    from src.train_models import (
        model_available, load_best_model_meta,
        MODEL_DISPLAY_NAMES,
    )
    statuses = {
        name: {
            "available":    model_available(name),
            "display_name": MODEL_DISPLAY_NAMES.get(name, name),
        }
        for name in MODEL_DISPLAY_NAMES
    }
    best = load_best_model_meta()
    return _json_response({
        "models":          statuses,
        "best_model":      best,
        "timestamp":       utc_now_iso(),
    })


@api.route("/models/comparison", methods=["GET"])
def models_comparison():
    from src.train_models import load_comparison_report
    report = load_comparison_report()
    if report is None:
        return _json_response({
            "available": False,
            "message": (
                "No model evaluation report found. "
                "Please run the training pipeline: python train_pipeline.py"
            ),
        }, 404)
    report["available"] = True
    return _json_response(report)


# ============================================================ PREDICT ========

@api.route("/predict", methods=["POST"])
def predict():
    """
    Predict fraud for a single transaction.

    Expected JSON body:
    {
        "Time": <float>,
        "Amount": <float>,
        "V1": <float>, ... "V28": <float>
    }
    """
    from backend.services import get_scaler_and_features, get_best_model
    from src.preprocessing import transform_single
    from src.train_models import load_best_model_meta

    # Validate that models are trained
    meta = load_best_model_meta()
    if meta is None:
        return _json_response({
            "error": "No trained model available. Run python train_pipeline.py first.",
            "available": False,
        }, 503)

    body = request.get_json(silent=True)
    if not body:
        return _json_response({"error": "JSON body required."}, 400)

    try:
        scaler, feature_names = get_scaler_and_features()
    except FileNotFoundError as e:
        return _json_response({"error": str(e), "available": False}, 503)

    # Validate all required features present
    missing = [f for f in feature_names if f not in body]
    if missing:
        return _json_response({
            "error": f"Missing features in request: {missing}",
            "required_features": feature_names,
        }, 400)

    # Validate numeric
    bad = []
    for f in feature_names:
        try:
            float(body[f])
        except (TypeError, ValueError):
            bad.append(f)
    if bad:
        return _json_response({"error": f"Non-numeric values for: {bad}"}, 400)

    try:
        X = transform_single(scaler, feature_names, {f: float(body[f]) for f in feature_names})
        model, model_name = get_best_model()

        pred_class = int(model.predict(X)[0])
        pred_prob  = None
        if hasattr(model, "predict_proba"):
            pred_prob = round(float(model.predict_proba(X)[0][1]), 6)

        risk = risk_category(pred_prob, pred_class)

        return _json_response({
            "prediction":      pred_class,
            "label":           "Fraud" if pred_class == 1 else "Legitimate",
            "probability":     pred_prob,
            "risk_category":   risk,
            "model_used":      model_name,
            "model_display":   meta.get("display_name", model_name),
            "disclaimer": (
                "This is an experimental model output. "
                "It is NOT a confirmed fraud finding."
            ),
        })

    except Exception as e:
        logger.exception("Prediction error")
        return _json_response({"error": str(e)}, 500)


# ============================================================ EXPLAIN ========

@api.route("/explain", methods=["POST"])
def explain():
    """
    SHAP explanation for a single transaction prediction.
    Same body format as /predict.
    """
    from backend.services import get_scaler_and_features, get_best_model, get_test_arrays
    from src.preprocessing import transform_single
    from src.explainability import build_explainer, local_explanation, global_feature_importance
    from src.train_models import load_best_model_meta
    import numpy as np

    meta = load_best_model_meta()
    if meta is None:
        return _json_response({"available": False,
                               "error": "No trained model. Run training first."}, 503)

    body = request.get_json(silent=True)
    if not body:
        return _json_response({"error": "JSON body required."}, 400)

    try:
        scaler, feature_names = get_scaler_and_features()
        model, model_name     = get_best_model()
        X_test, y_test        = get_test_arrays()

        missing = [f for f in feature_names if f not in body]
        if missing:
            return _json_response({"error": f"Missing features: {missing}"}, 400)

        X_instance = transform_single(
            scaler, feature_names,
            {f: float(body[f]) for f in feature_names}
        )

        # Build explainer using test set as background
        background = X_test[:200]
        explainer  = build_explainer(model, background, model_name)

        pred_class = int(model.predict(X_instance)[0])
        pred_prob  = None
        if hasattr(model, "predict_proba"):
            pred_prob = round(float(model.predict_proba(X_instance)[0][1]), 6)

        result = local_explanation(
            explainer, X_instance, feature_names, pred_class, pred_prob
        )
        return _json_response(result)

    except Exception as e:
        logger.exception("Explain error")
        return _json_response({"available": False, "error": str(e)}, 500)


@api.route("/explain/global", methods=["GET"])
def explain_global():
    """Global SHAP feature importance from the best model."""
    from backend.services import get_best_model, get_test_arrays
    from src.explainability import build_explainer, global_feature_importance
    from src.train_models import load_best_model_meta

    meta = load_best_model_meta()
    if meta is None:
        return _json_response({"available": False,
                               "error": "No trained model. Run training first."}, 503)

    try:
        model, model_name = get_best_model()
        X_test, _         = get_test_arrays()
        background        = X_test[:200]
        feature_names     = FEATURE_COLUMNS

        explainer = build_explainer(model, background, model_name)
        result    = global_feature_importance(explainer, X_test, feature_names)
        return _json_response(result)

    except Exception as e:
        logger.exception("Global SHAP error")
        return _json_response({"available": False, "error": str(e)}, 500)


# ============================================================ ANOMALY ========

@api.route("/anomaly", methods=["GET"])
def anomaly():
    """Anomaly detection results from Isolation Forest on the test set."""
    from backend.services import get_isolation_forest, get_test_arrays, get_dataset
    from src.anomaly_detection import score_transactions
    from src.data_loader import dataset_available
    from src.train_models import model_available

    if not model_available("isolation_forest"):
        return _json_response({
            "available": False,
            "message": "Isolation Forest not trained yet. Run training pipeline.",
        }, 404)

    try:
        if_model      = get_isolation_forest()
        X_test, y_test = get_test_arrays()

        # Build display frame from test set
        import pandas as pd
        import numpy as np
        df_test_display = pd.DataFrame(
            X_test,
            columns=FEATURE_COLUMNS,
        )
        df_test_display["Class"] = y_test

        result = score_transactions(if_model, X_test, df_test_display, n_top=50)
        result["available"] = True
        return _json_response(result)

    except Exception as e:
        logger.exception("Anomaly error")
        return _json_response({"available": False, "error": str(e)}, 500)


# ============================================================ BUSINESS ========

@api.route("/business-insights", methods=["GET"])
def business_insights():
    from src.train_models import load_comparison_report
    from src.business_insights import generate_insights

    report = load_comparison_report()
    if report is None:
        return _json_response({
            "available": False,
            "message": "No evaluation report. Run training pipeline first.",
        }, 404)

    insights = generate_insights(report)
    return _json_response(insights)


# ============================================================ TRANSACTIONS ===

@api.route("/transactions", methods=["GET"])
def transactions():
    from backend.services import get_transactions

    page     = request.args.get("page",     1,    type=int)
    per_page = request.args.get("per_page", 50,   type=int)
    cls      = request.args.get("class",    None, type=int)

    per_page = min(per_page, 200)  # cap for performance

    result = get_transactions(page=page, per_page=per_page, filter_class=cls)
    return _json_response(result)


# ============================================================ THRESHOLD ======

@api.route("/threshold-analysis", methods=["GET"])
def threshold_analysis():
    """Precision/Recall/F1 at multiple thresholds for the best supervised model."""
    from backend.services import get_best_model, get_test_arrays
    from src.evaluate_models import threshold_analysis as ta
    from src.train_models import load_best_model_meta

    meta = load_best_model_meta()
    if meta is None:
        return _json_response({"available": False,
                               "error": "No trained model."}, 503)

    try:
        model, _ = get_best_model()
        X_test, y_test = get_test_arrays()
        rows = ta(model, X_test, y_test)
        return _json_response({
            "available":    True,
            "model":        meta.get("display_name"),
            "thresholds":   rows,
        })
    except Exception as e:
        logger.exception("Threshold analysis error")
        return _json_response({"available": False, "error": str(e)}, 500)


# ============================================================ CASES ==========

@api.route("/cases", methods=["GET"])
def get_cases():
    """Retrieve investigation cases with optional filters."""
    from backend.cases import get_all_cases
    status = request.args.get("status")
    risk_level = request.args.get("risk_level")
    search = request.args.get("search")
    cases = get_all_cases(status=status, risk_level=risk_level, search=search)
    return _json_response({"cases": cases, "total": len(cases)})


@api.route("/cases", methods=["POST"])
def post_case():
    """Create a new investigation case."""
    from backend.cases import create_case
    body = request.get_json(silent=True)
    if not body:
        return _json_response({"error": "JSON payload required."}, 400)
    try:
        new_case = create_case(body)
        return _json_response(new_case, 201)
    except Exception as e:
        logger.exception("Failed to create case")
        return _json_response({"error": str(e)}, 500)


@api.route("/cases/<int:case_id>", methods=["PATCH"])
def patch_case(case_id: int):
    """Update case status or notes."""
    from backend.cases import update_case
    body = request.get_json(silent=True)
    if not body:
        return _json_response({"error": "JSON payload required."}, 400)
    res = update_case(case_id, body)
    if res is None:
        return _json_response({"error": "Case not found or nothing to update."}, 404)
    return _json_response(res)


@api.route("/cases/<int:case_id>", methods=["DELETE"])
def remove_case(case_id: int):
    """Delete a case."""
    from backend.cases import delete_case
    success = delete_case(case_id)
    if not success:
        return _json_response({"error": "Case not found."}, 404)
    return _json_response({"message": "Case deleted successfully.", "id": case_id})

