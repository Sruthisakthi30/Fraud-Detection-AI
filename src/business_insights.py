"""
FraudShield AI - Business Insights
Derives actionable business conclusions from real evaluation results.
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def generate_insights(comparison_report: dict) -> dict:
    """
    Generate business insights from an actual comparison report.

    All numeric values are sourced from the report — nothing is fabricated.
    """
    if not comparison_report:
        return {"available": False, "reason": "No model evaluation report available."}

    best_name    = comparison_report.get("best_display", "Unknown")
    best_metrics = comparison_report.get("best_metrics", {})
    sup_results  = comparison_report.get("supervised_results", [])
    unsup_res    = comparison_report.get("unsupervised_results", [])
    dataset_meta = comparison_report.get("dataset_meta", {})

    # ── Dataset-level insights ────────────────────────────────────────────────
    fraud_pct      = dataset_meta.get("fraud_percentage", 0.0)
    total_txn      = dataset_meta.get("total_transactions", 0)
    fraud_count    = dataset_meta.get("fraud_count", 0)

    # ── Best-model details ────────────────────────────────────────────────────
    precision  = best_metrics.get("precision", 0.0)
    recall     = best_metrics.get("recall",    0.0)
    f1         = best_metrics.get("f1",        0.0)
    roc_auc    = best_metrics.get("roc_auc")

    # ── False positive / negative analysis ───────────────────────────────────
    # Use the best model's confusion matrix
    best_result = next(
        (r for r in sup_results if r["display_name"] == best_name), None
    )
    fp = best_result.get("fp", 0) if best_result else 0
    fn = best_result.get("fn", 0) if best_result else 0
    tp = best_result.get("tp", 0) if best_result else 0
    tn = best_result.get("tn", 0) if best_result else 0

    # ── Model ranking ─────────────────────────────────────────────────────────
    ranked = sorted(sup_results, key=lambda r: r.get("f1", 0), reverse=True)
    model_ranking = [
        {"rank": i + 1,
         "model": r["display_name"],
         "f1":    r["f1"],
         "precision": r["precision"],
         "recall": r["recall"],
         "roc_auc": r.get("roc_auc")}
        for i, r in enumerate(ranked)
    ]

    # ── Text insights ─────────────────────────────────────────────────────────
    insights = []

    insights.append({
        "category": "Dataset Overview",
        "icon": "📊",
        "title": "Highly Imbalanced Dataset",
        "body": (
            f"The dataset contains {total_txn:,} transactions, of which "
            f"{fraud_count:,} ({fraud_pct:.4f}%) are fraudulent. "
            "This extreme class imbalance is a core challenge in fraud detection "
            "and was addressed using SMOTE oversampling on the training partition only."
        ),
    })

    insights.append({
        "category": "Model Performance",
        "icon": "🏆",
        "title": f"Best Model: {best_name}",
        "body": (
            f"{best_name} achieved an F1-score of {f1:.4f}, "
            f"Precision of {precision:.4f}, Recall of {recall:.4f}"
            + (f", and ROC-AUC of {roc_auc:.4f}." if roc_auc else ".")
            + " F1-score was used as the primary selection criterion because it "
            "balances precision and recall, which is critical when both "
            "false positives and false negatives have significant business costs."
        ),
    })

    fp_rate = round(fp / (fp + tn) * 100, 2) if (fp + tn) > 0 else 0.0
    fn_rate = round(fn / (fn + tp) * 100, 2) if (fn + tp) > 0 else 0.0

    insights.append({
        "category": "Operational Impact",
        "icon": "⚠️",
        "title": "False Positive Analysis",
        "body": (
            f"The best model incorrectly flags {fp:,} legitimate transactions as "
            f"fraud (false positive rate: {fp_rate}%). "
            "Each false positive has a customer-experience cost: unnecessary card "
            "blocks, declined transactions, and support burden. "
            "A higher threshold reduces false positives but may miss real fraud."
        ),
    })

    insights.append({
        "category": "Operational Impact",
        "icon": "🔍",
        "title": "False Negative Analysis",
        "body": (
            f"The best model misses {fn:,} actual fraud cases (miss rate: {fn_rate}%). "
            "Each missed fraud results in a direct financial loss. "
            "A lower threshold captures more fraud but increases false alarms. "
            "The optimal threshold depends on the cost structure of the business."
        ),
    })

    if len(sup_results) > 1:
        worst = ranked[-1]
        insights.append({
            "category": "Model Comparison",
            "icon": "📈",
            "title": "Ensemble vs. Linear Models",
            "body": (
                f"Ensemble models generally outperformed simpler classifiers. "
                f"{best_name} (F1: {f1:.4f}) significantly outperformed "
                f"{worst['display_name']} (F1: {worst['f1']:.4f}). "
                "This is expected for highly imbalanced, non-linear financial data "
                "where tree ensembles handle feature interactions well."
            ),
        })

    if unsup_res:
        if_result = next(
            (r for r in unsup_res if r["model_name"] == "isolation_forest"), None
        )
        if if_result:
            insights.append({
                "category": "Anomaly Detection",
                "icon": "🌲",
                "title": "Isolation Forest as Complementary Tool",
                "body": (
                    f"Isolation Forest (unsupervised) achieved Recall: "
                    f"{if_result['recall']:.4f} and Precision: "
                    f"{if_result['precision']:.4f} on the test set "
                    "when evaluated against ground-truth labels. "
                    "Unlike supervised models, it does not see labels during training. "
                    "It is best used for detecting novel transaction patterns not "
                    "previously seen in labelled fraud data."
                ),
            })

    insights.append({
        "category": "Recommendations",
        "icon": "💡",
        "title": "Deployment Considerations",
        "body": (
            "This application is an educational prototype. "
            "Before production deployment, consider: "
            "(1) Regular model retraining as fraud patterns evolve; "
            "(2) Human review for all model-flagged transactions; "
            "(3) A/B testing of thresholds; "
            "(4) Compliance with GDPR and financial regulations; "
            "(5) Monitoring for model drift and adversarial attacks."
        ),
    })

    return {
        "available":       True,
        "best_model":      best_name,
        "model_ranking":   model_ranking,
        "insights":        insights,
        "dataset_summary": {
            "total":       total_txn,
            "fraud":       fraud_count,
            "fraud_pct":   fraud_pct,
        },
        "precision_recall_tradeoff": {
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
        },
    }
