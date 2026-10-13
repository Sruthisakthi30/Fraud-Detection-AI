"""
FraudShield AI - Full Training Pipeline
Run this script to train all models and generate evaluation artifacts.

Usage:
    python train_pipeline.py
    python train_pipeline.py --sample 50000
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import setup_logging, utc_now_iso, ensure_dirs
from src.config import (
    DATASET_PATH, MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
    FEATURE_COLUMNS, RANDOM_STATE,
)

setup_logging()
logger = logging.getLogger(__name__)


def run(sample_size: int | None = None, use_demo: bool = False) -> None:
    from src.data_loader import (
        dataset_available, load_dataset, load_demo_dataset, get_dataset_summary
    )
    from src.preprocessing import (
        run_preprocessing, save_preprocessor
    )
    from src.train_models import (
        train_supervised, train_isolation_forest,
        save_comparison_report, save_best_model_meta, save_dataset_meta,
        MODEL_DISPLAY_NAMES,
    )
    from src.evaluate_models import (
        evaluate_supervised, evaluate_isolation_forest,
        build_comparison_report,
    )
    from src.eda import get_full_eda
    import json

    ensure_dirs(MODELS_DIR, REPORTS_DIR, FIGURES_DIR)

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    if use_demo:
        logger.warning(
            "\n" + "="*70 +
            "\nRUNNING WITH SYNTHETIC DEMONSTRATION DATA (--demo flag active)\n"
            "This synthetic dataset is strictly for smoke testing and UI verification.\n"
            "DO NOT use these metrics to claim real-world fraud detection performance.\n" +
            "="*70
        )
        df = load_demo_dataset()
        sampled = False
        summary = get_dataset_summary(df, sampled=sampled, is_demo=True)
    else:
        if not dataset_available():
            logger.error(
                "\n" + "="*60 +
                "\nDATASET NOT FOUND at: %s\n"
                "Please download 'creditcard.csv' from:\n"
                "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
                "and place it in the 'data/' directory.\n"
                "To run a demonstration test with synthetic data, use:\n"
                "  python train_pipeline.py --demo\n" +
                "="*60, DATASET_PATH
            )
            sys.exit(1)

        logger.info("Loading dataset...")
        df = load_dataset(sample_size=sample_size)
        sampled = sample_size is not None and sample_size < 284_807
        summary = get_dataset_summary(df, sampled=sampled, is_demo=False)

    # ── 2. EDA (persist for API) ─────────────────────────────────────────────
    logger.info("Computing EDA...")
    eda_data = get_full_eda(df)
    eda_path = REPORTS_DIR / "eda_cache.json"
    with open(eda_path, "w") as f:
        json.dump(eda_data, f)
    logger.info("EDA saved → %s", eda_path)

    # ── 3. Preprocessing ─────────────────────────────────────────────────────
    logger.info("Running preprocessing pipeline...")
    prep = run_preprocessing(df)
    save_preprocessor(prep["scaler"], prep["feature_names"])

    X_train     = prep["X_train"]      # SMOTE-applied, scaled
    X_test      = prep["X_test"]       # scaled, never touched by SMOTE
    y_train     = prep["y_train"]
    y_test      = prep["y_test"]
    X_train_raw = prep["X_train_raw"]  # for Isolation Forest (not SMOTE'd)
    meta        = prep["meta"]

    # Save preprocessing meta
    save_dataset_meta({**summary, **meta, "timestamp": utc_now_iso()})

    # Also persist test set for later inference demos
    import numpy as np
    np.save(MODELS_DIR / "X_test.npy", X_test)
    np.save(MODELS_DIR / "y_test.npy", y_test.values)

    # ── 4. Supervised classifiers ─────────────────────────────────────────────
    supervised_names = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
    ]

    results = []
    trained_models = {}

    for name in supervised_names:
        model, elapsed = train_supervised(name, X_train, y_train)
        trained_models[name] = model
        eval_result = evaluate_supervised(name, model, X_test, y_test, elapsed)
        results.append(eval_result)
        logger.info(
            "%s → F1: %.4f | Prec: %.4f | Recall: %.4f | ROC-AUC: %s",
            MODEL_DISPLAY_NAMES[name],
            eval_result["f1"],
            eval_result["precision"],
            eval_result["recall"],
            f"{eval_result['roc_auc']:.4f}" if eval_result["roc_auc"] else "N/A",
        )

    # ── 5. Isolation Forest ───────────────────────────────────────────────────
    if_model, if_elapsed = train_isolation_forest(X_train_raw)
    if_result = evaluate_isolation_forest(if_model, X_test, y_test, if_elapsed)
    results.append(if_result)
    logger.info(
        "Isolation Forest → F1: %.4f | Prec: %.4f | Recall: %.4f",
        if_result["f1"], if_result["precision"], if_result["recall"],
    )

    # ── 6. Comparison report ──────────────────────────────────────────────────
    dataset_meta = {**summary, **meta}
    report = build_comparison_report(results, dataset_meta)
    save_comparison_report(report)

    # ── 7. Best model meta ────────────────────────────────────────────────────
    best_name = report["best_model"]
    save_best_model_meta({
        "model_name":    best_name,
        "display_name":  report["best_display"],
        "selection_metric": report["selection_metric"],
        "metrics":       report["best_metrics"],
        "timestamp":     report["timestamp"],
        "is_sampled":    sampled,
        "sample_size":   sample_size,
    })

    logger.info("\n" + "="*60)
    logger.info("Training complete.")
    logger.info("Best model: %s (F1 = %.4f)", report["best_display"],
                report["best_metrics"]["f1"])
    logger.info("Artifacts saved to: %s", MODELS_DIR)
    logger.info("Reports saved to:   %s", REPORTS_DIR)
    logger.info("="*60)

    if sampled:
        logger.warning(
            "SAMPLED TRAINING: Results are based on %d rows, not the full dataset. "
            "Metrics may differ from full-dataset training. "
            "Re-run without --sample for production-quality results.",
            sample_size,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudShield AI Training Pipeline")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Optional: use only N rows (stratified). "
             "Useful for fast experimentation. "
             "Results will be labelled as SAMPLED."
    )
    parser.add_argument(
        "--demo", action="store_true", default=False,
        help="Optional: run training on isolated synthetic demonstration data. "
             "Results will be explicitly labelled as SYNTHETIC DEMONSTRATION DATA."
    )
    args = parser.parse_args()
    run(sample_size=args.sample, use_demo=args.demo)
