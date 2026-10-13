"""
FraudShield AI - Anomaly Detection
Isolation Forest scoring, ranking, and threshold analysis.
"""
import logging

import numpy as np
import pandas as pd

from src.config import TARGET_COLUMN, FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def score_transactions(
    model,
    X_scaled: np.ndarray,
    df_original: pd.DataFrame,
    n_top: int = 50,
) -> dict:
    """
    Compute Isolation Forest anomaly scores for a scaled feature matrix.

    Parameters
    ----------
    model       : fitted IsolationForest
    X_scaled    : scaled test features (numpy array)
    df_original : original (unscaled) test rows for display purposes
    n_top       : how many most anomalous rows to return

    Returns
    -------
    dict with anomaly_scores, threshold, top_anomalies
    """
    scores = model.decision_function(X_scaled)  # higher = more normal
    predictions = model.predict(X_scaled)        # +1 normal, -1 anomaly

    anomaly_scores = -scores                     # higher = more anomalous
    is_anomaly = (predictions == -1).astype(int)

    # Build ranked frame
    result_df = df_original.reset_index(drop=True).copy()
    result_df["anomaly_score"] = anomaly_scores
    result_df["is_anomaly"]    = is_anomaly

    top = (
        result_df
        .sort_values("anomaly_score", ascending=False)
        .head(n_top)
    )

    # Serialise only key columns to keep response small
    display_cols = (
        [c for c in [TARGET_COLUMN, "Amount", "Time"] if c in top.columns]
        + ["anomaly_score", "is_anomaly"]
    )

    rows = []
    for _, row in top[display_cols].iterrows():
        record = {}
        for col in display_cols:
            val = row[col]
            if isinstance(val, float):
                val = round(val, 6)
            elif hasattr(val, "item"):
                val = val.item()
            record[col] = val
        rows.append(record)

    total_anomalies = int(is_anomaly.sum())
    total_normal    = int((is_anomaly == 0).sum())

    return {
        "total_scored":      len(anomaly_scores),
        "total_anomalies":   total_anomalies,
        "total_normal":      total_normal,
        "anomaly_rate_pct":  round(total_anomalies / len(anomaly_scores) * 100, 4)
                             if len(anomaly_scores) else 0.0,
        "score_stats": {
            "min":  round(float(anomaly_scores.min()), 6),
            "max":  round(float(anomaly_scores.max()), 6),
            "mean": round(float(anomaly_scores.mean()), 6),
            "std":  round(float(anomaly_scores.std()), 6),
        },
        "top_anomalies":   rows,
        "methodology": (
            "Isolation Forest is an unsupervised anomaly detector. "
            "It does NOT use fraud labels during training. "
            "Anomaly scores are relative – they reflect statistical rarity, "
            "not confirmed fraud. A high anomaly score means the transaction "
            "is unusual compared to the training distribution."
        ),
    }
