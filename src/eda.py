"""
FraudShield AI - Exploratory Data Analysis
Computes statistics and chart-ready data from the loaded dataframe.
"""
import logging

import numpy as np
import pandas as pd

from src.config import (
    TARGET_COLUMN,
    TIME_COLUMN,
    AMOUNT_COLUMN,
    PCA_FEATURES,
    FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)


def class_distribution(df: pd.DataFrame) -> dict:
    """Fraud vs legitimate count and percentage."""
    total  = len(df)
    fraud  = int(df[TARGET_COLUMN].sum())
    legit  = total - fraud
    return {
        "fraud":             fraud,
        "legitimate":        legit,
        "total":             total,
        "fraud_pct":         round(fraud / total * 100, 4),
        "legitimate_pct":    round(legit / total * 100, 4),
        "imbalance_ratio":   round(legit / fraud, 2) if fraud else None,
    }


def amount_distribution(df: pd.DataFrame, bins: int = 30) -> dict:
    """
    Histogram data for transaction Amount, split by class.
    Returns bin edges and counts for legitimate and fraud transactions.
    """
    legit_amounts = df.loc[df[TARGET_COLUMN] == 0, AMOUNT_COLUMN]
    fraud_amounts = df.loc[df[TARGET_COLUMN] == 1, AMOUNT_COLUMN]

    # Use a shared range but cap at 99th percentile to reduce tail distortion
    cap = float(df[AMOUNT_COLUMN].quantile(0.99))
    bins_arr = np.linspace(0, cap, bins + 1)

    legit_counts, edges = np.histogram(legit_amounts.clip(upper=cap), bins=bins_arr)
    fraud_counts, _     = np.histogram(fraud_amounts.clip(upper=cap), bins=bins_arr)

    return {
        "bin_edges":    [round(e, 2) for e in edges.tolist()],
        "legitimate":   legit_counts.tolist(),
        "fraud":        fraud_counts.tolist(),
        "cap_value":    round(cap, 2),
    }


def time_distribution(df: pd.DataFrame, bins: int = 48) -> dict:
    """
    Transaction count over time (hours since start), split by class.
    """
    df_copy = df[[TIME_COLUMN, TARGET_COLUMN]].copy()
    df_copy["hour"] = (df_copy[TIME_COLUMN] / 3600).astype(int)

    max_hour = int(df_copy["hour"].max()) + 1
    hours    = list(range(max_hour))

    legit_by_hour = (
        df_copy[df_copy[TARGET_COLUMN] == 0]
        .groupby("hour").size().reindex(hours, fill_value=0).tolist()
    )
    fraud_by_hour = (
        df_copy[df_copy[TARGET_COLUMN] == 1]
        .groupby("hour").size().reindex(hours, fill_value=0).tolist()
    )

    return {
        "hours":      hours,
        "legitimate": legit_by_hour,
        "fraud":      fraud_by_hour,
    }


def feature_statistics(df: pd.DataFrame) -> list[dict]:
    """Per-feature descriptive statistics."""
    stats = []
    for col in FEATURE_COLUMNS:
        s = df[col].describe()
        stats.append({
            "feature": col,
            "mean":   round(float(s["mean"]), 6),
            "std":    round(float(s["std"]),  6),
            "min":    round(float(s["min"]),  6),
            "25%":    round(float(s["25%"]),  6),
            "50%":    round(float(s["50%"]),  6),
            "75%":    round(float(s["75%"]),  6),
            "max":    round(float(s["max"]),  6),
        })
    return stats


def correlation_matrix(df: pd.DataFrame, top_n: int = 15) -> dict:
    """
    Return a correlation matrix for the top_n features most correlated with
    the target, plus the target itself.
    """
    corr_with_target = (
        df[FEATURE_COLUMNS].corrwith(df[TARGET_COLUMN]).abs()
        .sort_values(ascending=False)
    )
    top_features = corr_with_target.head(top_n).index.tolist()
    sub = df[top_features + [TARGET_COLUMN]]
    mat = sub.corr().round(4)

    return {
        "features":    mat.columns.tolist(),
        "matrix":      mat.values.tolist(),
        "top_features": top_features,
    }


def feature_distribution_by_class(
    df: pd.DataFrame, feature: str, bins: int = 30
) -> dict:
    """
    Histogram of a single feature split by class.
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not in dataframe.")

    legit = df.loc[df[TARGET_COLUMN] == 0, feature]
    fraud = df.loc[df[TARGET_COLUMN] == 1, feature]

    lo = float(df[feature].quantile(0.01))
    hi = float(df[feature].quantile(0.99))
    edges = np.linspace(lo, hi, bins + 1)

    l_counts, _ = np.histogram(legit.clip(lower=lo, upper=hi), bins=edges)
    f_counts, _ = np.histogram(fraud.clip(lower=lo, upper=hi), bins=edges)

    return {
        "feature":    feature,
        "bin_edges":  [round(e, 4) for e in edges.tolist()],
        "legitimate": l_counts.tolist(),
        "fraud":      f_counts.tolist(),
    }


def get_full_eda(df: pd.DataFrame) -> dict:
    """
    Collect all EDA outputs into a single dict for the API.
    """
    logger.info("Computing EDA for %d rows...", len(df))
    return {
        "class_distribution":   class_distribution(df),
        "amount_distribution":  amount_distribution(df),
        "time_distribution":    time_distribution(df),
        "feature_statistics":   feature_statistics(df),
        "correlation_matrix":   correlation_matrix(df),
        "feature_list":         FEATURE_COLUMNS,
    }
