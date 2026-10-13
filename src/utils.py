"""
FraudShield AI - Utilities
Shared helpers: logging setup, JSON serialisation, risk categorisation.
"""
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
#  Logging                                                                     #
# --------------------------------------------------------------------------- #

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with timestamp and level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# --------------------------------------------------------------------------- #
#  JSON serialisation                                                          #
# --------------------------------------------------------------------------- #

class _SafeEncoder(json.JSONEncoder):
    """Handle numpy types, NaN/Inf, and Path objects."""
    def default(self, obj: Any) -> Any:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if math.isnan(obj) or math.isinf(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def safe_json_dumps(obj: Any, **kwargs) -> str:
    return json.dumps(obj, cls=_SafeEncoder, **kwargs)


# --------------------------------------------------------------------------- #
#  Risk categorisation                                                         #
# --------------------------------------------------------------------------- #

def risk_category(probability: float | None, predicted_class: int) -> str:
    """
    Convert predicted class and probability into a human-readable risk label.
    This is a heuristic for demonstration — not a financial compliance tool.
    """
    if predicted_class == 0:
        if probability is None or probability < 0.1:
            return "Low Risk"
        elif probability < 0.3:
            return "Low-Medium Risk"
        else:
            return "Medium Risk"
    else:
        if probability is None or probability >= 0.8:
            return "High Risk"
        elif probability >= 0.5:
            return "Medium-High Risk"
        else:
            return "Medium Risk"


# --------------------------------------------------------------------------- #
#  Timestamp helper                                                            #
# --------------------------------------------------------------------------- #

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
#  Ensure directories exist                                                    #
# --------------------------------------------------------------------------- #

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
