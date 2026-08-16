"""Evaluation metrics for ECG heartbeat classification."""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

from ecg_arrhythmia.data.preprocessor import CLASS_NAMES


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics tailored for imbalanced arrhythmia detection.

    Args:
        y_true: 1D array of ground truth labels
        y_pred: 1D array of predicted class indices
        target_names: List of class names (default: ['N', 'S', 'V', 'F', 'Q'])

    Returns:
        Dictionary containing overall and per-class metrics
    """
    if target_names is None:
        target_names = CLASS_NAMES

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class Sensitivity (Recall) and Precision
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    sens_dict = {
        f"sensitivity_{name}": float(per_class_rec[i]) if i < len(per_class_rec) else 0.0
        for i, name in enumerate(target_names)
    }
    f1_dict = {
        f"f1_{name}": float(per_class_f1[i]) if i < len(per_class_f1) else 0.0
        for i, name in enumerate(target_names)
    }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))

    results = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_sensitivity": macro_rec,
        "confusion_matrix": cm,
        **sens_dict,
        **f1_dict
    }

    return results
