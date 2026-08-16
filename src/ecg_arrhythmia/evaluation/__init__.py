"""Evaluation module exports."""

from ecg_arrhythmia.evaluation.metrics import compute_metrics
from ecg_arrhythmia.evaluation.significance import (
    compute_pairwise_pvalues,
    paired_ttest,
    wilcoxon_test,
)

__all__ = [
    "compute_metrics",
    "paired_ttest",
    "wilcoxon_test",
    "compute_pairwise_pvalues",
]
