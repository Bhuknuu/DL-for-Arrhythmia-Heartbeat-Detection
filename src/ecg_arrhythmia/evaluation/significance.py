"""Statistical significance testing across cross-validation folds."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def paired_ttest(
    scores_a: List[float], scores_b: List[float]
) -> Tuple[float, float]:
    """Computes paired Student's t-test between fold scores of two models.

    Returns:
        (t_statistic, p_value)
    """
    res = stats.ttest_rel(scores_a, scores_b)
    return float(res.statistic), float(res.pvalue)


def wilcoxon_test(
    scores_a: List[float], scores_b: List[float]
) -> Tuple[float, float]:
    """Computes Wilcoxon signed-rank test between fold scores of two models.

    Returns:
        (statistic, p_value)
    """
    diffs = np.array(scores_a) - np.array(scores_b)
    if np.all(diffs == 0):
        return 0.0, 1.0
    res = stats.wilcoxon(scores_a, scores_b, zero_method="wilcox")
    return float(res.statistic), float(res.pvalue)


def compute_pairwise_pvalues(
    model_fold_metrics: Dict[str, List[float]],
    test_type: str = "wilcoxon"
) -> pd.DataFrame:
    """Generates a square matrix of p-values for all pairs of models across CV folds.

    Args:
        model_fold_metrics: Dict mapping model_name -> list of fold scores (e.g. 10 Macro-F1 scores)
        test_type: 'wilcoxon' or 'ttest'

    Returns:
        DataFrame p-value matrix
    """
    model_names = list(model_fold_metrics.keys())
    n = len(model_names)
    matrix = np.ones((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i != j:
                scores_i = model_fold_metrics[model_names[i]]
                scores_j = model_fold_metrics[model_names[j]]
                if test_type == "ttest":
                    _, p = paired_ttest(scores_i, scores_j)
                else:
                    _, p = wilcoxon_test(scores_i, scores_j)
                matrix[i, j] = p

    df = pd.DataFrame(matrix, index=model_names, columns=model_names)
    return df
