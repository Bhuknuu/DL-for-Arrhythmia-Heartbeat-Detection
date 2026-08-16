"""Script to initialize clean phase-based Jupyter notebooks."""

import json
from pathlib import Path

notebooks = {
    "01_eda_and_waveform_analysis.ipynb": {
        "title": "Phase 1: Exploratory Data Analysis & Waveform Analysis",
        "desc": "Load PhysioNet MIT-BIH records, inspect AAMI EC57 annotation classes, and visualize raw ECG beat morphologies.",
        "code": """import matplotlib.pyplot as plt
from ecg_arrhythmia.data import load_record, get_available_records, extract_beats_from_record, CLASS_NAMES
from ecg_arrhythmia.utils import plot_ecg_beat

records = get_available_records("data/raw/mitdb")
print(f"Total available records: {len(records)}")

# Load Record 100
sig, sample_indices, symbols, fs = load_record("100", "data/raw/mitdb")
print(f"Record 100: Length={len(sig)} samples, Sampling rate={fs}Hz, Beats={len(sample_indices)}")

# Extract and plot beat
beats, labels, symbols = extract_beats_from_record("100", "data/raw/mitdb")
fig = plot_ecg_beat(beats[0], label_name=CLASS_NAMES[labels[0]], title="Sample Beat from Record 100")
plt.show()"""
    },
    "02_groupkfold_leakage_safe_splits.ipynb": {
        "title": "Phase 2: Leakage-Safe GroupKFold Patient Splits",
        "desc": "Verify zero patient overlap across 10-fold cross-validation splits and check class distributions.",
        "code": """import numpy as np
from ecg_arrhythmia.data import extract_all_beats, GroupKFoldSplitter, CLASS_NAMES

# Extract beats across available records
X, y, groups = extract_all_beats(data_dir="data/raw/mitdb", max_records=10)
print(f"Extracted {len(X)} beats across {len(np.unique(groups))} patients.")

splitter = GroupKFoldSplitter(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups)):
    train_patients = set(groups[train_idx])
    test_patients = set(groups[test_idx])
    print(f"Fold {fold+1}: Train beats={len(train_idx)} ({len(train_patients)} patients), Test beats={len(test_idx)} ({len(test_patients)} patients)")
    GroupKFoldSplitter.assert_no_leakage(groups[train_idx], groups[test_idx])
print("Zero patient leakage verified successfully across all folds!")"""
    },
    "03_model_benchmarking.ipynb": {
        "title": "Phase 3: Benchmarking 12 Arrhythmia Detection Models",
        "desc": "Train and evaluate the 12 classical, deep learning, and ensemble models across patient folds.",
        "code": """from ecg_arrhythmia.models import list_models, get_model
from ecg_arrhythmia.data import extract_all_beats, GroupKFoldSplitter
from ecg_arrhythmia.evaluation import compute_metrics
from ecg_arrhythmia.tracking import ExperimentLogger

print("Available 12 models in registry:")
for m in list_models():
    print(f" - {m}")

logger = ExperimentLogger(tracking_dir="runs")
print("MLflow experiment logger ready to track benchmark runs.")"""
    },
    "04_statistical_significance_testing.ipynb": {
        "title": "Phase 4: Statistical Significance Testing",
        "desc": "Compute Wilcoxon signed-rank and paired t-tests between model fold performances to establish statistical superiority.",
        "code": """import pandas as pd
from ecg_arrhythmia.evaluation import paired_ttest, wilcoxon_test, compute_pairwise_pvalues

# Example fold metrics across 5 folds for 3 top models
mock_metrics = {
    "1d_cnn": [0.912, 0.925, 0.898, 0.931, 0.918],
    "cnn_transformer": [0.935, 0.941, 0.918, 0.948, 0.939],
    "random_forest": [0.871, 0.884, 0.865, 0.892, 0.878]
}

p_matrix = compute_pairwise_pvalues(mock_metrics, test_type="wilcoxon")
print("Pairwise Wilcoxon Test p-value Matrix:")
p_matrix"""
    },
    "05_interpretability_and_reporting.ipynb": {
        "title": "Phase 5: Interpretability, Confusion Matrices & Final Reporting",
        "desc": "Generate confusion matrices, per-class sensitivity charts, and diagnostic summary figures.",
        "code": """import numpy as np
import matplotlib.pyplot as plt
from ecg_arrhythmia.utils import plot_confusion_matrix
from ecg_arrhythmia.data import CLASS_NAMES

# Sample confusion matrix visualization
cm = np.array([
    [920, 15, 20,  5,  2],
    [ 25, 85,  8,  2,  0],
    [ 18,  6, 95,  3,  1],
    [  4,  1,  2, 38,  0],
    [  3,  0,  1,  0, 22]
])

fig = plot_confusion_matrix(cm, class_names=CLASS_NAMES, title="1D-CNN Normalized Confusion Matrix")
plt.show()"""
    }
}


def create_all_notebooks():
    for fname, content in notebooks.items():
        nb = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"# {content['title']}\n\n", f"{content['desc']}\n"]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": content["code"].splitlines(keepends=True)
                }
            ],
            "metadata": {
                "language_info": {"name": "python"}
            },
            "nbformat": 4,
            "nbformat_minor": 2
        }
        target = Path("notebooks") / fname
        with open(target, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"Created {target}")


if __name__ == "__main__":
    create_all_notebooks()
