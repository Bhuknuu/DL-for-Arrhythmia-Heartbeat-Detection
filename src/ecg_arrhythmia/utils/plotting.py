"""Plotting and visualization helpers for ECG waveforms and evaluation metrics."""

from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ecg_arrhythmia.data.preprocessor import CLASS_NAMES


def plot_ecg_beat(
    beat_signal: np.ndarray,
    label_name: str = "Unknown",
    title: Optional[str] = None,
    fs: int = 360,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plots a single ECG beat waveform with millisecond time-axis."""
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=100)
    time_ms = (np.arange(len(beat_signal)) / fs) * 1000.0

    ax.plot(time_ms, beat_signal, color="#1f77b4", lw=1.8, label=f"Beat ({label_name})")
    ax.axvline(time_ms[len(beat_signal) // 2], color="red", linestyle="--", alpha=0.7, label="R-Peak")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalized Amplitude")
    ax.set_title(title or f"ECG Beat Morphology — Class {label_name}")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plots a normalized confusion matrix heatmap."""
    if class_names is None:
        class_names = CLASS_NAMES

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
