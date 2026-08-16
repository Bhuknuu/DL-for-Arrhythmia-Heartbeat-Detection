"""Data preprocessing, AAMI EC57 annotation mapping, and signal filtering."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import signal

# Standard AAMI EC57 class mapping from MIT-BIH annotation symbols
AAMI_MAPPING: Dict[str, str] = {
    # Normal / Bundle branch block beats -> 'N' (0)
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    # Supraventricular ectopic beats -> 'S' (1)
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    # Ventricular ectopic beats -> 'V' (2)
    'V': 'V', 'E': 'V',
    # Fusion beats -> 'F' (3)
    'F': 'F',
    # Paced / Unclassifiable beats -> 'Q' (4)
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}

CLASS_TO_IDX: Dict[str, int] = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}
CLASS_NAMES: List[str] = ['N', 'S', 'V', 'F', 'Q']


class AAMIMapper:
    """Helper to map PhysioNet MIT-BIH annotation symbols to AAMI EC57 classes."""

    @staticmethod
    def map_symbol(symbol: str) -> Optional[str]:
        """Maps a single annotation symbol to an AAMI class, or None if non-beat."""
        return AAMI_MAPPING.get(symbol, None)

    @staticmethod
    def map_symbol_to_idx(symbol: str) -> Optional[int]:
        """Maps a symbol to an integer index [0-4], or None if non-beat."""
        cls_name = AAMI_MAPPING.get(symbol, None)
        return CLASS_TO_IDX[cls_name] if cls_name else None

    @staticmethod
    def is_valid_beat(symbol: str) -> bool:
        """Returns True if the symbol corresponds to an AAMI evaluated beat."""
        return symbol in AAMI_MAPPING


def remove_baseline_wander(ecg_signal: np.ndarray, fs: int = 360) -> np.ndarray:
    """Removes baseline wander using two median filters (200ms and 600ms)."""
    # Filter 1: 200 ms window (width = 0.2 * fs)
    win1 = int(0.2 * fs)
    if win1 % 2 == 0:
        win1 += 1
    # Filter 2: 600 ms window (width = 0.6 * fs)
    win2 = int(0.6 * fs)
    if win2 % 2 == 0:
        win2 += 1

    baseline = signal.medfilt(ecg_signal, kernel_size=win1)
    baseline = signal.medfilt(baseline, kernel_size=win2)
    return ecg_signal - baseline


def normalize_signal(ecg_signal: np.ndarray, method: str = "z-score") -> np.ndarray:
    """Normalizes an ECG signal array using z-score (mean=0, std=1) or min-max scaling."""
    if method == "z-score":
        std = np.std(ecg_signal)
        if std == 0:
            return ecg_signal - np.mean(ecg_signal)
        return (ecg_signal - np.mean(ecg_signal)) / std
    elif method == "min-max":
        s_min, s_max = np.min(ecg_signal), np.max(ecg_signal)
        if s_max == s_min:
            return np.zeros_like(ecg_signal)
        return (ecg_signal - s_min) / (s_max - s_min)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
