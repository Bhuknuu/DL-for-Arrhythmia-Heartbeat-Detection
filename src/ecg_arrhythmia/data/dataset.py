"""Dataset creation, beat extraction, and leakage-safe GroupKFold splitting."""

from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

from ecg_arrhythmia.data.loader import get_available_records, load_record
from ecg_arrhythmia.data.preprocessor import (
    AAMIMapper,
    normalize_signal,
    remove_baseline_wander,
)


def extract_beats_from_record(
    record_name: str,
    data_dir: Union[str, Path] = "data/raw/mitdb",
    lead: int = 0,
    window_before: int = 90,
    window_after: int = 110,
    filter_baseline: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extracts fixed-length beat windows centered around R-peaks for valid AAMI classes.

    Returns:
        beats: 2D array of shape (num_beats, window_before + window_after)
        labels: 1D array of integer class labels [0-4]
        symbols: List of original annotation symbols
    """
    sig, sample_indices, symbols, fs = load_record(record_name, data_dir, lead=lead)

    if filter_baseline:
        sig = remove_baseline_wander(sig, fs=fs)
    if normalize:
        sig = normalize_signal(sig, method="z-score")

    beats_list = []
    labels_list = []
    valid_symbols = []
    total_len = len(sig)

    for idx, sym in zip(sample_indices, symbols, strict=False):
        if not AAMIMapper.is_valid_beat(sym):
            continue

        start = idx - window_before
        end = idx + window_after

        # Check boundary bounds
        if start < 0 or end > total_len:
            continue

        beat = sig[start:end]
        cls_idx = AAMIMapper.map_symbol_to_idx(sym)

        if cls_idx is not None and len(beat) == (window_before + window_after):
            beats_list.append(beat)
            labels_list.append(cls_idx)
            valid_symbols.append(sym)

    if not beats_list:
        return np.empty((0, window_before + window_after)), np.empty(0, dtype=int), []

    return np.array(beats_list, dtype=np.float32), np.array(labels_list, dtype=np.int64), valid_symbols


def extract_all_beats(
    records: Optional[List[str]] = None,
    data_dir: Union[str, Path] = "data/raw/mitdb",
    lead: int = 0,
    window_before: int = 90,
    window_after: int = 110,
    max_records: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts beats across all available records.

    Returns:
        X: Array of shape (total_beats, window_length)
        y: Array of shape (total_beats,) with class indices [0-4]
        groups: Array of patient/record IDs of shape (total_beats,)
    """
    if records is None:
        records = get_available_records(data_dir)

    if max_records is not None:
        records = records[:max_records]

    all_beats = []
    all_labels = []
    all_groups = []

    for rec in records:
        beats, labels, _ = extract_beats_from_record(
            rec,
            data_dir=data_dir,
            lead=lead,
            window_before=window_before,
            window_after=window_after
        )
        if len(beats) > 0:
            all_beats.append(beats)
            all_labels.append(labels)
            all_groups.extend([rec] * len(labels))

    if not all_beats:
        window_len = window_before + window_after
        return np.empty((0, window_len)), np.empty(0, dtype=int), np.empty(0, dtype=str)

    X = np.vstack(all_beats)
    y = np.concatenate(all_labels)
    groups = np.array(all_groups)

    return X, y, groups


class GroupKFoldSplitter:
    """Manages patient-grouped k-fold cross-validation with leakage assertions."""

    def __init__(self, n_splits: int = 10, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.gkf = GroupKFold(n_splits=n_splits)

    def split(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generates train/test index splits grouped by patient ID."""
        for train_idx, test_idx in self.gkf.split(X, y, groups=groups):
            # Assert zero leakage
            train_patients = set(groups[train_idx])
            test_patients = set(groups[test_idx])
            overlap = train_patients.intersection(test_patients)
            if overlap:
                raise ValueError(
                    f"DATA LEAKAGE DETECTED! Patients present in both train & test sets: {overlap}"
                )
            yield train_idx, test_idx

    @staticmethod
    def assert_no_leakage(train_groups: np.ndarray, test_groups: np.ndarray) -> bool:
        """Verifies that no patient ID appears in both sets."""
        overlap = set(train_groups).intersection(set(test_groups))
        if overlap:
            raise AssertionError(f"Leakage detected for patients: {overlap}")
        return True


class ECGDataset(Dataset):
    """PyTorch Dataset for 1D or 2D ECG beat tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray, add_channel_dim: bool = True):
        self.X = torch.tensor(X, dtype=torch.float32)
        if add_channel_dim and self.X.ndim == 2:
            self.X = self.X.unsqueeze(1)  # (N, 1, seq_len)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
