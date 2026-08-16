"""Unit tests for patient-grouped cross-validation and zero data leakage."""

import numpy as np
import pytest

from ecg_arrhythmia.data.dataset import GroupKFoldSplitter


def test_groupkfold_zero_leakage():
    """Verify that no patient appears in both train and validation sets across all folds."""
    # Synthetic dataset with 10 patients and 50 beats each
    n_patients = 10
    beats_per_patient = 50
    total_beats = n_patients * beats_per_patient

    X = np.random.randn(total_beats, 200).astype(np.float32)
    y = np.random.randint(0, 5, size=total_beats)
    groups = np.repeat([f"patient_{i:02d}" for i in range(n_patients)], beats_per_patient)

    splitter = GroupKFoldSplitter(n_splits=5)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups=groups)):
        train_patients = set(groups[train_idx])
        val_patients = set(groups[val_idx])

        # Assert no intersection
        overlap = train_patients.intersection(val_patients)
        assert len(overlap) == 0, f"Fold {fold} has leaking patients: {overlap}"

        # Assert helper method passes
        assert GroupKFoldSplitter.assert_no_leakage(groups[train_idx], groups[val_idx])


def test_leakage_assertion_failure():
    """Verify that assert_no_leakage raises an error when an overlap exists."""
    train_groups = np.array(["patient_01", "patient_02", "patient_03"])
    test_groups = np.array(["patient_03", "patient_04"])

    with pytest.raises(AssertionError):
        GroupKFoldSplitter.assert_no_leakage(train_groups, test_groups)
