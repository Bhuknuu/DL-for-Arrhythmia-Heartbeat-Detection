"""Unit tests for AAMI EC57 label mapping and signal preprocessor."""

import numpy as np

from ecg_arrhythmia.data.preprocessor import (
    AAMIMapper,
    normalize_signal,
)


def test_aami_mapping_completeness():
    """Verify that all standard MIT-BIH symbols map to the correct AAMI classes."""
    # N class: Normal beat, Left/Right bundle branch block, Atrial escape, Nodal escape
    for sym in ['N', 'L', 'R', 'e', 'j']:
        assert AAMIMapper.map_symbol(sym) == 'N', f"Symbol {sym} should map to N"
        assert AAMIMapper.map_symbol_to_idx(sym) == 0

    # S class: Atrial premature, Aberrated atrial premature, Nodal premature, Supraventricular premature
    for sym in ['A', 'a', 'J', 'S']:
        assert AAMIMapper.map_symbol(sym) == 'S', f"Symbol {sym} should map to S"
        assert AAMIMapper.map_symbol_to_idx(sym) == 1

    # V class: Premature ventricular contraction, Ventricular escape
    for sym in ['V', 'E']:
        assert AAMIMapper.map_symbol(sym) == 'V', f"Symbol {sym} should map to V"
        assert AAMIMapper.map_symbol_to_idx(sym) == 2

    # F class: Fusion of ventricular and normal beat
    assert AAMIMapper.map_symbol('F') == 'F'
    assert AAMIMapper.map_symbol_to_idx('F') == 3

    # Q class: Paced beat, Fusion of paced and normal, Unclassifiable
    for sym in ['/', 'f', 'Q']:
        assert AAMIMapper.map_symbol(sym) == 'Q', f"Symbol {sym} should map to Q"
        assert AAMIMapper.map_symbol_to_idx(sym) == 4


def test_non_beat_symbols():
    """Verify that non-beat annotations (e.g. rhythm changes, noise) return None."""
    non_beats = ['+', '~', '|', 'x', '[', ']', '!']
    for sym in non_beats:
        assert AAMIMapper.map_symbol(sym) is None
        assert not AAMIMapper.is_valid_beat(sym)


def test_normalization():
    """Test z-score and min-max signal normalization."""
    sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z_norm = normalize_signal(sig, method="z-score")
    assert np.isclose(np.mean(z_norm), 0.0, atol=1e-6)
    assert np.isclose(np.std(z_norm), 1.0, atol=1e-6)

    mm_norm = normalize_signal(sig, method="min-max")
    assert np.isclose(np.min(mm_norm), 0.0)
    assert np.isclose(np.max(mm_norm), 1.0)
