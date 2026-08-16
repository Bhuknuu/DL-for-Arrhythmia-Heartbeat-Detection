"""WFDB record loader for MIT-BIH Arrhythmia Database."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import wfdb

from ecg_arrhythmia.data.preprocessor import AAMIMapper


def load_record(
    record_name: str,
    data_dir: Union[str, Path] = "data/raw/mitdb",
    lead: int = 0
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """Loads a single WFDB ECG record and its annotations.

    Args:
        record_name: e.g. '100'
        data_dir: Directory containing .dat, .hea, and .atr files
        lead: Signal lead index (default 0, typically MLII)

    Returns:
        signal: 1D numpy array of ECG signal
        sample_indices: Array of R-peak sample indices
        symbols: List of annotation symbols at each sample index
        fs: Sampling frequency in Hz (typically 360)
    """
    record_path = Path(data_dir) / record_name
    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), 'atr')

    sig = record.p_signal[:, lead]
    sample_indices = np.array(annotation.sample)
    symbols = annotation.symbol
    fs = record.fs

    return sig, sample_indices, symbols, fs


def get_available_records(data_dir: Union[str, Path] = "data/raw/mitdb") -> List[str]:
    """Returns a sorted list of record names available in the data directory."""
    path = Path(data_dir)
    if not path.exists():
        return []
    
    hea_files = path.glob("*.hea")
    records = sorted([f.stem for f in hea_files if not f.stem.startswith(".")])
    return records
