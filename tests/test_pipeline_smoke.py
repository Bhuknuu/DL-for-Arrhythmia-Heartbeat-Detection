"""End-to-end integration and smoke tests."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ecg_arrhythmia.data.dataset import ECGDataset, extract_beats_from_record
from ecg_arrhythmia.data.preprocessor import CLASS_NAMES
from ecg_arrhythmia.evaluation.metrics import compute_metrics
from ecg_arrhythmia.inference.predictor import Predictor
from ecg_arrhythmia.models.registry import get_model


def test_pipeline_end_to_end_smoke():
    """Runs a complete end-to-end smoke test: data extraction -> training -> evaluation -> inference."""
    data_dir = Path("data/raw/mitdb")

    if (data_dir / "100.hea").exists():
        beats, labels, _ = extract_beats_from_record("100", data_dir=data_dir)
        # Take small subset for smoke test
        beats = beats[:40]
        labels = labels[:40]
    else:
        # Synthetic fallback
        beats = np.random.randn(40, 200).astype(np.float32)
        labels = np.random.randint(0, 5, size=40)

    # 1. Dataset & DataLoader
    ds = ECGDataset(beats, labels)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    # 2. Model instantiate & 1 step of training
    model = get_model("1d_cnn")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for X_b, y_b in loader:
        optimizer.zero_grad()
        out = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()

    # 3. Predict & Compute Metrics
    model.eval()
    with torch.no_grad():
        preds = []
        for X_b, _ in loader:
            p = torch.argmax(model(X_b), dim=-1).numpy()
            preds.extend(p)

    metrics = compute_metrics(labels, np.array(preds))
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics

    # 4. Predictor inference
    predictor = Predictor(model=model, model_name="1d_cnn")
    single_res = predictor.predict_beat(beats[0])
    assert single_res["predicted_class"] in CLASS_NAMES
    assert 0.0 <= single_res["confidence"] <= 1.0
    assert len(single_res["probabilities"]) == 5
