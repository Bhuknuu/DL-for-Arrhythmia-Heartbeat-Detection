"""Integration tests for FastAPI inference endpoints."""

import numpy as np
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model_loaded" in data


def test_predict_endpoint_valid_signal():
    """Verify POST /predict returns valid AAMI prediction and probabilities."""
    # Synthetic 200-sample ECG beat
    dummy_beat = np.sin(np.linspace(0, 10, 200)).tolist()

    response = client.post("/predict", json={"signal": dummy_beat, "sampling_rate": 360})
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] in ["N", "S", "V", "F", "Q"]
    assert 0 <= data["class_index"] <= 4
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 5
    assert np.isclose(sum(data["probabilities"].values()), 1.0, atol=1e-3)


def test_predict_endpoint_empty_signal():
    """Verify POST /predict returns 400 when signal is empty."""
    response = client.post("/predict", json={"signal": []})
    assert response.status_code == 400
