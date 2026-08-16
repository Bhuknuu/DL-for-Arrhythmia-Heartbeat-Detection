"""Unit tests for all 12 model architectures in the registry."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from ecg_arrhythmia.models.registry import get_model, list_models


@pytest.mark.parametrize("model_name", [
    "mlp",
    "1d_cnn",
    "fft_2d_cnn",
    "bilstm",
    "cnn_transformer",
    "autoencoder",
    "fusion_model"
])
def test_pytorch_models_forward_pass(model_name):
    """Verify forward pass and logit output shape (batch_size, num_classes) for PyTorch models."""
    batch_size = 4
    seq_len = 200
    num_classes = 5

    model = get_model(model_name)
    assert isinstance(model, nn.Module)

    dummy_input = torch.randn(batch_size, seq_len)
    output = model(dummy_input)

    assert output.shape == (batch_size, num_classes), (
        f"Model {model_name} output shape is {output.shape}, expected ({batch_size}, {num_classes})"
    )


@pytest.mark.parametrize("model_name", [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "svm"
])
def test_classical_models_fit_predict(model_name):
    """Verify fit, predict, and predict_proba on synthetic ECG data for classical models."""
    n_samples = 20
    seq_len = 200
    num_classes = 5

    X = np.random.randn(n_samples, seq_len).astype(np.float32)
    y = np.random.randint(0, num_classes, size=n_samples)

    model = get_model(model_name)
    model.fit(X, y)

    preds = model.predict(X)
    probs = model.predict_proba(X)

    assert len(preds) == n_samples
    assert probs.shape == (n_samples, num_classes)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-4)


def test_heterogeneous_ensemble():
    """Verify heterogeneous ensemble prediction."""
    n_samples = 10
    seq_len = 200
    X = np.random.randn(n_samples, seq_len).astype(np.float32)
    y = np.random.randint(0, 5, size=n_samples)

    m1 = get_model("random_forest")
    m1.fit(X, y)
    m2 = get_model("1d_cnn")

    ensemble = get_model("heterogeneous_ensemble", models=[m1, m2])
    probs = ensemble.predict_proba(X)
    preds = ensemble.predict(X)

    assert probs.shape == (n_samples, 5)
    assert len(preds) == n_samples
