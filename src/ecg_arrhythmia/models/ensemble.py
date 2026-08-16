"""Ensemble and feature fusion models for ECG heartbeat classification."""

from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy import stats


def extract_morphological_features(x: np.ndarray) -> np.ndarray:
    """Extracts handcrafted statistical & morphological features for each beat.

    Features per beat:
    - mean, std, skewness, kurtosis, min, max, peak-to-peak amplitude, energy, rms, zero-crossings
    """
    if x.ndim == 3:
        x = x.squeeze(1)

    feats = []
    for beat in x:
        mean_val = np.mean(beat)
        std_val = np.std(beat)
        skew_val = stats.skew(beat)
        kurt_val = stats.kurtosis(beat)
        min_val = np.min(beat)
        max_val = np.max(beat)
        p2p_val = max_val - min_val
        energy = np.sum(beat ** 2)
        rms = np.sqrt(np.mean(beat ** 2))
        zero_cross = np.sum(np.diff(np.sign(beat)) != 0)

        feats.append([
            mean_val, std_val, skew_val, kurt_val,
            min_val, max_val, p2p_val, energy, rms, zero_cross
        ])
    return np.array(feats, dtype=np.float32)


class HandcraftedDeepFusionModel(nn.Module):
    """Fuses 1D-CNN deep representations with handcrafted statistical features."""

    def __init__(
        self,
        handcrafted_dim: int = 10,
        deep_feature_dim: int = 64,
        fusion_dim: int = 64,
        dropout: float = 0.3,
        num_classes: int = 5
    ):
        super().__init__()
        self.cnn_backbone = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, deep_feature_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(deep_feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.handcrafted_fc = nn.Sequential(
            nn.Linear(handcrafted_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.classifier = nn.Sequential(
            nn.Linear(deep_feature_dim + 32, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, handcrafted_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim == 2:
            x_raw = x.detach().cpu().numpy()
            x = x.unsqueeze(1)
        elif x.ndim == 3:
            x_raw = x.squeeze(1).detach().cpu().numpy()
        else:
            x_raw = x.detach().cpu().numpy()

        if handcrafted_feats is None:
            feats_np = extract_morphological_features(x_raw)
            handcrafted_feats = torch.tensor(feats_np, dtype=torch.float32, device=x.device)

        deep_feat = self.cnn_backbone(x)
        hand_feat = self.handcrafted_fc(handcrafted_feats)
        combined = torch.cat([deep_feat, hand_feat], dim=1)
        return self.classifier(combined)


class HeterogeneousEnsemble:
    """Soft-voting heterogeneous ensemble combining trained models."""

    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights if weights is not None else [1.0 / len(models)] * len(models)
        # Normalize weights
        total_w = sum(self.weights)
        self.weights = [w / total_w for w in self.weights]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.zeros((len(X), 5), dtype=np.float32)

        for model, weight in zip(self.models, self.weights, strict=False):
            if hasattr(model, "predict_proba"):
                raw_p = model.predict_proba(X)
                p = np.zeros((len(X), 5), dtype=np.float32)
                cols = min(5, raw_p.shape[1])
                p[:, :cols] = raw_p[:, :cols]
            elif isinstance(model, nn.Module):
                model.eval()
                with torch.no_grad():
                    t_X = torch.tensor(X, dtype=torch.float32)
                    logits = model(t_X)
                    p = torch.softmax(logits, dim=-1).cpu().numpy()
            else:
                p = np.zeros((len(X), 5), dtype=np.float32)
            probs += weight * p

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
