"""Inference engine for real-time ECG beat classification."""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ecg_arrhythmia.data.preprocessor import CLASS_NAMES


class Predictor:
    """Production predictor wrapper for single beat or continuous ECG array classification."""

    def __init__(
        self,
        model: Any,
        model_name: str = "custom_model",
        device: str = "cpu",
        class_names: Optional[List[str]] = None
    ):
        self.model = model
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.class_names = class_names if class_names is not None else CLASS_NAMES

        if isinstance(self.model, nn.Module):
            self.model.to(self.device)
            self.model.eval()

    def predict_beat(self, beat_signal: np.ndarray) -> Dict[str, Any]:
        """Predicts class and probabilities for a single ECG beat window (e.g. 200 samples).

        Args:
            beat_signal: 1D array of shape (200,)

        Returns:
            Dict containing predicted_class, class_index, and probabilities dict
        """
        beat_arr = np.array(beat_signal, dtype=np.float32)
        if beat_arr.ndim == 1:
            beat_arr = beat_arr.reshape(1, -1)

        probs = self.predict_proba(beat_arr)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx]

        return {
            "predicted_class": pred_class,
            "class_index": pred_idx,
            "confidence": float(probs[pred_idx]),
            "probabilities": {name: float(probs[i]) for i, name in enumerate(self.class_names)}
        }

    def predict_proba(self, beats: np.ndarray) -> np.ndarray:
        """Predicts probabilities for a batch of beats. Shape (N, window_len) -> (N, 5)."""
        beats = np.array(beats, dtype=np.float32)
        if isinstance(self.model, nn.Module):
            with torch.no_grad():
                t_beats = torch.tensor(beats, dtype=torch.float32, device=self.device)
                logits = self.model(t_beats)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            return probs
        elif hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(beats)
        else:
            raise ValueError(f"Model {self.model} does not support probability prediction.")
