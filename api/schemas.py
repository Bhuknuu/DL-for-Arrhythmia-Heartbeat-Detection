"""Pydantic schemas for the ECG Arrhythmia Inference REST API."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    model_loaded: str = "1d_cnn"


class ECGBeatRequest(BaseModel):
    signal: List[float] = Field(
        ...,
        description="ECG beat sample values (typically 200 samples around R-peak)"
    )
    sampling_rate: Optional[int] = Field(360, description="Sampling rate in Hz")


class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., description="AAMI class: N, S, V, F, or Q")
    class_index: int = Field(..., description="Integer index [0-4]")
    confidence: float = Field(..., description="Probability of top prediction")
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution across all 5 AAMI classes"
    )
