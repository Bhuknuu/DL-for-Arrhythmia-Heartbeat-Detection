"""FastAPI REST service for serving ECG Arrhythmia predictions."""

from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
import numpy as np

from api.schemas import ECGBeatRequest, HealthResponse, PredictionResponse
from ecg_arrhythmia.inference.predictor import Predictor
from ecg_arrhythmia.models.registry import get_model

# Global predictor instance
predictor: Optional[Predictor] = None


def get_active_predictor() -> Predictor:
    global predictor
    if predictor is None:
        model = get_model("1d_cnn")
        predictor = Predictor(model=model, model_name="1d_cnn")
    return predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_active_predictor()
    print("ECG Inference Predictor initialized successfully.")
    yield


app = FastAPI(
    title="ECG Arrhythmia Detection API",
    description="Real-time AAMI EC57 Arrhythmia Beat Classification Service",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    pred = get_active_predictor()
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model_loaded=pred.model_name
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_beat(request: ECGBeatRequest):
    """Classifies a single ECG beat signal into N, S, V, F, or Q."""
    if not request.signal:
        raise HTTPException(status_code=400, detail="Empty signal received.")

    pred = get_active_predictor()

    # Resample or pad/slice to 200 samples if needed
    sig = np.array(request.signal, dtype=np.float32)
    if len(sig) != 200:
        # Interpolate or pad to exactly 200 samples
        x_old = np.linspace(0, 1, len(sig))
        x_new = np.linspace(0, 1, 200)
        sig = np.interp(x_new, x_old, sig).astype(np.float32)

    result = predictor.predict_beat(sig)
    return PredictionResponse(**result)
