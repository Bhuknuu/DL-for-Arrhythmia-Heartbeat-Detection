#!/usr/bin/env bash
echo "Starting FastAPI ECG Arrhythmia Inference Server..."
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
