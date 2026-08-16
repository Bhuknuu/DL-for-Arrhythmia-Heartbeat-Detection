"""CLI Training and Benchmark Runner for ECG Arrhythmia Models."""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from ecg_arrhythmia.data.dataset import ECGDataset, GroupKFoldSplitter, extract_all_beats
from ecg_arrhythmia.evaluation.metrics import compute_metrics
from ecg_arrhythmia.models.registry import get_model, list_models
from ecg_arrhythmia.tracking.experiment_logger import ExperimentLogger
from ecg_arrhythmia.utils.seed import set_seed


def train_pytorch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 0.001,
    device: str = "cpu"
) -> nn.Module:
    """Standard PyTorch training loop."""
    dev = torch.device(device)
    model.to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(dev), y_b.to(dev)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

    return model


def evaluate_pytorch_model(model: nn.Module, val_loader: DataLoader, device: str = "cpu") -> np.ndarray:
    """Computes predictions for PyTorch model."""
    dev = torch.device(device)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_b, _ in val_loader:
            X_b = X_b.to(dev)
            logits = model(X_b)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)


def run_benchmark(
    model_name: str,
    n_splits: int = 5,
    max_records: int = 10,
    epochs: int = 10,
    batch_size: int = 64,
    data_dir: str = "data/raw/mitdb"
) -> None:
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running Benchmark for Model: {model_name} (Device: {device}) ===")

    print(f"Extracting beats from {data_dir} (max records: {max_records})...")
    X, y, groups = extract_all_beats(data_dir=data_dir, max_records=max_records)
    print(f"Total extracted beats: {len(X)}, Patients: {len(np.unique(groups))}")

    splitter = GroupKFoldSplitter(n_splits=n_splits)
    logger = ExperimentLogger(tracking_dir="runs")

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups=groups)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = get_model(model_name)

        if isinstance(model, nn.Module):
            train_ds = ECGDataset(X_train, y_train)
            val_ds = ECGDataset(X_val, y_val)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            model = train_pytorch_model(model, train_loader, val_loader, epochs=epochs, device=device)
            y_pred = evaluate_pytorch_model(model, val_loader, device=device)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

        metrics = compute_metrics(y_val, y_pred)
        print(f"Fold {fold + 1} Macro-F1: {metrics['macro_f1']:.4f} | Accuracy: {metrics['accuracy']:.4f}")
        fold_metrics.append(metrics)

    mean_f1 = float(np.mean([m['macro_f1'] for m in fold_metrics]))
    mean_acc = float(np.mean([m['accuracy'] for m in fold_metrics]))
    print(f"\n==========================================")
    print(f"Final {model_name} - Mean Macro-F1: {mean_f1:.4f} | Mean Accuracy: {mean_acc:.4f}")
    print(f"==========================================")

    logger.log_run(
        model_name=model_name,
        params={"n_splits": n_splits, "max_records": max_records, "epochs": epochs},
        metrics={"mean_macro_f1": mean_f1, "mean_accuracy": mean_acc}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Model Benchmark Runner")
    parser.add_argument("--model", type=str, default="1d_cnn", choices=list_models(), help="Model to train")
    parser.add_argument("--splits", type=int, default=5, help="Number of GroupKFold splits")
    parser.add_argument("--records", type=int, default=10, help="Max records to use (for quick testing)")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs for DL models")
    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        n_splits=args.splits,
        max_records=args.records,
        epochs=args.epochs
    )
