"""Experiment tracking wrapper using local MLflow file store."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


class ExperimentLogger:
    """Logs training metrics, hyperparameters, and artifacts to local tracking store."""

    def __init__(self, tracking_dir: Union[str, Path] = "runs", experiment_name: str = "ECG_Arrhythmia_Benchmark"):
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.mlflow_available = False

        try:
            import mlflow
            mlflow.set_tracking_uri(f"file:///{self.tracking_dir.resolve().as_posix()}")
            mlflow.set_experiment(self.experiment_name)
            self.mlflow = mlflow
            self.mlflow_available = True
        except ImportError:
            self.mlflow = None

    def log_run(
        self,
        model_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        artifacts: Optional[Dict[str, str]] = None
    ) -> str:
        """Logs a single model evaluation run."""
        run_id = f"{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

        if self.mlflow_available:
            with self.mlflow.start_run(run_name=model_name) as run:
                # Log params
                for k, v in params.items():
                    self.mlflow.log_param(k, v)
                # Log scalar metrics
                for k, v in metrics.items():
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        self.mlflow.log_metric(k, float(v))
                # Log artifacts
                if artifacts:
                    for name, path in artifacts.items():
                        if Path(path).exists():
                            self.mlflow.log_artifact(path)
                return run.info.run_id
        else:
            # Fallback JSON record
            run_file = self.tracking_dir / f"{run_id}.json"
            clean_metrics = {
                k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)
                for k, v in metrics.items()
                if not isinstance(v, np.ndarray)
            }
            record = {
                "model_name": model_name,
                "params": params,
                "metrics": clean_metrics,
                "timestamp": str(pd.Timestamp.now())
            }
            with open(run_file, "w") as f:
                json.dump(record, f, indent=2)
            return run_id
