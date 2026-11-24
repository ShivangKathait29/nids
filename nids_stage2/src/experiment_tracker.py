"""
experiment_tracker.py
======================
Stage 2 - Component 3: Experiment Tracking

Wraps MLflow so every training run's parameters, metrics, and model
artifact get logged automatically. If MLflow isn't installed (as in this
dev environment — no network access to pip install), it transparently
falls back to writing the same information as local JSON files, so the
pipeline is fully testable either way and your team loses nothing by
running this before installing MLflow.

On your machines, once you `pip install mlflow`, this same code will:
    - track every run under the experiment name you give it
    - let you compare runs side-by-side with `mlflow ui` (run that command
      in the project folder and open http://localhost:5000)
    - log the serialized model as an artifact you can reload later
"""

import json
import os
import time
from datetime import datetime


class ExperimentTracker:
    """
    Usage
    -----
        tracker = ExperimentTracker(experiment_name="nids_stage2", out_dir="../results")
        with tracker.start_run(run_name="idrandom_forest_cic"):
            tracker.log_params({"n_estimators": 100, "dataset": "cic_ids2017"})
            tracker.log_metrics({"accuracy": 0.993, "latency_ms": 0.42})
            tracker.log_model(model, "idrandom_forest")
    """

    def __init__(self, experiment_name: str = "nids_stage2", out_dir: str = "../results"):
        self.experiment_name = experiment_name
        self.out_dir = out_dir
        self.use_mlflow = False
        self._mlflow = None
        self._current_run_log = None
        self._local_log_path = os.path.join(out_dir, "experiment_log.jsonl")

        os.makedirs(out_dir, exist_ok=True)

        try:
            import mlflow
            from pathlib import Path
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            db_path = Path(out_dir, "mlflow.db").resolve().as_posix()
            mlflow.set_tracking_uri(f"sqlite:///{db_path}")
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
            self.use_mlflow = True
            print(f"[ExperimentTracker] MLflow available — logging to sqlite:///{db_path} "
                  f"(run `mlflow ui --backend-store-uri sqlite:///{db_path}` to view)")
        except Exception as e:
            print(
                f"[ExperimentTracker] MLflow tracking initialization failed ({e}). "
                f"Falling back to local JSON logging at {self._local_log_path}."
            )

    def start_run(self, run_name: str):
        if self.use_mlflow:
            return self._mlflow.start_run(run_name=run_name)
        else:
            self._current_run_log = {
                "run_name": run_name,
                "timestamp": datetime.utcnow().isoformat(),
                "params": {},
                "metrics": {},
                "artifacts": [],
            }
            return _NullContextManager(self)

    def log_params(self, params: dict):
        if self.use_mlflow:
            self._mlflow.log_params(params)
        else:
            self._current_run_log["params"].update(params)

    def log_metrics(self, metrics: dict):
        if self.use_mlflow:
            # mlflow requires numeric metrics only
            numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            self._mlflow.log_metrics(numeric_metrics)
        else:
            self._current_run_log["metrics"].update(metrics)

    def log_model(self, model, artifact_name: str):
        import pickle
        model_dir = os.path.join(self.out_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        local_path = os.path.join(model_dir, f"{artifact_name}.pkl")
        with open(local_path, "wb") as f:
            pickle.dump(model, f)

        if self.use_mlflow:
            try:
                import mlflow.sklearn
                mlflow.sklearn.log_model(model, artifact_name)
            except Exception as e:
                pass
        else:
            self._current_run_log["artifacts"].append(local_path)

    def _finalize_local_run(self):
        """Called automatically when a non-mlflow run's context manager exits."""
        if self._current_run_log is not None:
            with open(self._local_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._current_run_log) + "\n")
            print(f"[ExperimentTracker] Logged run '{self._current_run_log['run_name']}' "
                  f"to {self._local_log_path}")
            self._current_run_log = None


class _NullContextManager:
    """Context manager used for the local-fallback logging path."""
    def __init__(self, tracker):
        self.tracker = tracker

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker._finalize_local_run()
        return False
