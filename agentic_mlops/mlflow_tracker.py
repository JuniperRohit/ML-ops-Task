"""MLflow experiment tracking integration for MLOps."""

import os
from typing import Dict, Any, Optional
from datetime import datetime

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLflowTracker:
    """Track ML experiments with MLflow."""
    
    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: str = "mlops-experiments"):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI (default: local)
            experiment_name: Experiment name for organization
        """
        if not MLFLOW_AVAILABLE:
            print("Warning: MLflow not installed. Install with: pip install mlflow")
            self.enabled = False
            return
        
        self.enabled = True
        self.tracking_uri = tracking_uri or "file:./mlruns"
        self.experiment_name = experiment_name
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set experiment
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Warning: Could not set experiment: {e}")

    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        if not self.enabled:
            return
        try:
            mlflow.log_params(params)
        except Exception as e:
            print(f"Warning: Could not log params: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: int = 0):
        """Log metrics."""
        if not self.enabled:
            return
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        except Exception as e:
            print(f"Warning: Could not log metrics: {e}")

    def log_model(self, model, artifact_path: str, model_type: str = "sklearn"):
        """Log model artifact."""
        if not self.enabled:
            return
        try:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, artifact_path)
            elif model_type == "pickle":
                mlflow.log_artifact(artifact_path)
            else:
                mlflow.log_artifact(artifact_path)
        except Exception as e:
            print(f"Warning: Could not log model: {e}")

    def log_artifact(self, local_path: str):
        """Log artifact file."""
        if not self.enabled:
            return
        try:
            mlflow.log_artifact(local_path)
        except Exception as e:
            print(f"Warning: Could not log artifact: {e}")

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        """Start a new MLflow run."""
        if not self.enabled:
            return None
        try:
            run = mlflow.start_run(run_name=run_name)
            if tags:
                mlflow.set_tags(tags)
            return run
        except Exception as e:
            print(f"Warning: Could not start run: {e}")
            return None

    def end_run(self, status: str = "FINISHED"):
        """End current MLflow run."""
        if not self.enabled:
            return
        try:
            mlflow.end_run(status=status)
        except Exception as e:
            print(f"Warning: Could not end run: {e}")

    def get_runs(self, max_results: int = 10) -> list:
        """Get recent experiment runs."""
        if not self.enabled:
            return []
        try:
            client = MlflowClient(self.tracking_uri)
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                return []
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results
            )
            return runs
        except Exception as e:
            print(f"Warning: Could not get runs: {e}")
            return []

    def log_summary(self, summary: Dict[str, Any]):
        """Log experiment summary."""
        if not self.enabled:
            return
        try:
            mlflow.log_dict(summary, "summary.json")
        except Exception as e:
            print(f"Warning: Could not log summary: {e}")


# Global tracker instance
_tracker = None


def get_mlflow_tracker() -> MLflowTracker:
    """Get or create global MLflow tracker."""
    global _tracker
    if _tracker is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
        _tracker = MLflowTracker(tracking_uri=tracking_uri)
    return _tracker
