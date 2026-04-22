import logging
import os
from typing import Dict, Any, Optional

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from core.registry import Registry

logger = logging.getLogger(__name__)

tracking_registry = Registry()

class ExperimentTracker:
    def log_summary(self, summary: Dict[str, Any]):
        """Logs a standardized summary.json payload to the tracking backend."""
        raise NotImplementedError


def init_tracker(config: Optional[Dict[str, Any]]) -> Optional[ExperimentTracker]:
    if not config:
        return None

    enabled = config.get("enabled", True)
    if not enabled:
        return None

    backend = config.get("backend", "mlflow")
    params = config.get("params", {})
    try:
        return tracking_registry.create(backend, **params)
    except Exception as exc:
        logger.warning(f"Failed to initialize tracker backend '{backend}': {exc}")
        return None

@tracking_registry.register("mlflow")
class MLflowTracker(ExperimentTracker):
    def __init__(self, tracking_uri: str = "file:./mlruns", experiment_name: str = "alife_platform"):
        if not HAS_MLFLOW:
            raise ImportError("mlflow is required to use MLflowTracker. Run: pip install mlflow")
            
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflowTracker initialized at {tracking_uri}, experiment: {experiment_name}")

    def log_summary(self, summary: Dict[str, Any]):
        """
        Takes the standard output from `core.logger.build_run_summary`
        and maps it to MLflow metrics, parameters, and tags.
        """
        # Make sure we have an active run or start one
        active_run = mlflow.active_run()
        if not active_run:
            mlflow.start_run()
            
        try:
            # Log basic metadata as tags/params
            mlflow.set_tag("system", summary.get("system", "unknown"))
            mlflow.set_tag("mode", summary.get("mode", "unknown"))
            mlflow.log_param("config_path", summary.get("config_path", ""))
            mlflow.log_param("run_dir", summary.get("run_dir", ""))
            
            # Log details as parameters (e.g. prompt, foundation_model)
            details = summary.get("details", {})
            for key, value in details.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(key, value)
                else:
                    # Convert lists/dicts to string representation for params
                    mlflow.log_param(key, str(value))
                    
            # Log metrics
            metrics = summary.get("metrics", {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))
                    
            # Try to log the whole summary.json as an artifact if the run_dir exists
            run_dir = summary.get("run_dir")
            if run_dir and os.path.exists(run_dir):
                # We can log all contents of the run_dir as artifacts
                mlflow.log_artifacts(run_dir)
                
            logger.info("Successfully logged summary to MLflow")
            
        finally:
            if not active_run:
                mlflow.end_run()
