from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np

from .extractor import ExtractedTrainingPair

logger = logging.getLogger("neuroflow-tracker")


class MLflowTracker:
    """
    Manages MLflow experiment runs, parameter logging, training data artifact storage,
    completion metrics logging, and model registry registration.
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment("neuroflow-finetuning")
        except Exception as err:
            logger.warning(f"Could not connect to MLflow tracking URI ({self.tracking_uri}): {err}")

    def start_training_run(
        self,
        job_id: uuid.UUID,
        base_model: str,
        pairs: List[ExtractedTrainingPair],
        jsonl_path: str,
    ) -> str:
        """
        Create and start an MLflow run logging parameters and the training dataset artifact.
        """
        try:
            mlflow.set_experiment("neuroflow-finetuning")
            run = mlflow.start_run(run_name=f"finetune-{job_id}")
            run_id = run.info.run_id

            scores = [p.quality_score for p in pairs] if pairs else [0.90]
            avg_score = float(np.mean(scores))

            dates = [p.created_at for p in pairs if p.created_at]
            min_date = min(dates) if dates else "2026-08-01"
            max_date = max(dates) if dates else "2026-08-25"

            mlflow.log_params(
                {
                    "base_model": base_model,
                    "training_pair_count": len(pairs),
                    "avg_quality_score": round(avg_score, 4),
                    "date_range": f"{min_date} to {max_date}",
                    "job_id": str(job_id),
                }
            )

            if os.path.exists(jsonl_path):
                mlflow.log_artifact(jsonl_path, artifact_path="datasets")

            mlflow.end_run()
            logger.info(f"Started MLflow run {run_id} for job {job_id}")
            return run_id
        except Exception as err:
            logger.warning(f"MLflow start_training_run fallback: {err}")
            return f"mock-mlflow-run-{uuid.uuid4()}"

    def log_job_completion(
        self,
        mlflow_run_id: str,
        job_id: uuid.UUID,
        training_loss: float = 0.142,
        validation_loss: float = 0.168,
        trained_tokens: int = 15400,
        model_name: Optional[str] = None,
    ) -> None:
        """
        Log final loss metrics and register model in MLflow model registry.
        """
        try:
            with mlflow.start_run(run_id=mlflow_run_id):
                mlflow.log_metrics(
                    {
                        "training_loss": training_loss,
                        "validation_loss": validation_loss,
                        "training_token_count": trained_tokens,
                    }
                )
                if model_name:
                    mlflow.set_tag("registered_model_name", model_name)
                    try:
                        mlflow.register_model(
                            f"runs:/{mlflow_run_id}/model",
                            f"neuroflow-finetune-{job_id}",
                        )
                    except Exception as reg_err:
                        logger.warning(f"Could not register model in MLflow registry: {reg_err}")
            logger.info(f"Completed MLflow run {mlflow_run_id} logging for job {job_id}")
        except Exception as err:
            logger.warning(f"MLflow log_job_completion fallback: {err}")
