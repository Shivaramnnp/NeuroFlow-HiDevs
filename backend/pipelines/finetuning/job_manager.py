from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import asyncpg
# pyrefly: ignore [missing-import]
import redis.asyncio as redis
from openai import AsyncOpenAI

try:
    from backend.config import settings
    from backend.db.pool import get_pool
    from .extractor import TrainingDataExtractor
    from .tracker import MLflowTracker
except ImportError:
    from config import settings
    from db.pool import get_pool
    from pipelines.finetuning.extractor import TrainingDataExtractor
    from pipelines.finetuning.tracker import MLflowTracker

logger = logging.getLogger("neuroflow-job-manager")


class FineTuningJobManager:
    """
    Coordinates dataset extraction, MLflow experiment tracking, fine-tuning submission,
    job status polling, and router model registration upon completion.
    """

    def __init__(
        self,
        pool: Optional[asyncpg.Pool] = None,
        tracker: Optional[MLflowTracker] = None,
        extractor: Optional[TrainingDataExtractor] = None,
    ):
        self.pool = pool
        self.tracker = tracker or MLflowTracker()
        self.extractor = extractor or TrainingDataExtractor(pool=pool)

    async def _get_db_pool(self) -> Optional[asyncpg.Pool]:
        if self.pool is not None:
            return self.pool
        try:
            self.pool = get_pool()
        except Exception:
            self.pool = None
        return self.pool

    async def submit_job(
        self,
        base_model: str = "gpt-4o-mini",
        task_type: str = "legal",
        min_quality_score: float = 0.82,
        format_type: str = "sft",
    ) -> Dict[str, Any]:
        """
        Trigger full fine-tuning workflow:
        1. Extract validated training pairs to JSONL
        2. Create MLflow tracking run
        3. Insert pending job in Postgres finetune_jobs table
        4. Submit fine-tuning job to provider
        """
        job_id = uuid.uuid4()
        pool = await self._get_db_pool()

        # Step 1: Extract and Validate Pairs
        extraction_res = await self.extractor.extract_and_export(
            job_id=job_id,
            min_quality_score=min_quality_score,
            format_type=format_type,
        )

        pairs = extraction_res["pairs"]
        jsonl_path = extraction_res["file_path"]

        # Step 2: Start MLflow Run
        mlflow_run_id = self.tracker.start_training_run(
            job_id=job_id,
            base_model=base_model,
            pairs=pairs,
            jsonl_path=jsonl_path,
        )

        # Step 3: Insert into Postgres finetune_jobs table
        if pool is not None:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO finetune_jobs (id, base_model, status, training_pair_count, mlflow_run_id, metrics)
                    VALUES ($1, $2, 'running', $3, $4, $5::jsonb);
                    """,
                    job_id,
                    base_model,
                    len(pairs),
                    mlflow_run_id,
                    json.dumps({"task_type": task_type, "jsonl_path": jsonl_path}),
                )

        # Step 4: Submit to OpenAI / Provider
        provider_job_id = f"ftjob-{job_id.hex[:12]}"
        try:
            api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if api_key and not api_key.startswith("mock") and os.path.exists(jsonl_path) and len(pairs) >= 10:
                client = AsyncOpenAI(api_key=api_key)
                with open(jsonl_path, "rb") as f:
                    file_resp = await client.files.create(file=f, purpose="fine-tune")
                ft_job = await client.fine_tuning.jobs.create(
                    training_file=file_resp.id,
                    model=base_model,
                )
                provider_job_id = ft_job.id
        except Exception as err:
            logger.info(f"Using simulated fine-tuning provider job ID: {err}")

        if pool is not None:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE finetune_jobs SET provider_job_id = $2 WHERE id = $1;",
                    job_id,
                    provider_job_id,
                )

        return {
            "job_id": str(job_id),
            "status": "running",
            "base_model": base_model,
            "task_type": task_type,
            "training_pair_count": len(pairs),
            "mlflow_run_id": mlflow_run_id,
            "provider_job_id": provider_job_id,
            "jsonl_path": jsonl_path,
        }

    async def complete_job_and_register_model(
        self,
        job_id: uuid.UUID,
        fine_tuned_model_name: Optional[str] = None,
        task_type: str = "legal",
        training_loss: float = 0.125,
        validation_loss: float = 0.148,
    ) -> Dict[str, Any]:
        """
        Mark fine-tuning job as succeeded, log completion to MLflow,
        and dynamically register the new model into Redis router:models.
        """
        pool = await self._get_db_pool()
        model_name = fine_tuned_model_name or f"ft:{job_id.hex[:8]}-neuroflow"
        mlflow_run_id = f"run-{job_id}"

        # 1. Update finetune_jobs table
        if pool is not None:
            async with pool.acquire() as conn:
                row = await conn.fetchrow("SELECT mlflow_run_id, metrics FROM finetune_jobs WHERE id = $1;", job_id)
                if row:
                    mlflow_run_id = row["mlflow_run_id"]
                    if row["metrics"]:
                        m = json.loads(row["metrics"]) if isinstance(row["metrics"], str) else row["metrics"]
                        task_type = m.get("task_type", task_type)

                await conn.execute(
                    """
                    UPDATE finetune_jobs
                    SET status = 'succeeded',
                        provider_job_id = $2,
                        completed_at = NOW(),
                        metrics = jsonb_set(COALESCE(metrics, '{}'::jsonb), '{completion}', $3::jsonb)
                    WHERE id = $1;
                    """,
                    job_id,
                    model_name,
                    json.dumps({"training_loss": training_loss, "validation_loss": validation_loss}),
                )

        # 2. Log MLflow Completion
        self.tracker.log_job_completion(
            mlflow_run_id=mlflow_run_id,
            job_id=job_id,
            training_loss=training_loss,
            validation_loss=validation_loss,
            model_name=model_name,
        )

        # 3. Register Model in Redis router:models
        registered_entry = {
            "model": model_name,
            "provider": "openai",
            "vision": False,
            "context_window": 128_000,
            "task_type": task_type,
            "fine_tuned": True,
            "cost_per_million_input": 3.00,
            "cost_per_million_output": 12.00,
        }

        try:
            redis_client = redis.from_url(settings.redis_url, socket_timeout=2.0)
            raw = await redis_client.get("router:models")
            models_list = json.loads(raw) if raw else []

            # Append or replace
            existing_idx = next((i for i, m in enumerate(models_list) if m.get("model") == model_name), None)
            if existing_idx is not None:
                models_list[existing_idx] = registered_entry
            else:
                models_list.append(registered_entry)

            await redis_client.set("router:models", json.dumps(models_list))
            await redis_client.aclose()
            logger.info(f"Registered fine-tuned model '{model_name}' in Redis router:models for domain '{task_type}'")
        except Exception as err:
            logger.warning(f"Could not register model in Redis: {err}")

        return {
            "job_id": str(job_id),
            "status": "succeeded",
            "registered_model": model_name,
            "task_type": task_type,
            "mlflow_run_id": mlflow_run_id,
        }
