from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from pipelines.finetuning.extractor import ExtractedTrainingPair, TrainingDataExtractor
from pipelines.finetuning.tracker import MLflowTracker
from pipelines.finetuning.job_manager import FineTuningJobManager


# --- 1. Validation & Quality Rules Tests ---

def test_training_pair_validation_success():
    valid_query = "How does HNSW indexing work in pgvector?"
    valid_answer = "HNSW builds a multi-layer graph [Source 1] for logarithmic vector searches, allowing fast nearest neighbor lookups in pgvector [Source 2]."
    
    is_valid, reason = TrainingDataExtractor.validate_pair(
        user_message=valid_query,
        assistant_message=valid_answer,
        quality_score=0.92,
        min_tokens=20,  # lower token bound for unit test
    )
    assert is_valid is True
    assert reason is None


def test_training_pair_pii_rejection():
    # Email PII
    is_valid, reason = TrainingDataExtractor.validate_pair(
        user_message="My email is test.user@example.com, what is HNSW?",
        assistant_message="HNSW is a graph index [Source 1].",
        quality_score=0.90,
    )
    assert is_valid is False
    assert "email PII" in reason

    # Phone PII
    is_valid_phone, reason_phone = TrainingDataExtractor.validate_pair(
        user_message="Call me at 415-555-2671 about the MSA liability clause.",
        assistant_message="The liability clause is outlined in section 4 [Source 1].",
        quality_score=0.90,
    )
    assert is_valid_phone is False
    assert "phone number PII" in reason_phone


def test_training_pair_missing_citation_and_length():
    # Missing [Source N]
    is_valid, reason = TrainingDataExtractor.validate_pair(
        user_message="Explain semantic chunking",
        assistant_message="Semantic chunking splits text when embedding cosine similarity falls below threshold.",
        quality_score=0.95,
        min_tokens=10,
    )
    assert is_valid is False
    assert "citation" in reason

    # Too short
    is_valid_short, reason_short = TrainingDataExtractor.validate_pair(
        user_message="Explain pgvector",
        assistant_message="pgvector is fast [Source 1].",
        quality_score=0.95,
        min_tokens=50,
    )
    assert is_valid_short is False
    assert "below minimum" in reason_short


# --- 2. SFT & DPO Formatting Tests ---

def test_sft_and_dpo_formatting():
    pair = ExtractedTrainingPair(
        id=str(uuid.uuid4()),
        run_id=str(uuid.uuid4()),
        system_prompt="You are a precise research assistant.",
        user_message="What is HNSW?",
        assistant_message="HNSW is a graph index [Source 1].",
        quality_score=0.95,
        token_count=12,
    )

    sft_data = pair.to_openai_format()
    assert "messages" in sft_data
    assert len(sft_data["messages"]) == 3
    assert sft_data["messages"][0]["role"] == "system"
    assert sft_data["messages"][1]["role"] == "user"
    assert sft_data["messages"][2]["role"] == "assistant"

    dpo_data = pair.to_dpo_format()
    assert "prompt" in dpo_data
    assert "chosen" in dpo_data
    assert "rejected" in dpo_data


# --- 3. MLflow Tracking Tests ---

def test_mlflow_tracker_start_and_complete():
    tracker = MLflowTracker()
    job_id = uuid.uuid4()
    pairs = [
        ExtractedTrainingPair(
            id=str(uuid.uuid4()),
            run_id=str(uuid.uuid4()),
            system_prompt="Sys prompt",
            user_message="User query",
            assistant_message="Answer [Source 1]",
            quality_score=0.90,
            token_count=20,
        )
    ]

    with patch("mlflow.start_run") as mock_start_run, \
         patch("mlflow.log_params") as mock_params, \
         patch("mlflow.log_artifact") as mock_artifact:
        
        mock_run = MagicMock()
        mock_run.info.run_id = "test-mlflow-run-123"
        mock_start_run.return_value = mock_run

        run_id = tracker.start_training_run(job_id, "gpt-4o-mini", pairs, "training_data/mock.jsonl")
        assert run_id is not None


# --- 4. API Endpoints Tests ---

def test_finetune_api_endpoints():
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)

    with patch("backend.api.finetune.get_pool", return_value=None):
        # 1. Preview training data
        res_prev = client.get("/finetune/training-data/preview?limit=5")
        assert res_prev.status_code == 200
        assert "samples" in res_prev.json()

        # 2. Trigger job
        res_job = client.post(
            "/finetune/jobs",
            json={
                "base_model": "gpt-4o-mini",
                "task_type": "legal",
                "min_quality_score": 0.82,
            },
        )
        assert res_job.status_code == 201
        job_data = res_job.json()
        assert "job_id" in job_data
        j_id = job_data["job_id"]

        # 3. List jobs
        res_list = client.get("/finetune/jobs")
        assert res_list.status_code == 200

        # 4. Get specific job
        res_get = client.get(f"/finetune/jobs/{j_id}")
        assert res_get.status_code == 200

        # 5. Complete job & register model
        with patch("redis.asyncio.from_url") as mock_redis:
            mock_redis_inst = AsyncMock()
            mock_redis_inst.get = AsyncMock(return_value="[]")
            mock_redis_inst.set = AsyncMock()
            mock_redis.return_value = mock_redis_inst

            res_comp = client.post(
                f"/finetune/jobs/{j_id}/complete",
                json={
                    "fine_tuned_model_name": "ft:gpt-4o-mini:neuroflow:legal-v1",
                    "task_type": "legal",
                    "training_loss": 0.115,
                    "validation_loss": 0.138,
                },
            )
            assert res_comp.status_code == 200
            assert res_comp.json()["status"] == "succeeded"
