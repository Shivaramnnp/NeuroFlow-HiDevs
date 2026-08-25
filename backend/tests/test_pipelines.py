from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from pydantic import ValidationError

from backend.models.pipeline import PipelineConfig
from backend.services.pipeline_optimizer import PipelineOptimizer


# --- 1. Schema Validation Tests ---

def test_pipeline_config_validation_success():
    valid_cfg = {
        "name": "legal-research-v2",
        "description": "Optimized for legal document analysis",
        "ingestion": {
            "chunking_strategy": "hierarchical",
            "chunk_size_tokens": 400,
            "chunk_overlap_tokens": 80,
            "extractors_enabled": ["pdf", "docx"],
        },
        "retrieval": {
            "dense_k": 30,
            "sparse_k": 20,
            "reranker": "cross-encoder",
            "top_k_after_rerank": 8,
            "query_expansion": True,
            "metadata_filters_enabled": True,
        },
        "generation": {
            "model_routing": {"task_type": "rag_generation", "max_cost_per_call": 0.05},
            "max_context_tokens": 6000,
            "temperature": 0.2,
            "system_prompt_variant": "precise",
        },
        "evaluation": {
            "auto_evaluate": True,
            "training_threshold": 0.82,
        },
    }

    config = PipelineConfig(**valid_cfg)
    assert config.name == "legal-research-v2"
    assert config.ingestion.chunking_strategy == "hierarchical"
    assert config.retrieval.dense_k == 30
    assert config.generation.max_context_tokens == 6000
    assert config.evaluation.training_threshold == 0.82


def test_pipeline_config_rejects_unknown_keys():
    invalid_cfg = {
        "name": "illegal-pipeline",
        "unknown_top_level_key": "bad_value",
    }
    with pytest.raises(ValidationError):
        PipelineConfig(**invalid_cfg)


# --- 2. Pipeline Optimizer Tests ---

def test_pipeline_optimizer_suggestions():
    config = {
        "retrieval": {"top_k_after_rerank": 10, "dense_k": 20, "query_expansion": False},
        "generation": {"temperature": 0.5, "system_prompt_variant": "factual"},
    }

    # Case 1: Low precision & low recall & low faithfulness
    degraded_metrics = {
        "faithfulness": 0.60,
        "answer_relevance": 0.85,
        "context_precision": 0.55,
        "context_recall": 0.50,
        "overall_score": 0.62,
    }

    suggestions = PipelineOptimizer.generate_suggestions(config, degraded_metrics, latency_ms=3000)
    assert len(suggestions) >= 3

    fields = [s["target_field"] for s in suggestions]
    assert "retrieval.top_k_after_rerank" in fields
    assert "retrieval.dense_k" in fields
    assert "generation.temperature" in fields


# --- 3. API CRUD & A/B Compare Tests ---

def test_pipeline_api_endpoints():
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)

    # 1. Create pipeline
    cfg_payload = {
        "name": "customer-support-v1",
        "description": "Fast support bot",
        "ingestion": {"chunking_strategy": "fixed_size", "chunk_size_tokens": 256},
        "retrieval": {"dense_k": 15, "top_k_after_rerank": 5},
        "generation": {"max_context_tokens": 2000, "temperature": 0.1},
        "evaluation": {"auto_evaluate": True, "training_threshold": 0.85},
    }

    with patch("backend.api.pipelines.get_pool", return_value=None):
        res = client.post("/pipelines", json=cfg_payload)
        assert res.status_code == 201
        data = res.json()
        assert data["version"] == 1
        p_id = data["id"]

        # 2. List pipelines
        res_list = client.get("/pipelines")
        assert res_list.status_code == 200

        # 3. Get pipeline
        res_get = client.get(f"/pipelines/{p_id}")
        assert res_get.status_code == 200

        # 4. Update pipeline
        cfg_payload["name"] = "customer-support-v2"
        res_patch = client.patch(f"/pipelines/{p_id}", json=cfg_payload)
        assert res_patch.status_code == 200
        assert res_patch.json()["version"] == 2

        # 5. Delete pipeline
        res_del = client.delete(f"/pipelines/{p_id}")
        assert res_del.status_code == 200
        assert res_del.json()["status"] == "archived"

        # 6. Analytics
        res_analytics = client.get(f"/pipelines/{p_id}/analytics")
        assert res_analytics.status_code == 200
        assert "latency" in res_analytics.json()


@pytest.mark.asyncio
async def test_pipeline_compare_endpoint():
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)

    mock_summary_a = {
        "run_id": str(uuid.uuid4()),
        "pipeline_id": str(uuid.uuid4()),
        "pipeline_name": "Pipeline A",
        "pipeline_version": 1,
        "generation": "Answer from pipeline A",
        "retrieval_latency_ms": 120,
        "total_latency_ms": 650,
        "chunks_used": 5,
        "eval_score": 0.90,
        "citations_count": 2,
    }
    mock_summary_b = {
        "run_id": str(uuid.uuid4()),
        "pipeline_id": str(uuid.uuid4()),
        "pipeline_name": "Pipeline B",
        "pipeline_version": 1,
        "generation": "Answer from pipeline B",
        "retrieval_latency_ms": 180,
        "total_latency_ms": 850,
        "chunks_used": 8,
        "eval_score": 0.82,
        "citations_count": 3,
    }

    with patch("backend.api.compare._run_single_pipeline", AsyncMock(side_effect=[mock_summary_a, mock_summary_b])):
        res = client.post(
            "/pipelines/compare",
            json={
                "query": "What is the liability clause in the MSA?",
                "pipeline_a_id": str(uuid.uuid4()),
                "pipeline_b_id": str(uuid.uuid4()),
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert "pipeline_a" in data
        assert "pipeline_b" in data
        assert data["winner"] == "Pipeline A"
