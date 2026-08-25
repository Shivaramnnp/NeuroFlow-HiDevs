from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from evaluation.metrics.faithfulness import evaluate_faithfulness
from evaluation.metrics.answer_relevance import evaluate_answer_relevance
from evaluation.metrics.context_precision import evaluate_context_precision
from evaluation.metrics.context_recall import evaluate_context_recall
from evaluation.judge import EvaluationJudge
from backend.providers.base import GenerationResult
from backend.providers.client import NeuroFlowClient


# --- 1. Metrics Tests ---

@pytest.mark.asyncio
async def test_faithfulness_metric():
    mock_client = MagicMock(spec=NeuroFlowClient)
    # Mock LLM verdict response
    mock_client.chat = AsyncMock(
        return_value=GenerationResult(
            content=json.dumps({"claims": [{"claim": "pgvector is fast", "verdict": "yes"}], "faithfulness_score": 1.0}),
            input_tokens=50,
            output_tokens=20,
            model="gpt-4o",
            cost_usd=0.001,
            latency_ms=100,
            finish_reason="stop",
        )
    )

    score = await evaluate_faithfulness("query", "pgvector is fast", "pgvector is fast and indexed", client=mock_client)
    assert score == 1.0

    # Empty context makes claims -> score 0.0
    score_empty = await evaluate_faithfulness("query", "pgvector is fast", "", client=mock_client)
    assert score_empty == 0.0


@pytest.mark.asyncio
async def test_answer_relevance_metric():
    mock_client = MagicMock(spec=NeuroFlowClient)
    mock_client.chat = AsyncMock(
        return_value=GenerationResult(
            content="What is pgvector?\nHow does pgvector work?",
            input_tokens=30,
            output_tokens=15,
            model="gpt-4o",
            cost_usd=0.001,
            latency_ms=50,
            finish_reason="stop",
        )
    )
    # Mock embedding vectors
    mock_client.embed = AsyncMock(return_value=[[1.0, 0.0, 0.0], [0.95, 0.05, 0.0], [0.98, 0.02, 0.0]])

    score = await evaluate_answer_relevance("What is pgvector?", "pgvector is a vector db extension.", client=mock_client)
    assert score > 0.90


@pytest.mark.asyncio
async def test_context_precision_and_recall_metrics():
    mock_client = MagicMock(spec=NeuroFlowClient)
    mock_client.chat = AsyncMock(
        return_value=GenerationResult(
            content=json.dumps(["yes", "no"]),
            input_tokens=40,
            output_tokens=10,
            model="gpt-4o",
            cost_usd=0.001,
            latency_ms=50,
            finish_reason="stop",
        )
    )

    precision = await evaluate_context_precision("query", ["chunk 1 useful", "chunk 2 noise"], "answer", client=mock_client)
    assert 0.0 <= precision <= 1.0

    recall = await evaluate_context_recall("query", ["chunk 1", "chunk 2"], "First sentence. Second sentence.", client=mock_client)
    assert 0.0 <= recall <= 1.0


# --- 2. Evaluation Judge Tests ---

@pytest.mark.asyncio
async def test_evaluation_judge_parallel_and_training_pairs():
    mock_client = MagicMock(spec=NeuroFlowClient)
    mock_client.chat = AsyncMock(
        return_value=GenerationResult(
            content=json.dumps({"faithfulness_score": 1.0, "verdict": "yes"}),
            input_tokens=50,
            output_tokens=20,
            model="gpt-4o",
            cost_usd=0.001,
            latency_ms=80,
            finish_reason="stop",
        )
    )
    mock_client.embed = AsyncMock(return_value=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

    judge = EvaluationJudge(client=mock_client, pool=mock_pool)
    run_id = uuid.uuid4()

    result = await judge.evaluate(
        query="Explain HNSW",
        answer="HNSW is a graph index.",
        context="HNSW is a graph index for vector search.",
        run_id=run_id,
    )

    assert result["run_id"] == str(run_id)
    assert 0.0 <= result["overall_score"] <= 1.0
    assert "faithfulness" in result
    assert "answer_relevance" in result
    assert "context_precision" in result
    assert "context_recall" in result


# --- 3. Ratings API Endpoint Tests ---

def test_ratings_api():
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    run_id = str(uuid.uuid4())

    # Send user rating 4/5
    res = client.patch(f"/runs/{run_id}/rating", json={"rating": 4})
    assert res.status_code == 200
    data = res.json()
    assert data["user_rating"] == 4
    assert data["run_id"] == run_id
