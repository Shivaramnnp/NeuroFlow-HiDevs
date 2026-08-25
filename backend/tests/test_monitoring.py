from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock
import uuid
import pytest
from prometheus_client import REGISTRY

from backend.monitoring.metrics import (
    queries_total,
    ingestion_docs_total,
    llm_calls_total,
    circuit_breaker_trips,
    retrieval_latency,
    generation_latency,
    llm_cost,
    eval_faithfulness,
    eval_overall,
    queue_depth,
    active_circuit_breakers_open,
)
from backend.monitoring.anomaly_detector import QualityAnomalyDetector


def test_prometheus_metrics_registration():
    """Verify all custom Prometheus metrics are defined and incrementable."""
    # Counters
    queries_total.labels(pipeline_id="test_p", status="complete").inc()
    ingestion_docs_total.labels(source_type="pdf").inc()
    llm_calls_total.labels(provider="openai", model="gpt-4o", task_type="rag_generation").inc()
    circuit_breaker_trips.labels(provider="openai").inc()

    # Histograms
    retrieval_latency.labels(strategy="hybrid").observe(0.12)
    generation_latency.labels(model="gpt-4o").observe(1.45)
    llm_cost.labels(model="gpt-4o").observe(0.005)

    # Gauges
    eval_faithfulness.labels(pipeline_id="test_p").set(0.92)
    eval_overall.labels(pipeline_id="test_p").set(0.88)
    queue_depth.set(12)
    active_circuit_breakers_open.set(0)

    # Verify metrics exist in global Prometheus registry
    sample_names = [m.name for m in REGISTRY.collect()]
    assert any("neuroflow_queries" in name for name in sample_names)
    assert any("neuroflow_ingestion_docs" in name for name in sample_names)
    assert any("neuroflow_retrieval_latency" in name for name in sample_names)
    assert any("neuroflow_eval_overall" in name for name in sample_names)


@pytest.mark.asyncio
async def test_anomaly_detector_detects_drop():
    """Verify anomaly detector flags a score > 2 standard deviations below mean."""
    mock_pool = MagicMock()
    mock_conn = AsyncMock()

    # Historical 7-day evaluations: average ~0.90 with std ~0.02
    # Latest evaluation: 0.50 (severe quality drop)
    mock_rows = [
        {"overall_score": 0.91},
        {"overall_score": 0.89},
        {"overall_score": 0.90},
        {"overall_score": 0.92},
        {"overall_score": 0.88},
        {"overall_score": 0.90},
        {"overall_score": 0.50},  # Latest drop
    ]
    mock_conn.fetch.return_value = mock_rows

    class MockAcquireContext:
        async def __aenter__(self):
            return mock_conn

        async def __aexit__(self, exc_type, exc, tb):
            pass

    mock_pool.acquire.return_value = MockAcquireContext()

    detector = QualityAnomalyDetector(pool=mock_pool)
    p_id = uuid.uuid4()
    result = await detector.check_pipeline_quality(p_id)

    assert result["is_anomalous"] is True
    assert result["latest_score"] == 0.50
    assert result["mean_7d"] > 0.85
