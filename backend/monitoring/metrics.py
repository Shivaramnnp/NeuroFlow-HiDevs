from __future__ import annotations

import logging
from prometheus_client import Counter, Gauge, Histogram, Summary

logger = logging.getLogger("neuroflow-monitoring")

# --- 1. Counters ---
queries_total = Counter(
    "neuroflow_queries_total",
    "Total queries executed across pipelines",
    ["pipeline_id", "status"],
)

ingestion_docs_total = Counter(
    "neuroflow_ingestion_docs_total",
    "Total documents processed by ingestion pipeline",
    ["source_type"],
)

llm_calls_total = Counter(
    "neuroflow_llm_calls_total",
    "Total LLM API calls executed across providers and models",
    ["provider", "model", "task_type"],
)

circuit_breaker_trips = Counter(
    "neuroflow_circuit_breaker_trips_total",
    "Total circuit breaker transition events to OPEN state",
    ["provider"],
)

# --- 2. Histograms ---
retrieval_latency = Histogram(
    "neuroflow_retrieval_latency_seconds",
    "End-to-end and strategy retrieval latency distribution in seconds",
    ["strategy"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

generation_latency = Histogram(
    "neuroflow_generation_latency_seconds",
    "LLM generation and streaming latency distribution in seconds",
    ["model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

llm_cost = Histogram(
    "neuroflow_llm_cost_usd",
    "LLM call cost distribution in USD",
    ["model"],
    buckets=[0.0001, 0.001, 0.01, 0.1, 1.0],
)

# --- 3. Gauges ---
eval_faithfulness = Gauge(
    "neuroflow_eval_faithfulness",
    "Rolling average faithfulness score per pipeline",
    ["pipeline_id"],
)

eval_overall = Gauge(
    "neuroflow_eval_overall",
    "Rolling average overall evaluation quality score per pipeline",
    ["pipeline_id"],
)

queue_depth = Gauge(
    "neuroflow_queue_depth",
    "Real-time background ingestion queue depth (LLEN queue:ingest)",
)

active_circuit_breakers_open = Gauge(
    "neuroflow_circuit_breakers_open",
    "Number of LLM provider circuit breakers currently in OPEN state",
)
