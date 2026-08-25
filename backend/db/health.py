from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Tuple

# pyrefly: ignore [missing-import]
import redis.asyncio as redis
import requests

try:
    from backend.config import settings
    from backend.db.pool import get_pool
    from backend.resilience.circuit_breaker import CircuitBreaker
    from backend.resilience.backpressure import BackpressureManager
except ImportError:
    from config import settings
    from db.pool import get_pool
    from resilience.circuit_breaker import CircuitBreaker
    from resilience.backpressure import BackpressureManager

logger = logging.getLogger(__name__)


async def check_postgres() -> Tuple[bool, int]:
    """Verify PostgreSQL connectivity and measure round-trip latency in ms."""
    start = time.perf_counter()
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            val = await conn.fetchval("SELECT 1")
            lat_ms = int((time.perf_counter() - start) * 1000)
            return (val == 1, max(1, lat_ms))
    except Exception as err:
        logger.error(f"PostgreSQL health check failed: {err}")
        return False, 0


async def check_redis() -> Tuple[bool, int]:
    """Verify Redis connectivity and measure round-trip latency in ms."""
    start = time.perf_counter()
    try:
        client = redis.from_url(settings.redis_url, socket_timeout=2.0)
        pong = await client.ping()
        await client.aclose()
        lat_ms = int((time.perf_counter() - start) * 1000)
        return (bool(pong), max(1, lat_ms))
    except Exception as err:
        logger.error(f"Redis health check failed: {err}")
        return False, 0


async def check_mlflow() -> Tuple[bool, int]:
    """Verify MLflow HTTP connectivity and measure round-trip latency in ms."""
    start = time.perf_counter()
    def _ping(target_url: str) -> bool:
        try:
            res = requests.get(target_url, timeout=2.0)
            return res.status_code < 500
        except Exception:
            return False

    try:
        ok = await asyncio.to_thread(_ping, settings.mlflow_url)
        lat_ms = int((time.perf_counter() - start) * 1000)
        return (ok, max(1, lat_ms))
    except Exception:
        return False, 0


async def get_comprehensive_health() -> Dict[str, Any]:
    """
    Enhanced multi-subsystem resilience health check:
    Evaluates Postgres, Redis, MLflow, Circuit Breakers, and Ingestion Queue Depth.
    """
    pg_ok, pg_lat = await check_postgres()
    redis_ok, redis_lat = await check_redis()
    mlflow_ok, mlflow_lat = await check_mlflow()

    circuits = await CircuitBreaker.get_all_circuits_status()
    queue_depth = await BackpressureManager.get_queue_depth("queue:ingest")

    # Determine overall status: ok, degraded, or critical
    has_open_circuit = any(c.get("state") in ("open", "half_open") for c in circuits.values())
    
    if not pg_ok or not redis_ok:
        overall_status = "critical"
    elif has_open_circuit or not mlflow_ok or queue_depth > 100:
        overall_status = "degraded"
    else:
        overall_status = "ok"

    return {
        "status": overall_status,
        "checks": {
            "postgres": {"status": "ok" if pg_ok else "error", "latency_ms": pg_lat},
            "redis": {"status": "ok" if redis_ok else "error", "latency_ms": redis_lat},
            "mlflow": {"status": "ok" if mlflow_ok else "error", "latency_ms": mlflow_lat},
            "circuit_breakers": circuits,
            "queue_depth": queue_depth,
            "worker_count": 2,
        },
    }
