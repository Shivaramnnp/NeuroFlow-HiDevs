from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import HTTPException

from backend.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError, circuit_breaker_guard
from backend.resilience.rate_limiter import SlidingWindowRateLimiter, TokenBucketRateLimiter
from backend.resilience.backpressure import BackpressureManager
from backend.resilience.timeouts import TimeoutError, TimeoutManager


# --- 1. Circuit Breaker State Transition Tests ---

@pytest.mark.asyncio
async def test_circuit_breaker_closed_to_open_transition():
    cb = CircuitBreaker(name="test_cb_1", failure_threshold=3, recovery_timeout=2)
    cb._get_redis = AsyncMock(return_value=None)  # Use in-memory state

    assert await cb.get_state() == CircuitBreaker.STATE_CLOSED

    # 1st failure
    await cb.record_failure(Exception("Fail 1"))
    assert await cb.get_state() == CircuitBreaker.STATE_CLOSED

    # 2nd failure
    await cb.record_failure(Exception("Fail 2"))
    assert await cb.get_state() == CircuitBreaker.STATE_CLOSED

    # 3rd failure -> should OPEN
    await cb.record_failure(Exception("Fail 3"))
    assert await cb.get_state() == CircuitBreaker.STATE_OPEN

    # Calls while OPEN should raise CircuitOpenError
    with pytest.raises(CircuitOpenError):
        await cb.check_can_execute()


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_and_recovery():
    cb = CircuitBreaker(name="test_cb_2", failure_threshold=2, recovery_timeout=1, half_open_max_calls=2)
    cb._get_redis = AsyncMock(return_value=None)

    # Force to OPEN
    await cb.record_failure(Exception("1"))
    await cb.record_failure(Exception("2"))
    assert await cb.get_state() == CircuitBreaker.STATE_OPEN

    # Wait for recovery timeout
    await asyncio.sleep(1.1)
    assert await cb.get_state() == CircuitBreaker.STATE_HALF_OPEN

    # Test call in HALF_OPEN succeeds -> transitions to CLOSED
    await cb.check_can_execute()
    await cb.record_success()
    assert await cb.get_state() == CircuitBreaker.STATE_CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_guard_context_manager():
    cb_guard = circuit_breaker_guard("test_guard", failure_threshold=2, recovery_timeout=5)

    async def mock_api(success=True):
        async with circuit_breaker_guard("test_guard_api", failure_threshold=2, recovery_timeout=5):
            if not success:
                raise ValueError("API Error")
            return "OK"

    # Success call
    res = await mock_api(success=True)
    assert res == "OK"

    # Failing calls
    with pytest.raises(ValueError):
        await mock_api(success=False)
    with pytest.raises(ValueError):
        await mock_api(success=False)

    # Subsequent call blocked by CircuitOpenError
    with pytest.raises(CircuitOpenError):
        await mock_api(success=True)


# --- 2. Rate Limiter Tests ---

@pytest.mark.asyncio
async def test_token_bucket_rate_limiter():
    limiter = TokenBucketRateLimiter(key_prefix="test_rpb", capacity=5, refill_rate=2.0)
    limiter._get_redis = AsyncMock(return_value=None)

    # Consume all tokens
    for _ in range(5):
        acquired = await limiter.acquire(tokens=1, timeout=0.1)
        assert acquired is True

    # 6th should time out quickly
    acquired = await limiter.acquire(tokens=1, timeout=0.1)
    assert acquired is False


# --- 3. Backpressure Tests ---

@pytest.mark.asyncio
async def test_backpressure_thresholds():
    # Normal queue (< 50)
    with patch.object(BackpressureManager, "get_queue_depth", AsyncMock(return_value=25)):
        has_warn, info = await BackpressureManager.check_ingestion_backpressure()
        assert has_warn is False
        assert info is None

    # Warning queue (50-100)
    with patch.object(BackpressureManager, "get_queue_depth", AsyncMock(return_value=75)):
        has_warn, info = await BackpressureManager.check_ingestion_backpressure()
        assert has_warn is True
        assert info["warning"] == "high_queue_depth"

    # Queue full (> 100) -> 503 HTTP Exception
    with patch.object(BackpressureManager, "get_queue_depth", AsyncMock(return_value=120)):
        with pytest.raises(HTTPException) as exc_info:
            await BackpressureManager.check_ingestion_backpressure()
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail["error"] == "ingestion_queue_full"


# --- 4. Timeout Manager Tests ---

@pytest.mark.asyncio
async def test_timeout_manager_execution_and_timeout():
    tm = TimeoutManager()

    async def fast_task():
        await asyncio.sleep(0.05)
        return "done"

    async def slow_task():
        await asyncio.sleep(0.5)
        return "done"

    # Fast task passes
    res = await tm.run_with_timeout(fast_task(), task_type="embedding", custom_timeout=0.2)
    assert res == "done"

    # Slow task times out
    with pytest.raises(TimeoutError):
        await tm.run_with_timeout(slow_task(), task_type="embedding", custom_timeout=0.1)


# --- 5. Health Check Resilience Status Tests ---

@pytest.mark.asyncio
async def test_comprehensive_health_status():
    from backend.db.health import get_comprehensive_health

    # Mock healthy services
    with patch("backend.db.health.check_postgres", AsyncMock(return_value=(True, 2))), \
         patch("backend.db.health.check_redis", AsyncMock(return_value=(True, 1))), \
         patch("backend.db.health.check_mlflow", AsyncMock(return_value=(True, 35))), \
         patch("backend.db.health.CircuitBreaker.get_all_circuits_status", AsyncMock(return_value={"openai": {"state": "closed"}})), \
         patch("backend.db.health.BackpressureManager.get_queue_depth", AsyncMock(return_value=10)):
        
        health = await get_comprehensive_health()
        assert health["status"] == "ok"
        assert "circuit_breakers" in health["checks"]
        assert health["checks"]["queue_depth"] == 10

    # Mock degraded due to open circuit
    with patch("backend.db.health.check_postgres", AsyncMock(return_value=(True, 2))), \
         patch("backend.db.health.check_redis", AsyncMock(return_value=(True, 1))), \
         patch("backend.db.health.check_mlflow", AsyncMock(return_value=(True, 35))), \
         patch("backend.db.health.CircuitBreaker.get_all_circuits_status", AsyncMock(return_value={"openai": {"state": "open"}})), \
         patch("backend.db.health.BackpressureManager.get_queue_depth", AsyncMock(return_value=10)):
        
        health_degraded = await get_comprehensive_health()
        assert health_degraded["status"] == "degraded"
