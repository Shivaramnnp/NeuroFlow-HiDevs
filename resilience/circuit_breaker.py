from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

# pyrefly: ignore [missing-import]
import redis.asyncio as redis

try:
    from backend.config import settings
except ImportError:
    from config import settings

logger = logging.getLogger("neuroflow-circuit-breaker")


class CircuitOpenError(Exception):
    """Raised when an external API call is attempted while the circuit breaker is in OPEN state."""
    pass


class CircuitBreaker:
    """
    Distributed, Redis-persisted Circuit Breaker protecting external LLM API calls.
    - CLOSED: Normal operation. If failure_threshold consecutive errors occur -> OPEN.
    - OPEN: All calls immediately fail with CircuitOpenError. After recovery_timeout -> HALF_OPEN.
    - HALF_OPEN: Allows up to half_open_max_calls. If successful -> CLOSED. If error -> OPEN.
    """

    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half_open"

    _MEMORY_REGISTRY: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
        redis_url: Optional[str] = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.redis_url = redis_url or settings.redis_url

        if self.name not in self._MEMORY_REGISTRY:
            self._MEMORY_REGISTRY[self.name] = {
                "state": self.STATE_CLOSED,
                "failure_count": 0,
                "opened_at": 0.0,
                "half_open_calls": 0,
            }

    @property
    def _local_data(self) -> Dict[str, Any]:
        if self.name not in self._MEMORY_REGISTRY:
            self._MEMORY_REGISTRY[self.name] = {
                "state": self.STATE_CLOSED,
                "failure_count": 0,
                "opened_at": 0.0,
                "half_open_calls": 0,
            }
        return self._MEMORY_REGISTRY[self.name]

    async def _get_redis(self) -> Optional[redis.Redis]:
        try:
            client = redis.from_url(self.redis_url, socket_timeout=1.5)
            await client.ping()
            return client
        except Exception:
            return None

    async def get_state(self) -> str:
        """Fetch current state from Redis or fallback, evaluating recovery timeout."""
        r = await self._get_redis()
        now = time.time()

        if r is not None:
            try:
                state_raw = await r.get(f"circuit:{self.name}:state")
                state = state_raw.decode("utf-8") if isinstance(state_raw, bytes) else (state_raw or self.STATE_CLOSED)

                if state == self.STATE_OPEN:
                    opened_at_raw = await r.get(f"circuit:{self.name}:opened_at")
                    opened_at = float(opened_at_raw) if opened_at_raw else 0.0
                    if now - opened_at >= self.recovery_timeout:
                        # Transition to HALF_OPEN
                        await r.set(f"circuit:{self.name}:state", self.STATE_HALF_OPEN)
                        await r.set(f"circuit:{self.name}:half_open_calls", 0)
                        state = self.STATE_HALF_OPEN
                        logger.info(f"Circuit '{self.name}' transitioned from OPEN to HALF_OPEN after {self.recovery_timeout}s timeout")

                await r.aclose()
                return state
            except Exception as err:
                logger.warning(f"Redis get_state error for circuit {self.name}: {err}")

        # Local fallback evaluation
        data = self._local_data
        if data["state"] == self.STATE_OPEN:
            if now - data["opened_at"] >= self.recovery_timeout:
                data["state"] = self.STATE_HALF_OPEN
                data["half_open_calls"] = 0
        return data["state"]

    async def check_can_execute(self) -> None:
        """Check if call is allowed to proceed or should be blocked immediately."""
        state = await self.get_state()
        if state == self.STATE_OPEN:
            raise CircuitOpenError(f"Circuit breaker '{self.name}' is OPEN. Requests blocked to protect provider.")

        if state == self.STATE_HALF_OPEN:
            r = await self._get_redis()
            if r is not None:
                try:
                    calls = await r.incr(f"circuit:{self.name}:half_open_calls")
                    await r.aclose()
                    if calls > self.half_open_max_calls:
                        raise CircuitOpenError(f"Circuit breaker '{self.name}' is HALF_OPEN and max test calls ({self.half_open_max_calls}) reached.")
                except CircuitOpenError:
                    raise
                except Exception:
                    pass
            else:
                self._local_data["half_open_calls"] += 1
                if self._local_data["half_open_calls"] > self.half_open_max_calls:
                    raise CircuitOpenError(f"Circuit breaker '{self.name}' is HALF_OPEN and max test calls reached.")

    async def record_success(self) -> None:
        """Record successful call: resets failure counter and closes circuit if in HALF_OPEN."""
        r = await self._get_redis()
        if r is not None:
            try:
                await r.set(f"circuit:{self.name}:state", self.STATE_CLOSED)
                await r.set(f"circuit:{self.name}:failure_count", 0)
                await r.delete(f"circuit:{self.name}:opened_at")
                await r.delete(f"circuit:{self.name}:half_open_calls")
                await r.aclose()
            except Exception as err:
                logger.warning(f"Redis record_success error for {self.name}: {err}")

        data = self._local_data
        data["state"] = self.STATE_CLOSED
        data["failure_count"] = 0
        data["opened_at"] = 0.0
        data["half_open_calls"] = 0

    async def record_failure(self, error: Exception) -> None:
        """Record call failure: increments failure count and opens circuit if threshold reached."""
        state = await self.get_state()
        r = await self._get_redis()
        now = time.time()

        if state == self.STATE_HALF_OPEN:
            # Immediate transition to OPEN on failure during testing
            if r is not None:
                try:
                    await r.set(f"circuit:{self.name}:state", self.STATE_OPEN)
                    await r.set(f"circuit:{self.name}:opened_at", str(now))
                    await r.aclose()
                except Exception:
                    pass
            data = self._local_data
            data["state"] = self.STATE_OPEN
            data["opened_at"] = now
            logger.warning(f"Circuit '{self.name}' failed in HALF_OPEN state -> Re-opening circuit due to error: {error}")
            return

        # Normal CLOSED state failure handling
        if r is not None:
            try:
                count = await r.incr(f"circuit:{self.name}:failure_count")
                if count >= self.failure_threshold:
                    await r.set(f"circuit:{self.name}:state", self.STATE_OPEN)
                    await r.set(f"circuit:{self.name}:opened_at", str(now))
                    logger.error(f"Circuit '{self.name}' OPENED after {count} consecutive failures! Error: {error}")
                    try:
                        from backend.monitoring.metrics import circuit_breaker_trips
                        circuit_breaker_trips.labels(provider=self.name).inc()
                    except Exception:
                        pass
                await r.aclose()
                return
            except Exception as err:
                logger.warning(f"Redis record_failure error for {self.name}: {err}")

        # Local fallback
        data = self._local_data
        data["failure_count"] += 1
        if data["failure_count"] >= self.failure_threshold:
            data["state"] = self.STATE_OPEN
            data["opened_at"] = now
            logger.error(f"Circuit '{self.name}' OPENED in local fallback after {data['failure_count']} failures!")
            try:
                from backend.monitoring.metrics import circuit_breaker_trips
                circuit_breaker_trips.labels(provider=self.name).inc()
            except Exception:
                pass

    @classmethod
    async def get_all_circuits_status(cls, redis_url: Optional[str] = None) -> Dict[str, Any]:
        """Fetch live status of all registered provider circuit breakers."""
        names = ["openai", "anthropic", "openrouter"]
        statuses = {}
        for name in names:
            cb = cls(name=name, redis_url=redis_url)
            state = await cb.get_state()
            statuses[name] = {
                "state": state,
                "failure_count": cb._local_failure_count,
            }
        return statuses


@asynccontextmanager
async def circuit_breaker_guard(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    half_open_max_calls: int = 3,
) -> AsyncGenerator[CircuitBreaker, None]:
    """
    Context manager protecting a block with a CircuitBreaker:
    async with circuit_breaker_guard("openai"):
        result = await provider.complete(messages)
    """
    cb = CircuitBreaker(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        half_open_max_calls=half_open_max_calls,
    )
    await cb.check_can_execute()
    try:
        yield cb
        await cb.record_success()
    except CircuitOpenError:
        raise
    except Exception as err:
        await cb.record_failure(err)
        raise
