from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

# pyrefly: ignore [missing-import]
import redis.asyncio as redis

try:
    from backend.config import settings
except ImportError:
    from config import settings

logger = logging.getLogger("neuroflow-timeouts")

T = TypeVar("T")


class TimeoutError(Exception):
    """Raised when an operation exceeds its configured or adaptive deadline."""
    pass


class TimeoutManager:
    """
    Manages explicit, task-aware timeouts with Redis metric telemetry
    and adaptive p95 multiplier adjustments.
    """

    DEFAULT_TIMEOUTS: Dict[str, float] = {
        "embedding": 10.0,
        "chat_completion": 60.0,
        "reranking": 15.0,
        "evaluation": 120.0,
        "file_extraction": 30.0,
        "url_fetch": 15.0,
    }

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.redis_url

    async def get_timeout_for_task(self, task_type: str) -> float:
        """
        Calculate adaptive timeout:
        Checks p95 latency over recent runs in Redis and scales timeout by 1.5x,
        falling back to DEFAULT_TIMEOUTS[task_type].
        """
        base_timeout = self.DEFAULT_TIMEOUTS.get(task_type, 30.0)
        try:
            r = redis.from_url(self.redis_url, socket_timeout=1.0)
            # Retrieve last 100 latency measurements from sorted set
            scores = await r.zrange(f"latencies:{task_type}", -100, -1, withscores=True)
            await r.aclose()

            if scores and len(scores) >= 10:
                latencies = [s[1] for s in scores]
                latencies.sort()
                p95 = latencies[int(len(latencies) * 0.95)] / 1000.0  # ms to seconds
                adaptive = max(base_timeout, p95 * 1.5)
                return adaptive
        except Exception:
            pass
        return base_timeout

    async def run_with_timeout(
        self,
        coro: Awaitable[T],
        task_type: str,
        custom_timeout: Optional[float] = None,
    ) -> T:
        """
        Execute an async coroutine with an explicit timeout deadline.
        On timeout, increments timeouts:{task_type} telemetry in Redis and raises TimeoutError.
        """
        timeout_sec = custom_timeout or await self.get_timeout_for_task(task_type)
        start = time.perf_counter()

        try:
            return await asyncio.wait_for(coro, timeout=timeout_sec)
        except asyncio.TimeoutError:
            duration_ms = int((time.perf_counter() - start) * 1000)
            logger.error(f"Task '{task_type}' timed out after {timeout_sec}s ({duration_ms}ms elapsed)")

            # Increment Redis timeout counter
            try:
                r = redis.from_url(self.redis_url, socket_timeout=1.0)
                await r.incr(f"timeouts:{task_type}")
                await r.aclose()
            except Exception:
                pass

            raise TimeoutError(f"Operation '{task_type}' timed out after {timeout_sec} seconds")
        except Exception:
            raise
