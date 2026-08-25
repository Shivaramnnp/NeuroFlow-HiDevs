from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, Tuple

from fastapi import HTTPException, Request, status
# pyrefly: ignore [missing-import]
import redis.asyncio as redis

try:
    from backend.config import settings
except ImportError:
    from config import settings

logger = logging.getLogger("neuroflow-rate-limiter")


class RateLimitExceededError(HTTPException):
    def __init__(self, detail: str = "Rate limit exceeded", retry_after: int = 60):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={"Retry-After": str(retry_after)},
        )


class TokenBucketRateLimiter:
    """
    Redis-persisted Token Bucket rate limiter for LLM provider and pipeline capacity:
    - capacity: maximum burst token bucket size (e.g. 3000 tokens)
    - refill_rate: tokens added per second (e.g. 50 tokens/sec)
    """

    def __init__(
        self,
        key_prefix: str,
        capacity: int = 3000,
        refill_rate: float = 50.0,
        redis_url: Optional[str] = None,
    ):
        self.key_prefix = key_prefix
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.redis_url = redis_url or settings.redis_url

        # Local fallback in-memory state
        self._local_tokens = float(capacity)
        self._local_last_refill = time.time()

    async def _get_redis(self) -> Optional[redis.Redis]:
        try:
            client = redis.from_url(self.redis_url, socket_timeout=1.5)
            await client.ping()
            return client
        except Exception:
            return None

    async def acquire(self, tokens: int = 1, timeout: float = 5.0) -> bool:
        """
        Acquire tokens from the bucket. If not enough tokens, waits up to timeout.
        Returns True if acquired, False/raises if timed out.
        """
        deadline = time.time() + timeout
        r = await self._get_redis()

        while time.time() <= deadline:
            now = time.time()

            if r is not None:
                try:
                    tok_key = f"{self.key_prefix}:tokens"
                    time_key = f"{self.key_prefix}:last_refill"

                    raw_tokens = await r.get(tok_key)
                    raw_time = await r.get(time_key)

                    current_tokens = float(raw_tokens) if raw_tokens else float(self.capacity)
                    last_time = float(raw_time) if raw_time else now

                    # Refill
                    elapsed = max(0.0, now - last_time)
                    current_tokens = min(float(self.capacity), current_tokens + (elapsed * self.refill_rate))

                    if current_tokens >= tokens:
                        current_tokens -= tokens
                        await r.set(tok_key, str(current_tokens))
                        await r.set(time_key, str(now))
                        await r.aclose()
                        return True

                    # Sleep fractionally and retry
                    wait_sec = min(0.2, (tokens - current_tokens) / max(0.1, self.refill_rate))
                    await asyncio.sleep(max(0.05, wait_sec))
                    continue
                except Exception as err:
                    logger.warning(f"Redis TokenBucket error: {err}")

            # Local fallback
            elapsed = max(0.0, now - self._local_last_refill)
            self._local_tokens = min(float(self.capacity), self._local_tokens + (elapsed * self.refill_rate))
            self._local_last_refill = now

            if self._local_tokens >= tokens:
                self._local_tokens -= tokens
                return True

            wait_sec = min(0.2, (tokens - self._local_tokens) / max(0.1, self.refill_rate))
            await asyncio.sleep(max(0.05, wait_sec))

        if r is not None:
            try:
                await r.aclose()
            except Exception:
                pass
        return False


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter enforcing per-IP request rates on public endpoints:
    - /ingest: 10 requests / hour (window: 3600s, max: 10)
    - /query: 60 requests / minute (window: 60s, max: 60)
    """

    @staticmethod
    async def check_rate_limit(
        identifier: str,
        endpoint_name: str,
        max_requests: int,
        window_seconds: int,
        redis_url: Optional[str] = None,
    ) -> Tuple[bool, int]:
        """
        Check sliding window counter.
        Returns (is_allowed: bool, retry_after_seconds: int).
        """
        r_url = redis_url or settings.redis_url
        now = time.time()
        key = f"ratelimit:{endpoint_name}:{identifier}"

        try:
            r = redis.from_url(r_url, socket_timeout=1.5)
            await r.ping()

            pipe = r.pipeline()
            # 1. Remove timestamps older than current window
            pipe.zremrangebyscore(key, 0, now - window_seconds)
            # 2. Count requests in current window
            pipe.zcard(key)
            # 3. Add current timestamp
            pipe.zadd(key, {str(now): now})
            # 4. Set key TTL
            pipe.expire(key, window_seconds)
            results = await pipe.execute()

            current_count = results[1]
            await r.aclose()

            if current_count >= max_requests:
                return False, int(window_seconds)
            return True, 0
        except Exception as err:
            logger.warning(f"SlidingWindowRateLimiter error (allowing request): {err}")
            return True, 0


# Dependency helpers for FastAPI routes
async def rate_limit_ingest(request: Request):
    client_ip = request.client.host if request.client else "127.0.0.1"
    allowed, retry_after = await SlidingWindowRateLimiter.check_rate_limit(
        identifier=client_ip,
        endpoint_name="ingest",
        max_requests=10,
        window_seconds=3600,
    )
    if not allowed:
        raise RateLimitExceededError(
            detail=f"Rate limit exceeded on /ingest. Maximum 10 uploads per hour.",
            retry_after=retry_after,
        )


async def rate_limit_query(request: Request):
    client_ip = request.client.host if request.client else "127.0.0.1"
    allowed, retry_after = await SlidingWindowRateLimiter.check_rate_limit(
        identifier=client_ip,
        endpoint_name="query",
        max_requests=60,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitExceededError(
            detail=f"Rate limit exceeded on /query. Maximum 60 queries per minute.",
            retry_after=retry_after,
        )
