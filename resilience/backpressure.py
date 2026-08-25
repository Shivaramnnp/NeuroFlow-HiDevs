from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException, status
# pyrefly: ignore [missing-import]
import redis.asyncio as redis

try:
    from backend.config import settings
except ImportError:
    from config import settings

logger = logging.getLogger("neuroflow-backpressure")


class BackpressureManager:
    """
    Monitors Redis background ingestion queue (queue:ingest) and enforces backpressure thresholds:
    - queue_depth > 100: Rejects with 503 Service Unavailable
    - queue_depth > 50: Accepts with 202 and high wait-time warning
    - queue_depth <= 50: Normal execution
    """

    MAX_QUEUE_DEPTH = 100
    WARNING_QUEUE_DEPTH = 50

    @classmethod
    async def get_queue_depth(cls, queue_name: str = "queue:ingest", redis_url: Optional[str] = None) -> int:
        r_url = redis_url or settings.redis_url
        try:
            r = redis.from_url(r_url, socket_timeout=1.5)
            depth = await r.llen(queue_name)
            await r.aclose()
            return int(depth)
        except Exception as err:
            logger.warning(f"Could not check queue depth for {queue_name}: {err}")
            return 0

    @classmethod
    async def check_ingestion_backpressure(cls, redis_url: Optional[str] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Evaluate queue depth for ingestion:
        - If > 100: raises 503 HTTP exception
        - If > 50: returns (True, warning_dict)
        - Else: returns (False, None)
        """
        depth = await cls.get_queue_depth("queue:ingest", redis_url=redis_url)

        if depth > cls.MAX_QUEUE_DEPTH:
            logger.error(f"Ingestion queue depth ({depth}) exceeds limit {cls.MAX_QUEUE_DEPTH}. Triggering 503 Backpressure!")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "ingestion_queue_full",
                    "queue_depth": depth,
                    "retry_after": 30,
                    "message": "Ingestion queue is currently at maximum capacity. Please retry in 30 seconds.",
                },
                headers={"Retry-After": "30"},
            )

        if depth > cls.WARNING_QUEUE_DEPTH:
            est_minutes = max(1, int(depth * 0.2))  # ~12s per document
            logger.warning(f"Ingestion queue depth ({depth}) elevated (> {cls.WARNING_QUEUE_DEPTH}). Returning 202 with wait warning.")
            return True, {
                "warning": "high_queue_depth",
                "queue_depth": depth,
                "estimated_wait_minutes": est_minutes,
            }

        return False, None
