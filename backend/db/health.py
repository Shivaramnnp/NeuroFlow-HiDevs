import logging
import asyncio
import redis.asyncio as redis
import requests

from config import settings
from db.pool import get_pool

logger = logging.getLogger(__name__)


async def check_postgres() -> bool:
    """
    Verify PostgreSQL connectivity using asyncpg pool.
    """
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            val = await conn.fetchval("SELECT 1")
            return val == 1
    except Exception as err:
        logger.error(f"PostgreSQL health check failed: {err}")
        return False


async def check_redis() -> bool:
    """
    Verify Redis connectivity using asyncio redis client.
    """
    try:
        # Try configured host first, fallback to localhost if needed
        urls = [settings.redis_url]
        if settings.REDIS_HOST != "localhost":
            fallback_url = f"redis://:{settings.REDIS_PASSWORD}@localhost:{settings.REDIS_PORT}/0" if settings.REDIS_PASSWORD else f"redis://localhost:{settings.REDIS_PORT}/0"
            urls.append(fallback_url)

        for url in urls:
            try:
                client = redis.from_url(url, socket_timeout=3.0)
                pong = await client.ping()
                await client.aclose()
                if pong:
                    return True
            except Exception:
                continue
        return False
    except Exception as err:
        logger.error(f"Redis health check failed: {err}")
        return False


async def check_mlflow() -> bool:
    """
    Verify MLflow HTTP connectivity.
    """
    urls = [settings.mlflow_url]
    if settings.MLFLOW_HOST != "localhost":
        urls.append(f"http://localhost:{settings.MLFLOW_PORT}")

    def _ping_mlflow(target_url: str) -> bool:
        try:
            # MLflow UI / server health check endpoint or root URL
            res = requests.get(target_url, timeout=3.0)
            return res.status_code < 500
        except Exception:
            return False

    for url in urls:
        try:
            ok = await asyncio.to_thread(_ping_mlflow, url)
            if ok:
                return True
        except Exception:
            continue

    logger.error("MLflow health check failed for all URLs")
    return False
