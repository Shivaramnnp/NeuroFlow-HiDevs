import logging
import asyncpg
from config import settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def init_pool() -> asyncpg.Pool:
    """
    Initialize global asyncpg database pool during application lifespan startup.
    """
    global _pool

    if _pool is None:
        logger.info(f"Initializing asyncpg pool connecting to {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}...")
        try:
            _pool = await asyncpg.create_pool(
                dsn=settings.postgres_dsn,
                min_size=2,
                max_size=10,
                command_timeout=30,
            )
        except Exception as err:
            # Fallback to localhost if host resolves fail (e.g., local dev outside docker)
            if settings.POSTGRES_HOST != "localhost":
                dsn = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@localhost:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
                logger.warning(f"Failed connecting with host '{settings.POSTGRES_HOST}', trying fallback DSN: {dsn}")
                _pool = await asyncpg.create_pool(
                    dsn=dsn,
                    min_size=2,
                    max_size=10,
                    command_timeout=30,
                )
            else:
                raise err
    return _pool


def get_pool() -> asyncpg.Pool:
    """
    Retrieve initialized asyncpg connection pool.
    """
    if _pool is None:
        raise RuntimeError("Database pool has not been initialized.")
    return _pool


async def close_pool() -> None:
    """
    Close global asyncpg pool during application lifespan shutdown.
    """
    global _pool

    if _pool is not None:
        logger.info("Closing asyncpg database connection pool...")
        await _pool.close()
        _pool = None