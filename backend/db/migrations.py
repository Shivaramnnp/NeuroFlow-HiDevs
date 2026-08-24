from __future__ import annotations

import os
import logging

try:
    from db.pool import get_pool
except ImportError:
    from backend.db.pool import get_pool

logger = logging.getLogger(__name__)


async def check_and_apply_migrations() -> None:
    """
    Check if required database tables exist, and apply initial schema if missing.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        table_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'documents'
            );
            """
        )
        if not table_exists:
            logger.info("Database schema not found. Applying migrations...")
            schema_path = os.path.join(os.path.dirname(__file__), "../../infra/init/001_schema.sql")
            if os.path.exists(schema_path):
                with open(schema_path, "r") as f:
                    sql = f.read()
                await conn.execute(sql)
                logger.info("Migrations applied successfully from 001_schema.sql.")
            else:
                logger.warning(f"Schema file not found at {schema_path}")
        else:
            logger.info("Database schema verified.")
