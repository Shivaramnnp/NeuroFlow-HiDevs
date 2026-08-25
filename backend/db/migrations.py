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

        # Apply schema upgrades for Task 38 if not present
        await conn.execute(
            """
            ALTER TABLE pipelines ADD COLUMN IF NOT EXISTS description TEXT;
            ALTER TABLE pipelines ADD COLUMN IF NOT EXISTS version INT NOT NULL DEFAULT 1;
            ALTER TABLE pipelines ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'active';
            ALTER TABLE pipelines ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

            CREATE TABLE IF NOT EXISTS pipeline_versions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                pipeline_id UUID NOT NULL REFERENCES pipelines(id),
                version INT NOT NULL,
                config JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(pipeline_id, version)
            );

            ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS pipeline_version INT NOT NULL DEFAULT 1;
            ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS retrieval_latency_ms INT;
            """
        )
