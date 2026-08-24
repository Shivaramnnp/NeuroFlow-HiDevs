from __future__ import annotations

import asyncio
import json
import logging
import signal
from typing import Any, Dict, Optional

# pyrefly: ignore [missing-import]
import redis.asyncio as redis

from backend.config import settings
from backend.db.pool import close_pool, init_pool
from backend.pipelines.ingestion.pipeline import IngestionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neuroflow-worker")

_running = True


def handle_stop_signals():
    global _running
    _running = False
    logger.info("Received termination signal. Gracefully shutting down worker...")


async def process_ingest_job(
    job_data: Dict[str, Any],
    pipeline: IngestionPipeline,
) -> None:
    """Process a single document ingestion job from the queue."""
    doc_id = job_data.get("document_id")
    file_path = job_data.get("file_path")
    source_type = job_data.get("source_type", "pdf")
    filename = job_data.get("filename")

    if not doc_id or not file_path:
        logger.error(f"Invalid job payload received: {job_data}")
        return

    logger.info(f"Processing ingestion for document {doc_id} (type: {source_type}, file: {filename})...")
    try:
        res = await pipeline.process_document(
            document_id=doc_id,
            source=file_path,
            source_type=source_type,
            filename=filename,
        )
        logger.info(f"Successfully finished ingestion for {doc_id}: {res}")
    except Exception as err:
        logger.error(f"Ingestion failed for document {doc_id}: {err}", exc_info=True)


async def arq_ingest_task(ctx: Dict[str, Any], document_id: str, file_path: str, source_type: str, filename: Optional[str] = None):
    """Arq task entry point for structured job management."""
    pipeline = IngestionPipeline()
    return await pipeline.process_document(
        document_id=document_id,
        source=file_path,
        source_type=source_type,
        filename=filename,
    )


class WorkerSettings:
    """Arq Worker settings."""
    functions = [arq_ingest_task]
    redis_settings = None


async def main():
    """Main worker event loop pulling from Redis queue:ingest."""
    global _running
    logger.info("Starting NeuroFlow async background worker...")

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_stop_signals)
        except (NotImplementedError, RuntimeError):
            pass

    # Initialize DB pool
    pool = None
    try:
        pool = await init_pool()
    except Exception as err:
        logger.warning(f"Could not connect to database pool: {err}")

    # Initialize Redis client
    redis_client = None
    try:
        redis_client = redis.from_url(settings.redis_url, socket_timeout=5.0)
    except Exception as err:
        logger.warning(f"Could not connect to Redis: {err}")

    pipeline = IngestionPipeline(pool=pool)

    logger.info("Worker initialized. Listening for jobs on 'queue:ingest'...")

    while _running:
        try:
            if redis_client is not None:
                item = await redis_client.brpop("queue:ingest", timeout=5)
                if item:
                    _, raw_payload = item
                    if isinstance(raw_payload, bytes):
                        raw_payload = raw_payload.decode("utf-8")
                    job_data = json.loads(raw_payload)
                    await process_ingest_job(job_data, pipeline)
            else:
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            break
        except Exception as err:
            logger.error(f"Worker queue processing error: {err}")
            await asyncio.sleep(1)

    logger.info("Closing worker connections...")
    if redis_client:
        await redis_client.aclose()
    await close_pool()
    logger.info("Worker shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
