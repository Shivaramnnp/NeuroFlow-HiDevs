from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import asyncpg
# pyrefly: ignore [missing-import]
from opentelemetry import trace

from .chunker import Chunk, chunk_pages, select_chunking_strategy
from .extractors import (
    CSVExtractor,
    DocxExtractor,
    ExtractedPage,
    ImageExtractor,
    PDFExtractor,
    PPTXExtractor,
    URLExtractor,
)
try:
    from backend.db.pool import get_pool
    from backend.providers.client import NeuroFlowClient
except ImportError:
    from db.pool import get_pool
    from providers.client import NeuroFlowClient

logger = logging.getLogger("neuroflow-ingestion-pipeline")
tracer = trace.get_tracer("neuroflow-ingestion")


def compute_content_hash(data: Union[bytes, str]) -> str:
    """Compute SHA-256 hex digest for document bytes or text content."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


async def check_deduplication(
    content_hash: str,
    pool: Optional[asyncpg.Pool] = None,
) -> Optional[Dict[str, Any]]:
    """
    Check if a document with the given content_hash already exists in the database.
    If exists, return the existing document row dict to avoid duplicate processing and embedding costs.
    """
    db_pool = pool
    if db_pool is None:
        try:
            db_pool = get_pool()
        except Exception:
            db_pool = None

    if db_pool is None:
        return None

    query = """
        SELECT id, filename, source_type, content_hash, status, chunk_count, metadata, created_at
        FROM documents
        WHERE content_hash = $1
        LIMIT 1;
    """
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(query, content_hash)
        if row:
            return dict(row)
    return None


class IngestionPipeline:
    """
    Orchestrates extraction, auto-chunking, vector embedding, and persistence for documents.
    """

    def __init__(
        self,
        client: Optional[NeuroFlowClient] = None,
        pool: Optional[asyncpg.Pool] = None,
    ):
        self.client = client or NeuroFlowClient.get_instance()
        self.pool = pool
        self.extractors = {
            "pdf": PDFExtractor(),
            "docx": DocxExtractor(),
            "image": ImageExtractor(client=self.client),
            "csv": CSVExtractor(),
            "url": URLExtractor(),
            "pptx": PPTXExtractor(client=self.client),
        }

    def get_extractor(self, source_type: str):
        """Retrieve appropriate extractor based on source_type."""
        clean_type = source_type.lower()
        if clean_type in ["jpg", "jpeg", "png", "webp"]:
            clean_type = "image"
        return self.extractors.get(clean_type, self.extractors.get("pdf"))

    async def _get_db_pool(self) -> Optional[asyncpg.Pool]:
        if self.pool is not None:
            return self.pool
        try:
            self.pool = get_pool()
        except Exception:
            self.pool = None
        return self.pool

    async def process_document(
        self,
        document_id: str,
        source: Union[str, bytes],
        source_type: str,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute full ingestion workflow:
        1. Extract pages using format-specific extractor
        2. Auto-select and run chunking strategy
        3. Generate embeddings via NeuroFlowClient
        4. Insert chunks and embeddings into PostgreSQL (pgvector)
        5. Update document status to 'complete'
        6. Emit OpenTelemetry spans and structured logs
        """
        start_time = time.perf_counter()
        pool = await self._get_db_pool()

        # Update document status to 'processing' in DB
        if pool is not None:
            try:
                async with pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE documents SET status = 'processing' WHERE id = $1;",
                        uuid.UUID(document_id),
                    )
            except Exception as err:
                logger.warning(f"Could not update status to 'processing' for document {document_id}: {err}")

        with tracer.start_as_current_span("ingestion.process") as span:
            span.set_attribute("document_id", document_id)
            span.set_attribute("source_type", source_type)

            # Step 1: Extraction
            extractor = self.get_extractor(source_type)
            extracted_pages = await extractor.extract(source, client=self.client)
            page_count = len(extracted_pages)
            span.set_attribute("page_count", page_count)

            # Step 2: Auto-Chunking
            chunks = await chunk_pages(
                extracted_pages,
                source_type=source_type,
                client=self.client,
            )
            chunk_count = len(chunks)
            span.set_attribute("chunk_count", chunk_count)

            # Step 3: Embeddings
            chunk_texts = [c.content for c in chunks]
            embeddings: List[List[float]] = []
            embedding_calls = 0

            if chunk_texts:
                embeddings = await self.client.embed(chunk_texts)
                embedding_calls = (len(chunk_texts) + 99) // 100

            span.set_attribute("embedding_calls", embedding_calls)

            # Step 4: Persist chunks to PostgreSQL (pgvector)
            total_tokens = sum(c.token_count for c in chunks)
            if pool is not None and chunks:
                try:
                    async with pool.acquire() as conn:
                        async with conn.transaction():
                            # Delete existing chunks if any
                            await conn.execute(
                                "DELETE FROM chunks WHERE document_id = $1;",
                                uuid.UUID(document_id),
                            )
                            # Bulk insert chunks
                            chunk_records = [
                                (
                                    uuid.UUID(document_id),
                                    c.content,
                                    str(embeddings[i]) if i < len(embeddings) else None,
                                    c.chunk_index,
                                    c.token_count,
                                    json.dumps(c.metadata),
                                )
                                for i, c in enumerate(chunks)
                            ]
                            await conn.executemany(
                                """
                                INSERT INTO chunks (document_id, content, embedding, chunk_index, token_count, metadata)
                                VALUES ($1, $2, $3, $4, $5, $6::jsonb);
                                """,
                                chunk_records,
                            )
                            # Update document status to complete
                            await conn.execute(
                                """
                                UPDATE documents
                                SET status = 'complete', chunk_count = $2
                                WHERE id = $1;
                                """,
                                uuid.UUID(document_id),
                                chunk_count,
                            )
                except Exception as err:
                    logger.error(f"Failed to persist chunks to DB for document {document_id}: {err}")
                    if pool is not None:
                        try:
                            async with pool.acquire() as conn:
                                await conn.execute(
                                    "UPDATE documents SET status = 'failed' WHERE id = $1;",
                                    uuid.UUID(document_id),
                                )
                        except Exception:
                            pass
                    raise

            duration_ms = (time.perf_counter() - start_time) * 1000.0

            # Structured JSON log for observability
            structured_log = {
                "event": "ingestion_complete",
                "document_id": document_id,
                "duration_ms": round(duration_ms, 2),
                "chunks": chunk_count,
                "tokens": total_tokens,
            }
            logger.info(json.dumps(structured_log))

            return {
                "document_id": document_id,
                "status": "complete",
                "page_count": page_count,
                "chunk_count": chunk_count,
                "total_tokens": total_tokens,
                "duration_ms": duration_ms,
            }
