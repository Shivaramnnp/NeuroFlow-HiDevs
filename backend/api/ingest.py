from __future__ import annotations

import json
import logging
import os
import pathlib
import uuid
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile, status
from pydantic import BaseModel, Field
# pyrefly: ignore [missing-import]
import redis.asyncio as redis

try:
    from backend.config import settings
    from backend.db.pool import get_pool
    from backend.pipelines.ingestion.pipeline import compute_content_hash, check_deduplication
    from backend.resilience.backpressure import BackpressureManager
    from backend.resilience.rate_limiter import rate_limit_ingest
except ImportError:
    from config import settings
    from db.pool import get_pool
    from pipelines.ingestion.pipeline import compute_content_hash, check_deduplication
    from resilience.backpressure import BackpressureManager
    from resilience.rate_limiter import rate_limit_ingest

logger = logging.getLogger("neuroflow-ingest-api")

router = APIRouter(tags=["Ingestion"])

UPLOAD_DIR = pathlib.Path("/tmp/neuroflow_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".csv": "csv",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".webp": "image",
    ".pptx": "pptx",
    ".txt": "text",
}


class URLIngestRequest(BaseModel):
    url: str = Field(..., description="Web URL to ingest")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: str
    status: str
    duplicate: bool
    message: Optional[str] = None


class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    source_type: str
    status: str
    chunk_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


async def get_redis_client() -> Optional[Any]:
    """Retrieve async Redis connection."""
    try:
        return redis.from_url(settings.redis_url, socket_timeout=2.0)
    except Exception as err:
        logger.warning(f"Could not connect to Redis: {err}")
        return None


async def enqueue_ingest_job(
    document_id: str,
    file_path: str,
    source_type: str,
    filename: str,
    redis_client: Optional[Any] = None,
) -> None:
    """Enqueue document ingestion task to Redis queue:ingest."""
    client = redis_client or await get_redis_client()
    if client is not None:
        try:
            payload = json.dumps(
                {
                    "document_id": document_id,
                    "file_path": file_path,
                    "source_type": source_type,
                    "filename": filename,
                }
            )
            await client.lpush("queue:ingest", payload)
            logger.info(f"Enqueued document {document_id} to 'queue:ingest'")
        except Exception as err:
            logger.error(f"Failed to enqueue document {document_id} to Redis: {err}")
        finally:
            await client.aclose()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_document(
    request: Request,
    file: Optional[UploadFile] = File(None),
):
    """
    Ingest a document from uploaded file (multipart/form-data) or JSON body with {"url": "..."}.
    - Validates file size (max 100MB) and allowed extensions.
    - Computes content SHA-256 hash.
    - Checks deduplication: if file hash already exists, returns existing document ID with duplicate=true.
    - If new: creates document with status='queued', enqueues to Redis queue:ingest, and returns immediately.
    """
    # 1. Rate Limiting Check
    await rate_limit_ingest(request)

    # 2. Backpressure Check
    has_warning, warning_info = await BackpressureManager.check_ingestion_backpressure()

    pool = None
    try:
        pool = get_pool()
    except Exception:
        pool = None

    # Handle URL ingest via JSON body
    if file is None:
        try:
            body = await request.json()
            url = body.get("url")
            if not url:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Either 'file' or JSON 'url' must be provided.",
                )
            source_type = "url"
            filename = url
            file_bytes = url.encode("utf-8")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'file' or valid JSON 'url' is required.",
            )
    else:
        # File upload
        filename = file.filename or "uploaded_file"
        ext = pathlib.Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{ext}'. Allowed: {list(ALLOWED_EXTENSIONS.keys())}",
            )
        source_type = ALLOWED_EXTENSIONS[ext]
        file_bytes = await file.read()

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of 100MB.",
            )

    # 1. Compute Content Hash & Deduplication check
    content_hash = compute_content_hash(file_bytes)
    existing_doc = await check_deduplication(content_hash, pool=pool)

    if existing_doc:
        logger.info(f"Duplicate document detected (hash={content_hash}). Returning existing ID: {existing_doc['id']}")
        return IngestResponse(
            document_id=str(existing_doc["id"]),
            status=existing_doc.get("status", "complete"),
            duplicate=True,
            message="Document already ingested. Duplicate avoided.",
        )

    # 2. Save file to disk for worker processing
    document_id = str(uuid.uuid4())
    saved_file_path = str(UPLOAD_DIR / f"{document_id}_{filename}")
    if source_type != "url":
        with open(saved_file_path, "wb") as f:
            f.write(file_bytes)
    else:
        saved_file_path = filename

    # 3. Create document record with status='queued' in PostgreSQL
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO documents (id, filename, source_type, content_hash, status, metadata)
                    VALUES ($1, $2, $3, $4, 'queued', $5::jsonb);
                    """,
                    uuid.UUID(document_id),
                    filename,
                    source_type,
                    content_hash,
                    json.dumps({"saved_file_path": saved_file_path}),
                )
        except Exception as err:
            logger.error(f"Failed to create document in database: {err}")

    # 4. Enqueue to Redis queue:ingest
    await enqueue_ingest_job(
        document_id=document_id,
        file_path=saved_file_path,
        source_type=source_type,
        filename=filename,
    )

    return IngestResponse(
        document_id=document_id,
        status="queued",
        duplicate=False,
        message="Document queued for asynchronous processing.",
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentStatusResponse,
)
async def get_document_status(document_id: str):
    """
    Retrieve document processing status, chunk count, and metadata.
    """
    try:
        pool = get_pool()
    except Exception:
        pool = None

    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection is unavailable.",
        )

    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid UUID format.",
        )

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, filename, source_type, status, chunk_count, metadata, created_at
            FROM documents
            WHERE id = $1;
            """,
            doc_uuid,
        )
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found.",
            )

        meta = row["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)

        return DocumentStatusResponse(
            document_id=str(row["id"]),
            filename=row["filename"],
            source_type=row["source_type"],
            status=row["status"],
            chunk_count=row["chunk_count"],
            metadata=meta or {},
            created_at=row["created_at"].isoformat() if row["created_at"] else None,
        )
