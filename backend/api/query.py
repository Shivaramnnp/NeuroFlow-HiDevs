from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

try:
    from backend.db.pool import get_pool
    from backend.pipelines.generation.generator import RAGGenerator
    from backend.pipelines.retrieval.context_assembler import ContextAssembler
    from backend.pipelines.retrieval.query_processor import QueryProcessor
    from backend.pipelines.retrieval.retriever import HybridRetriever
    from backend.providers.client import NeuroFlowClient
except ImportError:
    from db.pool import get_pool
    from pipelines.generation.generator import RAGGenerator
    from pipelines.retrieval.context_assembler import ContextAssembler
    from pipelines.retrieval.query_processor import QueryProcessor
    from pipelines.retrieval.retriever import HybridRetriever
    from providers.client import NeuroFlowClient

logger = logging.getLogger("neuroflow-query-api")

router = APIRouter(prefix="/query", tags=["Query & RAG Generation"])

# In-memory registry for pending streaming runs
_STREAM_SESSIONS: Dict[str, Dict[str, Any]] = {}


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question / search prompt")
    pipeline_id: Optional[uuid.UUID] = Field(default=None, description="Target pipeline ID")
    stream: bool = Field(default=False, description="Whether to stream response via SSE")
    max_tokens: Optional[int] = Field(default=4000, description="Token budget for context assembly")
    enable_cot: bool = Field(default=False, description="Enable chain-of-thought reasoning")
    use_hyde: bool = Field(default=False, description="Enable HyDE retrieval expansion")


class QueryResponse(BaseModel):
    run_id: str
    query: str
    generation: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    latency_ms: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    status: str = "complete"
    stream_url: Optional[str] = None


@router.post("", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_pipeline(request: QueryRequest):
    """
    Execute end-to-end RAG query:
    - If stream=false: executes retrieval, context assembly, generation, and returns complete JSON response.
    - If stream=true: initializes run_id, registers streaming session, and returns stream_url.
    """
    run_id = uuid.uuid4()
    run_id_str = str(run_id)

    if request.stream:
        # Register session for GET /query/{run_id}/stream
        _STREAM_SESSIONS[run_id_str] = {
            "run_id": run_id,
            "query": request.query,
            "pipeline_id": request.pipeline_id,
            "max_tokens": request.max_tokens or 4000,
            "enable_cot": request.enable_cot,
            "use_hyde": request.use_hyde,
            "created_at": time.time(),
        }
        return QueryResponse(
            run_id=run_id_str,
            query=request.query,
            status="started",
            stream_url=f"/query/{run_id_str}/stream",
        )

    # Synchronous Execution
    start_time = time.perf_counter()
    pool = None
    try:
        pool = get_pool()
    except Exception:
        pool = None

    client = NeuroFlowClient.get_instance()
    retriever = HybridRetriever(pool=pool, client=client)
    generator = RAGGenerator(client=client, pool=pool)

    # 1. Retrieve & Assemble
    query_proc = await retriever.query_processor.process(request.query, use_hyde=request.use_hyde)
    retrieval_results = await retriever.retrieve(request.query, k=20, use_hyde=request.use_hyde)
    assembled = retriever.context_assembler.assemble(retrieval_results, max_tokens=request.max_tokens)

    # 2. Generate
    gen_result = await generator.generate(
        query=request.query,
        context=assembled["context"],
        sources=assembled["sources"],
        run_id=run_id,
        pipeline_id=request.pipeline_id,
        query_type=query_proc.query_type,
        enable_cot=request.enable_cot,
    )

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    return QueryResponse(
        run_id=run_id_str,
        query=request.query,
        generation=gen_result["generation"],
        citations=gen_result["citations"],
        sources=assembled["sources"],
        latency_ms=latency_ms,
        input_tokens=gen_result["input_tokens"],
        output_tokens=gen_result["output_tokens"],
        status="complete",
    )


async def _sse_event_generator(session_data: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Asynchronous event generator yielding SSE messages with 15s keepalive support:
    - data: {"type": "retrieval_start"}
    - data: {"type": "retrieval_complete", "chunk_count": N, "sources": [...]}
    - data: {"type": "token", "delta": "..."}
    - data: {"type": "done", "run_id": "...", "citations": [...]}
    """
    run_id: uuid.UUID = session_data["run_id"]
    query: str = session_data["query"]
    pipeline_id = session_data.get("pipeline_id")
    max_tokens = session_data.get("max_tokens", 4000)
    enable_cot = session_data.get("enable_cot", False)
    use_hyde = session_data.get("use_hyde", False)

    pool = None
    try:
        pool = get_pool()
    except Exception:
        pool = None

    client = NeuroFlowClient.get_instance()
    retriever = HybridRetriever(pool=pool, client=client)
    generator = RAGGenerator(client=client, pool=pool)

    # Event 1: Retrieval Start
    yield {"data": json.dumps({"type": "retrieval_start"})}

    # Step 1: Retrieval & Context Assembly
    query_proc = await retriever.query_processor.process(query, use_hyde=use_hyde)
    retrieval_results = await retriever.retrieve(query, k=20, use_hyde=use_hyde)
    assembled = retriever.context_assembler.assemble(retrieval_results, max_tokens=max_tokens)

    source_names = list(dict.fromkeys(s.get("filename", "") for s in assembled["sources"]))

    # Event 2: Retrieval Complete
    yield {
        "data": json.dumps(
            {
                "type": "retrieval_complete",
                "chunk_count": len(assembled["chunks_used"]),
                "sources": source_names,
            }
        )
    }

    # Step 2: Stream Tokens from Generator
    stream_iter = generator.generate_stream(
        query=query,
        context=assembled["context"],
        sources=assembled["sources"],
        run_id=run_id,
        pipeline_id=pipeline_id,
        query_type=query_proc.query_type,
        enable_cot=enable_cot,
    )

    async for event in stream_iter:
        yield {"data": json.dumps(event)}


@router.get("/{run_id}/stream")
async def stream_query_sse(run_id: str, request: Request):
    """
    SSE streaming endpoint for real-time progressive token delivery with 15s keepalive.
    """
    session = _STREAM_SESSIONS.get(run_id)
    if not session:
        # Create ad-hoc session if not pre-registered
        session = {
            "run_id": uuid.UUID(run_id) if len(run_id) == 36 else uuid.uuid4(),
            "query": request.query_params.get("query", "Summarize retrieved context"),
            "max_tokens": 4000,
            "enable_cot": False,
            "use_hyde": False,
        }

    return EventSourceResponse(
        _sse_event_generator(session),
        ping=15,  # 15s keepalive ping
    )
