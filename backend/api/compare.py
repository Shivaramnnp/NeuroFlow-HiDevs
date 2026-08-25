from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

try:
    from backend.db.pool import get_pool
    from backend.evaluation.judge import EvaluationJudge
    from backend.pipelines.generation.generator import RAGGenerator
    from backend.pipelines.retrieval.context_assembler import ContextAssembler
    from backend.pipelines.retrieval.query_processor import QueryProcessor
    from backend.pipelines.retrieval.retriever import HybridRetriever
    from backend.providers.client import NeuroFlowClient
except ImportError:
    from db.pool import get_pool
    from evaluation.judge import EvaluationJudge
    from pipelines.generation.generator import RAGGenerator
    from pipelines.retrieval.context_assembler import ContextAssembler
    from pipelines.retrieval.query_processor import QueryProcessor
    from pipelines.retrieval.retriever import HybridRetriever
    from providers.client import NeuroFlowClient

logger = logging.getLogger("neuroflow-compare-api")

router = APIRouter(prefix="/pipelines", tags=["A/B Pipeline Comparison"])


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search prompt to evaluate across both pipelines")
    pipeline_a_id: str = Field(..., description="UUID of Pipeline A")
    pipeline_b_id: str = Field(..., description="UUID of Pipeline B")


class PipelineRunSummary(BaseModel):
    run_id: str
    pipeline_id: str
    pipeline_name: Optional[str] = None
    pipeline_version: int = 1
    generation: str
    retrieval_latency_ms: int
    total_latency_ms: int
    chunks_used: int
    eval_score: Optional[float] = None
    citations_count: int = 0


class CompareResponse(BaseModel):
    query: str
    pipeline_a: PipelineRunSummary
    pipeline_b: PipelineRunSummary
    winner: Optional[str] = None


async def _run_single_pipeline(
    pipeline_id_str: str,
    query: str,
    client: NeuroFlowClient,
    pool: Any,
) -> Dict[str, Any]:
    """Execute end-to-end RAG run for a specific pipeline config."""
    try:
        p_uuid = uuid.UUID(pipeline_id_str)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid pipeline UUID: {pipeline_id_str}")

    config_dict = {}
    p_name = f"pipeline_{pipeline_id_str[:8]}"
    p_version = 1

    if pool is not None:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT name, version, config FROM pipelines WHERE id = $1;",
                p_uuid,
            )
            if row:
                p_name = row["name"]
                p_version = row["version"]
                config_dict = json.loads(row["config"]) if isinstance(row["config"], str) else row["config"]

    # Extract pipeline parameters
    retrieval_cfg = config_dict.get("retrieval", {})
    generation_cfg = config_dict.get("generation", {})
    dense_k = retrieval_cfg.get("dense_k", 20)
    top_k = retrieval_cfg.get("top_k_after_rerank", 8)
    use_expansion = retrieval_cfg.get("query_expansion", True)
    max_tokens = generation_cfg.get("max_context_tokens", 4000)
    sys_variant = generation_cfg.get("system_prompt_variant", "precise")

    run_id = uuid.uuid4()
    total_start = time.perf_counter()

    retriever = HybridRetriever(pool=pool, client=client)
    generator = RAGGenerator(client=client, pool=pool)
    judge = EvaluationJudge(client=client, pool=pool)

    # 1. Retrieval
    ret_start = time.perf_counter()
    ret_results = await retriever.retrieve(query, k=dense_k, use_reranker=True)
    assembled = retriever.context_assembler.assemble(ret_results[:top_k], max_tokens=max_tokens)
    ret_latency_ms = int((time.perf_counter() - ret_start) * 1000)

    # 2. Generation
    gen_result = await generator.generate(
        query=query,
        context=assembled["context"],
        sources=assembled["sources"],
        run_id=run_id,
        pipeline_id=p_uuid,
        query_type=sys_variant,
    )

    total_latency_ms = int((time.perf_counter() - total_start) * 1000)

    # Record retrieval_latency_ms in pipeline_runs
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE pipeline_runs
                    SET pipeline_version = $2, retrieval_latency_ms = $3
                    WHERE id = $1;
                    """,
                    run_id,
                    p_version,
                    ret_latency_ms,
                )
        except Exception as err:
            logger.warning(f"Could not update retrieval latency: {err}")

    # 3. Judge evaluation
    eval_res = await judge.evaluate(
        query=query,
        answer=gen_result["generation"],
        context=assembled["context"],
        run_id=run_id,
    )

    return {
        "run_id": str(run_id),
        "pipeline_id": pipeline_id_str,
        "pipeline_name": p_name,
        "pipeline_version": p_version,
        "generation": gen_result["generation"],
        "retrieval_latency_ms": ret_latency_ms,
        "total_latency_ms": total_latency_ms,
        "chunks_used": len(assembled["chunks_used"]),
        "eval_score": eval_res.get("overall_score", 0.85),
        "citations_count": len(gen_result.get("citations", [])),
    }


@router.post("/compare", response_model=CompareResponse, status_code=status.HTTP_200_OK)
async def compare_pipelines(request: CompareRequest):
    """
    A/B Test two pipelines side-by-side simultaneously via asyncio.gather:
    Executes retrieval, generation, and automated evaluation on both pipelines concurrently.
    """
    client = NeuroFlowClient.get_instance()
    pool = None
    try:
        pool = get_pool()
    except Exception:
        pool = None

    # Run both pipelines simultaneously
    res_a, res_b = await asyncio.gather(
        _run_single_pipeline(request.pipeline_a_id, request.query, client, pool),
        _run_single_pipeline(request.pipeline_b_id, request.query, client, pool),
    )

    # Determine winner
    score_a = res_a.get("eval_score") or 0.0
    score_b = res_b.get("eval_score") or 0.0
    winner = None
    if abs(score_a - score_b) > 0.02:
        winner = res_a["pipeline_name"] if score_a > score_b else res_b["pipeline_name"]

    return CompareResponse(
        query=request.query,
        pipeline_a=PipelineRunSummary(**res_a),
        pipeline_b=PipelineRunSummary(**res_b),
        winner=winner,
    )
