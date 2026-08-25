from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
# pyrefly: ignore [missing-import]
import redis.asyncio as aioredis
from sse_starlette.sse import EventSourceResponse

try:
    from backend.config import settings
    from backend.db.pool import get_pool
except ImportError:
    from config import settings
    from db.pool import get_pool

logger = logging.getLogger("neuroflow-evaluations-api")

router = APIRouter(prefix="/evaluations", tags=["Real-time Evaluations Feed"])


class EvaluationItemResponse(BaseModel):
    eval_id: str
    run_id: str
    pipeline_name: Optional[str] = "default"
    query: str
    answer: Optional[str] = ""
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    overall_score: float
    judge_model: str
    user_rating: Optional[int] = None
    evaluated_at: Optional[str] = None


@router.get("", response_model=List[EvaluationItemResponse])
async def list_evaluations(
    pipeline_id: Optional[str] = None,
    min_overall_score: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    min_faithfulness: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    limit: int = Query(default=50, ge=1, le=200),
):
    """
    Get historical list of evaluation records with optional quality/metric filters.
    """
    pool = None
    try:
        pool = get_pool()
    except Exception:
        pool = None

    if pool is None:
        return [
            EvaluationItemResponse(
                eval_id=str(uuid.uuid4()),
                run_id=str(uuid.uuid4()),
                pipeline_name="legal-research-v2",
                query="What is the liability limitation in standard MSAs?",
                answer="Standard MSAs typically limit liability to fees paid within the prior 12 months [Source 1].",
                faithfulness=0.95,
                answer_relevance=0.92,
                context_precision=0.88,
                context_recall=0.90,
                overall_score=0.917,
                judge_model="gpt-4o",
                user_rating=5,
                evaluated_at="2026-08-25T14:30:00Z",
            )
        ]

    query_sql = """
        SELECT e.id as eval_id, e.run_id, e.faithfulness, e.answer_relevance,
               e.context_precision, e.context_recall, e.overall_score, e.judge_model,
               e.user_rating, e.evaluated_at,
               pr.query, pr.generation as answer, p.name as pipeline_name
        FROM evaluations e
        JOIN pipeline_runs pr ON pr.id = e.run_id
        LEFT JOIN pipelines p ON p.id = pr.pipeline_id
        WHERE 1=1
    """
    params = []
    idx = 1

    if pipeline_id:
        try:
            p_uuid = uuid.UUID(pipeline_id)
            query_sql += f" AND pr.pipeline_id = ${idx}"
            params.append(p_uuid)
            idx += 1
        except ValueError:
            pass

    if min_overall_score is not None:
        query_sql += f" AND e.overall_score >= ${idx}"
        params.append(min_overall_score)
        idx += 1

    if min_faithfulness is not None:
        query_sql += f" AND e.faithfulness >= ${idx}"
        params.append(min_faithfulness)
        idx += 1

    query_sql += f" ORDER BY e.evaluated_at DESC LIMIT ${idx};"
    params.append(limit)

    async with pool.acquire() as conn:
        rows = await conn.fetch(query_sql, *params)
        results = []
        for r in rows:
            results.append(
                EvaluationItemResponse(
                    eval_id=str(r["eval_id"]),
                    run_id=str(r["run_id"]),
                    pipeline_name=r["pipeline_name"] or "default",
                    query=r["query"] or "",
                    answer=r["answer"] or "",
                    faithfulness=float(r["faithfulness"] or 0.0),
                    answer_relevance=float(r["answer_relevance"] or 0.0),
                    context_precision=float(r["context_precision"] or 0.0),
                    context_recall=float(r["context_recall"] or 0.0),
                    overall_score=float(r["overall_score"] or 0.0),
                    judge_model=r["judge_model"] or "gpt-4o",
                    user_rating=r["user_rating"],
                    evaluated_at=str(r["evaluated_at"]) if r["evaluated_at"] else None,
                )
            )
        return results


@router.get("/stream")
async def stream_evaluations(request: Request):
    """
    Real-time SSE event stream for newly completed evaluations.
    Subscribes to Redis channel 'evaluations:new' and broadcasts JSON events.
    """
    async def event_generator() -> AsyncGenerator[Dict[str, str], None]:
        r_client = None
        pubsub = None
        try:
            r_client = aioredis.from_url(settings.redis_url, socket_timeout=2.0)
            pubsub = r_client.pubsub()
            await pubsub.subscribe("evaluations:new")
            logger.info("SSE client connected to /evaluations/stream")

            # Yield initial connection heartbeat
            yield {
                "event": "connected",
                "data": json.dumps({"message": "Connected to evaluations feed stream", "timestamp": time.time()}),
            }

            while True:
                if await request.is_disconnected():
                    logger.info("SSE client disconnected from /evaluations/stream")
                    break

                try:
                    msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if msg and msg.get("data"):
                        payload = msg["data"]
                        if isinstance(payload, bytes):
                            payload = payload.decode("utf-8")
                        yield {
                            "event": "evaluation",
                            "data": payload,
                        }
                    else:
                        # Send periodic keepalive ping
                        yield {"event": "ping", "data": json.dumps({"ping": int(time.time())})}
                        await asyncio.sleep(2.0)
                except Exception as loop_err:
                    logger.debug(f"Pubsub stream tick: {loop_err}")
                    await asyncio.sleep(1.0)
        except Exception as err:
            logger.warning(f"Error in evaluations SSE stream: {err}")
            # Fallback simulator heartbeat for disconnected environments
            while not await request.is_disconnected():
                yield {"event": "ping", "data": json.dumps({"ping": int(time.time())})}
                await asyncio.sleep(5.0)
        finally:
            if pubsub is not None:
                await pubsub.unsubscribe("evaluations:new")
                await pubsub.aclose()
            if r_client is not None:
                await r_client.aclose()

    return EventSourceResponse(event_generator(), ping=15)
