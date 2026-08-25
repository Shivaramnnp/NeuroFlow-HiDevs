from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import asyncpg
# pyrefly: ignore [missing-import]
import redis.asyncio as redis

from .citations import Citation, CitationProcessor, strip_thinking
from .prompt_builder import PromptBuilder
try:
    from backend.config import settings
    from backend.db.pool import get_pool
    from backend.pipelines.retrieval.context_assembler import count_tokens
    from backend.providers.base import ChatMessage, GenerationResult
    from backend.providers.client import NeuroFlowClient
    from backend.providers.router import RoutingCriteria
except ImportError:
    from config import settings
    from db.pool import get_pool
    from pipelines.retrieval.context_assembler import count_tokens
    from providers.base import ChatMessage, GenerationResult
    from providers.client import NeuroFlowClient
    from providers.router import RoutingCriteria

logger = logging.getLogger("neuroflow-generator")


class RAGGenerator:
    """
    Orchestrates prompt assembly, token streaming, citation post-processing,
    database logging to pipeline_runs, and non-blocking asynchronous evaluation queuing.
    """

    def __init__(
        self,
        client: Optional[NeuroFlowClient] = None,
        pool: Optional[asyncpg.Pool] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        citation_processor: Optional[CitationProcessor] = None,
    ):
        self.client = client or NeuroFlowClient.get_instance()
        self.pool = pool
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.citation_processor = citation_processor or CitationProcessor()

    async def _get_db_pool(self) -> Optional[asyncpg.Pool]:
        if self.pool is not None:
            return self.pool
        try:
            self.pool = get_pool()
        except Exception:
            self.pool = None
        return self.pool

    async def _ensure_pipeline_record(self, pipeline_id: Optional[uuid.UUID], conn: asyncpg.Connection) -> uuid.UUID:
        """Ensure a valid pipeline row exists in pipelines table for foreign key integrity."""
        if pipeline_id is None:
            pipeline_id = uuid.UUID("00000000-0000-0000-0000-000000000001")

        await conn.execute(
            """
            INSERT INTO pipelines (id, name, config)
            VALUES ($1, 'default_rag_pipeline', '{"type": "rag"}'::jsonb)
            ON CONFLICT (id) DO NOTHING;
            """,
            pipeline_id,
        )
        return pipeline_id

    async def _log_run_start(
        self,
        run_id: uuid.UUID,
        pipeline_id: Optional[uuid.UUID],
        query: str,
        retrieved_chunk_ids: List[str],
    ) -> None:
        """Insert initial pipeline_runs record with status='running'."""
        pool = await self._get_db_pool()
        if pool is not None:
            try:
                chunk_uuids = [uuid.UUID(cid) for cid in retrieved_chunk_ids if cid]
            except Exception:
                chunk_uuids = []

            try:
                async with pool.acquire() as conn:
                    p_id = await self._ensure_pipeline_record(pipeline_id, conn)
                    await conn.execute(
                        """
                        INSERT INTO pipeline_runs (id, pipeline_id, query, retrieved_chunk_ids, status)
                        VALUES ($1, $2, $3, $4, 'running')
                        ON CONFLICT (id) DO UPDATE
                        SET query = EXCLUDED.query, retrieved_chunk_ids = EXCLUDED.retrieved_chunk_ids, status = 'running';
                        """,
                        run_id,
                        p_id,
                        query,
                        chunk_uuids,
                    )
            except Exception as err:
                logger.warning(f"Could not log pipeline_run start for {run_id}: {err}")

    async def _log_run_complete(
        self,
        run_id: uuid.UUID,
        generation: str,
        input_tokens: int,
        output_tokens: int,
        model_used: str,
        latency_ms: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update pipeline_runs record with final generation results and status='complete'."""
        pool = await self._get_db_pool()
        if pool is not None:
            try:
                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE pipeline_runs
                        SET generation = $2,
                            input_tokens = $3,
                            output_tokens = $4,
                            model_used = $5,
                            latency_ms = $6,
                            status = 'complete'
                        WHERE id = $1;
                        """,
                        run_id,
                        generation,
                        input_tokens,
                        output_tokens,
                        model_used,
                        latency_ms,
                    )
            except Exception as err:
                logger.warning(f"Could not update pipeline_run complete for {run_id}: {err}")

    def _enqueue_eval_job_async(self, run_id: uuid.UUID, query: str, generation: str, sources: List[Dict[str, Any]]) -> None:
        """Enqueue evaluation job in Redis queue:eval asynchronously without blocking."""
        async def _do_enqueue():
            try:
                redis_client = redis.from_url(settings.redis_url, socket_timeout=2.0)
                payload = json.dumps({
                    "run_id": str(run_id),
                    "query": query,
                    "generation": generation,
                    "sources": sources,
                    "timestamp": time.time(),
                })
                await redis_client.lpush("queue:eval", payload)
                await redis_client.aclose()
                logger.info(f"Enqueued evaluation job for run {run_id}")
            except Exception as err:
                logger.warning(f"Could not enqueue evaluation job for {run_id}: {err}")

        # Fire and forget task
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_do_enqueue())
        except Exception:
            pass

    async def generate_stream(
        self,
        query: str,
        context: str,
        sources: List[Dict[str, Any]],
        run_id: Optional[uuid.UUID] = None,
        pipeline_id: Optional[uuid.UUID] = None,
        query_type: str = "factual",
        enable_cot: bool = False,
        model: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream generation token by token:
        1. Logs run start to pipeline_runs
        2. Streams token events
        3. Parses citations and updates pipeline_runs
        4. Enqueues evaluation job asynchronously
        """
        active_run_id = run_id or uuid.uuid4()
        chunk_ids = [s.get("chunk_id", "") for s in sources if s.get("chunk_id")]
        
        # 1. Log run start
        await self._log_run_start(active_run_id, pipeline_id, query, chunk_ids)

        # 2. Build messages
        messages = self.prompt_builder.build_chat_messages(
            query=query,
            context=context,
            query_type=query_type,
            enable_cot=enable_cot,
        )

        input_tokens = sum(count_tokens(m.content if isinstance(m.content, str) else "") for m in messages)
        full_tokens: List[str] = []
        start_time = time.perf_counter()
        criteria = RoutingCriteria(task_type="rag_generation")

        # 3. Stream from provider
        if hasattr(self.client, "chat"):
            chat_res = self.client.chat(messages, criteria=criteria, model=model, stream=True)
            if asyncio.iscoroutine(chat_res):
                stream_gen = await chat_res
            else:
                stream_gen = chat_res
        elif hasattr(self.client, "stream"):
            stream_gen = self.client.stream(messages, criteria=criteria, model=model)
            if asyncio.iscoroutine(stream_gen):
                stream_gen = await stream_gen
        else:
            stream_gen = []

        async for token in stream_gen:
            full_tokens.append(token)
            yield {"type": "token", "delta": token}

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        raw_generation = "".join(full_tokens)

        # 4. Citation post-processing and reasoning extraction
        clean_generation, thinking = strip_thinking(raw_generation)
        output_tokens = count_tokens(raw_generation)
        citations = self.citation_processor.parse_citations(clean_generation, sources)
        model_used = model or "gpt-4o"

        # 5. Log run complete to DB
        await self._log_run_complete(
            run_id=active_run_id,
            generation=clean_generation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_used=model_used,
            latency_ms=latency_ms,
            metadata={"thinking": thinking} if thinking else None,
        )

        # 6. Enqueue evaluation asynchronously
        self._enqueue_eval_job_async(active_run_id, query, clean_generation, sources)

        # 7. Final done event
        yield {
            "type": "done",
            "run_id": str(active_run_id),
            "generation": clean_generation,
            "citations": [c.to_dict() for c in citations],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "thinking": thinking,
        }

    async def generate(
        self,
        query: str,
        context: str,
        sources: List[Dict[str, Any]],
        run_id: Optional[uuid.UUID] = None,
        pipeline_id: Optional[uuid.UUID] = None,
        query_type: str = "factual",
        enable_cot: bool = False,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous complete generation."""
        active_run_id = run_id or uuid.uuid4()
        chunk_ids = [s.get("chunk_id", "") for s in sources if s.get("chunk_id")]

        await self._log_run_start(active_run_id, pipeline_id, query, chunk_ids)

        messages = self.prompt_builder.build_chat_messages(
            query=query,
            context=context,
            query_type=query_type,
            enable_cot=enable_cot,
        )

        input_tokens = sum(count_tokens(m.content if isinstance(m.content, str) else "") for m in messages)
        start_time = time.perf_counter()
        criteria = RoutingCriteria(task_type="rag_generation")

        res = await self.client.chat(messages, criteria=criteria, model=model)
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        raw_generation = res.content
        clean_generation, thinking = strip_thinking(raw_generation)
        output_tokens = res.output_tokens or count_tokens(raw_generation)
        citations = self.citation_processor.parse_citations(clean_generation, sources)
        model_used = res.model or model or "gpt-4o"

        await self._log_run_complete(
            run_id=active_run_id,
            generation=clean_generation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_used=model_used,
            latency_ms=latency_ms,
            metadata={"thinking": thinking} if thinking else None,
        )

        self._enqueue_eval_job_async(active_run_id, query, clean_generation, sources)

        return {
            "run_id": str(active_run_id),
            "generation": clean_generation,
            "citations": [c.to_dict() for c in citations],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "model_used": model_used,
            "thinking": thinking,
        }
