from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

import asyncpg

from .context_assembler import ContextAssembler
from .fusion import RetrievalResult, reciprocal_rank_fusion
from .query_processor import ProcessedQuery, QueryProcessor
from .reranker import CrossEncoderReranker

try:
    from backend.db.pool import get_pool
    from backend.providers.client import NeuroFlowClient
except ImportError:
    from db.pool import get_pool
    from providers.client import NeuroFlowClient

logger = logging.getLogger("neuroflow-retriever")


class HybridRetriever:
    """
    Production-grade Multi-Stage Hybrid Retrieval Pipeline:
    1. Query Processing: Query expansion, metadata filter extraction, query classification, HyDE.
    2. Parallel Retrieval via asyncio.gather:
       - Dense pgvector cosine search (HNSW index <=>)
       - Sparse full-text search (plainto_tsquery + ts_rank_cd)
       - Metadata JSONB containment search (metadata @> $1::jsonb)
    3. Reciprocal Rank Fusion (RRF, k=60) combining multi-channel scores.
    4. Cross-Encoder Reranking: Joint scoring on top-40 candidates.
    5. Context Assembly: Token budget enforcement with source citation headers.
    """

    def __init__(
        self,
        pool: Optional[asyncpg.Pool] = None,
        client: Optional[NeuroFlowClient] = None,
        query_processor: Optional[QueryProcessor] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        context_assembler: Optional[ContextAssembler] = None,
    ):
        self.pool = pool
        self.client = client or NeuroFlowClient.get_instance()
        self.query_processor = query_processor or QueryProcessor(client=self.client)
        self.reranker = reranker or CrossEncoderReranker(client=self.client)
        self.context_assembler = context_assembler or ContextAssembler()

    async def _get_db_pool(self) -> Optional[asyncpg.Pool]:
        if self.pool is not None:
            return self.pool
        try:
            self.pool = get_pool()
        except Exception:
            self.pool = None
        return self.pool

    async def _dense_retrieval(
        self,
        query_or_queries: Union[str, List[str]],
        k: int = 20,
    ) -> List[RetrievalResult]:
        """
        Dense vector search using pgvector cosine distance operator (<=>).
        Supports multi-query embedding union for expanded queries.
        """
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        if not queries:
            return []

        try:
            embeddings = await self.client.embed(queries)
        except Exception as err:
            logger.warning(f"Dense retrieval embedding generation failed: {err}")
            return []

        pool = await self._get_db_pool()
        if pool is None:
            return []

        all_results: Dict[str, RetrievalResult] = {}

        async with pool.acquire() as conn:
            for emb in embeddings:
                try:
                    rows = await conn.fetch(
                        """
                        SELECT id, document_id, content, metadata, 1 - (embedding <=> $1::vector) AS score
                        FROM chunks
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> $1::vector
                        LIMIT $2;
                        """,
                        str(emb),
                        k,
                    )
                    for row in rows:
                        cid = str(row["id"])
                        meta = row["metadata"]
                        if isinstance(meta, str):
                            meta = json.loads(meta)
                        score = float(row["score"] or 0.0)
                        if cid not in all_results or score > all_results[cid].score:
                            all_results[cid] = RetrievalResult(
                                chunk_id=cid,
                                document_id=str(row["document_id"]),
                                content=row["content"],
                                score=score,
                                metadata=meta or {},
                            )
                except Exception as err:
                    logger.warning(f"Dense vector query execution failed: {err}")

        results = list(all_results.values())
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    async def _sparse_retrieval(
        self,
        query: str,
        k: int = 20,
    ) -> List[RetrievalResult]:
        """
        Sparse full-text retrieval using PostgreSQL plainto_tsquery and ts_rank_cd cover density.
        """
        pool = await self._get_db_pool()
        if pool is None:
            return []

        results: List[RetrievalResult] = []
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, document_id, content, metadata,
                           ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', $1)) AS score
                    FROM chunks
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
                    ORDER BY score DESC
                    LIMIT $2;
                    """,
                    query,
                    k,
                )
                for row in rows:
                    meta = row["metadata"]
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    results.append(
                        RetrievalResult(
                            chunk_id=str(row["id"]),
                            document_id=str(row["document_id"]),
                            content=row["content"],
                            score=float(row["score"] or 0.0),
                            metadata=meta or {},
                        )
                    )
        except Exception as err:
            logger.warning(f"Sparse full-text query execution failed: {err}")

        return results

    async def _metadata_retrieval(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 20,
    ) -> List[RetrievalResult]:
        """
        Filtered retrieval using JSONB metadata containment (@>) and vector similarity.
        """
        if not filters:
            return []

        pool = await self._get_db_pool()
        if pool is None:
            return []

        try:
            embeddings = await self.client.embed([query])
            emb_str = str(embeddings[0]) if embeddings else None
        except Exception:
            emb_str = None

        results: List[RetrievalResult] = []
        filter_json = json.dumps(filters)

        try:
            async with pool.acquire() as conn:
                if emb_str:
                    rows = await conn.fetch(
                        """
                        SELECT id, document_id, content, metadata, 1 - (embedding <=> $2::vector) AS score
                        FROM chunks
                        WHERE metadata @> $1::jsonb
                        ORDER BY embedding <=> $2::vector
                        LIMIT $3;
                        """,
                        filter_json,
                        emb_str,
                        k,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT id, document_id, content, metadata, 1.0 AS score
                        FROM chunks
                        WHERE metadata @> $1::jsonb
                        LIMIT $2;
                        """,
                        filter_json,
                        k,
                    )

                for row in rows:
                    meta = row["metadata"]
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    results.append(
                        RetrievalResult(
                            chunk_id=str(row["id"]),
                            document_id=str(row["document_id"]),
                            content=row["content"],
                            score=float(row["score"] or 0.0),
                            metadata=meta or {},
                        )
                    )
        except Exception as err:
            logger.warning(f"Metadata filtered query execution failed: {err}")

        return results

    def _fuse(self, result_lists: List[List[RetrievalResult]]) -> List[RetrievalResult]:
        """Apply Reciprocal Rank Fusion on multi-channel results."""
        return reciprocal_rank_fusion(result_lists, k=60)

    async def retrieve(
        self,
        query: str,
        k: int = 20,
        use_hyde: bool = False,
        use_reranker: bool = True,
    ) -> List[RetrievalResult]:
        """
        Execute full parallel hybrid retrieval workflow:
        1. Process query (expansions, metadata filters, classification, HyDE)
        2. Parallel search across Dense, Sparse, and Metadata channels
        3. RRF Fusion
        4. Cross-Encoder Reranking
        """
        start_time = time.perf_counter()
        from opentelemetry import trace
        tracer = trace.get_tracer("neuroflow-retriever")

        with tracer.start_as_current_span("retrieval.pipeline") as span:
            span.set_attribute("query", query)
            span.set_attribute("k", k)

            # 1. Query Analysis
            processed = await self.query_processor.process(query, use_hyde=use_hyde)

            # Dense queries: original + expansions (+ hyde doc if requested)
            dense_queries = [processed.original_query] + processed.expanded_queries
            if processed.hypothetical_document:
                dense_queries.append(processed.hypothetical_document)

            # 2. Parallel Multi-Strategy Retrieval with Spans
            async def _run_dense():
                with tracer.start_as_current_span("retrieval.dense") as dspan:
                    dstart = time.perf_counter()
                    res = await self._dense_retrieval(dense_queries, k=k)
                    dspan.set_attribute("results_count", len(res))
                    try:
                        from backend.monitoring.metrics import retrieval_latency
                        retrieval_latency.labels(strategy="dense").observe(time.perf_counter() - dstart)
                    except Exception:
                        pass
                    return res

            async def _run_sparse():
                with tracer.start_as_current_span("retrieval.sparse") as sspan:
                    sstart = time.perf_counter()
                    res = await self._sparse_retrieval(processed.original_query, k=k)
                    sspan.set_attribute("results_count", len(res))
                    try:
                        from backend.monitoring.metrics import retrieval_latency
                        retrieval_latency.labels(strategy="sparse").observe(time.perf_counter() - sstart)
                    except Exception:
                        pass
                    return res

            async def _run_meta():
                with tracer.start_as_current_span("retrieval.metadata") as mspan:
                    res = await self._metadata_retrieval(processed.original_query, filters=processed.metadata_filters, k=k)
                    mspan.set_attribute("results_count", len(res))
                    return res

            dense_res, sparse_res, meta_res = await asyncio.gather(
                _run_dense(), _run_sparse(), _run_meta()
            )

            # 3. Reciprocal Rank Fusion with Span
            with tracer.start_as_current_span("retrieval.fusion") as fspan:
                fused = self._fuse([dense_res, sparse_res, meta_res])
                fspan.set_attribute("fused_count", len(fused))

            # 4. Cross-Encoder Reranking with Span
            final_res = fused
            if use_reranker and fused:
                with tracer.start_as_current_span("retrieval.rerank") as rspan:
                    rstart = time.perf_counter()
                    reranked = await self.reranker.rerank(processed.original_query, fused, top_n=min(40, len(fused)))
                    rspan.set_attribute("reranked_count", len(reranked))
                    try:
                        from backend.monitoring.metrics import retrieval_latency
                        retrieval_latency.labels(strategy="cross_encoder").observe(time.perf_counter() - rstart)
                    except Exception:
                        pass
                    final_res = reranked

            total_lat = time.perf_counter() - start_time
            span.set_attribute("total_latency_ms", int(total_lat * 1000))
            try:
                from backend.monitoring.metrics import retrieval_latency
                retrieval_latency.labels(strategy="hybrid").observe(total_lat)
            except Exception:
                pass

            return final_res[:k]

    async def retrieve_and_assemble(
        self,
        query: str,
        k: int = 20,
        token_budget: int = 4000,
        use_hyde: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute retrieval and assemble formatted context window.
        """
        results = await self.retrieve(query, k=k, use_hyde=use_hyde)
        assembled = self.context_assembler.assemble(results, max_tokens=token_budget)
        return assembled
