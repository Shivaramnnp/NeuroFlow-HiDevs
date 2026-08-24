from __future__ import annotations

import asyncio
import json
import logging
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
        # 1. Query Analysis
        processed = await self.query_processor.process(query, use_hyde=use_hyde)

        # Dense queries: original + expansions (+ hyde doc if requested)
        dense_queries = [processed.original_query] + processed.expanded_queries
        if processed.hypothetical_document:
            dense_queries.append(processed.hypothetical_document)

        # 2. Parallel Multi-Strategy Retrieval
        dense_task = self._dense_retrieval(dense_queries, k=k)
        sparse_task = self._sparse_retrieval(processed.original_query, k=k)
        meta_task = self._metadata_retrieval(processed.original_query, filters=processed.metadata_filters, k=k)

        dense_res, sparse_res, meta_res = await asyncio.gather(
            dense_task, sparse_task, meta_task
        )

        # 3. Reciprocal Rank Fusion
        fused = self._fuse([dense_res, sparse_res, meta_res])

        # 4. Cross-Encoder Reranking
        if use_reranker and fused:
            reranked = await self.reranker.rerank(processed.original_query, fused, top_n=min(40, len(fused)))
            return reranked[:k]

        return fused[:k]

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
