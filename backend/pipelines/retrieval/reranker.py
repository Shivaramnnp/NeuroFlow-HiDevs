from __future__ import annotations

import asyncio
import logging
import re
from typing import List, Optional

from .fusion import RetrievalResult
try:
    from backend.providers.base import ChatMessage
    from backend.providers.client import NeuroFlowClient
    from backend.providers.router import RoutingCriteria
except ImportError:
    from providers.base import ChatMessage
    from providers.client import NeuroFlowClient
    from providers.router import RoutingCriteria

logger = logging.getLogger("neuroflow-reranker")


class CrossEncoderReranker:
    """
    Cross-Encoder Reranker for fused retrieval candidates:
    - Takes top candidates (e.g. top-40 from RRF)
    - Scores (query, chunk) pairs jointly for high-precision semantic matching
    - Supports API-based parallel LLM evaluation and optional local cross-encoder model
    """

    def __init__(
        self,
        client: Optional[NeuroFlowClient] = None,
        use_local_model: bool = False,
        local_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.client = client
        self.use_local_model = use_local_model
        self.local_model_name = local_model_name
        self._local_model = None

        if self.use_local_model:
            try:
                from sentence_transformers import CrossEncoder
                self._local_model = CrossEncoder(self.local_model_name)
                logger.info(f"Loaded local CrossEncoder model '{self.local_model_name}'")
            except Exception as err:
                logger.warning(f"Could not load local CrossEncoder: {err}. Falling back to API reranker.")

    def _get_client(self) -> Optional[NeuroFlowClient]:
        if self.client is not None:
            return self.client
        try:
            return NeuroFlowClient.get_instance()
        except Exception:
            return None

    async def _score_single_pair_api(
        self,
        client: NeuroFlowClient,
        query: str,
        item: RetrievalResult,
    ) -> float:
        """Score single (query, passage) pair on 0-10 scale via LLM."""
        prompt = (
            f"Rate the relevance of this passage to the query on a scale of 0-10.\n"
            f"Query: {query}\n"
            f"Passage: {item.content}\n"
            f"Return only the number."
        )
        try:
            messages = [ChatMessage(role="user", content=prompt)]
            criteria = RoutingCriteria(task_type="classification")
            res = await client.chat(messages, criteria=criteria, max_tokens=10)
            # Parse number from response
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)", res.content)
            if match:
                val = float(match.group(1))
                return min(10.0, max(0.0, val))
        except Exception as err:
            logger.warning(f"Reranking score error for chunk {item.chunk_id}: {err}")

        # Fallback to normalized RRF score
        return float(item.score)

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_n: int = 40,
    ) -> List[RetrievalResult]:
        """
        Rerank top-N fused candidates using joint cross-encoder scoring.
        """
        if not candidates:
            return []

        to_rerank = candidates[:top_n]
        remaining = candidates[top_n:]

        # 1. Local CrossEncoder if available
        if self._local_model is not None:
            try:
                pairs = [[query, item.content] for item in to_rerank]
                scores = await asyncio.to_thread(self._local_model.predict, pairs)
                reranked = []
                for item, score in zip(to_rerank, scores):
                    meta = dict(item.metadata)
                    meta["cross_encoder_score"] = float(score)
                    reranked.append(
                        RetrievalResult(
                            chunk_id=item.chunk_id,
                            document_id=item.document_id,
                            content=item.content,
                            score=float(score),
                            metadata=meta,
                        )
                    )
                reranked.sort(key=lambda x: x.score, reverse=True)
                return reranked + remaining
            except Exception as err:
                logger.warning(f"Local CrossEncoder inference failed: {err}")

        # 2. Parallel API-based Cross-Encoder
        client = self._get_client()
        if client is not None:
            tasks = [
                self._score_single_pair_api(client, query, item)
                for item in to_rerank
            ]
            scores = await asyncio.gather(*tasks)

            reranked = []
            for item, score in zip(to_rerank, scores):
                meta = dict(item.metadata)
                meta["cross_encoder_score"] = score
                reranked.append(
                    RetrievalResult(
                        chunk_id=item.chunk_id,
                        document_id=item.document_id,
                        content=item.content,
                        score=score,
                        metadata=meta,
                    )
                )
            reranked.sort(key=lambda x: x.score, reverse=True)
            return reranked + remaining

        # Default fallback if no client or model
        return candidates
