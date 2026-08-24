from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("neuroflow-fusion")


@dataclass
class RetrievalResult:
    """Represents a retrieved document chunk with relevance scoring and metadata."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def reciprocal_rank_fusion(
    result_lists: List[List[RetrievalResult]],
    k: int = 60,
) -> List[RetrievalResult]:
    """
    Reciprocal Rank Fusion (RRF) algorithm:
    RRF_score(chunk) = sum_{list in result_lists} 1 / (k + rank(chunk, list))
    where rank is 1-indexed.
    Chunks appearing across multiple retrievers receive boosted scores.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, RetrievalResult] = {}
    appearance_counts: Dict[str, int] = {}

    for r_list in result_lists:
        for rank, item in enumerate(r_list, start=1):
            cid = str(item.chunk_id)
            score_contribution = 1.0 / (k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contribution
            appearance_counts[cid] = appearance_counts.get(cid, 0) + 1

            if cid not in chunk_map:
                chunk_map[cid] = item

    # Build fused results list sorted by RRF score descending
    fused_results: List[RetrievalResult] = []
    for cid, fused_score in rrf_scores.items():
        original_item = chunk_map[cid]
        merged_meta = dict(original_item.metadata)
        merged_meta["rrf_appearances"] = appearance_counts[cid]
        merged_meta["original_retrieval_score"] = original_item.score

        fused_results.append(
            RetrievalResult(
                chunk_id=cid,
                document_id=original_item.document_id,
                content=original_item.content,
                score=fused_score,
                metadata=merged_meta,
            )
        )

    fused_results.sort(key=lambda x: x.score, reverse=True)
    return fused_results
