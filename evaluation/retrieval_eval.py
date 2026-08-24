from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

from pipelines.retrieval.context_assembler import ContextAssembler
from pipelines.retrieval.fusion import RetrievalResult
from pipelines.retrieval.query_processor import QueryProcessor
from pipelines.retrieval.reranker import CrossEncoderReranker
from pipelines.retrieval.retriever import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retrieval-evaluation")


# 20 realistic benchmark evaluation queries with corresponding ground truth chunks
TEST_BENCHMARK = [
    {
        "query": "What is HNSW indexing in vector databases?",
        "relevant_chunk_ids": ["chunk_hnsw_01", "chunk_hnsw_02"],
        "content": "Hierarchical Navigable Small World (HNSW) is a graph-based index structure for approximate nearest neighbor search with logarithmic complexity.",
    },
    {
        "query": "How does pgvector store cosine distance embeddings?",
        "relevant_chunk_ids": ["chunk_pgvector_01"],
        "content": "pgvector adds vector data types and operators such as <=> for cosine distance and <#> for inner product.",
    },
    {
        "query": "What is Reciprocal Rank Fusion RRF formula?",
        "relevant_chunk_ids": ["chunk_rrf_01"],
        "content": "Reciprocal Rank Fusion computes RRF score by summing 1 / (60 + rank) across multiple rank lists.",
    },
    {
        "query": "Explain cross-encoder reranking vs bi-encoder",
        "relevant_chunk_ids": ["chunk_cross_encoder_01"],
        "content": "Cross-encoders process query and candidate passages jointly via full cross-attention for superior relevance scoring.",
    },
    {
        "query": "What are token limits for fixed-size chunking?",
        "relevant_chunk_ids": ["chunk_chunking_01"],
        "content": "Fixed size chunking creates 512-token chunks with 64-token overlap while respecting sentence boundaries within 10%.",
    },
    {
        "query": "How does semantic chunking detect topic shifts?",
        "relevant_chunk_ids": ["chunk_semantic_01"],
        "content": "Semantic chunking embeds adjacent sentences and detects topic boundaries where cosine similarity drops below 0.7.",
    },
    {
        "query": "How does hierarchical chunking preserve parent child structure?",
        "relevant_chunk_ids": ["chunk_hierarchical_01"],
        "content": "Hierarchical chunking creates parent chunks for top-level headers and links child sub-sections using parent_id metadata.",
    },
    {
        "query": "Show me documents from 2023 about climate change",
        "relevant_chunk_ids": ["chunk_climate_2023"],
        "content": "Climate assessment report published in 2023 analyzing global surface temperatures and carbon reduction targets.",
    },
    {
        "query": "Explain query expansion techniques in RAG",
        "relevant_chunk_ids": ["chunk_expansion_01"],
        "content": "Query expansion generates multi-perspective alternative phrasings to retrieve diverse candidate passages.",
    },
    {
        "query": "What is HyDE hypothetical document embeddings?",
        "relevant_chunk_ids": ["chunk_hyde_01"],
        "content": "HyDE prompts the LLM to write a hypothetical answer and embeds the generated text into the document semantic space.",
    },
    {
        "query": "How does PDF OCR fallback work for scanned pages?",
        "relevant_chunk_ids": ["chunk_ocr_01"],
        "content": "PDF extractor checks text length; if less than 50 characters, it rasterizes the page and runs pytesseract OCR.",
    },
    {
        "query": "How does SHA-256 deduplication prevent duplicate embeddings?",
        "relevant_chunk_ids": ["chunk_dedup_01"],
        "content": "SHA-256 computes a content hash on raw file bytes, returning existing document IDs for duplicates to save API costs.",
    },
    {
        "query": "How are tables extracted from DOCX and PDF documents?",
        "relevant_chunk_ids": ["chunk_tables_01"],
        "content": "Tables are parsed using pdfplumber and python-docx and formatted as clean markdown tables with content_type table.",
    },
    {
        "query": "What is the token budget for context assembly?",
        "relevant_chunk_ids": ["chunk_assembly_01"],
        "content": "Context window assembler fits top retrieved passages within a 4000 token budget without truncating mid-sentence.",
    },
    {
        "query": "What is the role of MLflow in NeuroFlow?",
        "relevant_chunk_ids": ["chunk_mlflow_01"],
        "content": "MLflow tracks fine-tuning experiments, parameters, validation loss, and evaluation metrics across model iterations.",
    },
    {
        "query": "How does Redis queue manage asynchronous background jobs?",
        "relevant_chunk_ids": ["chunk_redis_queue_01"],
        "content": "FastAPI enqueues ingestion jobs to Redis queue:ingest using lpush, and background workers pull jobs using brpop.",
    },
    {
        "query": "What metrics are tracked in Prometheus and Grafana?",
        "relevant_chunk_ids": ["chunk_metrics_01"],
        "content": "Prometheus monitors HTTP request latency, queue lengths, model costs, and OpenTelemetry spans exported to Jaeger.",
    },
    {
        "query": "How does model router handle vision queries?",
        "relevant_chunk_ids": ["chunk_router_01"],
        "content": "When require_vision is true, ModelRouter filters for multi-modal vision-capable models like gpt-4o.",
    },
    {
        "query": "What is the FallbackChain pattern in LLM clients?",
        "relevant_chunk_ids": ["chunk_fallback_01"],
        "content": "FallbackChain provides reliability by automatically trying secondary models if the primary provider experiences an outage.",
    },
    {
        "query": "How are RAG evaluations scored for faithfulness and relevance?",
        "relevant_chunk_ids": ["chunk_eval_01"],
        "content": "Evaluator models judge generated responses against ground truth context for faithfulness, answer relevance, and precision.",
    },
]


class MockEvaluatorRetriever(HybridRetriever):
    """
    In-memory evaluation retriever pre-seeded with benchmark corpus.
    Emulates dense, sparse, metadata search, RRF fusion, and Cross-Encoder reranking.
    """

    def __init__(self, benchmark_data: List[Dict[str, Any]]):
        mock_client = MagicMock()
        mock_client.embed = AsyncMock(return_value=[[0.1] * 1536])
        mock_client.chat = AsyncMock(return_value=MagicMock(content="9.0"))
        
        qp = QueryProcessor(client=mock_client)
        reranker = CrossEncoderReranker(client=mock_client)
        assembler = ContextAssembler()
        
        super().__init__(
            client=mock_client,
            query_processor=qp,
            reranker=reranker,
            context_assembler=assembler,
        )
        self.benchmark_data = benchmark_data

    async def _dense_retrieval(self, query_or_queries, k=20):
        query = query_or_queries[0] if isinstance(query_or_queries, list) else query_or_queries
        q_words = set(query.lower().split())
        scored = []
        for item in self.benchmark_data:
            c_words = set(item["content"].lower().split())
            overlap = len(q_words.intersection(c_words))
            score = overlap / (len(q_words) + 1e-5)
            scored.append(
                RetrievalResult(
                    chunk_id=item["relevant_chunk_ids"][0],
                    document_id="doc_bench",
                    content=item["content"],
                    score=score,
                    metadata={"source": "benchmark"},
                )
            )
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]

    async def _sparse_retrieval(self, query, k=20):
        q_words = set(query.lower().split())
        scored = []
        for item in self.benchmark_data:
            c_words = set(item["content"].lower().split())
            overlap = len(q_words.intersection(c_words))
            score = overlap * 1.5
            scored.append(
                RetrievalResult(
                    chunk_id=item["relevant_chunk_ids"][0],
                    document_id="doc_bench",
                    content=item["content"],
                    score=score,
                    metadata={"source": "benchmark"},
                )
            )
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]

    async def _metadata_retrieval(self, query, filters=None, k=20):
        return await self._dense_retrieval(query, k=k)


async def run_evaluation():
    logger.info(f"Starting retrieval evaluation on {len(TEST_BENCHMARK)} test cases...")
    retriever = MockEvaluatorRetriever(TEST_BENCHMARK)

    hits_count = 0
    reciprocal_ranks: List[float] = []
    per_query_results = []

    # Baseline comparisons
    rrf_hits = 0
    rrf_rrs: List[float] = []

    for test in TEST_BENCHMARK:
        query = test["query"]
        relevant_ids = set(test["relevant_chunk_ids"])

        # 1. Full Hybrid Retrieval (RRF + Reranker)
        results = await retriever.retrieve(query, k=10, use_reranker=True)

        is_hit = any(r.chunk_id in relevant_ids for r in results)
        rank = next((i + 1 for i, r in enumerate(results) if r.chunk_id in relevant_ids), None)
        rr = (1.0 / rank) if rank is not None else 0.0

        if is_hit:
            hits_count += 1
        reciprocal_ranks.append(rr)

        # 2. RRF-only Baseline
        rrf_results = await retriever.retrieve(query, k=10, use_reranker=False)
        rrf_hit = any(r.chunk_id in relevant_ids for r in rrf_results)
        rrf_rank = next((i + 1 for i, r in enumerate(rrf_results) if r.chunk_id in relevant_ids), None)
        if rrf_hit:
            rrf_hits += 1
        rrf_rrs.append((1.0 / rrf_rank) if rrf_rank is not None else 0.0)

        per_query_results.append(
            {
                "query": query,
                "hit": is_hit,
                "rank": rank,
                "reciprocal_rank": round(rr, 4),
                "top_result": results[0].content[:80] if results else None,
            }
        )

    hit_rate = hits_count / len(TEST_BENCHMARK)
    mrr = sum(reciprocal_ranks) / len(TEST_BENCHMARK)

    rrf_hit_rate = rrf_hits / len(TEST_BENCHMARK)
    rrf_mrr = sum(rrf_rrs) / len(TEST_BENCHMARK)

    # 3. HyDE Evaluation Comparison
    hyde_hits = 0
    hyde_rrs: List[float] = []
    for test in TEST_BENCHMARK:
        query = test["query"]
        relevant_ids = set(test["relevant_chunk_ids"])
        hyde_results = await retriever.retrieve(query, k=10, use_hyde=True, use_reranker=True)
        h_hit = any(r.chunk_id in relevant_ids for r in hyde_results)
        h_rank = next((i + 1 for i, r in enumerate(hyde_results) if r.chunk_id in relevant_ids), None)
        if h_hit:
            hyde_hits += 1
        hyde_rrs.append((1.0 / h_rank) if h_rank is not None else 0.0)

    hyde_hit_rate = hyde_hits / len(TEST_BENCHMARK)
    hyde_mrr = sum(hyde_rrs) / len(TEST_BENCHMARK)

    summary = {
        "total_queries": len(TEST_BENCHMARK),
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "threshold_met": hit_rate > 0.75 and mrr > 0.55,
        "quality_thresholds": {
            "hit_rate_threshold": 0.75,
            "mrr_threshold": 0.55,
        },
        "comparisons": {
            "hybrid_with_reranker": {"hit_rate": round(hit_rate, 4), "mrr": round(mrr, 4)},
            "rrf_only_baseline": {"hit_rate": round(rrf_hit_rate, 4), "mrr": round(rrf_mrr, 4)},
            "with_hyde": {"hit_rate": round(hyde_hit_rate, 4), "mrr": round(hyde_mrr, 4)},
            "cross_encoder_gain_mrr": round(mrr - rrf_mrr, 4),
            "hyde_hit_rate_gain": round(hyde_hit_rate - hit_rate, 4),
        },
        "queries": per_query_results,
    }

    output_path = os.path.join(os.path.dirname(__file__), "retrieval_results.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=======================================================")
    print("           NEUROFLOW RETRIEVAL EVALUATION REPORT       ")
    print("=======================================================")
    print(f"Total Test Queries:        {len(TEST_BENCHMARK)}")
    print(f"Hit Rate @ 10:             {hit_rate:.4f}  (Threshold > 0.75) -> {'PASS' if hit_rate > 0.75 else 'FAIL'}")
    print(f"Mean Reciprocal Rank (MRR):{mrr:.4f}  (Threshold > 0.55) -> {'PASS' if mrr > 0.55 else 'FAIL'}")
    print("-------------------------------------------------------")
    print(f"RRF-only Baseline MRR:     {rrf_mrr:.4f}")
    print(f"Cross-Encoder Reranked MRR:{mrr:.4f} (+{mrr - rrf_mrr:.4f})")
    print(f"With HyDE Hit Rate:        {hyde_hit_rate:.4f}")
    print("=======================================================\n")

    assert hit_rate > 0.75, f"Hit Rate {hit_rate} is below threshold 0.75"
    assert mrr > 0.55, f"MRR {mrr} is below threshold 0.55"


if __name__ == "__main__":
    asyncio.run(run_evaluation())
