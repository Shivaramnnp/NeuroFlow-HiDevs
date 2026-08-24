from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from pipelines.retrieval.query_processor import ProcessedQuery, QueryProcessor
from pipelines.retrieval.fusion import RetrievalResult, reciprocal_rank_fusion
from pipelines.retrieval.reranker import CrossEncoderReranker
from pipelines.retrieval.context_assembler import ContextAssembler
from pipelines.retrieval.retriever import HybridRetriever
from backend.providers.base import GenerationResult
from backend.providers.client import NeuroFlowClient


# --- 1. Query Processor Tests ---

@pytest.mark.asyncio
async def test_query_processor_expansion_and_filters():
    mock_client = MagicMock(spec=NeuroFlowClient)
    mock_client.chat = AsyncMock(
        return_value=GenerationResult(
            content="explain self-attention mechanism\ntransformer attention weights calculation",
            model="gpt-4o-mini",
            input_tokens=20,
            output_tokens=15,
            latency_ms=80.0,
            cost_usd=0.0001,
            finish_reason="stop",
        )
    )

    qp = QueryProcessor(client=mock_client)
    processed = await qp.process("Show me documents from 2023 about climate change")

    assert "explain self-attention mechanism" in processed.expanded_queries
    # Verify metadata extraction for year and topic
    assert processed.metadata_filters.get("year") == 2023
    assert processed.metadata_filters.get("topic") == "climate"


@pytest.mark.asyncio
async def test_query_processor_classification():
    qp = QueryProcessor()
    assert await qp.classify_query("Compare GPT-4 vs Claude 3.5 Sonnet") == "comparative"
    assert await qp.classify_query("How to install pgvector on PostgreSQL") == "procedural"
    assert await qp.classify_query("Analyze the impact of interest rates on tech stocks") == "analytical"


@pytest.mark.asyncio
async def test_query_processor_hyde():
    mock_client = MagicMock(spec=NeuroFlowClient)
    mock_client.chat = AsyncMock(
        return_value=GenerationResult(
            content="HNSW (Hierarchical Navigable Small World) constructs multi-layer graphs for fast nearest neighbor search.",
            model="gpt-4o-mini",
            input_tokens=20,
            output_tokens=25,
            latency_ms=90.0,
            cost_usd=0.0001,
            finish_reason="stop",
        )
    )
    qp = QueryProcessor(client=mock_client)
    processed = await qp.process("What is HNSW indexing?", use_hyde=True)

    assert processed.hypothetical_document is not None
    assert "HNSW" in processed.hypothetical_document


# --- 2. Reciprocal Rank Fusion Tests ---

def test_reciprocal_rank_fusion_boosting():
    # Chunk A appears in 2 lists (rank 1 and rank 2)
    # Chunk B appears in 1 list (rank 1)
    # Chunk C appears in 1 list (rank 2)
    list1 = [
        RetrievalResult(chunk_id="chunk_a", document_id="doc1", content="Text A", score=0.9),
        RetrievalResult(chunk_id="chunk_c", document_id="doc1", content="Text C", score=0.8),
    ]
    list2 = [
        RetrievalResult(chunk_id="chunk_b", document_id="doc2", content="Text B", score=0.95),
        RetrievalResult(chunk_id="chunk_a", document_id="doc1", content="Text A", score=0.85),
    ]

    fused = reciprocal_rank_fusion([list1, list2], k=60)

    # Chunk A should be #1 because it appeared in both lists: 1/61 + 1/62 = 0.0325
    # Chunk B score: 1/61 = 0.01639
    # Chunk C score: 1/62 = 0.01612
    assert fused[0].chunk_id == "chunk_a"
    assert fused[0].metadata["rrf_appearances"] == 2
    assert fused[1].chunk_id == "chunk_b"
    assert fused[2].chunk_id == "chunk_c"


# --- 3. Cross-Encoder Reranking Tests ---

@pytest.mark.asyncio
async def test_cross_encoder_reranking():
    mock_client = MagicMock(spec=NeuroFlowClient)
    # Return higher score for chunk 2 than chunk 1
    mock_client.chat = AsyncMock(
        side_effect=[
            GenerationResult(content="3.5", model="gpt-4o", input_tokens=10, output_tokens=2, latency_ms=50.0, cost_usd=0.0001, finish_reason="stop"),
            GenerationResult(content="9.5", model="gpt-4o", input_tokens=10, output_tokens=2, latency_ms=50.0, cost_usd=0.0001, finish_reason="stop"),
        ]
    )

    candidates = [
        RetrievalResult(chunk_id="c1", document_id="d1", content="Irrelevant passage", score=0.9),
        RetrievalResult(chunk_id="c2", document_id="d2", content="Highly relevant passage", score=0.7),
    ]

    reranker = CrossEncoderReranker(client=mock_client)
    reranked = await reranker.rerank("relevant query", candidates, top_n=2)

    # Chunk 2 should be boosted to top rank due to higher score 9.5
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].score == 9.5
    assert reranked[1].chunk_id == "c1"
    assert reranked[1].score == 3.5


# --- 4. Context Assembler Tests ---

def test_context_assembler_budget_and_sources():
    assembler = ContextAssembler(default_max_tokens=100)

    chunks = [
        RetrievalResult(
            chunk_id="c1",
            document_id="doc_a",
            content="First chunk content discussing machine learning and transformers.",
            score=0.95,
            metadata={"filename": "paper.pdf", "page_number": 2},
        ),
        RetrievalResult(
            chunk_id="c2",
            document_id="doc_b",
            content="Second chunk content discussing neural attention mechanisms.",
            score=0.88,
            metadata={"filename": "book.pdf", "page_number": 5},
        ),
    ]

    assembled = assembler.assemble(chunks, max_tokens=100)

    assert "[Source 1 — paper.pdf, page 2]" in assembled["context"]
    assert "[Source 2 — book.pdf, page 5]" in assembled["context"]
    assert len(assembled["chunks_used"]) == 2
    assert assembled["total_tokens"] > 0
    assert len(assembled["sources"]) == 2


# --- 5. Hybrid Retriever Parallel Search Tests ---

@pytest.mark.asyncio
async def test_hybrid_retriever_parallel_gather():
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

    # Dense query returns c1
    mock_conn.fetch.side_effect = [
        [{"id": "c1", "document_id": "d1", "content": "Dense result", "metadata": "{}", "score": 0.9}],  # dense
        [{"id": "c2", "document_id": "d2", "content": "Sparse result", "metadata": "{}", "score": 0.8}],  # sparse
        [],  # meta
    ]

    mock_client = MagicMock(spec=NeuroFlowClient)
    mock_client.embed = AsyncMock(return_value=[[0.1] * 1536])
    mock_client.chat = AsyncMock(
        return_value=GenerationResult(content="8.0", model="gpt-4o", input_tokens=10, output_tokens=2, latency_ms=50.0, cost_usd=0.0001, finish_reason="stop")
    )

    retriever = HybridRetriever(pool=mock_pool, client=mock_client)
    
    with patch.object(retriever.query_processor, "process", AsyncMock(return_value=ProcessedQuery(
        original_query="What is vector search?",
        expanded_queries=[],
        metadata_filters={},
        query_type="factual",
    ))):
        results = await retriever.retrieve("What is vector search?", k=5)
        assert len(results) >= 2
        chunk_ids = [r.chunk_id for r in results]
        assert "c1" in chunk_ids
        assert "c2" in chunk_ids
