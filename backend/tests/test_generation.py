from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from pipelines.generation.prompt_builder import PromptBuilder, BASE_SYSTEM_PROMPT
from pipelines.generation.citations import Citation, CitationProcessor, strip_thinking
from pipelines.generation.generator import RAGGenerator
from backend.providers.base import GenerationResult
from backend.providers.client import NeuroFlowClient


# --- 1. Prompt Builder Tests ---

def test_prompt_builder_query_types():
    pb = PromptBuilder()

    # Factual
    factual_prompt = pb.build_system_prompt("factual")
    assert "Provide a direct, concise answer" in factual_prompt
    assert BASE_SYSTEM_PROMPT in factual_prompt

    # Analytical
    analytical_prompt = pb.build_system_prompt("analytical")
    assert "Analyze and synthesize across the provided sources" in analytical_prompt
    assert "<think>" in analytical_prompt

    # Comparative
    comparative_prompt = pb.build_system_prompt("comparative")
    assert "Organize your response as a structured comparison" in comparative_prompt

    # Procedural
    procedural_prompt = pb.build_system_prompt("procedural")
    assert "Provide numbered steps. Each step must be cited." in procedural_prompt

    # User message formatting with <context> tags
    msg = pb.format_user_message("What is HNSW?", "HNSW is a graph vector index.")
    assert "<context>\nHNSW is a graph vector index.\n</context>\n\nWhat is HNSW?" in msg


# --- 2. Citation Processor Tests ---

def test_citations_processor_valid_and_hallucinated():
    sources = [
        {"chunk_id": str(uuid.uuid4()), "filename": "doc1.pdf", "page_number": 3, "content_preview": "HNSW index explanation"},
        {"chunk_id": str(uuid.uuid4()), "filename": "doc2.pdf", "page_number": 7, "content_preview": "pgvector cosine similarity"},
    ]

    # Generation references Source 1, Source 2, and non-existent Source 5
    text = "Based on [Source 1], HNSW is fast. pgvector uses cosine [Source 2]. Also mentioned in [Source 5]."

    citations = CitationProcessor.parse_citations(text, sources)
    assert len(citations) == 3

    c1 = citations[0]
    assert c1.reference == "Source 1"
    assert c1.document_name == "doc1.pdf"
    assert c1.page_number == 3
    assert c1.invalid_citation is False

    c2 = citations[1]
    assert c2.reference == "Source 2"
    assert c2.document_name == "doc2.pdf"
    assert c2.invalid_citation is False

    c5 = citations[2]
    assert c5.reference == "Source 5"
    assert c5.invalid_citation is True  # Hallucinated citation!


def test_strip_thinking():
    raw_text = "<think>Let me analyze the sources carefully.</think>According to [Source 1], vector search is scalable."
    clean, thinking = strip_thinking(raw_text)

    assert clean == "According to [Source 1], vector search is scalable."
    assert thinking == "Let me analyze the sources carefully."


# --- 3. Generator Streaming & DB Logging Tests ---

@pytest.mark.asyncio
async def test_rag_generator_streaming():
    async def mock_stream_tokens(*args, **kwargs):
        tokens = ["Based ", "on ", "[Source 1]", ", pgvector is efficient."]
        for t in tokens:
            yield t

    mock_client = MagicMock()
    mock_client.chat = MagicMock(return_value=mock_stream_tokens())
    mock_client.stream = mock_stream_tokens

    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

    generator = RAGGenerator(client=mock_client, pool=mock_pool)

    sources = [
        {"chunk_id": str(uuid.uuid4()), "filename": "db.pdf", "page_number": 1, "content_preview": "pgvector details"}
    ]

    events = []
    async for ev in generator.generate_stream("Tell me about pgvector", "pgvector context", sources):
        events.append(ev)

    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 4
    assert "".join(e["delta"] for e in token_events) == "Based on [Source 1], pgvector is efficient."

    done_event = [e for e in events if e["type"] == "done"][0]
    assert len(done_event["citations"]) == 1
    assert done_event["citations"][0]["reference"] == "Source 1"


# --- 4. SSE Query API Endpoint Tests ---

@pytest.mark.asyncio
async def test_api_query_endpoints():
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)

    # 1. Test POST /query (stream=false)
    mock_rag_gen = {
        "run_id": str(uuid.uuid4()),
        "generation": "HNSW is an index [Source 1].",
        "citations": [{"reference": "Source 1", "chunk_id": str(uuid.uuid4()), "document_name": "a.pdf", "page_number": 1}],
        "input_tokens": 50,
        "output_tokens": 15,
        "latency_ms": 120,
        "model_used": "gpt-4o",
        "thinking": None,
    }

    with patch("backend.api.query.get_pool", return_value=None), \
         patch("backend.api.query.RAGGenerator.generate", AsyncMock(return_value=mock_rag_gen)), \
         patch("backend.api.query.HybridRetriever.retrieve", AsyncMock(return_value=[])), \
         patch("backend.api.query.ContextAssembler.assemble", return_value={"context": "HNSW docs", "sources": [], "chunks_used": []}):

        res = client.post("/query", json={"query": "Explain HNSW", "stream": False})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "complete"
        assert "HNSW is an index" in data["generation"]

    # 2. Test POST /query (stream=true)
    res_stream = client.post("/query", json={"query": "Explain HNSW", "stream": True})
    assert res_stream.status_code == 200
    data_stream = res_stream.json()
    assert data_stream["status"] == "started"
    assert "stream_url" in data_stream
