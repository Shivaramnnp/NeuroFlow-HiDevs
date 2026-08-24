import asyncio
import io
import os
import sys

from pipelines.ingestion.extractors.base import ExtractedPage
from pipelines.ingestion.extractors.pdf_extractor import PDFExtractor
from pipelines.ingestion.extractors.docx_extractor import DocxExtractor
from pipelines.ingestion.extractors.csv_extractor import CSVExtractor
from pipelines.ingestion.chunker import (
    count_tokens,
    fixed_size_chunking,
    select_chunking_strategy,
    chunk_pages,
)
from pipelines.ingestion.pipeline import compute_content_hash


async def main():
    print("=== Testing NeuroFlow Ingestion Pipeline Standalone ===")

    # 1. Deduplication Hash
    sample_bytes = b"NeuroFlow test document bytes"
    h = compute_content_hash(sample_bytes)
    print(f"\n1. Content Hash (SHA-256): {h}")
    assert len(h) == 64

    # 2. CSV Extraction & Auto-Chunking
    print("\n2. Testing CSV Extractor & Fixed-Size Chunker...")
    csv_data = "id,name,role\n1,Alice,Admin\n2,Bob,User\n3,Charlie,Reviewer\n"
    csv_extractor = CSVExtractor()
    pages = await csv_extractor.extract(csv_data.encode("utf-8"))
    print(f"Extracted {len(pages)} CSV pages. Page 1 Content:\n{pages[0].content}")

    strategy = select_chunking_strategy(pages, source_type="csv")
    print(f"Auto-selected chunking strategy: '{strategy}'")
    assert strategy == "fixed_size"

    chunks = await chunk_pages(pages, strategy=strategy, source_type="csv")
    print(f"Generated {len(chunks)} chunks. Chunk 1 tokens: {chunks[0].token_count}")

    # 3. Fixed size chunking sentence boundary check
    print("\n3. Testing Sentence Boundary Preservation...")
    long_text = "NeuroFlow is an enterprise RAG platform. It processes multi-modal documents asynchronously. " \
                "Embeddings are stored in PostgreSQL using pgvector with HNSW indexing."
    chunks_fixed = fixed_size_chunking(long_text, target_tokens=15, overlap_tokens=5)
    for i, c in enumerate(chunks_fixed):
        print(f"Chunk {i+1} ({c.token_count} tok): {c.content}")

    print("\n=== All Standalone Ingestion Tests Completed Successfully! ===")


if __name__ == "__main__":
    asyncio.run(main())
