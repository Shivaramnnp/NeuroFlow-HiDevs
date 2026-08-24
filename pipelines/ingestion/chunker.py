from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import tiktoken

from .extractors.base import ExtractedPage
try:
    from backend.providers.client import NeuroFlowClient
except ImportError:
    from providers.client import NeuroFlowClient

logger = logging.getLogger("neuroflow-chunker")

# Initialize tiktoken encoder
try:
    _encoder = tiktoken.get_encoding("cl100k_base")
except Exception:
    _encoder = None


def count_tokens(text: str) -> int:
    """Accurately count tokens using tiktoken (cl100k_base)."""
    if _encoder is not None:
        return len(_encoder.encode(text))
    return max(1, len(text) // 4)


@dataclass
class Chunk:
    """Represents a discrete indexed text chunk with embeddings metadata."""
    content: str
    chunk_index: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences cleanly."""
    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vector embeddings."""
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def fixed_size_chunking(
    text: str,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
    metadata: Optional[Dict[str, Any]] = None,
    start_index: int = 0,
) -> List[Chunk]:
    """
    Fixed-size chunker: 512 tokens with 64-token overlap.
    Never splits mid-sentence: finds nearest sentence boundary within 10% of target size.
    """
    if not text.strip():
        return []

    sentences = split_sentences(text)
    if not sentences:
        sentences = [text]

    chunks: List[Chunk] = []
    current_sentences: List[str] = []
    current_tokens = 0
    min_tokens = int(target_tokens * 0.90)  # 10% lower bound (460 tokens)
    max_tokens = int(target_tokens * 1.10)  # 10% upper bound (563 tokens)

    chunk_idx = start_index

    for sent in sentences:
        sent_tokens = count_tokens(sent)
        
        # If single sentence is bigger than max_tokens, break by words
        if sent_tokens > max_tokens and not current_sentences:
            words = sent.split()
            buf = []
            buf_tok = 0
            for w in words:
                w_tok = count_tokens(w + " ")
                if buf_tok + w_tok > target_tokens and buf:
                    chunk_text = " ".join(buf)
                    chunks.append(
                        Chunk(
                            content=chunk_text,
                            chunk_index=chunk_idx,
                            token_count=count_tokens(chunk_text),
                            metadata=dict(metadata or {}),
                        )
                    )
                    chunk_idx += 1
                    buf = []
                    buf_tok = 0
                buf.append(w)
                buf_tok += w_tok
            if buf:
                current_sentences = [" ".join(buf)]
                current_tokens = buf_tok
            continue

        if current_tokens + sent_tokens > max_tokens and current_tokens >= min_tokens:
            # Reached boundary window - cut chunk here
            chunk_text = " ".join(current_sentences)
            chunks.append(
                Chunk(
                    content=chunk_text,
                    chunk_index=chunk_idx,
                    token_count=count_tokens(chunk_text),
                    metadata=dict(metadata or {}),
                )
            )
            chunk_idx += 1

            # Compute overlap sentences
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_sentences):
                s_tok = count_tokens(s)
                if overlap_count + s_tok <= overlap_tokens:
                    overlap_sentences.insert(0, s)
                    overlap_count += s_tok
                else:
                    break

            current_sentences = overlap_sentences + [sent]
            current_tokens = sum(count_tokens(s) for s in current_sentences)
        else:
            current_sentences.append(sent)
            current_tokens += sent_tokens

    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append(
            Chunk(
                content=chunk_text,
                chunk_index=chunk_idx,
                token_count=count_tokens(chunk_text),
                metadata=dict(metadata or {}),
            )
        )

    return chunks


async def semantic_chunking(
    text: str,
    similarity_threshold: float = 0.7,
    client: Optional[NeuroFlowClient] = None,
    metadata: Optional[Dict[str, Any]] = None,
    start_index: int = 0,
) -> List[Chunk]:
    """
    Semantic chunker:
    - Uses sliding window to find natural topic shifts.
    - Embeds sentences, computes cosine similarity between adjacent sentences,
      and splits where similarity drops below 0.7.
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return fixed_size_chunking(text, metadata=metadata, start_index=start_index)

    chunks: List[Chunk] = []
    chunk_idx = start_index

    # If client is provided, compute real embeddings
    if client is not None:
        try:
            embeddings = await client.embed(sentences)
            current_group: List[str] = [sentences[0]]

            for i in range(len(sentences) - 1):
                sim = cosine_similarity(embeddings[i], embeddings[i + 1])
                # If topic shifts (similarity < 0.7) and current group has reasonable size
                if sim < similarity_threshold and len(current_group) > 0:
                    group_text = " ".join(current_group)
                    chunks.append(
                        Chunk(
                            content=group_text,
                            chunk_index=chunk_idx,
                            token_count=count_tokens(group_text),
                            metadata={"strategy": "semantic", "split_similarity": sim, **(metadata or {})},
                        )
                    )
                    chunk_idx += 1
                    current_group = [sentences[i + 1]]
                else:
                    current_group.append(sentences[i + 1])

            if current_group:
                group_text = " ".join(current_group)
                chunks.append(
                    Chunk(
                        content=group_text,
                        chunk_index=chunk_idx,
                        token_count=count_tokens(group_text),
                        metadata={"strategy": "semantic", **(metadata or {})},
                    )
                )
            return chunks
        except Exception as err:
            logger.warning(f"Semantic chunking embedding failed, falling back to fixed_size: {err}")

    # Fallback to fixed size if no client or embedding fails
    return fixed_size_chunking(text, metadata=metadata, start_index=start_index)


def hierarchical_chunking(
    pages: List[ExtractedPage],
    start_index: int = 0,
) -> List[Chunk]:
    """
    Hierarchical chunker:
    - For documents with heading structures (DOCX with headings, PDF with chapters).
    - Top-level sections become parent chunks; sub-sections become child chunks.
    - Stores parent-child relationship in chunks.metadata (parent_id, level, section).
    """
    chunks: List[Chunk] = []
    chunk_idx = start_index

    current_parent_id: Optional[str] = None
    current_parent_title: Optional[str] = None

    for page in pages:
        level = page.metadata.get("level")
        section = page.metadata.get("section")
        
        # When an H1 heading or section header is encountered, create parent chunk
        if level == "h1" or (section and level is None and current_parent_id is None):
            current_parent_id = str(uuid.uuid4())
            current_parent_title = section or page.content
            chunks.append(
                Chunk(
                    content=page.content,
                    chunk_index=chunk_idx,
                    token_count=count_tokens(page.content),
                    metadata={
                        "is_parent": True,
                        "chunk_id": current_parent_id,
                        "level": "h1",
                        "section": current_parent_title,
                        **page.metadata,
                    },
                )
            )
            chunk_idx += 1
        elif level in ["h2", "h3"] or current_parent_id is not None:
            # Sub-section child chunk
            child_chunks = fixed_size_chunking(
                page.content,
                target_tokens=512,
                overlap_tokens=64,
                metadata={
                    "is_child": True,
                    "parent_id": current_parent_id,
                    "level": level or "content",
                    "section": current_parent_title,
                    **page.metadata,
                },
                start_index=chunk_idx,
            )
            chunks.extend(child_chunks)
            chunk_idx += len(child_chunks)
        else:
            # Standard chunk
            sub_chunks = fixed_size_chunking(
                page.content,
                metadata=page.metadata,
                start_index=chunk_idx,
            )
            chunks.extend(sub_chunks)
            chunk_idx += len(sub_chunks)

    return chunks


def select_chunking_strategy(
    pages: List[ExtractedPage],
    source_type: str = "text",
) -> str:
    """
    Auto-select chunking strategy based on content types and document characteristics:
    - table content type in pages or CSV -> always 'fixed_size'
    - DOCX with headings -> 'hierarchical'
    - PDF with > 50 pages -> 'semantic'
    - Default -> 'fixed_size'
    """
    if any(p.content_type == "table" for p in pages) or source_type == "csv":
        return "fixed_size"

    if source_type == "docx" and any(p.metadata.get("level") for p in pages):
        return "hierarchical"

    # Check total page count for PDF
    total_pages = len(pages)
    if pages and "total_pages" in pages[0].metadata:
        total_pages = max(total_pages, pages[0].metadata["total_pages"])

    if source_type == "pdf" and total_pages > 50:
        return "semantic"

    return "fixed_size"


async def chunk_pages(
    pages: List[ExtractedPage],
    strategy: Optional[str] = None,
    source_type: str = "text",
    client: Optional[NeuroFlowClient] = None,
) -> List[Chunk]:
    """
    Execute chunking on extracted pages using auto-selected or specified strategy.
    """
    chosen_strategy = strategy or select_chunking_strategy(pages, source_type=source_type)
    logger.info(f"Chunking {len(pages)} pages using strategy: '{chosen_strategy}' (source_type={source_type})")

    if chosen_strategy == "hierarchical":
        return hierarchical_chunking(pages)

    all_chunks: List[Chunk] = []
    chunk_index = 0

    if chosen_strategy == "semantic":
        full_text = "\n\n".join(p.content for p in pages if p.content.strip())
        return await semantic_chunking(full_text, client=client, metadata={"source_type": source_type})

    # Fixed size chunking per page
    for page in pages:
        page_chunks = fixed_size_chunking(
            page.content,
            target_tokens=512,
            overlap_tokens=64,
            metadata={
                "page_number": page.page_number,
                "content_type": page.content_type,
                **page.metadata,
            },
            start_index=chunk_index,
        )
        all_chunks.extend(page_chunks)
        chunk_index += len(page_chunks)

    return all_chunks
