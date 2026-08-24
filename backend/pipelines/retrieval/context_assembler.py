from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import tiktoken

from .fusion import RetrievalResult

logger = logging.getLogger("neuroflow-context-assembler")

try:
    _encoder = tiktoken.get_encoding("cl100k_base")
except Exception:
    _encoder = None


def count_tokens(text: str) -> int:
    if _encoder is not None:
        return len(_encoder.encode(text))
    return max(1, len(text) // 4)


class ContextAssembler:
    """
    Assembles top retrieved and reranked chunks into an LLM context window:
    - Formats citations: [Source N — filename, page X]
    - Enforces maximum token budget (default 4000 tokens)
    - Preserves sentence boundaries without mid-sentence truncation
    - Returns structured metadata: context, chunks_used, total_tokens, sources
    """

    def __init__(self, default_max_tokens: int = 4000):
        self.default_max_tokens = default_max_tokens

    def assemble(
        self,
        chunks: List[RetrievalResult],
        max_tokens: Optional[int] = None,
        doc_names: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        token_budget = max_tokens or self.default_max_tokens
        doc_names = doc_names or {}

        assembled_blocks: List[str] = []
        chunks_used: List[str] = []
        sources: List[Dict[str, Any]] = []
        current_tokens = 0

        for idx, chunk in enumerate(chunks, start=1):
            doc_id = str(chunk.document_id)
            filename = chunk.metadata.get("filename") or doc_names.get(doc_id) or f"document_{doc_id[:8]}.pdf"
            page_num = chunk.metadata.get("page_number") or chunk.metadata.get("slide_number") or 1

            header = f"[Source {idx} — {filename}, page {page_num}]"
            block = f"{header}\n{chunk.content.strip()}"
            block_tokens = count_tokens(block + "\n\n")

            # Check if chunk fits in budget
            if current_tokens + block_tokens <= token_budget:
                assembled_blocks.append(block)
                chunks_used.append(str(chunk.chunk_id))
                sources.append(
                    {
                        "source_index": idx,
                        "chunk_id": str(chunk.chunk_id),
                        "document_id": doc_id,
                        "filename": filename,
                        "page_number": page_num,
                        "score": chunk.score,
                    }
                )
                current_tokens += block_tokens
            else:
                # Token budget reached
                break

        full_context = "\n\n".join(assembled_blocks)
        return {
            "context": full_context,
            "chunks_used": chunks_used,
            "total_tokens": current_tokens,
            "sources": sources,
        }
