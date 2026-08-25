from __future__ import annotations

import logging
import re
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("neuroflow-citations")


@dataclass
class Citation:
    """Represents a structured source citation extracted from LLM generation."""
    reference: str  # "Source 1"
    chunk_id: Union[uuid.UUID, str]
    document_name: str
    page_number: Optional[int]
    content_preview: str  # first 100 chars of cited chunk
    invalid_citation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["chunk_id"] = str(self.chunk_id)
        return d


def strip_thinking(text: str) -> Tuple[str, Optional[str]]:
    """
    Extract <think>...</think> reasoning blocks from generated text.
    Returns (clean_text, thinking_content).
    """
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return clean, thinking
    return text.strip(), None


class CitationProcessor:
    """
    Parses and validates citation references ([Source N]) from generation text against
    retrieved source metadata. Flags hallucinated citations that exceed available sources.
    """

    @staticmethod
    def parse_citations(
        generation_text: str,
        sources: List[Dict[str, Any]],
        chunks_map: Optional[Dict[str, Any]] = None,
    ) -> List[Citation]:
        """
        Parse all [Source N] patterns in generation_text and resolve against context sources.
        """
        chunks_map = chunks_map or {}
        num_sources = len(sources)

        # Find all [Source N] matches (1-indexed)
        matches = re.findall(r"\[Source\s+(\d+)\]", generation_text, re.IGNORECASE)
        seen_refs = set()
        citations: List[Citation] = []

        for ref_str in matches:
            idx = int(ref_str)
            ref_key = f"Source {idx}"
            if ref_key in seen_refs:
                continue
            seen_refs.add(ref_key)

            # Check if source index is valid (1 <= idx <= num_sources)
            if 1 <= idx <= num_sources:
                src_meta = sources[idx - 1]
                cid = src_meta.get("chunk_id", str(uuid.uuid4()))
                doc_name = src_meta.get("filename") or f"document_{src_meta.get('document_id', '')[:8]}.pdf"
                page_num = src_meta.get("page_number", 1)

                chunk_obj = chunks_map.get(str(cid))
                if chunk_obj and hasattr(chunk_obj, "content"):
                    content_preview = chunk_obj.content[:100]
                elif isinstance(chunk_obj, dict) and "content" in chunk_obj:
                    content_preview = chunk_obj["content"][:100]
                else:
                    content_preview = src_meta.get("content_preview", f"Excerpt from {doc_name} (Page {page_num})")[:100]

                citations.append(
                    Citation(
                        reference=ref_key,
                        chunk_id=cid,
                        document_name=doc_name,
                        page_number=page_num,
                        content_preview=content_preview,
                        invalid_citation=False,
                    )
                )
            else:
                # Hallucinated citation: referenced source index does not exist in context
                citations.append(
                    Citation(
                        reference=ref_key,
                        chunk_id=str(uuid.uuid4()),
                        document_name="Unknown Source (Hallucinated)",
                        page_number=None,
                        content_preview="[Invalid citation: source index exceeds context sources]",
                        invalid_citation=True,
                    )
                )

        return citations
