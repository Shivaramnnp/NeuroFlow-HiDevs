from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from backend.providers.base import ChatMessage
    from backend.providers.client import NeuroFlowClient
    from backend.providers.router import RoutingCriteria
except ImportError:
    from providers.base import ChatMessage
    from providers.client import NeuroFlowClient
    from providers.router import RoutingCriteria

logger = logging.getLogger("neuroflow-query-processor")


@dataclass
class ProcessedQuery:
    """Represents an analyzed and expanded search query."""
    original_query: str
    expanded_queries: List[str] = field(default_factory=list)
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    query_type: str = "factual"  # "factual" | "analytical" | "comparative" | "procedural"
    hypothetical_document: Optional[str] = None


class QueryProcessor:
    """
    Analyzes raw user queries before retrieval:
    1. Generates 2-3 query expansions for multi-angle retrieval
    2. Extracts implicit metadata filters (e.g. year, topic, doc type)
    3. Classifies query type (factual, analytical, comparative, procedural)
    4. Generates HyDE (Hypothetical Document Embedding) answers when requested
    """

    def __init__(self, client: Optional[NeuroFlowClient] = None):
        self.client = client

    def _get_client(self) -> Optional[NeuroFlowClient]:
        if self.client is not None:
            return self.client
        try:
            return NeuroFlowClient.get_instance()
        except Exception:
            return None

    async def expand_query(self, query: str) -> List[str]:
        """
        Generate 2-3 alternative search phrasings for the input query.
        """
        client = self._get_client()
        if client is not None:
            try:
                prompt = (
                    f"Generate 2 to 3 concise, alternative search query phrasings for the following query. "
                    f"Return ONLY the alternative queries, one per line, without numbering or bullets.\n\n"
                    f"Query: {query}"
                )
                messages = [ChatMessage(role="user", content=prompt)]
                criteria = RoutingCriteria(task_type="classification")
                res = await client.chat(messages, criteria=criteria, max_tokens=150)
                lines = [line.strip().lstrip("-*0123456789. ") for line in res.content.splitlines() if line.strip()]
                if lines:
                    return lines[:3]
            except Exception as err:
                logger.warning(f"LLM query expansion failed: {err}")

        # Rule-based fallback expansions
        words = query.split()
        expansions = []
        if "how" in query.lower() or "why" in query.lower():
            expansions.append(f"explain {query.lower().replace('how does ', '').replace('how do ', '').replace('how ', '')}")
        if len(words) > 2:
            expansions.append(" ".join(words[:4]) + " explanation")
        return expansions or [query]

    def extract_metadata_filters_heuristics(self, query: str) -> Dict[str, Any]:
        """Rule-based extraction of metadata filters (e.g. year, topic, source_type)."""
        filters: Dict[str, Any] = {}
        q_lower = query.lower()

        # Extract 4-digit year (e.g. 2020-2029)
        year_match = re.search(r"\b(20[12][0-9])\b", query)
        if year_match:
            filters["year"] = int(year_match.group(1))

        # Extract source type (pdf, docx, csv, etc.)
        for st in ["pdf", "docx", "csv", "image", "url"]:
            if f"{st} document" in q_lower or f"{st} file" in q_lower or f"in {st}" in q_lower:
                filters["source_type"] = st

        # Topic detection
        for topic in ["climate", "finance", "medical", "legal", "technology", "security", "transformer"]:
            if topic in q_lower:
                filters["topic"] = topic

        return filters

    async def extract_metadata_filters(self, query: str) -> Dict[str, Any]:
        """Extract structured metadata filters using heuristics + LLM."""
        filters = self.extract_metadata_filters_heuristics(query)
        client = self._get_client()
        if client is not None and ("from" in query.lower() or "about" in query.lower()):
            try:
                prompt = (
                    f"Extract any implicit metadata filter attributes (e.g. year as integer, topic, author, source_type) "
                    f"from the query. Return a valid JSON object only (e.g. {{\"year\": 2023, \"topic\": \"climate\"}}). "
                    f"If none found, return {{}}.\n\nQuery: {query}"
                )
                messages = [ChatMessage(role="user", content=prompt)]
                criteria = RoutingCriteria(task_type="classification")
                res = await client.chat(messages, criteria=criteria, max_tokens=100)
                json_match = re.search(r"\{.*?\}", res.content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    filters.update(parsed)
            except Exception as err:
                logger.warning(f"LLM metadata filter extraction failed: {err}")

        return filters

    async def classify_query(self, query: str) -> str:
        """
        Classify query as 'factual', 'analytical', 'comparative', or 'procedural'.
        """
        q_lower = query.lower()
        if any(w in q_lower for w in ["compare", "vs", "versus", "difference between", "better than"]):
            return "comparative"
        if any(w in q_lower for w in ["how to", "steps to", "procedure", "guide", "implement", "create"]):
            return "procedural"
        if any(w in q_lower for w in ["why", "analyze", "impact of", "explain the relationship", "evaluate"]):
            return "analytical"

        client = self._get_client()
        if client is not None:
            try:
                prompt = (
                    f"Classify this user query into exactly ONE of these categories: factual, analytical, comparative, procedural.\n"
                    f"Return ONLY the category name in lowercase.\n\nQuery: {query}"
                )
                res = await client.chat([ChatMessage(role="user", content=prompt)], max_tokens=20)
                category = res.content.strip().lower()
                if category in ["factual", "analytical", "comparative", "procedural"]:
                    return category
            except Exception:
                pass

        return "factual"

    async def generate_hyde_document(self, query: str) -> str:
        """
        Generate a hypothetical document answering the query for HyDE retrieval.
        """
        client = self._get_client()
        if client is not None:
            try:
                prompt = (
                    f"Write a comprehensive, factual paragraph that directly and accurately answers the following query as if it were an excerpt from an authoritative technical document.\n\n"
                    f"Query: {query}"
                )
                res = await client.chat([ChatMessage(role="user", content=prompt)], max_tokens=250)
                return res.content.strip()
            except Exception as err:
                logger.warning(f"HyDE document generation failed: {err}")

        return f"Authoritative documentation and comprehensive explanation answering {query}."

    async def process(self, query: str, use_hyde: bool = False) -> ProcessedQuery:
        """Execute full query processing pipeline."""
        exp_task = self.expand_query(query)
        filt_task = self.extract_metadata_filters(query)
        cls_task = self.classify_query(query)

        if use_hyde:
            hyde_task = self.generate_hyde_document(query)
            expansions, filters, q_type, hyde_doc = await asyncio.gather(
                exp_task, filt_task, cls_task, hyde_task
            )
        else:
            expansions, filters, q_type = await asyncio.gather(
                exp_task, filt_task, cls_task
            )
            hyde_doc = None

        return ProcessedQuery(
            original_query=query,
            expanded_queries=expansions,
            metadata_filters=filters,
            query_type=q_type,
            hypothetical_document=hyde_doc,
        )
