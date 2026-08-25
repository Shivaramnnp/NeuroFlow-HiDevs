from __future__ import annotations

import logging
import re
from typing import List, Optional

import numpy as np

try:
    from backend.providers.base import ChatMessage
    from backend.providers.client import NeuroFlowClient
    from backend.providers.router import RoutingCriteria
except ImportError:
    from providers.base import ChatMessage
    from providers.client import NeuroFlowClient
    from providers.router import RoutingCriteria

logger = logging.getLogger("neuroflow-answer-relevance")


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


async def evaluate_answer_relevance(
    query: str,
    answer: str,
    client: Optional[NeuroFlowClient] = None,
) -> float:
    """
    Evaluates answer relevance: Does the answer address what was asked?
    - Generates 3-5 questions that the answer could be a response to.
    - Embeds the original query and all generated questions.
    - Score = mean cosine similarity between original query and generated question embeddings.
    """
    if not answer or not answer.strip() or not query or not query.strip():
        return 0.0

    llm_client = client or NeuroFlowClient.get_instance()
    criteria = RoutingCriteria(task_type="evaluation")

    # Step 1: Generate oracle questions
    prompt = (
        f"Generate 3 to 5 candidate questions that the following answer directly addresses or answers.\n"
        f"Return ONLY the generated questions, one per line, without numbering.\n\n"
        f"Answer: {answer}"
    )

    generated_questions: List[str] = []
    try:
        res = await llm_client.chat(
            [ChatMessage(role="user", content=prompt)],
            criteria=criteria,
            max_tokens=200,
        )
        lines = [line.strip().lstrip("-*0123456789. ") for line in res.content.splitlines() if line.strip()]
        generated_questions = lines[:5]
    except Exception as err:
        logger.warning(f"Answer relevance question generation fallback: {err}")
        generated_questions = [query]

    if not generated_questions:
        generated_questions = [query]

    # Step 2: Embed query and generated questions
    all_texts = [query] + generated_questions
    try:
        embeddings = await llm_client.embed(all_texts)
        q_emb = embeddings[0]
        gen_embs = embeddings[1:]
        sims = [cosine_similarity(q_emb, g_emb) for g_emb in gen_embs]
        return round(float(np.mean(sims)), 4) if sims else 0.0
    except Exception as err:
        logger.warning(f"Answer relevance embedding fallback: {err}")
        # Word overlap Jaccard fallback
        q_words = set(query.lower().split())
        overlaps = [
            len(q_words.intersection(set(g.lower().split()))) / (len(q_words) + 1e-5)
            for g in generated_questions
        ]
        return round(float(np.mean(overlaps)), 4) if overlaps else 0.5
