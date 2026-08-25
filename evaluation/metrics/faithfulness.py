from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

try:
    from backend.providers.base import ChatMessage
    from backend.providers.client import NeuroFlowClient
    from backend.providers.router import RoutingCriteria
except ImportError:
    from providers.base import ChatMessage
    from providers.client import NeuroFlowClient
    from providers.router import RoutingCriteria

logger = logging.getLogger("neuroflow-faithfulness")


async def evaluate_faithfulness(
    query: str,
    answer: str,
    context: str,
    client: Optional[NeuroFlowClient] = None,
) -> float:
    """
    Evaluates faithfulness: Are all claims in the answer grounded in the retrieved context?
    - Extracts claims from answer
    - For each claim, evaluates: 'yes' (1.0), 'partial' (0.5), 'no' (0.0)
    - Returns supported_claims / total_claims
    - Returns 0.0 if answer makes claims but context is empty
    """
    if not answer or not answer.strip():
        return 1.0

    if not context or not context.strip():
        # Answer makes claims but context is empty -> ungrounded
        return 0.0

    llm_client = client or NeuroFlowClient.get_instance()
    criteria = RoutingCriteria(task_type="evaluation")

    prompt = (
        "You are an expert evaluator assessing the faithfulness and factual grounding of an answer relative to a context.\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "Instructions:\n"
        "1. Identify the key factual assertions in the Answer.\n"
        "2. Verify whether each assertion is supported by the Context (synonyms and reasonable domain paraphrases count as supported):\n"
        "   - 'yes' (1.0) if supported and consistent with the Context\n"
        "   - 'partial' (0.5) if partly supported\n"
        "   - 'no' (0.0) if unsupported, contradictory, or fabricated\n"
        "3. Scoring Guidelines:\n"
        "   - If all claims accurately reflect the context -> score is 1.0\n"
        "   - If one claim is accurate and one claim is hallucinated/fabricated -> score is 0.5\n"
        "   - If completely unrelated, fabricated, or contradictory -> score is 0.0\n\n"
        "Return ONLY a JSON object in this exact schema:\n"
        "{\n"
        "  \"claims\": [{\"claim\": \"...\", \"verdict\": \"yes\"}],\n"
        "  \"faithfulness_score\": 1.0\n"
        "}"
    )

    try:
        res = await llm_client.chat(
            [ChatMessage(role="user", content=prompt)],
            criteria=criteria,
            model="gpt-4o-mini",
            max_tokens=300,
        )
        content = res.content.strip()
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if "faithfulness_score" in data:
                return round(float(data["faithfulness_score"]), 4)
            elif "claims" in data and isinstance(data["claims"], list) and data["claims"]:
                total = len(data["claims"])
                score_sum = sum(
                    1.0 if c.get("verdict") == "yes" else 0.5 if c.get("verdict") == "partial" else 0.0
                    for c in data["claims"]
                )
                return round(score_sum / max(1, total), 4)
    except Exception as err:
        logger.warning(f"Faithfulness LLM evaluation fallback: {err}")

    # Heuristic fallback: clause-by-clause overlap analysis
    clauses = [c.strip() for c in re.split(r"[,;]|(?:\s+and\s+)|\.\s+", answer) if len(c.strip()) > 5]
    if not clauses:
        clauses = [answer.strip()]

    ctx_words = set(re.findall(r"\w+", context.lower()))
    clause_scores = []
    for cl in clauses:
        cl_words = set(re.findall(r"\w+", cl.lower()))
        overlap = len(cl_words.intersection(ctx_words)) / max(1, len(cl_words))
        if overlap >= 0.50:
            clause_scores.append(1.0)
        elif overlap >= 0.20:
            clause_scores.append(0.5)
        else:
            clause_scores.append(0.0)

    return round(float(sum(clause_scores) / max(1, len(clause_scores))), 4)
