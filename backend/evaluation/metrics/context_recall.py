from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

try:
    from backend.providers.base import ChatMessage
    from backend.providers.client import NeuroFlowClient
    from backend.providers.router import RoutingCriteria
except ImportError:
    from providers.base import ChatMessage
    from providers.client import NeuroFlowClient
    from providers.router import RoutingCriteria

logger = logging.getLogger("neuroflow-context-recall")


async def evaluate_context_recall(
    query: str,
    chunks: List[str],
    answer: str,
    client: Optional[NeuroFlowClient] = None,
) -> float:
    """
    Evaluates context recall: Were the relevant sources retrieved?
    - Breaks answer into sentences.
    - For each sentence, checks if it can be attributed to the provided context.
    - Score = attributable_sentences / total_sentences.
    """
    if not answer or not answer.strip():
        return 1.0

    if not chunks:
        return 0.0

    combined_context = "\n\n".join(chunks)

    # Break answer into sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", answer) if len(s.strip()) > 5]
    if not sentences:
        sentences = [answer.strip()]

    llm_client = client or NeuroFlowClient.get_instance()
    criteria = RoutingCriteria(task_type="evaluation")

    prompt = (
        f"Context:\n{combined_context[:3000]}\n\n"
        f"For each of the following sentences from the answer, determine if it can be attributed "
        f"to the context provided above.\n"
        f"Answer for each sentence with 'yes' or 'no'.\n"
        f"Return ONLY a JSON array of strings, e.g. [\"yes\", \"no\", \"yes\"].\n\n"
        f"Sentences:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    )

    verdicts: List[str] = []
    try:
        res = await llm_client.chat(
            [ChatMessage(role="user", content=prompt)],
            criteria=criteria,
            max_tokens=200,
        )
        json_match = re.search(r"\[.*?\]", res.content, re.DOTALL)
        if json_match:
            verdicts = json.loads(json_match.group(0))
        else:
            verdicts = ["yes" if "yes" in line.lower() else "no" for line in res.content.splitlines() if line.strip()]
    except Exception as err:
        logger.warning(f"Context recall evaluation fallback: {err}")
        ctx_words = set(combined_context.lower().split())
        verdicts = []
        for s in sentences:
            s_words = set(s.lower().split())
            overlap = len(s_words.intersection(ctx_words)) / (len(s_words) + 1e-5)
            verdicts.append("yes" if overlap > 0.4 else "no")

    attributable = sum(1 for v in verdicts[:len(sentences)] if str(v).strip().lower() == "yes")
    return round(attributable / max(1, len(sentences)), 4)
