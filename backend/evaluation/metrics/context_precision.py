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

logger = logging.getLogger("neuroflow-context-precision")


async def evaluate_context_precision(
    query: str,
    chunks: List[str],
    answer: str,
    client: Optional[NeuroFlowClient] = None,
) -> float:
    """
    Evaluates context precision: Were the retrieved chunks actually useful?
    - For each retrieved chunk, checks if useful in generating the answer.
    - Computes rank-weighted precision: sum(useful[i] * (1/i)) / sum(1/i).
    """
    if not chunks:
        return 0.0

    if not answer or not answer.strip():
        return 0.0

    llm_client = client or NeuroFlowClient.get_instance()
    criteria = RoutingCriteria(task_type="evaluation")

    prompt = (
        f"Query: {query}\n"
        f"Answer: {answer}\n\n"
        f"For each of the following retrieved passages, determine if it contains information "
        f"that was actually useful in generating the answer above.\n"
        f"Answer for each passage with 'yes' or 'no'.\n"
        f"Return ONLY a JSON array of strings, e.g. [\"yes\", \"no\", \"yes\"].\n\n"
        f"Passages:\n" + "\n".join(f"Passage {i+1}:\n{chunk[:300]}" for i, chunk in enumerate(chunks))
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
        logger.warning(f"Context precision evaluation fallback: {err}")
        # Overlap heuristic
        ans_words = set(answer.lower().split())
        verdicts = []
        for c in chunks:
            c_words = set(c.lower().split())
            overlap = len(ans_words.intersection(c_words)) / (len(ans_words) + 1e-5)
            verdicts.append("yes" if overlap > 0.2 else "no")

    # Compute rank-weighted precision
    weights_sum = 0.0
    weighted_useful = 0.0

    for i, chunk in enumerate(chunks, start=1):
        v = verdicts[i - 1] if i - 1 < len(verdicts) else "no"
        weight = 1.0 / i
        weights_sum += weight
        if str(v).strip().lower() == "yes":
            weighted_useful += weight

    if weights_sum == 0.0:
        return 0.0

    return round(weighted_useful / weights_sum, 4)
