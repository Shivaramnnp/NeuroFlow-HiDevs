from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("neuroflow-optimizer")


class PipelineOptimizer:
    """
    Analyzes historical evaluation metrics and latencies to generate actionable,
    rule-based configuration optimization suggestions.
    """

    @staticmethod
    def generate_suggestions(
        config: Dict[str, Any],
        metrics: Dict[str, float],
        latency_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        suggestions: List[Dict[str, Any]] = []

        retrieval_cfg = config.get("retrieval", {})
        generation_cfg = config.get("generation", {})

        faithfulness = metrics.get("faithfulness", 1.0)
        answer_relevance = metrics.get("answer_relevance", 1.0)
        context_precision = metrics.get("context_precision", 1.0)
        context_recall = metrics.get("context_recall", 1.0)
        overall = metrics.get("overall_score", 1.0)

        # Rule 1: Low Context Precision -> Too much noise in top chunks
        if context_precision < 0.70:
            current_top_k = retrieval_cfg.get("top_k_after_rerank", 10)
            suggested_top_k = max(4, current_top_k - 3)
            suggestions.append(
                {
                    "category": "retrieval",
                    "metric": "context_precision",
                    "current_value": context_precision,
                    "target_field": "retrieval.top_k_after_rerank",
                    "suggested_value": suggested_top_k,
                    "reason": f"Context precision is low ({context_precision:.2f}). Reducing top_k_after_rerank from {current_top_k} to {suggested_top_k} filters out irrelevant retrieved chunks.",
                }
            )

        # Rule 2: Low Context Recall -> Missing critical source documents
        if context_recall < 0.70:
            current_dense = retrieval_cfg.get("dense_k", 20)
            suggested_dense = current_dense + 15
            suggestions.append(
                {
                    "category": "retrieval",
                    "metric": "context_recall",
                    "current_value": context_recall,
                    "target_field": "retrieval.dense_k",
                    "suggested_value": suggested_dense,
                    "reason": f"Context recall is low ({context_recall:.2f}). Increasing dense_k from {current_dense} to {suggested_dense} expands candidate search radius.",
                }
            )
            if not retrieval_cfg.get("query_expansion", False):
                suggestions.append(
                    {
                        "category": "retrieval",
                        "metric": "context_recall",
                        "current_value": context_recall,
                        "target_field": "retrieval.query_expansion",
                        "suggested_value": True,
                        "reason": "Enable query expansion to generate alternative phrasings and improve multi-perspective recall.",
                    }
                )

        # Rule 3: Low Faithfulness -> Hallucination / Drift
        if faithfulness < 0.75:
            current_temp = generation_cfg.get("temperature", 0.2)
            suggestions.append(
                {
                    "category": "generation",
                    "metric": "faithfulness",
                    "current_value": faithfulness,
                    "target_field": "generation.temperature",
                    "suggested_value": 0.0,
                    "reason": f"Faithfulness is below threshold ({faithfulness:.2f}). Lowering temperature to 0.0 enforces strict adherence to retrieved context.",
                }
            )
            if generation_cfg.get("system_prompt_variant") != "precise":
                suggestions.append(
                    {
                        "category": "generation",
                        "metric": "faithfulness",
                        "current_value": faithfulness,
                        "target_field": "generation.system_prompt_variant",
                        "suggested_value": "precise",
                        "reason": "Switching system prompt variant to 'precise' adds strict grounding constraints against unverified claims.",
                    }
                )

        # Rule 4: High Latency -> Optimization for context budgeting
        if latency_ms and latency_ms > 2500:
            current_budget = generation_cfg.get("max_context_tokens", 4000)
            if current_budget > 3000:
                suggestions.append(
                    {
                        "category": "latency",
                        "metric": "total_latency_ms",
                        "current_value": latency_ms,
                        "target_field": "generation.max_context_tokens",
                        "suggested_value": 2500,
                        "reason": f"Total latency ({latency_ms}ms) is elevated. Reducing context token budget to 2500 accelerates TTFT and generation streaming.",
                    }
                )

        if not suggestions:
            suggestions.append(
                {
                    "category": "general",
                    "metric": "overall_score",
                    "current_value": overall,
                    "target_field": None,
                    "suggested_value": None,
                    "reason": "Pipeline is operating within optimal quality thresholds. No parameter adjustments required.",
                }
            )

        return suggestions
