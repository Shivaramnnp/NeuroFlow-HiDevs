from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger("neuroflow-model-router")


@dataclass
class RoutingCriteria:
    task_type: str = "rag_generation"          # "rag_generation" | "evaluation" | "embedding" | "classification"
    max_cost_per_call: Optional[float] = None
    require_vision: bool = False
    require_long_context: bool = False         # > 100k tokens
    latency_budget_ms: Optional[int] = None
    prefer_fine_tuned: bool = False


@dataclass
class ModelConfig:
    model: str
    provider: str                               # "openai" | "anthropic"
    vision: bool = False
    context_window: int = 128_000
    cost_per_input_token: float = 0.0          # USD per token
    cost_per_output_token: float = 0.0         # USD per token
    is_judge: bool = False
    is_fine_tuned: bool = False
    fine_tuned_for: Optional[str] = None       # task_type e.g. "rag_generation"
    task_types: List[str] = field(default_factory=list)
    latency_p50_ms: Optional[float] = None

    def estimated_cost(self, input_tokens: int = 1000, output_tokens: int = 500) -> float:
        """Estimate the cost in USD for a call with given token counts."""
        return (input_tokens * self.cost_per_input_token) + (output_tokens * self.cost_per_output_token)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        return cls(**data)


# Default registered models if Redis router:models is empty
DEFAULT_MODELS: List[ModelConfig] = [
    ModelConfig(
        model="gpt-4o-mini",
        provider="openai",
        vision=True,
        context_window=128_000,
        cost_per_input_token=0.15 / 1_000_000,
        cost_per_output_token=0.60 / 1_000_000,
        is_judge=False,
        is_fine_tuned=False,
        task_types=["rag_generation", "classification"],
    ),
    ModelConfig(
        model="claude-3-5-haiku-20241022",
        provider="anthropic",
        vision=True,
        context_window=200_000,
        cost_per_input_token=0.80 / 1_000_000,
        cost_per_output_token=4.00 / 1_000_000,
        is_judge=False,
        is_fine_tuned=False,
        task_types=["rag_generation", "classification"],
    ),
    ModelConfig(
        model="gpt-4o",
        provider="openai",
        vision=True,
        context_window=128_000,
        cost_per_input_token=2.50 / 1_000_000,
        cost_per_output_token=10.00 / 1_000_000,
        is_judge=True,
        is_fine_tuned=False,
        task_types=["rag_generation", "evaluation", "classification"],
    ),
    ModelConfig(
        model="claude-3-5-sonnet-20241022",
        provider="anthropic",
        vision=True,
        context_window=200_000,
        cost_per_input_token=3.00 / 1_000_000,
        cost_per_output_token=15.00 / 1_000_000,
        is_judge=True,
        is_fine_tuned=False,
        task_types=["rag_generation", "evaluation", "classification"],
    ),
]


class ModelRouter:
    """
    Rule-based model router for NeuroFlow.
    Reads registered models from Redis key `router:models` with fallback to default configs.
    Applies criteria: vision, long context (>100k), fine-tuned routing, judge models for evaluation,
    cost filtering, and defaults to the cheapest eligible model.
    """

    REDIS_KEY = "router:models"

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        default_models: Optional[List[ModelConfig]] = None,
    ):
        self.redis_client = redis_client
        self.default_models = list(default_models or DEFAULT_MODELS)

    async def get_registered_models(self) -> List[ModelConfig]:
        """
        Fetch registered models from Redis key `router:models` or fallback to defaults.
        """
        if self.redis_client is not None:
            try:
                raw_data = await self.redis_client.get(self.REDIS_KEY)
                if raw_data:
                    if isinstance(raw_data, bytes):
                        raw_data = raw_data.decode("utf-8")
                    models_list = json.loads(raw_data)
                    return [ModelConfig.from_dict(item) for item in models_list]
            except Exception as err:
                logger.warning(f"Failed to read models from Redis ({self.REDIS_KEY}): {err}")

        return list(self.default_models)

    async def register_models(self, models: List[ModelConfig]) -> None:
        """
        Save/update list of registered models in Redis key `router:models`.
        """
        payload = json.dumps([m.to_dict() for m in models])
        if self.redis_client is not None:
            try:
                await self.redis_client.set(self.REDIS_KEY, payload)
            except Exception as err:
                logger.error(f"Failed to write models to Redis ({self.REDIS_KEY}): {err}")
        self.default_models = list(models)

    async def register_model(self, model: ModelConfig) -> None:
        """
        Add or update a single model config in Redis.
        """
        current_models = await self.get_registered_models()
        # Replace existing or append
        updated = [m for m in current_models if m.model != model.model]
        updated.append(model)
        await self.register_models(updated)

    async def get_eligible_models(
        self,
        criteria: RoutingCriteria,
        estimated_input_tokens: int = 1000,
        estimated_output_tokens: int = 500,
    ) -> List[ModelConfig]:
        """
        Filter and sort registered models according to routing rules.
        """
        models = await self.get_registered_models()
        candidates = list(models)

        # Rule 1: Evaluation task constraint
        # "If task_type='evaluation' -> always use a capable judge model, never a fine-tuned one"
        if criteria.task_type == "evaluation":
            candidates = [m for m in candidates if m.is_judge and not m.is_fine_tuned]
        else:
            # Rule 2: Prefer fine-tuned model for specific task_type
            # "If prefer_fine_tuned=True AND a fine-tuned model is registered for this task_type -> route to it"
            if criteria.prefer_fine_tuned:
                ft_matches = [
                    m for m in candidates
                    if m.is_fine_tuned and (m.fine_tuned_for == criteria.task_type or criteria.task_type in m.task_types)
                ]
                if ft_matches:
                    candidates = ft_matches

        # Rule 3: Vision requirement
        # "If require_vision=True -> route to a vision-capable model"
        if criteria.require_vision:
            candidates = [m for m in candidates if m.vision]

        # Rule 4: Long context requirement
        # "If require_long_context=True -> route to a model with >100k context"
        if criteria.require_long_context:
            candidates = [m for m in candidates if m.context_window > 100_000]

        # Rule 5: Cost limit filter
        # "If max_cost_per_call is set -> filter out models that would exceed it for an estimated call"
        if criteria.max_cost_per_call is not None:
            candidates = [
                m for m in candidates
                if m.estimated_cost(estimated_input_tokens, estimated_output_tokens) <= criteria.max_cost_per_call
            ]

        # Rule 6: Default sort by cheapest estimated cost
        # "Default: route to the cheapest model that satisfies all hard constraints"
        candidates.sort(
            key=lambda m: m.estimated_cost(estimated_input_tokens, estimated_output_tokens)
        )

        return candidates

    async def route(
        self,
        criteria: RoutingCriteria,
        estimated_input_tokens: int = 1000,
        estimated_output_tokens: int = 500,
    ) -> ModelConfig:
        """
        Route to the best model matching the given criteria.
        """
        candidates = await self.get_eligible_models(
            criteria,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
        )

        if not candidates:
            raise ValueError(
                f"No registered model satisfies the routing criteria: {criteria}"
            )

        selected = candidates[0]
        logger.info(
            f"Routed criteria (task={criteria.task_type}, vision={criteria.require_vision}, "
            f"long_ctx={criteria.require_long_context}, ft={criteria.prefer_fine_tuned}) "
            f"-> model '{selected.model}' ({selected.provider})"
        )
        return selected

    async def get_fallback_chain(
        self,
        criteria: RoutingCriteria,
        estimated_input_tokens: int = 1000,
        estimated_output_tokens: int = 500,
    ) -> List[ModelConfig]:
        """
        Get ordered list of candidate models for fallback execution.
        """
        candidates = await self.get_eligible_models(
            criteria,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
        )
        if candidates:
            return candidates
        # If no strict matches, return all registered models sorted by cost as last resort
        models = await self.get_registered_models()
        models.sort(key=lambda m: m.estimated_cost(estimated_input_tokens, estimated_output_tokens))
        return models
