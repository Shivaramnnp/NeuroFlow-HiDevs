from __future__ import annotations

import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

# pyrefly: ignore [missing-import]
import redis.asyncio as redis
# pyrefly: ignore [missing-import]
from opentelemetry import trace

try:
    from backend.config import settings
except ImportError:
    try:
        from config import settings
    except ImportError:
        settings = None

from .anthropic_provider import AnthropicProvider
from .base import BaseLLMProvider, ChatMessage, GenerationResult
from .openai_provider import OpenAIProvider
from .router import ModelConfig, ModelRouter, RoutingCriteria

logger = logging.getLogger("neuroflow-client")
tracer = trace.get_tracer("neuroflow-client")


class FallbackChain:
    """
    Executes completion and streaming across an ordered sequence of candidate models/providers.
    If the primary provider fails, automatically falls back to subsequent candidates in the chain.
    """

    def __init__(self, client: "NeuroFlowClient"):
        self.client = client

    async def complete(
        self,
        candidates: List[ModelConfig],
        messages: List[ChatMessage],
        **kwargs,
    ) -> GenerationResult:
        errors = []
        for config in candidates:
            provider = self.client.get_provider(config.provider)
            if provider is None:
                continue

            try:
                logger.info(f"Attempting completion via {config.provider} ({config.model})...")
                result = await provider.complete(messages, model=config.model, **kwargs)
                return result
            except Exception as exc:
                logger.warning(
                    f"Candidate model '{config.model}' ({config.provider}) failed: {exc}. Trying next fallback..."
                )
                errors.append((config.model, exc))

        error_details = "; ".join(f"[{m}: {e}]" for m, e in errors)
        raise RuntimeError(f"All providers in FallbackChain failed: {error_details}")

    async def stream(
        self,
        candidates: List[ModelConfig],
        messages: List[ChatMessage],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        errors = []
        for config in candidates:
            provider = self.client.get_provider(config.provider)
            if provider is None:
                continue

            try:
                logger.info(f"Attempting stream via {config.provider} ({config.model})...")
                # Try opening the stream generator
                stream_gen = provider.stream(messages, model=config.model, **kwargs)
                # Test the first yield or return generator
                async for chunk in stream_gen:
                    yield chunk
                return
            except Exception as exc:
                logger.warning(
                    f"Candidate stream for model '{config.model}' failed: {exc}. Trying next fallback..."
                )
                errors.append((config.model, exc))

        error_details = "; ".join(f"[{m}: {e}]" for m, e in errors)
        raise RuntimeError(f"All providers in FallbackChain stream failed: {error_details}")


class NeuroFlowClient:
    """
    Unified client for LLM providers in NeuroFlow.
    Features:
    - Centralized router matching criteria to optimal models
    - OpenTelemetry spans on every provider call
    - Redis call count & cost metrics tracking
    - FallbackChain for high reliability and failover
    """

    _instance: Optional["NeuroFlowClient"] = None

    def __init__(
        self,
        openai_provider: Optional[OpenAIProvider] = None,
        anthropic_provider: Optional[AnthropicProvider] = None,
        router: Optional[ModelRouter] = None,
        redis_client: Optional[Any] = None,
    ):
        self.openai_provider = openai_provider or OpenAIProvider()
        self.anthropic_provider = anthropic_provider or AnthropicProvider()
        self.providers: Dict[str, BaseLLMProvider] = {
            "openai": self.openai_provider,
            "anthropic": self.anthropic_provider,
        }

        self.redis_client = redis_client
        self.router = router or ModelRouter(redis_client=self.redis_client)
        self.fallback_chain = FallbackChain(self)

    @classmethod
    def get_instance(cls, **kwargs) -> "NeuroFlowClient":
        """Singleton accessor for NeuroFlowClient."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (primarily for testing)."""
        cls._instance = None

    async def _init_redis_if_needed(self) -> Optional[Any]:
        """Lazy initialization of Redis client if not injected."""
        if self.redis_client is None and settings is not None:
            try:
                self.redis_client = redis.from_url(
                    settings.redis_url,
                    decode_responses=False,
                    socket_timeout=2.0,
                )
                self.router.redis_client = self.redis_client
            except Exception as err:
                logger.warning(f"Could not connect to Redis: {err}")
        return self.redis_client

    def get_provider(self, provider_name: str) -> Optional[BaseLLMProvider]:
        """Retrieve provider by name."""
        return self.providers.get(provider_name.lower())

    def get_provider_for_model(self, model_name: str) -> BaseLLMProvider:
        """Infer provider from model name if unknown."""
        if model_name.startswith("claude"):
            return self.anthropic_provider
        return self.openai_provider

    async def record_metrics(
        self,
        model_name: str,
        cost_usd: float,
        calls: int = 1,
    ) -> None:
        """
        Increment call counts and costs in Redis for the given model:
        - metrics:model:{model_name}:calls
        - metrics:model:{model_name}:cost_usd
        """
        await self._init_redis_if_needed()
        if self.redis_client is not None:
            calls_key = f"metrics:model:{model_name}:calls"
            cost_key = f"metrics:model:{model_name}:cost_usd"
            try:
                pipeline = self.redis_client.pipeline()
                pipeline.incrby(calls_key, calls)
                pipeline.incrbyfloat(cost_key, cost_usd)
                await pipeline.execute()
            except Exception as err:
                logger.warning(f"Failed to record Redis metrics for '{model_name}': {err}")

    def _estimate_tokens(self, messages: List[ChatMessage]) -> int:
        """Estimate token count from message text."""
        total_chars = 0
        for msg in messages:
            if isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                total_chars += sum(len(str(b)) for b in msg.content)
        return max(1, total_chars // 4)

    async def chat(
        self,
        messages: List[ChatMessage],
        criteria: Optional[RoutingCriteria] = None,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[GenerationResult, AsyncGenerator[str, None]]:
        """
        Execute chat completion or streaming with automatic routing,
        OpenTelemetry tracing, Redis metrics, and FallbackChain.
        """
        await self._init_redis_if_needed()
        estimated_input_tokens = self._estimate_tokens(messages)
        estimated_output_tokens = kwargs.get("max_tokens", 500)

        # Resolve candidate models for FallbackChain
        candidates: List[ModelConfig]
        if model is not None:
            provider_name = "anthropic" if model.startswith("claude") else "openai"
            candidates = [
                ModelConfig(
                    model=model,
                    provider=provider_name,
                    vision=True,
                    context_window=128_000,
                )
            ]
        else:
            routing_crit = criteria or RoutingCriteria()
            candidates = await self.router.get_fallback_chain(
                routing_crit,
                estimated_input_tokens=estimated_input_tokens,
                estimated_output_tokens=estimated_output_tokens,
            )

        if stream:
            return self._stream_with_telemetry(candidates, messages, **kwargs)

        with tracer.start_as_current_span("neuroflow.llm.chat") as span:
            result = await self.fallback_chain.complete(candidates, messages, **kwargs)

            # Record OpenTelemetry Span Attributes
            span.set_attribute("model", result.model)
            span.set_attribute("input_tokens", result.input_tokens)
            span.set_attribute("output_tokens", result.output_tokens)
            span.set_attribute("cost_usd", result.cost_usd)
            span.set_attribute("latency_ms", result.latency_ms)
            span.set_attribute("finish_reason", result.finish_reason)

            # Infer provider
            provider_name = "anthropic" if result.model.startswith("claude") else "openai"
            span.set_attribute("provider", provider_name)

            # Record Redis metrics
            await self.record_metrics(result.model, result.cost_usd, calls=1)

            return result

    async def _stream_with_telemetry(
        self,
        candidates: List[ModelConfig],
        messages: List[ChatMessage],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Wrap streaming in OpenTelemetry span."""
        with tracer.start_as_current_span("neuroflow.llm.stream") as span:
            primary_model = candidates[0].model if candidates else "unknown"
            span.set_attribute("model", primary_model)

            async for token in self.fallback_chain.stream(candidates, messages, **kwargs):
                yield token

            # Increment call count for stream
            await self.record_metrics(primary_model, cost_usd=0.0, calls=1)

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: int = 100,
        **kwargs,
    ) -> List[List[float]]:
        """
        Generate embeddings with OpenTelemetry instrumentation.
        """
        with tracer.start_as_current_span("neuroflow.llm.embed") as span:
            start_time = time.perf_counter()
            embeddings = await self.openai_provider.embed(
                texts,
                model=model,
                batch_size=batch_size,
                **kwargs,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            used_model = model or self.openai_provider.embedding_model
            span.set_attribute("model", used_model)
            span.set_attribute("input_count", len(texts))
            span.set_attribute("latency_ms", latency_ms)
            span.set_attribute("provider", "openai")

            return embeddings


def get_client(**kwargs) -> NeuroFlowClient:
    """Convenience accessor for NeuroFlowClient singleton."""
    return NeuroFlowClient.get_instance(**kwargs)
