from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

# pyrefly: ignore [missing-import]
from openai import AsyncOpenAI, RateLimitError

try:
    from .base import BaseLLMProvider, ChatMessage, GenerationResult
except ImportError:
    from backend.providers.base import BaseLLMProvider, ChatMessage, GenerationResult

logger = logging.getLogger("neuroflow-openai-provider")


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI / OpenAI-compatible API provider.
    Supports chat completion, streaming, chunked batch embeddings,
    cost tracking with per-model pricing tables, and rate-limit retries.
    """

    # Price table per model in USD per 1,000,000 tokens
    PRICE_TABLE: Dict[str, Dict[str, float]] = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }

    CONTEXT_WINDOWS: Dict[str, int] = {
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        **client_kwargs: Any,
    ):
        self.model = default_model
        self.embedding_model = embedding_model
        self.pricing = dict(self.PRICE_TABLE)
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

        # Initialize AsyncOpenAI client (supports base_url for compatible APIs)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            **client_kwargs,
        )

    @staticmethod
    def _extract_retry_after(exc: Exception) -> Optional[float]:
        """Extract retry_after seconds from RateLimitError headers or attributes."""
        if hasattr(exc, "retry_after") and isinstance(exc.retry_after, (int, float)):
            return float(exc.retry_after)
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", {})
            if "retry-after-ms" in headers:
                try:
                    return float(headers["retry-after-ms"]) / 1000.0
                except (ValueError, TypeError):
                    pass
            if "retry-after" in headers:
                try:
                    return float(headers["retry-after"])
                except (ValueError, TypeError):
                    pass
        return None

    async def _execute_with_retry(self, coro_func, *args, **kwargs) -> Any:
        """
        Execute an async callable, catching RateLimitError and retrying
        up to max_retries times with exponential backoff.
        """
        for attempt in range(self.max_retries + 1):
            try:
                return await coro_func(*args, **kwargs)
            except RateLimitError as exc:
                if attempt == self.max_retries:
                    logger.error(
                        f"OpenAI RateLimitError persisted after {self.max_retries} retries: {exc}"
                    )
                    raise

                retry_after = self._extract_retry_after(exc)
                if retry_after is None or retry_after <= 0:
                    retry_after = self.retry_base_delay * (2 ** attempt)

                logger.warning(
                    f"OpenAI rate limit encountered (attempt {attempt + 1}/{self.max_retries + 1}). "
                    f"Retrying after {retry_after:.2f}s..."
                )
                await asyncio.sleep(retry_after)

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate total USD cost for given token counts."""
        pricing = self.pricing.get(model)
        if not pricing:
            pricing = self.pricing.get(self.model, {"input": 2.50, "output": 10.00})
        input_cost = (input_tokens / 1_000_000.0) * pricing["input"]
        output_cost = (output_tokens / 1_000_000.0) * pricing["output"]
        return input_cost + output_cost

    @property
    def cost_per_input_token(self) -> float:
        pricing = self.pricing.get(self.model, {"input": 2.50, "output": 10.00})
        return pricing["input"] / 1_000_000.0

    @property
    def cost_per_output_token(self) -> float:
        pricing = self.pricing.get(self.model, {"input": 2.50, "output": 10.00})
        return pricing["output"] / 1_000_000.0

    @property
    def context_window(self) -> int:
        return self.CONTEXT_WINDOWS.get(self.model, 128_000)

    async def complete(self, messages: List[ChatMessage], **kwargs) -> GenerationResult:
        """
        Generate completion for chat messages.
        """
        model = kwargs.pop("model", self.model)
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        start_time = time.perf_counter()
        response = await self._execute_with_retry(
            self.client.chat.completions.create,
            model=model,
            messages=formatted_messages,
            **kwargs,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        choice = response.choices[0] if response.choices else None
        content = choice.message.content if (choice and choice.message) else ""
        finish_reason = choice.finish_reason if choice else "stop"

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        cost_usd = self.calculate_cost(model, input_tokens, output_tokens)

        return GenerationResult(
            content=content or "",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            finish_reason=finish_reason or "stop",
        )

    async def stream(self, messages: List[ChatMessage], **kwargs) -> AsyncGenerator[str, None]:
        """
        Yield tokens from stream response using client.chat.completions.create(..., stream=True).
        """
        model = kwargs.pop("model", self.model)
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        response = await self._execute_with_retry(
            self.client.chat.completions.create,
            model=model,
            messages=formatted_messages,
            stream=True,
            **kwargs,
        )

        async for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: int = 100,
        **kwargs,
    ) -> List[List[float]]:
        """
        Generate text embeddings in batches of 100 by default using text-embedding-3-small.
        """
        if not texts:
            return []

        embedding_model = model or self.embedding_model
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self._execute_with_retry(
                self.client.embeddings.create,
                input=batch,
                model=embedding_model,
                **kwargs,
            )
            # Ensure correct order based on index
            sorted_data = sorted(response.data, key=lambda item: item.index)
            all_embeddings.extend([item.embedding for item in sorted_data])

        return all_embeddings