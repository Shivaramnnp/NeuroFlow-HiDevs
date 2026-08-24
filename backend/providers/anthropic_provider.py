from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

# pyrefly: ignore [missing-import]
from anthropic import AsyncAnthropic, RateLimitError

try:
    from .base import BaseLLMProvider, ChatMessage, GenerationResult
except ImportError:
    from backend.providers.base import BaseLLMProvider, ChatMessage, GenerationResult

logger = logging.getLogger("neuroflow-anthropic-provider")


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude API provider.
    Supports chat completion, streaming, cost tracking with per-model pricing tables,
    rate-limit retries with exponential backoff, and correct message role mapping
    (top-level system parameter separation).
    """

    # Price table per model in USD per 1,000,000 tokens
    PRICE_TABLE: Dict[str, Dict[str, float]] = {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }

    CONTEXT_WINDOWS: Dict[str, int] = {
        "claude-3-5-sonnet-20241022": 200_000,
        "claude-3-5-haiku-20241022": 200_000,
        "claude-3-opus-20240229": 200_000,
        "claude-3-sonnet-20240229": 200_000,
        "claude-3-haiku-20240307": 200_000,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "claude-3-5-sonnet-20241022",
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        default_max_tokens: int = 4096,
        **client_kwargs: Any,
    ):
        self.model = default_model
        self.pricing = dict(self.PRICE_TABLE)
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.default_max_tokens = default_max_tokens

        # Initialize AsyncAnthropic client
        self.client = AsyncAnthropic(
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
                        f"Anthropic RateLimitError persisted after {self.max_retries} retries: {exc}"
                    )
                    raise

                retry_after = self._extract_retry_after(exc)
                if retry_after is None or retry_after <= 0:
                    retry_after = self.retry_base_delay * (2 ** attempt)

                logger.warning(
                    f"Anthropic rate limit encountered (attempt {attempt + 1}/{self.max_retries + 1}). "
                    f"Retrying after {retry_after:.2f}s..."
                )
                await asyncio.sleep(retry_after)

    def _prepare_messages(
        self, messages: List[ChatMessage]
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Extract system messages into a top-level system prompt string
        and return formatted user/assistant messages for Anthropic Messages API.
        """
        system_prompts: List[str] = []
        anthropic_messages: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                if isinstance(msg.content, str):
                    system_prompts.append(msg.content)
                elif isinstance(msg.content, list):
                    # Multi-modal / block system messages if applicable
                    for block in msg.content:
                        if isinstance(block, str):
                            system_prompts.append(block)
                        elif isinstance(block, dict) and "text" in block:
                            system_prompts.append(block["text"])
            else:
                # Anthropic expects 'user' or 'assistant'
                anthropic_messages.append({"role": msg.role, "content": msg.content})

        system_str = "\n\n".join(system_prompts) if system_prompts else None
        return system_str, anthropic_messages

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate total USD cost for given token counts."""
        pricing = self.pricing.get(model)
        if not pricing:
            pricing = self.pricing.get(self.model, {"input": 3.00, "output": 15.00})
        input_cost = (input_tokens / 1_000_000.0) * pricing["input"]
        output_cost = (output_tokens / 1_000_000.0) * pricing["output"]
        return input_cost + output_cost

    @property
    def cost_per_input_token(self) -> float:
        pricing = self.pricing.get(self.model, {"input": 3.00, "output": 15.00})
        return pricing["input"] / 1_000_000.0

    @property
    def cost_per_output_token(self) -> float:
        pricing = self.pricing.get(self.model, {"input": 3.00, "output": 15.00})
        return pricing["output"] / 1_000_000.0

    @property
    def context_window(self) -> int:
        return self.CONTEXT_WINDOWS.get(self.model, 200_000)

    async def complete(self, messages: List[ChatMessage], **kwargs) -> GenerationResult:
        """
        Generate completion for chat messages using Anthropic Messages API.
        """
        model = kwargs.pop("model", self.model)
        max_tokens = kwargs.pop("max_tokens", self.default_max_tokens)
        system_str, formatted_messages = self._prepare_messages(messages)

        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if system_str:
            request_kwargs["system"] = system_str

        start_time = time.perf_counter()
        response = await self._execute_with_retry(
            self.client.messages.create,
            **request_kwargs,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Extract text from content blocks
        content_text = ""
        if hasattr(response, "content") and response.content:
            text_blocks = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_blocks.append(block.text)
                elif isinstance(block, dict) and "text" in block:
                    text_blocks.append(block["text"])
            content_text = "".join(text_blocks)

        finish_reason = getattr(response, "stop_reason", "end_turn") or "end_turn"

        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "input_tokens", 0)
            output_tokens = getattr(response.usage, "output_tokens", 0)

        cost_usd = self.calculate_cost(model, input_tokens, output_tokens)

        return GenerationResult(
            content=content_text,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            finish_reason=finish_reason,
        )

    async def stream(self, messages: List[ChatMessage], **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream tokens asynchronously using Anthropic Messages API.
        """
        model = kwargs.pop("model", self.model)
        max_tokens = kwargs.pop("max_tokens", self.default_max_tokens)
        system_str, formatted_messages = self._prepare_messages(messages)

        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if system_str:
            request_kwargs["system"] = system_str

        # Execute stream connection with rate limit retry
        async def _open_stream():
            return self.client.messages.stream(**request_kwargs)

        stream_ctx = await self._execute_with_retry(_open_stream)
        async with stream_ctx as stream_manager:
            async for text in stream_manager.text_stream:
                yield text

    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Anthropic Claude does not natively provide an embedding endpoint.
        """
        raise NotImplementedError(
            "Anthropic does not offer a native text embedding endpoint. "
            "Please use OpenAIProvider or an external embedding model."
        )
