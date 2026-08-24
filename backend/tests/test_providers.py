import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from openai import RateLimitError as OpenAIRateLimitError
from anthropic import RateLimitError as AnthropicRateLimitError
import httpx

from backend.providers.base import ChatMessage, GenerationResult
from backend.providers.openai_provider import OpenAIProvider
from backend.providers.anthropic_provider import AnthropicProvider


@pytest.fixture
def openai_provider():
    return OpenAIProvider(api_key="test-openai-key", retry_base_delay=0.01)


@pytest.fixture
def anthropic_provider():
    return AnthropicProvider(api_key="test-anthropic-key", retry_base_delay=0.01)


# --- OpenAI Provider Tests ---

def test_openai_pricing_and_properties(openai_provider):
    assert openai_provider.model == "gpt-4o"
    # gpt-4o: $2.50 per 1M input tokens => 0.0000025
    assert openai_provider.cost_per_input_token == pytest.approx(2.50 / 1_000_000)
    # gpt-4o: $10.00 per 1M output tokens => 0.000010
    assert openai_provider.cost_per_output_token == pytest.approx(10.00 / 1_000_000)
    assert openai_provider.context_window == 128_000

    # Test gpt-4o cost calculation
    cost_4o = openai_provider.calculate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=500_000)
    assert cost_4o == pytest.approx(2.50 + 5.00)

    # Test gpt-4o-mini cost calculation ($0.15 input / $0.60 output per 1M)
    cost_mini = openai_provider.calculate_cost("gpt-4o-mini", input_tokens=2_000_000, output_tokens=1_000_000)
    assert cost_mini == pytest.approx((2 * 0.15) + (1 * 0.60))


@pytest.mark.asyncio
async def test_openai_complete(openai_provider):
    mock_choice = MagicMock()
    mock_choice.message.content = "Hello from OpenAI"
    mock_choice.finish_reason = "stop"

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    openai_provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Hi!"),
    ]

    result = await openai_provider.complete(messages)

    assert isinstance(result, GenerationResult)
    assert result.content == "Hello from OpenAI"
    assert result.model == "gpt-4o"
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.finish_reason == "stop"
    assert result.latency_ms > 0
    expected_cost = (100 / 1_000_000 * 2.50) + (50 / 1_000_000 * 10.00)
    assert result.cost_usd == pytest.approx(expected_cost)

    openai_provider.client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi!"},
        ],
    )


@pytest.mark.asyncio
async def test_openai_stream(openai_provider):
    # Mock streaming chunks
    chunks = []
    for token in ["Hello", " ", "world", "!"]:
        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content=token))]
        chunks.append(chunk)

    async def mock_chunk_generator():
        for c in chunks:
            yield c

    openai_provider.client.chat.completions.create = AsyncMock(return_value=mock_chunk_generator())

    messages = [ChatMessage(role="user", content="Say hello")]
    collected = []
    async for token in openai_provider.stream(messages):
        collected.append(token)

    assert "".join(collected) == "Hello world!"
    openai_provider.client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Say hello"}],
        stream=True,
    )


@pytest.mark.asyncio
async def test_openai_embed_batching(openai_provider):
    # Test batching with 250 texts and default batch size 100 -> 3 batches
    texts = [f"text-{i}" for i in range(250)]

    def create_mock_embedding_batch(input=None, model=None, **kwargs):
        input_batch = input or []
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(index=idx, embedding=[float(idx), 0.1, 0.2])
            for idx in range(len(input_batch))
        ]
        return mock_response

    openai_provider.client.embeddings.create = AsyncMock(side_effect=create_mock_embedding_batch)

    embeddings = await openai_provider.embed(texts, batch_size=100)

    assert len(embeddings) == 250
    assert openai_provider.client.embeddings.create.await_count == 3
    # Check first batch call had 100 items, last had 50
    first_call_input = openai_provider.client.embeddings.create.await_args_list[0].kwargs["input"]
    assert len(first_call_input) == 100
    last_call_input = openai_provider.client.embeddings.create.await_args_list[2].kwargs["input"]
    assert len(last_call_input) == 50


@pytest.mark.asyncio
async def test_openai_rate_limit_retry_success(openai_provider):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response_429 = httpx.Response(429, headers={"retry-after": "0.01"}, request=request)
    rate_limit_err = OpenAIRateLimitError(
        message="Rate limit reached",
        response=response_429,
        body={"error": {"message": "Rate limit reached"}},
    )

    mock_choice = MagicMock()
    mock_choice.message.content = "Recovered after rate limit"
    mock_choice.finish_reason = "stop"
    mock_success = MagicMock(choices=[mock_choice], usage=MagicMock(prompt_tokens=10, completion_tokens=5))

    # Fail twice with RateLimitError, then succeed
    openai_provider.client.chat.completions.create = AsyncMock(
        side_effect=[rate_limit_err, rate_limit_err, mock_success]
    )

    result = await openai_provider.complete([ChatMessage(role="user", content="Test")])
    assert result.content == "Recovered after rate limit"
    assert openai_provider.client.chat.completions.create.await_count == 3


@pytest.mark.asyncio
async def test_openai_rate_limit_retry_exhausted(openai_provider):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response_429 = httpx.Response(429, headers={"retry-after-ms": "10"}, request=request)
    rate_limit_err = OpenAIRateLimitError(
        message="Rate limit reached",
        response=response_429,
        body={"error": {"message": "Rate limit reached"}},
    )

    openai_provider.client.chat.completions.create = AsyncMock(side_effect=rate_limit_err)

    with pytest.raises(OpenAIRateLimitError):
        await openai_provider.complete([ChatMessage(role="user", content="Test")])

    # Initial attempt + 3 retries = 4 attempts total
    assert openai_provider.client.chat.completions.create.await_count == 4


# --- Anthropic Provider Tests ---

def test_anthropic_pricing_and_properties(anthropic_provider):
    assert anthropic_provider.model == "claude-3-5-sonnet-20241022"
    assert anthropic_provider.cost_per_input_token == pytest.approx(3.00 / 1_000_000)
    assert anthropic_provider.cost_per_output_token == pytest.approx(15.00 / 1_000_000)
    assert anthropic_provider.context_window == 200_000

    cost = anthropic_provider.calculate_cost("claude-3-5-sonnet-20241022", 1_000_000, 1_000_000)
    assert cost == pytest.approx(3.00 + 15.00)


def test_anthropic_role_mapping(anthropic_provider):
    messages = [
        ChatMessage(role="system", content="System instruction 1"),
        ChatMessage(role="user", content="User prompt"),
        ChatMessage(role="system", content="System instruction 2"),
        ChatMessage(role="assistant", content="Assistant reply"),
        ChatMessage(role="user", content="Next user prompt"),
    ]

    system_str, formatted_msgs = anthropic_provider._prepare_messages(messages)

    # System messages concatenated at top-level
    assert system_str == "System instruction 1\n\nSystem instruction 2"
    # System messages omitted from messages array
    assert formatted_msgs == [
        {"role": "user", "content": "User prompt"},
        {"role": "assistant", "content": "Assistant reply"},
        {"role": "user", "content": "Next user prompt"},
    ]


@pytest.mark.asyncio
async def test_anthropic_complete(anthropic_provider):
    mock_block = MagicMock()
    mock_block.text = "Hello from Claude"

    mock_usage = MagicMock()
    mock_usage.input_tokens = 80
    mock_usage.output_tokens = 40

    mock_response = MagicMock()
    mock_response.content = [mock_block]
    mock_response.usage = mock_usage
    mock_response.stop_reason = "end_turn"

    anthropic_provider.client.messages.create = AsyncMock(return_value=mock_response)

    messages = [
        ChatMessage(role="system", content="Be concise."),
        ChatMessage(role="user", content="Hello"),
    ]

    result = await anthropic_provider.complete(messages)

    assert isinstance(result, GenerationResult)
    assert result.content == "Hello from Claude"
    assert result.model == "claude-3-5-sonnet-20241022"
    assert result.input_tokens == 80
    assert result.output_tokens == 40
    assert result.finish_reason == "end_turn"
    assert result.latency_ms > 0
    expected_cost = (80 / 1_000_000 * 3.00) + (40 / 1_000_000 * 15.00)
    assert result.cost_usd == pytest.approx(expected_cost)

    anthropic_provider.client.messages.create.assert_awaited_once_with(
        model="claude-3-5-sonnet-20241022",
        system="Be concise.",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=4096,
    )


@pytest.mark.asyncio
async def test_anthropic_stream(anthropic_provider):
    class MockStreamContext:
        async def __aenter__(self):
            class StreamManager:
                @property
                def text_stream(self):
                    async def _gen():
                        for token in ["Claude", " ", "streams", "!"]:
                            yield token
                    return _gen()
            return StreamManager()

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    anthropic_provider.client.messages.stream = MagicMock(return_value=MockStreamContext())

    messages = [
        ChatMessage(role="system", content="System instruction"),
        ChatMessage(role="user", content="Stream test"),
    ]

    tokens = []
    async for token in anthropic_provider.stream(messages):
        tokens.append(token)

    assert "".join(tokens) == "Claude streams!"
    anthropic_provider.client.messages.stream.assert_called_once_with(
        model="claude-3-5-sonnet-20241022",
        system="System instruction",
        messages=[{"role": "user", "content": "Stream test"}],
        max_tokens=4096,
    )


@pytest.mark.asyncio
async def test_anthropic_embed_raises_not_implemented(anthropic_provider):
    with pytest.raises(NotImplementedError):
        await anthropic_provider.embed(["Some text"])


@pytest.mark.asyncio
async def test_anthropic_rate_limit_retry(anthropic_provider):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response_429 = httpx.Response(429, headers={"retry-after": "0.01"}, request=request)
    rate_limit_err = AnthropicRateLimitError(
        message="Rate limit reached",
        response=response_429,
        body={"error": {"message": "Rate limit reached"}},
    )

    mock_block = MagicMock()
    mock_block.text = "Success after retry"
    mock_success = MagicMock(
        content=[mock_block],
        usage=MagicMock(input_tokens=10, output_tokens=5),
        stop_reason="end_turn",
    )

    anthropic_provider.client.messages.create = AsyncMock(
        side_effect=[rate_limit_err, mock_success]
    )

    result = await anthropic_provider.complete([ChatMessage(role="user", content="Hello")])
    assert result.content == "Success after retry"
    assert anthropic_provider.client.messages.create.await_count == 2


@pytest.mark.asyncio
async def test_openai_embed_empty(openai_provider):
    embeddings = await openai_provider.embed([])
    assert embeddings == []


@pytest.mark.asyncio
async def test_openai_embed_out_of_order_preservation(openai_provider):
    # If API returns responses out of order, sort by index must fix it
    def mock_unordered_embeddings(input=None, model=None, **kwargs):
        mock_resp = MagicMock()
        mock_resp.data = [
            MagicMock(index=1, embedding=[1.0, 1.0]),
            MagicMock(index=0, embedding=[0.0, 0.0]),
        ]
        return mock_resp

    openai_provider.client.embeddings.create = AsyncMock(side_effect=mock_unordered_embeddings)
    embeddings = await openai_provider.embed(["a", "b"])
    assert embeddings == [[0.0, 0.0], [1.0, 1.0]]


@pytest.mark.asyncio
async def test_openai_custom_model_override(openai_provider):
    mock_choice = MagicMock()
    mock_choice.message.content = "gpt-4o-mini response"
    mock_choice.finish_reason = "stop"

    mock_resp = MagicMock(
        choices=[mock_choice],
        usage=MagicMock(prompt_tokens=1_000_000, completion_tokens=1_000_000),
    )
    openai_provider.client.chat.completions.create = AsyncMock(return_value=mock_resp)

    res = await openai_provider.complete(
        [ChatMessage(role="user", content="Hi")],
        model="gpt-4o-mini",
    )
    assert res.model == "gpt-4o-mini"
    # Cost with mini: $0.15 + $0.60 = $0.75
    assert res.cost_usd == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_anthropic_custom_model_override(anthropic_provider):
    mock_block = MagicMock()
    mock_block.text = "Haiku reply"

    mock_resp = MagicMock(
        content=[mock_block],
        usage=MagicMock(input_tokens=1_000_000, output_tokens=1_000_000),
        stop_reason="end_turn",
    )
    anthropic_provider.client.messages.create = AsyncMock(return_value=mock_resp)

    res = await anthropic_provider.complete(
        [ChatMessage(role="user", content="Hi")],
        model="claude-3-5-haiku-20241022",
    )
    assert res.model == "claude-3-5-haiku-20241022"
    # Cost with 3.5 haiku: $0.80 + $4.00 = $4.80
    assert res.cost_usd == pytest.approx(4.80)


# --- Model Router Tests ---

from backend.providers.router import ModelConfig, ModelRouter, RoutingCriteria, DEFAULT_MODELS
from backend.providers.client import NeuroFlowClient, FallbackChain, get_client


@pytest.fixture
def custom_models():
    return [
        ModelConfig(
            model="cheap-text-model",
            provider="openai",
            vision=False,
            context_window=32_000,
            cost_per_input_token=0.05 / 1_000_000,
            cost_per_output_token=0.10 / 1_000_000,
            is_judge=False,
            is_fine_tuned=False,
            task_types=["rag_generation", "classification"],
        ),
        ModelConfig(
            model="ft:custom-rag-model",
            provider="openai",
            vision=False,
            context_window=32_000,
            cost_per_input_token=0.10 / 1_000_000,
            cost_per_output_token=0.20 / 1_000_000,
            is_judge=False,
            is_fine_tuned=True,
            fine_tuned_for="rag_generation",
            task_types=["rag_generation"],
        ),
        ModelConfig(
            model="vision-capable-model",
            provider="openai",
            vision=True,
            context_window=64_000,
            cost_per_input_token=0.50 / 1_000_000,
            cost_per_output_token=1.00 / 1_000_000,
            is_judge=False,
            is_fine_tuned=False,
            task_types=["rag_generation"],
        ),
        ModelConfig(
            model="long-context-model",
            provider="anthropic",
            vision=True,
            context_window=200_000,
            cost_per_input_token=1.00 / 1_000_000,
            cost_per_output_token=2.00 / 1_000_000,
            is_judge=False,
            is_fine_tuned=False,
            task_types=["rag_generation"],
        ),
        ModelConfig(
            model="judge-evaluator-model",
            provider="openai",
            vision=True,
            context_window=128_000,
            cost_per_input_token=2.50 / 1_000_000,
            cost_per_output_token=10.00 / 1_000_000,
            is_judge=True,
            is_fine_tuned=False,
            task_types=["evaluation"],
        ),
        ModelConfig(
            model="ft:eval-judge-disallowed",
            provider="openai",
            vision=True,
            context_window=128_000,
            cost_per_input_token=1.00 / 1_000_000,
            cost_per_output_token=2.00 / 1_000_000,
            is_judge=True,
            is_fine_tuned=True,
            fine_tuned_for="evaluation",
            task_types=["evaluation"],
        ),
    ]


@pytest.fixture
def mock_redis():
    storage = {}

    class MockRedis:
        async def get(self, key):
            return storage.get(key)

        async def set(self, key, val):
            storage[key] = val

        async def incrby(self, key, amount):
            storage[key] = storage.get(key, 0) + amount

        async def incrbyfloat(self, key, amount):
            storage[key] = float(storage.get(key, 0.0)) + float(amount)

        def pipeline(self):
            pipe_self = self
            class Pipeline:
                def __init__(self):
                    self.commands = []
                def incrby(self, key, amount):
                    self.commands.append(("incrby", key, amount))
                    return self
                def incrbyfloat(self, key, amount):
                    self.commands.append(("incrbyfloat", key, amount))
                    return self
                async def execute(self):
                    for cmd, k, amt in self.commands:
                        if cmd == "incrby":
                            await pipe_self.incrby(k, amt)
                        elif cmd == "incrbyfloat":
                            await pipe_self.incrbyfloat(k, amt)
            return Pipeline()

    return MockRedis(), storage


@pytest.mark.asyncio
async def test_router_default_cheapest(custom_models):
    router = ModelRouter(default_models=custom_models)
    criteria = RoutingCriteria(task_type="rag_generation")
    routed = await router.route(criteria)
    # Default without extra constraints should pick cheapest
    assert routed.model == "cheap-text-model"


@pytest.mark.asyncio
async def test_router_require_vision(custom_models):
    router = ModelRouter(default_models=custom_models)
    criteria = RoutingCriteria(task_type="rag_generation", require_vision=True)
    routed = await router.route(criteria)
    assert routed.vision is True
    # Cheapest vision model among candidates
    assert routed.model == "vision-capable-model"


@pytest.mark.asyncio
async def test_router_require_long_context(custom_models):
    router = ModelRouter(default_models=custom_models)
    criteria = RoutingCriteria(task_type="rag_generation", require_long_context=True)
    routed = await router.route(criteria)
    assert routed.context_window > 100_000
    assert routed.model == "long-context-model"


@pytest.mark.asyncio
async def test_router_prefer_fine_tuned(custom_models):
    router = ModelRouter(default_models=custom_models)
    criteria = RoutingCriteria(task_type="rag_generation", prefer_fine_tuned=True)
    routed = await router.route(criteria)
    assert routed.is_fine_tuned is True
    assert routed.model == "ft:custom-rag-model"


@pytest.mark.asyncio
async def test_router_evaluation_judge_rule(custom_models):
    router = ModelRouter(default_models=custom_models)
    # Even if prefer_fine_tuned=True, task_type='evaluation' must use judge and never fine-tuned
    criteria = RoutingCriteria(task_type="evaluation", prefer_fine_tuned=True)
    routed = await router.route(criteria)
    assert routed.is_judge is True
    assert routed.is_fine_tuned is False
    assert routed.model == "judge-evaluator-model"


@pytest.mark.asyncio
async def test_router_max_cost_per_call(custom_models):
    router = ModelRouter(default_models=custom_models)
    # For 1M tokens in + 1M out:
    # cheap-text-model cost: $0.15
    # ft:custom-rag-model cost: $0.30
    # vision-capable-model cost: $1.50
    criteria = RoutingCriteria(
        require_vision=True,
        max_cost_per_call=0.000000001,  # extremely low, should filter all
    )
    with pytest.raises(ValueError, match="No registered model satisfies the routing criteria"):
        await router.route(criteria, estimated_input_tokens=1000, estimated_output_tokens=500)


@pytest.mark.asyncio
async def test_router_redis_sync(mock_redis, custom_models):
    redis_instance, storage = mock_redis
    router = ModelRouter(redis_client=redis_instance, default_models=custom_models)

    # Initial register
    await router.register_models(custom_models)
    assert ModelRouter.REDIS_KEY in storage

    # Register a new fine-tuned model dynamically
    new_ft = ModelConfig(
        model="ft:gpt-4o-mini:dynamic-fine-tune",
        provider="openai",
        vision=True,
        context_window=128_000,
        cost_per_input_token=0.01 / 1_000_000,
        cost_per_output_token=0.02 / 1_000_000,
        is_judge=False,
        is_fine_tuned=True,
        fine_tuned_for="classification",
        task_types=["classification"],
    )
    await router.register_model(new_ft)

    # Verify retrieval from Redis
    models = await router.get_registered_models()
    assert any(m.model == "ft:gpt-4o-mini:dynamic-fine-tune" for m in models)

    # Route for classification with prefer_fine_tuned
    crit = RoutingCriteria(task_type="classification", prefer_fine_tuned=True)
    routed = await router.route(crit)
    assert routed.model == "ft:gpt-4o-mini:dynamic-fine-tune"


# --- NeuroFlowClient & FallbackChain Tests ---

@pytest.mark.asyncio
async def test_neuroflow_client_chat_and_telemetry(mock_redis, openai_provider, anthropic_provider):
    redis_instance, storage = mock_redis
    client = NeuroFlowClient(
        openai_provider=openai_provider,
        anthropic_provider=anthropic_provider,
        redis_client=redis_instance,
    )

    mock_choice = MagicMock()
    mock_choice.message.content = "Client chat completion ok"
    mock_choice.finish_reason = "stop"
    mock_resp = MagicMock(
        choices=[mock_choice],
        usage=MagicMock(prompt_tokens=100, completion_tokens=50),
    )
    openai_provider.client.chat.completions.create = AsyncMock(return_value=mock_resp)

    messages = [ChatMessage(role="user", content="Test client call")]

    with patch("backend.providers.client.tracer") as mock_tracer:
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        result = await client.chat(messages)

        assert result.content == "Client chat completion ok"
        assert result.input_tokens == 100
        assert result.output_tokens == 50

        # Verify OpenTelemetry span attributes
        mock_span.set_attribute.assert_any_call("model", "gpt-4o-mini")
        mock_span.set_attribute.assert_any_call("input_tokens", 100)
        mock_span.set_attribute.assert_any_call("output_tokens", 50)
        mock_span.set_attribute.assert_any_call("provider", "openai")
        mock_span.set_attribute.assert_any_call("finish_reason", "stop")

    # Verify Redis metrics
    calls_val = storage.get("metrics:model:gpt-4o-mini:calls")
    cost_val = storage.get("metrics:model:gpt-4o-mini:cost_usd")
    assert calls_val == 1
    assert cost_val > 0


@pytest.mark.asyncio
async def test_neuroflow_client_stream(mock_redis, openai_provider, anthropic_provider):
    redis_instance, storage = mock_redis
    client = NeuroFlowClient(
        openai_provider=openai_provider,
        anthropic_provider=anthropic_provider,
        redis_client=redis_instance,
    )

    chunks = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content=t))])
        for t in ["Streaming", " ", "works", "!"]
    ]
    async def mock_gen():
        for c in chunks:
            yield c

    openai_provider.client.chat.completions.create = AsyncMock(return_value=mock_gen())

    messages = [ChatMessage(role="user", content="Stream please")]
    stream_gen = await client.chat(messages, stream=True)
    tokens = []
    async for t in stream_gen:
        tokens.append(t)

    assert "".join(tokens) == "Streaming works!"


@pytest.mark.asyncio
async def test_fallback_chain_failover(mock_redis, openai_provider, anthropic_provider):
    redis_instance, storage = mock_redis
    client = NeuroFlowClient(
        openai_provider=openai_provider,
        anthropic_provider=anthropic_provider,
        redis_client=redis_instance,
    )

    # Primary provider (OpenAI) fails with connection error
    openai_provider.client.chat.completions.create = AsyncMock(
        side_effect=RuntimeError("OpenAI API service outage 503")
    )

    # Secondary provider (Anthropic) succeeds
    mock_block = MagicMock()
    mock_block.text = "Fallback to Claude succeeded"
    mock_claude_resp = MagicMock(
        content=[mock_block],
        usage=MagicMock(input_tokens=20, output_tokens=10),
        stop_reason="end_turn",
    )
    anthropic_provider.client.messages.create = AsyncMock(return_value=mock_claude_resp)

    candidates = [
        ModelConfig(model="gpt-4o-mini", provider="openai", cost_per_input_token=0.0001, cost_per_output_token=0.0002),
        ModelConfig(model="claude-3-5-haiku-20241022", provider="anthropic", cost_per_input_token=0.0001, cost_per_output_token=0.0002),
    ]

    result = await client.fallback_chain.complete(candidates, [ChatMessage(role="user", content="Hi")])
    assert result.content == "Fallback to Claude succeeded"
    assert result.model == "claude-3-5-haiku-20241022"


@pytest.mark.asyncio
async def test_fallback_chain_all_fail(mock_redis, openai_provider, anthropic_provider):
    redis_instance, storage = mock_redis
    client = NeuroFlowClient(
        openai_provider=openai_provider,
        anthropic_provider=anthropic_provider,
        redis_client=redis_instance,
    )

    openai_provider.client.chat.completions.create = AsyncMock(
        side_effect=RuntimeError("OpenAI failed")
    )
    anthropic_provider.client.messages.create = AsyncMock(
        side_effect=RuntimeError("Anthropic failed")
    )

    candidates = [
        ModelConfig(model="gpt-4o-mini", provider="openai"),
        ModelConfig(model="claude-3-5-haiku-20241022", provider="anthropic"),
    ]

    with pytest.raises(RuntimeError, match="All providers in FallbackChain failed"):
        await client.fallback_chain.complete(candidates, [ChatMessage(role="user", content="Hi")])
