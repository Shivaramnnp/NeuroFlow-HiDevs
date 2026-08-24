import asyncio
import os
import sys

from providers.base import ChatMessage
from providers.openai_provider import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider
from providers.client import NeuroFlowClient
from providers.router import RoutingCriteria


async def test_openai_standalone():
    print("=== Testing OpenAI Provider ===")
    api_key = os.environ.get("OPENAI_API_KEY", "mock-key")
    provider = OpenAIProvider(api_key=api_key)

    print("\n1. Testing embed(['hello world'])...")
    try:
        embeddings = await provider.embed(["hello world"])
        print(f"Embedding generated successfully! Vector dimension: {len(embeddings[0]) if embeddings else 0}")
    except Exception as e:
        print(f"Embed call info (API key may be required for live endpoint): {e}")

    print("\n2. Testing stream([ChatMessage(role='user', content='Say one word')])...")
    messages = [ChatMessage(role="user", content="Say one word")]
    try:
        print("Stream output: ", end="", flush=True)
        async for token in provider.stream(messages):
            print(token, end="", flush=True)
        print()
    except Exception as e:
        print(f"\nStream call info (API key may be required for live endpoint): {e}")


async def test_client_standalone():
    print("\n=== Testing NeuroFlowClient Wrapper & Router ===")
    client = NeuroFlowClient()
    messages = [
        ChatMessage(role="system", content="You are a concise assistant."),
        ChatMessage(role="user", content="Say hello in one word."),
    ]
    criteria = RoutingCriteria(task_type="rag_generation")

    print("\nTesting client.chat streaming...")
    try:
        stream_gen = await client.chat(messages, criteria=criteria, stream=True)
        print("Client stream: ", end="", flush=True)
        async for token in stream_gen:
            print(token, end="", flush=True)
        print()
    except Exception as e:
        print(f"\nClient chat info: {e}")


async def main():
    await test_openai_standalone()
    await test_client_standalone()


if __name__ == "__main__":
    asyncio.run(main())
