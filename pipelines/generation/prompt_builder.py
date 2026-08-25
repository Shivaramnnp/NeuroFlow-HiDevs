from __future__ import annotations

import logging
from typing import List, Optional

try:
    from backend.providers.base import ChatMessage
except ImportError:
    from providers.base import ChatMessage

logger = logging.getLogger("neuroflow-prompt-builder")

BASE_SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question using ONLY the provided context.\n"
    "If the context does not contain enough information to answer fully, say so explicitly.\n"
    "For every factual claim, include a citation in the format [Source N].\n"
    "Do not introduce information not present in the context."
)

QUERY_TYPE_INSTRUCTIONS = {
    "factual": "Provide a direct, concise answer. If multiple sources agree, cite all of them.",
    "analytical": "Analyze and synthesize across the provided sources. Identify agreements and contradictions.",
    "comparative": "Organize your response as a structured comparison. Use a table if appropriate.",
    "procedural": "Provide numbered steps. Each step must be cited.",
}


class PromptBuilder:
    """
    Constructs grounded, cited RAG prompts dynamically based on query type classification
    and chain-of-thought reasoning modes.
    """

    def __init__(self, base_prompt: str = BASE_SYSTEM_PROMPT):
        self.base_prompt = base_prompt

    def build_system_prompt(
        self,
        query_type: str = "factual",
        enable_cot: bool = False,
    ) -> str:
        """Construct tailored system prompt with query-type directives and optional reasoning."""
        clean_type = query_type.lower() if query_type else "factual"
        type_instruction = QUERY_TYPE_INSTRUCTIONS.get(
            clean_type,
            QUERY_TYPE_INSTRUCTIONS["factual"],
        )

        prompt_parts = [self.base_prompt, type_instruction]

        # Chain-of-thought support for analytical/comparative queries
        if enable_cot or clean_type in ["analytical", "comparative"]:
            prompt_parts.append(
                "Before presenting your final answer, formulate your reasoning step-by-step inside <think>...</think> tags. "
                "Ensure your final response outside <think> tags is clean, well-structured, and explicitly cited."
            )

        return "\n\n".join(prompt_parts)

    def format_user_message(self, query: str, context: str) -> str:
        """Inject assembled context between <context> tags followed by user query."""
        clean_context = context.strip() if context else "No context provided."
        return f"<context>\n{clean_context}\n</context>\n\n{query.strip()}"

    def build_chat_messages(
        self,
        query: str,
        context: str,
        query_type: str = "factual",
        enable_cot: bool = False,
    ) -> List[ChatMessage]:
        """Build list of ChatMessage objects for provider consumption."""
        system_prompt = self.build_system_prompt(query_type=query_type, enable_cot=enable_cot)
        user_content = self.format_user_message(query=query, context=context)

        return [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_content),
        ]
