"""Anthropic implementation of LLMProvider.

Uses the official ``anthropic`` async SDK (``AsyncAnthropic``). Anthropic has no
embeddings API, so this module provides an LLM provider only; pair it with an
OpenAI/Voyage/Bedrock embedder. An optional ``anthropic_base_url`` points the
SDK at a compatible gateway (e.g. Grove). The SDK is an opt-in extra
(``pip install agent-memory[anthropic]``).
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from agent_memory.core.config import MCPConfig
from agent_memory.exceptions import ConfigError
from agent_memory.providers.base import LLMProvider

try:  # optional dependency
    from anthropic import AsyncAnthropic
except ImportError:  # pragma: no cover - exercised via the ConfigError guard
    AsyncAnthropic = None  # type: ignore[assignment]

_MAX_TOKENS = 1024


def _build_client(config: MCPConfig):
    if AsyncAnthropic is None:
        raise ConfigError(
            "Provider 'anthropic' requires its SDK. "
            "Install it with: pip install agent-memory[anthropic]"
        )
    kwargs: dict = {"api_key": config.anthropic_api_key}
    if config.anthropic_base_url:
        kwargs["base_url"] = config.anthropic_base_url
    return AsyncAnthropic(**kwargs)


class AnthropicLLMProvider(LLMProvider):
    def __init__(self, config: MCPConfig, client=None) -> None:
        self._config = config
        self._model = config.anthropic_model
        self._client = client if client is not None else _build_client(config)

    async def chat(self, messages: list[dict], **kwargs) -> str:
        kwargs.setdefault("max_tokens", _MAX_TOKENS)
        response = await self._client.messages.create(
            model=self._model, messages=messages, **kwargs
        )
        return response.content[0].text

    async def chat_stream(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        """Stream text deltas via the SDK's ``messages.stream`` helper.

        ``text_stream`` yields text only, which is exactly this method's
        contract — no filtering of tool-use or thinking blocks needed here.
        """
        kwargs.setdefault("max_tokens", _MAX_TOKENS)
        async with self._client.messages.stream(
            model=self._model, messages=messages, **kwargs
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def assess_importance(self, content: str) -> float:
        text = (
            "Rate the importance of the following memory on a scale of 1-10, "
            "where 1 is trivial and 10 is critically important. "
            "Respond with ONLY a single integer.\n\n"
            f"Memory: {content}"
        )
        response = await self.chat([{"role": "user", "content": text}])
        match = re.search(r"\d+", response or "")
        if match:
            return max(0.1, min(1.0, int(match.group()) / 10.0))
        return 0.5

    async def generate_summary(self, content: str, max_length: int = 100) -> str:
        text = (
            f"Summarize the following text in {max_length} words or fewer. "
            "Be concise and capture the key points.\n\n"
            f"Text: {content}"
        )
        return await self.chat([{"role": "user", "content": text}])
