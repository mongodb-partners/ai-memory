"""OpenAI implementations of LLMProvider and EmbeddingProvider.

Uses the official ``openai`` async SDK (``AsyncOpenAI``). An optional
``openai_base_url`` points the same SDK at a compatible gateway (e.g. MongoDB's
Grove), exactly as Voyage uses ``voyage_base_url``. The SDK is an opt-in extra
(``pip install agent-memory[openai]``); ``ProviderManager`` raises ``ConfigError``
with that hint if it is missing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from agent_memory.core.config import MCPConfig
from agent_memory.exceptions import ConfigError
from agent_memory.providers.base import (
    EmbeddingProvider,
    LLMProvider,
    parse_importance,
    render_prompt,
)

try:  # optional dependency
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - exercised via the ConfigError guard
    AsyncOpenAI = None  # type: ignore[assignment]


def _build_client(config: MCPConfig):
    if AsyncOpenAI is None:
        raise ConfigError(
            "Provider 'openai' requires its SDK. "
            "Install it with: pip install agent-memory[openai]"
        )
    kwargs: dict = {"api_key": config.openai_api_key}
    if config.openai_base_url:
        kwargs["base_url"] = config.openai_base_url
    return AsyncOpenAI(**kwargs)


class OpenAILLMProvider(LLMProvider):
    def __init__(self, config: MCPConfig, client=None) -> None:
        self._config = config
        self._model = config.openai_model
        self._client = client if client is not None else _build_client(config)

    async def chat(self, messages: list[dict], **kwargs) -> str:
        response = await self._client.chat.completions.create(
            model=self._model, messages=messages, **kwargs
        )
        return response.choices[0].message.content

    async def chat_stream(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        """Stream text deltas from a chat-completions stream.

        A chunk can carry an empty ``choices`` list (the usage-only final chunk
        when ``stream_options`` is set) and ``delta.content`` is ``None`` on the
        role-announcing first chunk, so both are guarded rather than indexed.
        """
        kwargs.pop("stream", None)
        stream = await self._client.chat.completions.create(
            model=self._model, messages=messages, stream=True, **kwargs
        )
        async for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            text = getattr(choices[0].delta, "content", None)
            if text:
                yield text

    async def assess_importance(self, content: str, prompt: str | None = None) -> float:
        text = render_prompt(
            prompt,
            content,
            "Rate the importance of the following memory on a scale of 1-10, "
            "where 1 is trivial and 10 is critically important. "
            "Respond with ONLY a single integer.\n\n"
            f"Memory: {content}",
        )
        response = await self.complete(text)
        # Handles both the 1-10 scale prompted above and a 0.0-1.0 reply, which a
        # custom prompt may ask for. See `parse_importance`.
        return parse_importance(response)

    async def generate_summary(
        self, content: str, max_length: int = 100, prompt: str | None = None
    ) -> str:
        text = render_prompt(
            prompt,
            content,
            f"Summarize the following text in {max_length} words or fewer. "
            "Be concise and capture the key points.\n\n"
            f"Text: {content}",
        )
        return await self.complete(text)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: MCPConfig, client=None) -> None:
        self._config = config
        self._model = config.openai_embedding_model
        self._client = client if client is not None else _build_client(config)

    async def generate_embedding(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(model=self._model, input=[text])
        return response.data[0].embedding

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(model=self._model, input=texts)
        items = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in items]
