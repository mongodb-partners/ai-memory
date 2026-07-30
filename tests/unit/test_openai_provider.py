"""Tests for OpenAI providers. REQ-E-050, REQ-E-051.

The AsyncOpenAI client is injected as a mock — no live API calls.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_memory.config import MemoryConfig
from agent_memory.providers.openai import (
    OpenAIEmbeddingProvider,
    OpenAILLMProvider,
)


def _config(**overrides) -> MemoryConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MemoryConfig(**defaults, _env_file=None)


def _chat_client(text: str):
    client = SimpleNamespace()
    client.chat = SimpleNamespace()
    client.chat.completions = SimpleNamespace()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )
    )
    return client


def _embed_client(vectors: list[list[float]]):
    client = SimpleNamespace()
    client.embeddings = SimpleNamespace()
    client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(embedding=v, index=i) for i, v in enumerate(vectors)]
        )
    )
    return client


class TestOpenAILLMProvider:
    async def test_chat_calls_client(self):
        # TC-PROV-001
        p = OpenAILLMProvider(_config(openai_api_key="sk"), client=_chat_client("hello"))
        out = await p.chat([{"role": "user", "content": "hi"}])
        assert out == "hello"

    async def test_assess_importance_parses_score(self):
        p = OpenAILLMProvider(_config(openai_api_key="sk"), client=_chat_client("8"))
        score = await p.assess_importance("important thing")
        assert score == pytest.approx(0.8)

    async def test_generate_summary(self):
        p = OpenAILLMProvider(_config(openai_api_key="sk"), client=_chat_client("a summary"))
        assert await p.generate_summary("long text") == "a summary"

    def test_base_url_forwarded_to_client(self, monkeypatch):
        # TC-PROV-001: Grove gateway via base_url
        captured = {}

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        import agent_memory.providers.openai as mod
        monkeypatch.setattr(mod, "AsyncOpenAI", FakeAsyncOpenAI)
        OpenAILLMProvider(_config(openai_api_key="sk", openai_base_url="https://grove/v1"))
        assert captured.get("base_url") == "https://grove/v1"
        assert captured.get("api_key") == "sk"


class TestOpenAIEmbeddingProvider:
    async def test_generate_embedding_small(self):
        # TC-PROV-002
        p = OpenAIEmbeddingProvider(
            _config(openai_embedding_model="text-embedding-3-small"),
            client=_embed_client([[0.1] * 1536]),
        )
        vec = await p.generate_embedding("hi")
        assert len(vec) == 1536

    async def test_generate_embeddings_batch_large_model(self):
        p = OpenAIEmbeddingProvider(
            _config(openai_embedding_model="text-embedding-3-large"),
            client=_embed_client([[0.1] * 3072, [0.2] * 3072]),
        )
        vecs = await p.generate_embeddings_batch(["a", "b"])
        assert len(vecs) == 2 and len(vecs[0]) == 3072
