"""Tests for the Anthropic LLM provider. REQ-E-052.

The AsyncAnthropic client is injected as a mock — no live API calls.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_memory.config import MemoryConfig
from agent_memory.providers.anthropic import AnthropicLLMProvider


def _config(**overrides) -> MemoryConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MemoryConfig(**defaults, _env_file=None)


def _client(text: str):
    client = SimpleNamespace()
    client.messages = SimpleNamespace()
    client.messages.create = AsyncMock(
        return_value=SimpleNamespace(content=[SimpleNamespace(text=text)])
    )
    return client


class TestAnthropicLLMProvider:
    async def test_chat_calls_client(self):
        # TC-PROV-003
        p = AnthropicLLMProvider(_config(anthropic_api_key="ak"), client=_client("hi there"))
        out = await p.chat([{"role": "user", "content": "hi"}])
        assert out == "hi there"

    async def test_assess_importance_parses_score(self):
        p = AnthropicLLMProvider(_config(anthropic_api_key="ak"), client=_client("9"))
        assert await p.assess_importance("x") == pytest.approx(0.9)

    async def test_generate_summary(self):
        p = AnthropicLLMProvider(_config(anthropic_api_key="ak"), client=_client("sum"))
        assert await p.generate_summary("text") == "sum"

    def test_base_url_forwarded(self, monkeypatch):
        captured = {}

        class FakeAsyncAnthropic:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        import agent_memory.providers.anthropic as mod
        monkeypatch.setattr(mod, "AsyncAnthropic", FakeAsyncAnthropic)
        AnthropicLLMProvider(_config(anthropic_api_key="ak", anthropic_base_url="https://grove/anthropic"))
        assert captured.get("base_url") == "https://grove/anthropic"
        assert captured.get("api_key") == "ak"
