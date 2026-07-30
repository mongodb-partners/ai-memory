"""LLMProvider.chat_stream across providers.

Streaming was added because the sample UI needs tokens as they are produced, and
the interface had no way to express that: ``chat()`` returns a complete string.
The method is deliberately concrete rather than abstract — a provider that
cannot stream inherits a correct one-chunk implementation instead of breaking.

These tests cover the three things that can go wrong: the fallback silently not
firing, a provider yielding non-text events as text, and (for Bedrock) an
exception in the pump thread being swallowed into a truncated answer that looks
like a complete one.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_memory.providers.base import LLMProvider


class _NonStreaming(LLMProvider):
    """A provider that implements only the required surface."""

    def __init__(self, answer: str = "one shot") -> None:
        self.answer = answer
        self.chat_calls: list[dict] = []

    async def chat(self, messages: list[dict], **kwargs) -> str:
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return self.answer

    async def assess_importance(self, content: str) -> float:
        return 0.5

    async def generate_summary(self, content: str, max_length: int = 100) -> str:
        return content[:max_length]


class TestDefaultImplementation:
    """The fallback is what makes this a non-breaking change."""

    async def test_a_provider_without_streaming_yields_one_chunk(self):
        provider = _NonStreaming("the whole answer")
        chunks = [c async for c in provider.chat_stream([{"role": "user"}])]
        assert chunks == ["the whole answer"]

    async def test_the_fallback_forwards_kwargs_to_chat(self):
        """A caller's system prompt must not be dropped by the shim."""
        provider = _NonStreaming()
        _ = [
            c
            async for c in provider.chat_stream(
                [{"role": "user"}], temperature=0.0, max_tokens=64
            )
        ]
        assert provider.chat_calls[0]["kwargs"] == {
            "temperature": 0.0,
            "max_tokens": 64,
        }


class TestBedrockConverseKwargs:
    """The non-streaming path silently discarded every caller override."""

    def _provider(self):
        from agent_memory.providers.bedrock import BedrockLLMProvider

        config = MagicMock()
        config.llm_model = "test-model"
        config.aws_region = "us-east-1"
        config.aws_access_key_id = None
        config.aws_secret_access_key = None
        provider = BedrockLLMProvider.__new__(BedrockLLMProvider)
        provider._config = config
        provider._client = MagicMock()
        provider._unsupported_sampling = set()
        return provider

    async def test_system_prompt_reaches_converse(self):
        provider = self._provider()
        provider._client.converse.return_value = {
            "output": {"message": {"content": [{"text": "hi"}]}}
        }
        await provider.chat(
            [{"role": "user", "content": [{"text": "q"}]}],
            system=[{"text": "You are terse."}],
            inferenceConfig={"maxTokens": 64},
        )
        sent = provider._client.converse.call_args.kwargs
        assert sent["system"] == [{"text": "You are terse."}]
        assert sent["inferenceConfig"] == {"maxTokens": 64}
        assert sent["modelId"] == "test-model"

    async def test_model_id_is_overridable_per_call(self):
        provider = self._provider()
        provider._client.converse.return_value = {
            "output": {"message": {"content": [{"text": "hi"}]}}
        }
        await provider.chat([], modelId="other-model")
        assert provider._client.converse.call_args.kwargs["modelId"] == "other-model"


class TestBedrockStreaming:
    """converse_stream, drained off the event loop."""

    def _provider(self, events):
        from agent_memory.providers.bedrock import BedrockLLMProvider

        config = MagicMock()
        config.llm_model = "test-model"
        provider = BedrockLLMProvider.__new__(BedrockLLMProvider)
        provider._config = config
        provider._client = MagicMock()
        provider._unsupported_sampling = set()
        provider._client.converse_stream.return_value = {"stream": iter(events)}
        return provider

    async def test_only_text_deltas_are_yielded(self):
        """messageStart/metadata/contentBlockStop carry no text and must not
        surface as empty chunks — a consumer appending them is fine, but one
        counting them as tokens is not."""
        provider = self._provider(
            [
                {"messageStart": {"role": "assistant"}},
                {"contentBlockDelta": {"delta": {"text": "Hello"}}},
                {"contentBlockDelta": {"delta": {"text": " world"}}},
                {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}},
                {"contentBlockStop": {}},
                {"messageStop": {"stopReason": "end_turn"}},
                {"metadata": {"usage": {"inputTokens": 5}}},
            ]
        )
        chunks = [c async for c in provider.chat_stream([])]
        assert chunks == ["Hello", " world"]

    async def test_a_failure_mid_stream_is_raised_not_swallowed(self):
        """The failure mode this guards: a truncated answer that looks whole.

        The exception is raised inside a worker thread, so without the explicit
        error channel it would be lost and the consumer would see a clean end of
        stream after two tokens.
        """

        def _explode():
            yield {"contentBlockDelta": {"delta": {"text": "partial"}}}
            raise RuntimeError("bedrock throttled")

        provider = self._provider(_explode())
        seen = []
        with pytest.raises(RuntimeError, match="bedrock throttled"):
            async for chunk in provider.chat_stream([]):
                seen.append(chunk)
        assert seen == ["partial"]

    async def test_stream_kwargs_are_forwarded(self):
        provider = self._provider([{"contentBlockDelta": {"delta": {"text": "x"}}}])
        _ = [c async for c in provider.chat_stream([], system=[{"text": "sys"}])]
        sent = provider._client.converse_stream.call_args.kwargs
        assert sent["system"] == [{"text": "sys"}]
