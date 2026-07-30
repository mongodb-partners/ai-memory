"""Bedrock sampling-parameter negotiation.

The newest Claude models on Bedrock reject ``temperature`` and ``top_p``: the
request fails with ``ValidationException: `temperature` is deprecated for this
model``. Every earlier Claude accepts them. A static per-model table would be
wrong within a release cycle, so the provider learns from the model's own error,
retries once without the offending key, and remembers the answer.

These tests cover what can actually break: the retry not firing, the retry
firing on unrelated errors, the learned state not being reused (a retry per
call), and — the subtle one — the caller's ``inferenceConfig`` dict being
mutated in place, which would reconfigure every other call sharing it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_memory.providers.bedrock import BedrockLLMProvider


class ValidationException(Exception):
    """Stands in for botocore's generated error class.

    The provider matches on ``type(exc).__name__``, not on a botocore import,
    because the real class is synthesized at runtime from the service model and
    is not importable by path.
    """


def _provider(model: str = "global.anthropic.claude-sonnet-5"):
    config = MagicMock()
    config.llm_model = model
    provider = BedrockLLMProvider.__new__(BedrockLLMProvider)
    provider._config = config
    provider._client = MagicMock()
    provider._unsupported_sampling = set()
    return provider


def _ok(text: str = "hi") -> dict:
    return {"output": {"message": {"content": [{"text": text}]}}}


class TestChatRetry:
    async def test_temperature_rejection_is_retried_without_it(self):
        provider = _provider()
        provider._client.converse.side_effect = [
            ValidationException(
                "The model returned the following errors: "
                "`temperature` is deprecated for this model."
            ),
            _ok("answer"),
        ]

        result = await provider.chat(
            [], inferenceConfig={"maxTokens": 64, "temperature": 0.0}
        )

        assert result == "answer"
        assert provider._client.converse.call_count == 2
        first, second = provider._client.converse.call_args_list
        assert first.kwargs["inferenceConfig"] == {"maxTokens": 64, "temperature": 0.0}
        # maxTokens survives; only the rejected key is dropped.
        assert second.kwargs["inferenceConfig"] == {"maxTokens": 64}

    async def test_top_p_is_recognized_under_its_snake_case_name(self):
        """The error says ``top_p``; the request field is ``topP``."""
        provider = _provider()
        provider._client.converse.side_effect = [
            ValidationException("The model returned: `top_p` is deprecated."),
            _ok(),
        ]

        await provider.chat([], inferenceConfig={"maxTokens": 8, "topP": 0.9})

        assert provider._client.converse.call_args.kwargs["inferenceConfig"] == {
            "maxTokens": 8
        }

    async def test_the_learned_parameter_is_dropped_on_later_calls(self):
        """Without caching, every call pays a doomed request first."""
        provider = _provider()
        provider._client.converse.side_effect = [
            ValidationException("`temperature` is deprecated for this model."),
            _ok(),
            _ok(),
        ]

        await provider.chat([], inferenceConfig={"maxTokens": 8, "temperature": 0.0})
        provider._client.converse.reset_mock()
        provider._client.converse.side_effect = None
        provider._client.converse.return_value = _ok()

        await provider.chat([], inferenceConfig={"maxTokens": 8, "temperature": 0.0})

        assert provider._client.converse.call_count == 1
        assert provider._client.converse.call_args.kwargs["inferenceConfig"] == {
            "maxTokens": 8
        }

    async def test_an_unrelated_validation_error_is_not_retried(self):
        """Retrying a real error would double every failing request and hide the
        cause behind a second, identical exception."""
        provider = _provider()
        provider._client.converse.side_effect = ValidationException(
            "messages: at least one message is required"
        )

        with pytest.raises(ValidationException, match="at least one message"):
            await provider.chat([], inferenceConfig={"maxTokens": 8})

        assert provider._client.converse.call_count == 1

    async def test_a_rejection_with_nothing_to_strip_is_raised(self):
        """The model named a parameter the request did not send — there is no
        recovery, so the error must surface instead of looping."""
        provider = _provider()
        provider._client.converse.side_effect = ValidationException(
            "`temperature` is deprecated for this model."
        )

        with pytest.raises(ValidationException):
            await provider.chat([], inferenceConfig={"maxTokens": 8})

        assert provider._client.converse.call_count == 1

    async def test_the_callers_inference_config_is_not_mutated(self):
        """The failure this prevents: one shared config dict, silently stripped
        of temperature for every other model using it."""
        provider = _provider()
        provider._client.converse.side_effect = [
            ValidationException("`temperature` is deprecated for this model."),
            _ok(),
        ]
        shared = {"maxTokens": 64, "temperature": 0.0}

        await provider.chat([], inferenceConfig=shared)

        assert shared == {"maxTokens": 64, "temperature": 0.0}


class TestResponseTextExtraction:
    """Reasoning-capable models put a ``reasoningContent`` block first."""

    async def test_text_after_a_reasoning_block_is_returned(self):
        """``content[0]["text"]`` raised ``KeyError: 'text'`` on Opus 5 — the
        answer is at index 1, behind the model's deliberation."""
        provider = _provider()
        provider._client.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {"reasoningContent": {"reasoningText": {"text": "hmm..."}}},
                        {"text": "9"},
                    ]
                }
            },
            "stopReason": "end_turn",
        }

        assert await provider.chat([]) == "9"

    async def test_reasoning_text_is_not_included_in_the_answer(self):
        """assess_importance regexes the first integer it finds. If the model's
        deliberation leaked through, it would score on a number from the
        reasoning, not the conclusion."""
        provider = _provider()
        provider._client.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {
                            "reasoningContent": {
                                "reasoningText": {"text": "Maybe 3, maybe 4..."}
                            }
                        },
                        {"text": "9"},
                    ]
                }
            }
        }

        assert await provider.assess_importance("x") == 0.9

    async def test_multiple_text_blocks_are_joined(self):
        provider = _provider()
        provider._client.converse.return_value = {
            "output": {"message": {"content": [{"text": "part one "}, {"text": "two"}]}}
        }

        assert await provider.chat([]) == "part one two"

    async def test_a_budget_exhausted_by_reasoning_returns_empty_not_an_error(self):
        """Observed live: Opus 5 at maxTokens=64 spends the whole budget
        thinking and returns only a reasoning block. Empty is the honest answer;
        raising would make a low token cap look like an outage."""
        provider = _provider()
        provider._client.converse.return_value = {
            "output": {
                "message": {
                    "content": [{"reasoningContent": {"reasoningText": {"text": ""}}}]
                }
            },
            "stopReason": "max_tokens",
        }

        assert await provider.chat([]) == ""


class TestStreamRetry:
    def _stream(self, text: str):
        return {"stream": iter([{"contentBlockDelta": {"delta": {"text": text}}}])}

    async def test_stream_retries_without_the_rejected_parameter(self):
        provider = _provider()
        provider._client.converse_stream.side_effect = [
            ValidationException("`temperature` is deprecated for this model."),
            self._stream("streamed"),
        ]

        chunks = [
            c
            async for c in provider.chat_stream(
                [], inferenceConfig={"maxTokens": 64, "temperature": 0.0}
            )
        ]

        assert chunks == ["streamed"]
        assert provider._client.converse_stream.call_count == 2
        second = provider._client.converse_stream.call_args
        assert second.kwargs["inferenceConfig"] == {"maxTokens": 64}

    async def test_stream_surfaces_an_unrelated_error(self):
        provider = _provider()
        provider._client.converse_stream.side_effect = ValidationException(
            "modelId: not found"
        )

        with pytest.raises(ValidationException, match="not found"):
            async for _ in provider.chat_stream([], inferenceConfig={"maxTokens": 8}):
                pass

        assert provider._client.converse_stream.call_count == 1
