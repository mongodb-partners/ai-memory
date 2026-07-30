"""Every LLM provider must accept the arguments enrichment actually passes.

This file exists because the whole suite agreed the code worked while two of the
three providers could not run enrichment at all.

`EnrichmentWorker._process_standard_enrichment` calls
`assess_importance(content, prompt=...)`. Only Bedrock declared `prompt`, so on
OpenAI and Anthropic that call raised `TypeError` at argument binding — before any
network I/O, on every memory. The worker catches `Exception`, increments
`enrichment_retries`, and after `enrichment_max_retries` marks the memory
`failed`. No importance, no summary, no error surfaced to the caller.

Two things conspired to hide it:

1. The `prompt` branch looked conditional. `if importance_prompt:` reads like an
   opt-in, but `PromptLibrary.get_prompt` falls back to `_HARDCODED_PROMPTS` on
   every path — experiment disabled, DB miss, or library exception — so it returns
   a non-empty template essentially always. The branch is unconditional in
   practice.
2. The tests mocked it away. `_make_providers()` in `test_enrichment.py` builds
   `providers.llm = AsyncMock()`, and an `AsyncMock` accepts any keyword argument.
   The mock was strictly more permissive than every real implementation, so a
   green suite proved nothing about the call it was standing in for.

So these tests deliberately avoid mocks for the signature checks: they inspect the
real classes, and they call real (network-stubbed) instances. A test that asserts a
contract against a mock of that contract cannot fail.

REQ-E-140 (provider prompt contract), REQ-E-141 (template rendering is safe).
"""

from __future__ import annotations

import inspect

import pytest

from agent_memory.providers.base import LLMProvider, render_prompt

# Every concrete provider, imported directly. The optional SDKs are only needed to
# *construct* these, not to inspect them — and construction is bypassed below.
from agent_memory.providers.anthropic import AnthropicLLMProvider
from agent_memory.providers.bedrock import BedrockLLMProvider
from agent_memory.providers.openai import OpenAILLMProvider

_PROVIDERS = [
    pytest.param(BedrockLLMProvider, id="bedrock"),
    pytest.param(OpenAILLMProvider, id="openai"),
    pytest.param(AnthropicLLMProvider, id="anthropic"),
]

# The two methods the enrichment worker calls with a `prompt=` keyword.
_PROMPTED_METHODS = ["assess_importance", "generate_summary"]


class TestSignaturesMatchWhatEnrichmentCalls:
    """Static contract: the argument exists on every implementation."""

    @pytest.mark.parametrize("provider", _PROVIDERS)
    @pytest.mark.parametrize("method", _PROMPTED_METHODS)
    def test_provider_accepts_prompt_keyword(self, provider, method) -> None:
        sig = inspect.signature(getattr(provider, method))
        assert "prompt" in sig.parameters, (
            f"{provider.__name__}.{method} does not accept `prompt=`; "
            "EnrichmentWorker passes it on every call, so enrichment fails "
            "with TypeError on this provider"
        )

    @pytest.mark.parametrize("method", _PROMPTED_METHODS)
    def test_the_abstract_base_declares_it(self, method) -> None:
        """The ABC is what makes a new provider's omission a visible error."""
        sig = inspect.signature(getattr(LLMProvider, method))
        assert "prompt" in sig.parameters

    @pytest.mark.parametrize("provider", _PROVIDERS)
    @pytest.mark.parametrize("method", _PROMPTED_METHODS)
    def test_prompt_is_optional(self, provider, method) -> None:
        """Callers without a template must still work — `_get_prompt` can return None."""
        param = inspect.signature(getattr(provider, method)).parameters["prompt"]
        assert param.default is None


class TestTheCallActuallyBinds:
    """Dynamic contract: inspection can pass while the call still fails.

    A signature check alone would miss a positional-only declaration or a
    decorator that re-wraps with different parameters. These call the real methods
    on real instances, stubbing only the network boundary (`complete`).
    """

    @pytest.mark.parametrize("provider", _PROVIDERS)
    @pytest.mark.asyncio
    async def test_assess_importance_binds_and_uses_the_template(
        self, provider
    ) -> None:
        seen: list[str] = []

        class Stub(provider):  # type: ignore[valid-type,misc]
            # Bypass __init__: constructing these requires SDKs and credentials,
            # and neither is relevant to argument binding.
            def __init__(self) -> None:
                pass

            async def complete(self, text: str, **kwargs) -> str:
                seen.append(text)
                return "7"

        score = await Stub().assess_importance(
            "the memory", prompt="Score this: {content}"
        )

        assert seen == ["Score this: the memory"], (
            "the template was accepted but not rendered with the content"
        )
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("provider", _PROVIDERS)
    @pytest.mark.asyncio
    async def test_generate_summary_binds_and_uses_the_template(self, provider) -> None:
        seen: list[str] = []

        class Stub(provider):  # type: ignore[valid-type,misc]
            def __init__(self) -> None:
                pass

            async def complete(self, text: str, **kwargs) -> str:
                seen.append(text)
                return "a summary"

        result = await Stub().generate_summary(
            "the memory", prompt="Summarize: {content}"
        )

        assert seen == ["Summarize: the memory"]
        assert result == "a summary"

    @pytest.mark.parametrize("provider", _PROVIDERS)
    @pytest.mark.asyncio
    async def test_omitting_prompt_falls_back_to_the_builtin(self, provider) -> None:
        """The `prompt=None` path must still produce a usable instruction."""
        seen: list[str] = []

        class Stub(provider):  # type: ignore[valid-type,misc]
            def __init__(self) -> None:
                pass

            async def complete(self, text: str, **kwargs) -> str:
                seen.append(text)
                return "7"

        await Stub().assess_importance("the memory")

        assert len(seen) == 1
        assert "the memory" in seen[0], "the built-in prompt lost the content"


class TestRenderPromptIsSafeOnHostileTemplates:
    """Templates live in the database, so rendering must not be able to raise.

    `str.format` on operator-editable text is the same silent-failure shape as the
    missing argument: any raise here is swallowed by the enrichment worker and
    turns into a memory stuck at `failed`.
    """

    def test_the_normal_case(self) -> None:
        assert render_prompt("Rate: {content}", "abc", "FB") == "Rate: abc"

    def test_none_and_empty_fall_back(self) -> None:
        assert render_prompt(None, "abc", "FB") == "FB"
        assert render_prompt("", "abc", "FB") == "FB"

    @pytest.mark.parametrize(
        "template",
        [
            "Rate {content} and {unknown_key}",  # KeyError
            "Rate {content} and {}",  # IndexError
            "Rate {content} and { unbalanced",  # ValueError
        ],
        ids=["unknown-key", "positional", "unbalanced-brace"],
    )
    def test_broken_templates_fall_back_instead_of_raising(self, template) -> None:
        assert render_prompt(template, "abc", "FB") == "FB"

    def test_a_template_without_the_placeholder_falls_back(self) -> None:
        """Formatting would succeed and silently omit the memory.

        The model would then score the instructions rather than the content, which
        is worse than using the built-in: it returns a plausible number for the
        wrong input, so nothing anywhere looks broken.
        """
        assert render_prompt("Rate the memory please", "abc", "FB") == "FB"

    def test_literal_braces_survive_when_escaped(self) -> None:
        """Doubled braces are valid `format` syntax and must not be a fallback."""
        assert render_prompt("{{json}}: {content}", "abc", "FB") == "{json}: abc"
