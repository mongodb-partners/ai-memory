"""Tests for ``user_turn`` / ``complete`` — the provider-agnostic prompt path.

REQ-E-120. These exist because ``LLMProvider.chat`` takes **provider-native**
messages, and nothing in its signature says so. Both shapes are a
``list[dict]``:

    OpenAI / Anthropic:  [{"role": "user", "content": "text"}]
    Bedrock Converse:    [{"role": "user", "content": [{"text": "text"}]}]

So a library-internal caller that hand-builds one of them type-checks, reads
correctly, passes under ``AsyncMock``, and fails against exactly one provider at
runtime. That happened: ``EnrichmentWorker._process_merge`` built the OpenAI
shape and sent it to Bedrock — the *default* provider — where botocore raised
``Invalid type for parameter messages[0].content``. ``_enrich_memory`` catches
every exception, so the only symptom was memories stuck in ``merge_pending``
indefinitely and a traceback in a log nobody was reading.

The mock-shaped tests could not catch it. ``providers.llm.chat`` as an
``AsyncMock`` accepts any argument at all, so the assertion "chat was called"
passed while the real call was malformed. Hence the shape assertions below run
against the concrete providers, not mocks of them.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory.providers.base import LLMProvider
from agent_memory.providers.bedrock import BedrockLLMProvider


class _Recorder(LLMProvider):
    """A provider that records what ``chat`` received and inherits the default."""

    def __init__(self) -> None:
        self.seen: list[dict] | None = None
        self.kwargs: dict = {}

    async def chat(self, messages: list[dict], **kwargs) -> str:
        self.seen = messages
        self.kwargs = kwargs
        return "reply"

    async def assess_importance(self, content: str) -> float:  # pragma: no cover
        return 0.5

    async def generate_summary(  # pragma: no cover
        self, content: str, max_length: int = 100
    ) -> str:
        return ""


def _bedrock() -> BedrockLLMProvider:
    """A Bedrock provider without a real boto3 client or credentials."""
    config = MagicMock()
    config.llm_model = "global.anthropic.claude-sonnet-5"
    provider = BedrockLLMProvider.__new__(BedrockLLMProvider)
    provider._config = config
    provider._client = MagicMock()
    provider._unsupported_sampling = set()
    return provider


class TestTheDefaultShape:
    """REQ-E-120: the string form, which OpenAI and Anthropic both accept."""

    def test_content_is_a_plain_string(self):
        assert _Recorder().user_turn("hello") == [
            {"role": "user", "content": "hello"}
        ]

    async def test_complete_routes_through_chat(self):
        provider = _Recorder()

        assert await provider.complete("hello") == "reply"
        assert provider.seen == [{"role": "user", "content": "hello"}]


class TestBedrockOverridesIt:
    """REQ-E-120: Converse requires content blocks — the regression."""

    def test_content_is_a_list_of_blocks(self):
        assert _bedrock().user_turn("hello") == [
            {"role": "user", "content": [{"text": "hello"}]}
        ]

    def test_the_shapes_are_not_interchangeable(self):
        """Both are ``list[dict]``, which is why the bug was invisible. The
        difference is one level down, where no type annotation reaches."""
        default = _Recorder().user_turn("x")[0]["content"]
        bedrock = _bedrock().user_turn("x")[0]["content"]

        assert isinstance(default, str)
        assert isinstance(bedrock, list)

    async def test_complete_sends_blocks_to_converse(self):
        provider = _bedrock()
        provider._client.converse.return_value = {
            "output": {"message": {"content": [{"text": "merged"}]}}
        }

        assert await provider.complete("merge these") == "merged"

        sent = provider._client.converse.call_args.kwargs["messages"]
        assert sent == [{"role": "user", "content": [{"text": "merge these"}]}]


class TestKwargsSurvive:
    """REQ-E-120: ``complete`` is a wrapper, not a narrowing.

    Anthropic's ``chat`` supplies a ``max_tokens`` default via ``setdefault``, and
    callers pass ``system`` and ``inferenceConfig`` through. A ``complete`` that
    dropped ``**kwargs`` would silently truncate replies on one provider.
    """

    async def test_kwargs_reach_chat(self):
        provider = _Recorder()

        await provider.complete("hello", max_tokens=64, system="be brief")

        assert provider.kwargs == {"max_tokens": 64, "system": "be brief"}


class TestEveryProviderSuppliesTheHelper:
    """REQ-E-120: a new provider gets a working ``complete`` for free.

    The default lives on the ABC rather than on each subclass, so the failure
    mode for a future provider is a *wrong shape it has to override*, not a
    missing method. Bedrock is the one that must override; asserting that here
    means a refactor which drops the override fails a test instead of a demo.
    """

    def test_bedrock_does_not_inherit_the_string_form(self):
        assert BedrockLLMProvider.user_turn is not LLMProvider.user_turn

    @pytest.mark.parametrize("module,name", [
        ("agent_memory.providers.anthropic", "AnthropicLLMProvider"),
        ("agent_memory.providers.openai", "OpenAILLMProvider"),
    ])
    def test_the_string_form_providers_inherit_it(self, module, name):
        provider_module = pytest.importorskip(module)
        cls = getattr(provider_module, name)

        assert cls.user_turn is LLMProvider.user_turn


class TestInternalCallersUseIt:
    """REQ-E-120: the enrichment merge path, which is where this went wrong.

    Asserted at the call site rather than only in the provider, because the
    provider was never the problem — the caller bypassing it was.
    """

    async def test_merge_does_not_build_its_own_message(self):
        from agent_memory.services.enrichment import EnrichmentWorker

        providers = MagicMock()
        providers.llm.complete = AsyncMock(return_value="merged")
        # If the worker calls `chat` directly it is hand-building a message shape
        # again, and this raises rather than passing on an accommodating mock.
        providers.llm.chat = AsyncMock(
            side_effect=AssertionError(
                "merge must go through `complete`, not `chat` — see REQ-E-120"
            )
        )

        collection = MagicMock()
        collection.update_one = AsyncMock()
        collection.find_one = AsyncMock(
            return_value={"_id": "target", "content": "existing", "importance": 0.6}
        )

        config = MagicMock()
        config.prompt_experiment_enabled = False
        # A real int: `asyncio.Semaphore` compares it against 0 and a MagicMock
        # raises TypeError on `<`.
        config.enrichment_concurrency = 1
        worker = EnrichmentWorker(collection, config, providers, MagicMock())
        worker.prompt_library = None

        await worker._process_merge(
            {"_id": "new", "content": "incoming", "merge_target_id": "target"}
        )

        providers.llm.complete.assert_awaited_once()
