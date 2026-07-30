"""Abstract base classes for embedding and LLM providers."""

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

# Matches an integer or a decimal. The decimal branch has to come first —
# alternation is ordered, so `\d+|\d*\.\d+` would match the "0" of "0.9" and stop.
_SCORE_RE = re.compile(r"\d*\.\d+|\d+")


def parse_importance(response: str | None, *, default: float = 0.5) -> float:
    """Parse an importance score from a model reply, on either scale.

    Providers prompt for a 1–10 integer, but the prompt is overridable — and the
    prompt shipped in ``services/prompt_library.py`` asks for 0.0–1.0 instead.
    Both have to work, because a caller who customizes the prompt has no way to
    know which parser the provider uses.

    Scale is inferred from the value: anything above 1 is treated as an N-of-10
    rating and divided; anything at or below 1 is already a fraction.

    ``1`` is the one genuinely ambiguous input — "least important" on a 1-10 scale,
    "most important" as a fraction — and it resolves to ``1.0``. That asymmetry is
    deliberate. Reading it as 0.1 puts the memory at the forgetting threshold, so a
    misread of "keep this forever" ends in deletion; reading it as 1.0 keeps a
    trivial memory around, which costs a little storage and a slightly worse
    ranking. Only one of those is recoverable.

    Getting this wrong is quiet and expensive. A ``\\d+`` parse of the reply "0.9"
    matches the leading ``0``, yields ``0.0``, and clamps to the floor — so the
    most important memory in the store is scored one step above forgettable and
    the consolidation worker eventually soft-deletes it. Nothing errors; the
    memory just stops being recalled.
    """
    match = _SCORE_RE.search(response or "")
    if match is None:
        return default
    try:
        value = float(match.group())
    except ValueError:  # pragma: no cover — the regex cannot produce this
        return default
    if value > 1.0:
        value = value / 10.0
    # Floor at 0.1 rather than 0.0: a memory the model rated as worthless is
    # still evidence, and `forgetting_score_threshold` is what decides whether it
    # survives — that call belongs to the operator's config, not to a parser.
    return max(0.1, min(1.0, value))


# Below this many characters, a memory is its own best summary. Summarizing a
# single conversational turn does not compress it — it asks the model to
# summarize a fragment it has no context for, and the reply is often a *refusal*
# rather than a summary. 120 characters is roughly a sentence and a half.
MIN_SUMMARIZABLE_CHARS = 120

# A reply containing one of these is the model saying it cannot summarize what it
# was given. Matching on phrasing is blunt, but the alternative — trusting the
# reply — is what put "This text fragment is too brief and lacks sufficient
# context" into the sample UI's memory panel, in place of the memory.
_NON_SUMMARY_MARKERS = (
    "don't see the original",
    "do not see the original",
    "too brief",
    "appears to be incomplete",
    "no text provided",
    "please provide",
    "but it appears to be",
    "cannot summarize",
    "unable to summarize",
)


def is_usable_summary(summary: str | None, content: str) -> bool:
    """Whether a model reply is a summary rather than a complaint about one.

    ``generate_summary`` returns whatever the model said, and on short input what
    it says is frequently "I don't see the original text that needs to be
    summarized". Nothing about that reply is exceptional — it is a successful call
    returning a well-formed string — so a worker that stores the result
    unconditionally stores the refusal, and every reader that prefers ``summary``
    over ``content`` then shows it instead of the memory.

    Three rejections: the empty reply, the reply at least as long as its source
    (compression that expands is not compression), and the recognizable refusals.

    Deliberately conservative in one direction. A memory with no summary falls
    back to its content, which is always readable; a memory with a bad summary
    displays the bad summary everywhere. So a false reject costs nothing and a
    false accept is visible on screen.
    """
    if not summary or not summary.strip():
        return False
    text = summary.strip()
    if len(text) >= len(content):
        return False
    lowered = text.casefold()
    return not any(marker in lowered for marker in _NON_SUMMARY_MARKERS)


class EmbeddingProvider(ABC):
    @abstractmethod
    async def generate_embedding(self, text: str) -> list[float]:
        ...

    @abstractmethod
    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        ...


class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs) -> str:
        ...

    async def chat_stream(
        self, messages: list[dict], **kwargs
    ) -> AsyncIterator[str]:
        """Yield text deltas as the model produces them.

        Concrete, not abstract: the default implementation awaits :meth:`chat`
        and yields the whole answer as one chunk. That keeps every existing
        provider working and makes streaming an optimization rather than a
        breaking change — a caller written against this interface behaves
        correctly on a provider that cannot stream, it just sees one large
        delta instead of many small ones.

        Yields text only. Tool-call and usage events are deliberately out of
        scope: they would make the return type provider-shaped, and the whole
        point of this seam is that the caller does not know which provider it
        has.
        """
        yield await self.chat(messages, **kwargs)

    def user_turn(self, text: str) -> list[dict]:
        """Wrap one text prompt as this provider's ``messages`` list.

        ``chat`` takes provider-native messages — that is deliberate, since it is
        the escape hatch for callers who need tool configs or multi-turn history
        the interface does not model. But it means a caller with nothing but a
        string cannot use ``chat`` without knowing which provider it holds, and
        the shapes are not compatible: OpenAI and Anthropic take ``content`` as a
        string, Bedrock's Converse API requires a list of content blocks.

        The default is the string form, which OpenAI and Anthropic both accept.
        Bedrock overrides it. See :meth:`complete` for the intended entry point.
        """
        return [{"role": "user", "content": text}]

    async def complete(self, text: str, **kwargs) -> str:
        """Send one text prompt and return the reply, whatever the provider.

        This is what library-internal callers should use. Building the message
        inline instead is how enrichment's merge path came to send an
        OpenAI-shaped message to Bedrock — the default provider — where botocore
        rejected it as ``Invalid type for parameter messages[0].content``. The
        enrichment worker catches and logs that, so the only symptom was memories
        stuck in ``merge_pending`` forever, with the failure visible in a log
        nobody reads.
        """
        return await self.chat(self.user_turn(text), **kwargs)

    @abstractmethod
    async def assess_importance(self, content: str, prompt: str | None = None) -> float:
        """Score `content` 0.0-1.0, optionally with a caller-supplied template.

        ``prompt`` is part of the interface because ``LLMScorer`` passes it
        whenever a custom template is configured. Leaving it off two of the three
        implementations made that path raise ``TypeError`` at call binding on
        OpenAI and Anthropic — swallowed by the enrichment worker's ``except``,
        retried to ``failed``, and invisible in tests that mock the LLM with a
        bare ``AsyncMock``, which accepts any keyword. Declaring it here makes
        that class of drift a signature error instead of a silent outage.

        Implementations must accept it. Ignoring its value is acceptable; not
        accepting the argument is not.
        """
        ...

    @abstractmethod
    async def generate_summary(self, content: str, max_length: int = 100) -> str:
        ...
