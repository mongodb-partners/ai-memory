"""Scoring implementations for the pluggable importance seam.
REQ-E-160, REQ-E-161, REQ-E-165, REQ-E-171.

Two properties dominate these tests:

1. **The floor.** No scorer may emit below 0.1. 0.0 sits at
   `forgetting_score_threshold`, so returning it is an instruction to delete the
   memory — see `providers/base.py:44-51`. Asserted independently of the maths so
   a bad artifact cannot order a deletion.
2. **Refusing a mismatched embedding.** A 1024-vector against 1536 coefficients
   would happily score the overlapping prefix. That is the failure mode worth
   engineering against, because it produces plausible numbers. So does returning
   the intercept for everything, which is why this raises instead.
"""

import math
from unittest.mock import AsyncMock, create_autospec

import pytest

from agent_memory.exceptions import ConfigError
from agent_memory.providers.base import LLMProvider
from agent_memory.services.importance import (
    LEXICAL_FEATURE_COUNT,
    MAX_IMPORTANCE,
    MIN_IMPORTANCE,
    Artifact,
    ImportanceScorer,
    LLMScorer,
    LocalScorer,
    logistic,
)


def _embedding_artifact(coefficients=(1.0, 0.0, 0.0), intercept=0.0) -> Artifact:
    return Artifact(
        kind="embedding_linear",
        coefficients=tuple(coefficients),
        intercept=intercept,
        provider="bedrock",
        model="test-model",
        dimension=len(coefficients),
        training={},
    )


def _lexical_artifact(coefficients=None, intercept=0.0) -> Artifact:
    coefficients = coefficients or [0.0] * LEXICAL_FEATURE_COUNT
    return Artifact(
        kind="lexical",
        coefficients=tuple(coefficients),
        intercept=intercept,
        training={},
    )


class TestLogistic:
    def test_midpoint(self):
        assert logistic(0.0) == pytest.approx(0.5)

    def test_monotone(self):
        assert logistic(-1.0) < logistic(0.0) < logistic(1.0)

    @pytest.mark.parametrize("x", [1e9, -1e9, 800.0, -800.0])
    def test_no_overflow_on_extremes(self, x):
        """`math.exp(-(-800))` raises OverflowError. A saturating model must not
        crash the enrichment worker — a crash there retries the memory to
        `failed`."""
        value = logistic(x)
        assert 0.0 <= value <= 1.0
        assert not math.isnan(value)


class TestProtocolConformance:
    def test_llm_scorer_is_a_scorer(self):
        assert isinstance(LLMScorer(AsyncMock()), ImportanceScorer)

    def test_local_scorer_is_a_scorer(self):
        assert isinstance(LocalScorer(_lexical_artifact()), ImportanceScorer)


class TestLLMScorer:
    async def test_delegates_to_provider(self):
        llm = create_autospec(LLMProvider, instance=True)
        llm.assess_importance.return_value = 0.7
        assert await LLMScorer(llm).score("hello") == 0.7

    async def test_passes_prompt_from_getter(self):
        llm = create_autospec(LLMProvider, instance=True)
        llm.assess_importance.return_value = 0.7
        getter = AsyncMock(return_value="Rate this: {content}")
        await LLMScorer(llm, prompt_getter=getter).score("hello")
        getter.assert_awaited_once_with("importance_assessment")
        llm.assess_importance.assert_awaited_once_with(
            "hello", prompt="Rate this: {content}"
        )

    async def test_omits_prompt_when_getter_returns_none(self):
        """Today's behaviour: no prompt kwarg at all, so the provider's own
        default template applies. Passing `prompt=None` explicitly would be a
        different call and is not what the current worker does."""
        llm = create_autospec(LLMProvider, instance=True)
        llm.assess_importance.return_value = 0.7
        await LLMScorer(llm, prompt_getter=AsyncMock(return_value=None)).score("hi")
        llm.assess_importance.assert_awaited_once_with("hi")

    async def test_omits_prompt_when_no_getter(self):
        llm = create_autospec(LLMProvider, instance=True)
        llm.assess_importance.return_value = 0.7
        await LLMScorer(llm).score("hi")
        llm.assess_importance.assert_awaited_once_with("hi")

    async def test_ignores_embedding_and_metadata(self):
        """The LLM path takes text only. Accepting the wider signature without
        using it is what makes the two implementations substitutable."""
        llm = create_autospec(LLMProvider, instance=True)
        llm.assess_importance.return_value = 0.7
        await LLMScorer(llm).score(
            "hi", [0.1] * 1536, tags=["a"], message_type="human"
        )
        llm.assess_importance.assert_awaited_once_with("hi")


class TestLocalScorerEmbeddingPath:
    async def test_dot_product_then_logistic(self):
        art = _embedding_artifact(coefficients=(2.0, 0.0, 0.0), intercept=0.0)
        got = await LocalScorer(art).score("x", [1.0, 5.0, 5.0])
        assert got == pytest.approx(logistic(2.0))

    async def test_intercept_applied(self):
        art = _embedding_artifact(coefficients=(0.0, 0.0, 0.0), intercept=1.5)
        assert await LocalScorer(art).score("x", [1.0, 1.0, 1.0]) == pytest.approx(
            logistic(1.5)
        )

    async def test_higher_dot_product_scores_higher(self):
        scorer = LocalScorer(_embedding_artifact(coefficients=(1.0, 1.0, 1.0)))
        low = await scorer.score("x", [0.0, 0.0, 0.0])
        high = await scorer.score("x", [1.0, 1.0, 1.0])
        assert high > low

    async def test_content_is_ignored_on_the_embedding_path(self):
        """The vector is the input. If content leaked in, an embedding artifact's
        coefficients would be crossed with lexical features."""
        scorer = LocalScorer(_embedding_artifact(coefficients=(1.0, 1.0, 1.0)))
        a = await scorer.score("short", [0.5, 0.5, 0.5])
        b = await scorer.score("a" * 5000 + "?", [0.5, 0.5, 0.5])
        assert a == b


class TestLocalScorerRefusesUnusableInput:
    """REQ-E-165. An embedding artifact handed an unusable vector raises.

    The tempting alternative — return the intercept, log a warning — is worse. It
    produces the same plausible number for every memory in the store, and the
    only symptom is recall quality drifting weeks later. Raising routes into
    `_enrich_memory`'s existing retry path, which leaves the memories in
    `enrichment_status: "failed"` where they can be counted and queried. Loud and
    inspectable beats quiet and uniform.
    """

    async def test_wrong_dimension_raises(self):
        art = _embedding_artifact(coefficients=(1.0, 1.0, 1.0))
        with pytest.raises(ConfigError, match="does not match"):
            await LocalScorer(art).score("x", [1.0, 1.0])

    async def test_error_names_both_dimensions(self):
        """'dimension mismatch' sends an operator to read source; '2 does not
        match model dimension 3' sends them to their config."""
        art = _embedding_artifact(coefficients=(1.0, 1.0, 1.0))
        with pytest.raises(ConfigError) as exc:
            await LocalScorer(art).score("x", [1.0, 1.0])
        assert "2" in str(exc.value) and "3" in str(exc.value)

    async def test_missing_embedding_raises(self):
        art = _embedding_artifact(coefficients=(1.0, 1.0, 1.0))
        with pytest.raises(ConfigError):
            await LocalScorer(art).score("x", None)

    async def test_empty_embedding_raises(self):
        art = _embedding_artifact(coefficients=(1.0, 1.0, 1.0))
        with pytest.raises(ConfigError):
            await LocalScorer(art).score("x", [])

    async def test_non_numeric_embedding_raises(self):
        """A vector read back from Mongo can contain None if a write half-failed.
        `None * float` would raise TypeError anyway — this raises the error that
        says what is actually wrong."""
        art = _embedding_artifact(coefficients=(1.0, 1.0, 1.0))
        with pytest.raises(ConfigError, match="numeric"):
            await LocalScorer(art).score("x", [1.0, None, 1.0])

    async def test_lexical_artifact_does_not_require_an_embedding(self):
        """Only `embedding_linear` needs a vector. A lexical artifact scoring
        text must not be dragged into this."""
        got = await LocalScorer(_lexical_artifact(intercept=0.3)).score("x", None)
        assert got == pytest.approx(logistic(0.3))


class TestLocalScorerLexicalPath:
    async def test_ignores_embedding_entirely(self):
        """A lexical artifact must not read the embedding — its coefficients are
        indexed by feature, and a 1536-vector would silently be scored against
        the wrong weights."""
        coeffs = [1.0] + [0.0] * (LEXICAL_FEATURE_COUNT - 1)
        scorer = LocalScorer(_lexical_artifact(coeffs))
        content = "a" * 1000
        with_emb = await scorer.score(content, [9.0] * 1536)
        without = await scorer.score(content, None)
        assert with_emb == without == pytest.approx(logistic(1.0))

    async def test_uses_content(self):
        coeffs = [1.0] + [0.0] * (LEXICAL_FEATURE_COUNT - 1)
        scorer = LocalScorer(_lexical_artifact(coeffs))
        short = await scorer.score("a" * 100, None)
        long = await scorer.score("a" * 1000, None)
        assert long > short


class TestClamping:
    """REQ-E-165. Asserted independently of the maths: a trained artifact with a
    large negative intercept is a plausible accident, and its consequence would
    be silent deletion of every memory it scores."""

    async def test_never_below_floor(self):
        art = _embedding_artifact(coefficients=(0.0,) * 3, intercept=-1000.0)
        assert await LocalScorer(art).score("x", [1.0, 1.0, 1.0]) == MIN_IMPORTANCE

    async def test_never_above_ceiling(self):
        art = _embedding_artifact(coefficients=(0.0,) * 3, intercept=1000.0)
        assert await LocalScorer(art).score("x", [1.0, 1.0, 1.0]) == MAX_IMPORTANCE

    async def test_floor_is_not_zero(self):
        assert MIN_IMPORTANCE == 0.1

    @pytest.mark.parametrize("intercept", [-50.0, -5.0, 0.0, 5.0, 50.0])
    async def test_always_in_range(self, intercept):
        art = _embedding_artifact(coefficients=(3.0,) * 3, intercept=intercept)
        got = await LocalScorer(art).score("x", [10.0, -10.0, 7.0])
        assert MIN_IMPORTANCE <= got <= MAX_IMPORTANCE


class TestSubstitutability:
    """Both implementations must satisfy the same caller. If this passes for one
    and not the other, the worker cannot treat them as interchangeable."""

    @pytest.mark.parametrize("kind", ["llm", "local"])
    async def test_same_call_shape(self, kind):
        if kind == "llm":
            llm = create_autospec(LLMProvider, instance=True)
            llm.assess_importance.return_value = 0.7
            scorer: ImportanceScorer = LLMScorer(llm)
        else:
            scorer = LocalScorer(_lexical_artifact(intercept=0.4))
        got = await scorer.score(
            "some content", [0.1] * LEXICAL_FEATURE_COUNT,
            tags=["work"], message_type="human",
        )
        assert MIN_IMPORTANCE <= got <= MAX_IMPORTANCE
