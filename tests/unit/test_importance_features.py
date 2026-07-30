"""Lexical feature extraction for the local importance scorer. REQ-E-166.

The feature *order* is a wire format: coefficients in a shipped artifact are
positional. Reordering or inserting a feature without retraining silently
reassigns every weight, and the model keeps returning plausible numbers. Hence
the explicit index assertions below — they exist to fail loudly on a refactor
that looks harmless.
"""

import pytest

from agent_memory.services.importance import (
    LEXICAL_FEATURE_COUNT,
    LEXICAL_FEATURE_NAMES,
    lexical_features,
)

DURABLE = "My manager always wants the release notes in Markdown, never plain text."
EPHEMERAL = "Can you deploy branch fix-3 today?"


class TestContract:
    def test_names_match_count(self):
        assert len(LEXICAL_FEATURE_NAMES) == LEXICAL_FEATURE_COUNT

    def test_expected_order(self):
        """Pinned to catch a reorder. Changing this requires retraining."""
        assert LEXICAL_FEATURE_NAMES == (
            "length",
            "digit_ratio",
            "preference",
            "identity",
            "temporal",
            "interrogative",
            "entity",
        )

    @pytest.mark.parametrize(
        "content", ["", " ", "x", DURABLE, EPHEMERAL, "?" * 5000, "123456"]
    )
    def test_always_returns_bounded_vector(self, content):
        feats = lexical_features(content)
        assert len(feats) == LEXICAL_FEATURE_COUNT
        assert all(0.0 <= f <= 1.0 for f in feats), feats

    def test_empty_content_does_not_divide_by_zero(self):
        assert lexical_features("") == [0.0] * LEXICAL_FEATURE_COUNT

    def test_none_content_is_treated_as_empty(self):
        """`memory["content"]` is non-null in practice, but a scorer that raises
        sends the memory down the retry path to `failed`."""
        assert lexical_features(None) == [0.0] * LEXICAL_FEATURE_COUNT


class TestIndividualFeatures:
    def _f(self, content: str, name: str) -> float:
        return lexical_features(content)[LEXICAL_FEATURE_NAMES.index(name)]

    def test_length_saturates(self):
        assert self._f("a" * 500, "length") == pytest.approx(0.5)
        assert self._f("a" * 1000, "length") == 1.0
        assert self._f("a" * 9000, "length") == 1.0

    def test_digit_ratio(self):
        assert self._f("1234", "digit_ratio") == 1.0
        assert self._f("ab12", "digit_ratio") == pytest.approx(0.5)
        assert self._f("abcd", "digit_ratio") == 0.0

    def test_preference_terms_counted_and_capped(self):
        assert self._f("The sky is blue.", "preference") == 0.0
        assert self._f("I always use tabs.", "preference") > 0.0
        many = "I always prefer tabs, never spaces, and must have trailing commas."
        assert self._f(many, "preference") == 1.0

    def test_preference_matching_is_case_insensitive(self):
        assert self._f("ALWAYS use tabs", "preference") > 0.0

    def test_preference_requires_whole_words(self):
        """'preferential' and 'somewhere' are not preference statements."""
        assert self._f("A preferential ballot is somewhere in the docs.", "preference") == 0.0

    def test_identity_terms(self):
        assert self._f("My team owns billing.", "identity") > 0.0
        assert self._f("The team owns billing.", "identity") == 0.0

    def test_temporal_terms(self):
        assert self._f("Ship it today.", "temporal") > 0.0
        assert self._f("Ship it.", "temporal") == 0.0

    def test_interrogative_is_binary(self):
        assert self._f("What is this?", "interrogative") == 1.0
        assert self._f("This is that.", "interrogative") == 0.0
        assert self._f("Really?? Yes??", "interrogative") == 1.0

    def test_entity_ignores_sentence_initial_capitals(self):
        """'The' at position 0 is grammar, not an entity."""
        assert self._f("The cat sat on the mat.", "entity") == 0.0
        assert self._f("The cat belongs to Priya.", "entity") > 0.0

    def test_entity_counts_multiple(self):
        one = self._f("we deploy via Terraform", "entity")
        two = self._f("we deploy Atlas via Terraform", "entity")
        assert two > one

    def test_single_word_has_no_entity_signal(self):
        """One token is all sentence-initial, so there is nothing to measure."""
        assert self._f("Priya", "entity") == 0.0


class TestDiscrimination:
    """The features have to separate the two cases the scorer exists to separate."""

    def test_durable_and_ephemeral_differ(self):
        assert lexical_features(DURABLE) != lexical_features(EPHEMERAL)

    def test_durable_scores_higher_on_preference(self):
        i = LEXICAL_FEATURE_NAMES.index("preference")
        assert lexical_features(DURABLE)[i] > lexical_features(EPHEMERAL)[i]

    def test_ephemeral_scores_higher_on_temporal_and_question(self):
        t = LEXICAL_FEATURE_NAMES.index("temporal")
        q = LEXICAL_FEATURE_NAMES.index("interrogative")
        d, e = lexical_features(DURABLE), lexical_features(EPHEMERAL)
        assert e[t] > d[t]
        assert e[q] > d[q]
