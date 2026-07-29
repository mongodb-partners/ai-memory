"""Tests for ``is_usable_summary`` — telling a summary from a refusal.

REQ-E-121. ``LLMProvider.generate_summary`` returns whatever the model said. On a
short conversational turn what it says is often not a summary but a complaint
about being asked for one:

    content:  "Understood. I'll keep shellfish out of everything."
    summary:  "I don't see the original text that needs to be summarized—only..."

Nothing about that is an error. The call succeeded, the string is well-formed,
and a worker that stores it unconditionally stores it. Every reader that prefers
``summary`` over ``content`` then displays the complaint in place of the memory —
which is what the sample UI's memory panel did, on the screen for a booth talk.

The asymmetry that shapes these tests: a memory with **no** summary falls back to
its content, which is always readable, while a memory with a **bad** summary
shows the bad summary everywhere. So the guard is tuned to over-reject. That is
why ``test_a_short_paraphrase_is_accepted`` matters as much as the refusal
cases — over-rejecting is cheap, but rejecting *everything* would quietly turn
the summary field off.
"""

import pytest

from agent_memory.providers.base import MIN_SUMMARIZABLE_CHARS, is_usable_summary

SOURCE = (
    "I cook for four most nights and for six when guests come over, so I plan "
    "around sheet-pan roasts and one-pot pasta that scale without extra work."
)


class TestRefusalsAreRejected:
    """REQ-E-121: the replies that were actually observed in the seeded data."""

    @pytest.mark.parametrize(
        "reply",
        [
            "I don't see the original text that needs to be summarized—only a...",
            "I do not see the original text you'd like me to summarize.",
            "This text fragment is too brief and lacks sufficient context.",
            "I'll help summarize this text, but it appears to be incomplete.",
            "Please provide the text you would like summarized.",
            "There is no text provided to summarize.",
            "I cannot summarize this.",
            "I'm unable to summarize a fragment this short.",
        ],
    )
    def test_a_complaint_is_not_a_summary(self, reply):
        assert is_usable_summary(reply, SOURCE) is False

    def test_matching_is_case_insensitive(self):
        """The model does not capitalize consistently across calls."""
        assert is_usable_summary("TOO BRIEF to summarize usefully.", SOURCE) is False


class TestEmptyRepliesAreRejected:
    """A blank summary is worse than no summary: it renders as an empty row."""

    @pytest.mark.parametrize("reply", [None, "", "   ", "\n\t"])
    def test_blank_is_rejected(self, reply):
        assert is_usable_summary(reply, SOURCE) is False


class TestLengthIsTheStructuralCheck:
    """The one rejection that needs no phrase list, and so catches novel refusals.

    Any reply as long as its source failed to compress, whatever it says. This is
    the guard that still works when a model version changes its wording and the
    marker list goes stale.
    """

    def test_longer_than_source_is_rejected(self):
        assert is_usable_summary("x" * (len(SOURCE) + 1), SOURCE) is False

    def test_exactly_as_long_is_rejected(self):
        assert is_usable_summary("x" * len(SOURCE), SOURCE) is False

    def test_one_character_shorter_passes_the_length_check(self):
        """Pinning the boundary as ``>=``, not ``>``: equal length is no summary."""
        assert is_usable_summary("x" * (len(SOURCE) - 1), SOURCE) is True


class TestRealSummariesAreAccepted:
    """REQ-E-121: over-rejecting is cheap, but rejecting everything is a silent
    feature removal — the field would simply never be set again."""

    @pytest.mark.parametrize(
        "reply",
        [
            "Cooks for four nightly, six with guests; prefers scalable one-pan meals.",
            "Household of four to six. Wants recipes that scale.",
            "Plans meals around sheet-pan roasts and one-pot pasta.",
        ],
    )
    def test_a_short_paraphrase_is_accepted(self, reply):
        assert is_usable_summary(reply, SOURCE) is True

    def test_surrounding_whitespace_does_not_disqualify(self):
        assert is_usable_summary("\n  Cooks for four.  \n", SOURCE) is True

    def test_a_summary_mentioning_brevity_of_its_subject_is_kept(self):
        """The markers target the model's *own* refusals. A summary that happens
        to describe something as brief is still a summary — the phrase has to be
        one of the refusal forms, not merely contain a similar word."""
        assert is_usable_summary("Prefers brief, quick weeknight meals.", SOURCE) is True


class TestTheThreshold:
    """REQ-E-121: ``MIN_SUMMARIZABLE_CHARS`` is the pre-call skip.

    Asserted as a property rather than a literal so a tuned value stays valid:
    what matters is that it is long enough to exclude a one-line turn and short
    enough not to exclude a real paragraph.
    """

    def test_it_excludes_a_single_conversational_turn(self):
        assert len("Understood. I'll keep shellfish out of everything.") < (
            MIN_SUMMARIZABLE_CHARS
        )

    def test_it_admits_a_multi_sentence_memory(self):
        assert len(SOURCE) >= MIN_SUMMARIZABLE_CHARS
