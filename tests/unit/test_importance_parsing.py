"""Tests for importance-score parsing across both prompt scales.

REQ-E-119. These exist because of a bug that was invisible in every way that
matters: nothing raised, nothing logged, and the stored score was a plausible
number. The default provider prompt asks for a 1-10 integer, but the prompt in
``services/prompt_library.py`` — which the enrichment worker prefers when the
library is available — asks for 0.0-1.0. A ``\\d+`` parse of the reply ``"0.9"``
matched the leading ``0``, produced ``0.0``, and clamped to the 0.1 floor. So the
most important memory in the store was scored one step above forgettable, and
``ConsolidationWorker._forget_low_importance`` would eventually soft-delete it for
being below ``forgetting_score_threshold``.

The symptom is a memory that stops being recalled weeks later. That is why the
scale-inference cases below are pinned rather than left to the parser's
discretion.
"""

import pytest

from agent_memory.providers.base import parse_importance


class TestTenPointScale:
    """REQ-E-119: integer replies on the 1-10 scale."""

    @pytest.mark.parametrize(
        "reply,expected",
        [
            ("9", 0.9),
            ("10", 1.0),
            ("5", 0.5),
            ("3", 0.3),
            # Below the floor after scaling; clamped, not zeroed.
            ("0", 0.1),
        ],
    )
    def test_integer_reply_scales_by_ten(self, reply, expected):
        assert parse_importance(reply) == pytest.approx(expected)

    def test_above_ten_clamps_to_one(self):
        """A model that ignores the scale must not produce importance > 1."""
        assert parse_importance("47") == 1.0


class TestFractionScale:
    """REQ-E-119: decimal replies on the 0.0-1.0 scale — the regression."""

    @pytest.mark.parametrize(
        "reply,expected",
        [
            # The exact case that was silently wrong: `\d+` matched "0".
            ("0.9", 0.9),
            ("0.85", 0.85),
            ("0.05", 0.1),  # below the floor
            (".7", 0.7),  # no leading zero
            ("1.0", 1.0),
        ],
    )
    def test_decimal_reply_is_not_divided(self, reply, expected):
        assert parse_importance(reply) == pytest.approx(expected)

    def test_bare_one_resolves_to_maximum_not_minimum(self):
        """`1` is genuinely ambiguous, and resolves in the recoverable direction.

        On a 1-10 scale it is the lowest rating; as a fraction it is the highest.
        The parser reads it as 1.0, because reading it as 0.1 would place the
        memory at ``forgetting_score_threshold`` and a misread of "keep this
        forever" would end in a soft delete. The opposite error keeps a trivial
        memory, which costs storage and a slightly worse ranking. Pinned so that a
        future change to the inference rule has to confront the asymmetry.
        """
        assert parse_importance("1") == pytest.approx(1.0)


class TestProseReplies:
    """REQ-E-119: models do not always obey "respond with ONLY a number"."""

    @pytest.mark.parametrize(
        "reply,expected",
        [
            ("Importance: 0.8", 0.8),
            ("I'd rate this 7 out of 10.", 0.7),
            ("The score is 9/10", 0.9),
            ("**0.95**", 0.95),
        ],
    )
    def test_first_number_wins(self, reply, expected):
        assert parse_importance(reply) == pytest.approx(expected)


class TestUnparseable:
    """REQ-E-119: no number means the default, never a crash."""

    @pytest.mark.parametrize(
        "reply", ["", None, "high", "I cannot assess this.", "N/A"]
    )
    def test_returns_default(self, reply):
        assert parse_importance(reply) == 0.5

    def test_default_is_overridable(self):
        assert parse_importance("unknown", default=0.42) == 0.42


class TestRange:
    """REQ-E-119: the output is always a usable importance."""

    @pytest.mark.parametrize(
        "reply",
        ["0", "0.0", "-3", "999", "10", "1", "0.5", "7", "not a number"],
    )
    def test_always_within_bounds(self, reply):
        value = parse_importance(reply)
        assert 0.1 <= value <= 1.0
