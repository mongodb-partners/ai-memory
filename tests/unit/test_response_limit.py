"""Tests for `max_response_bytes` enforcement. REQ-E-089.

The knob was declared in the config, documented in the README table, asserted by
`test_config.py`, and read by nothing. That is worse than having no limit: it is
the setting an operator reaches for when a client is choking on a response, and
lowering it changed nothing at all.
"""

from datetime import UTC
from unittest.mock import MagicMock

from agent_memory.core.response_limit import cap_results


def _doc(n: int, size: int = 100) -> dict:
    return {"_id": f"m{n}", "text": "x" * size}


class TestCapResults:
    def test_a_list_that_fits_is_returned_unchanged_with_no_metadata(self):
        """TC-RESP-001: the common path must be byte-identical to before.

        An empty `meta` is what lets the caller splat it into the envelope without
        adding a `truncated: false` key to every response ever returned.
        """
        docs = [_doc(i) for i in range(5)]
        kept, meta = cap_results(docs, 16_777_216)
        assert kept is docs
        assert meta == {}

    def test_truncation_drops_whole_documents_from_the_tail(self):
        """TC-RESP-002: never cut mid-document.

        A response sliced at a byte offset is invalid JSON at best, and once
        re-serialised, a document with silently missing fields at worst — the same
        class of failure as an unmarked mid-value truncation. Every document that
        survives survives intact.
        """
        docs = [_doc(i, size=100) for i in range(50)]
        kept, _ = cap_results(docs, 1000)

        assert 0 < len(kept) < 50
        assert kept == docs[: len(kept)]
        assert all(len(d["text"]) == 100 for d in kept)

    def test_truncation_is_reported_not_silent(self):
        """TC-RESP-003: a shortened list is indistinguishable from a short one.

        "recall returned 12 results" versus "12 of 40, capped at 16 MiB" is the
        difference between an operator raising the cap and an operator concluding
        the memory store is empty.
        """
        docs = [_doc(i, size=100) for i in range(50)]
        _kept, meta = cap_results(docs, 1000)

        assert meta["truncated"] is True
        assert meta["total_count"] == 50
        assert meta["max_response_bytes"] == 1000

    def test_at_least_one_document_survives_an_impossible_cap(self):
        """TC-RESP-004: a too-large answer must not become no answer.

        The caller has no way to ask for a smaller one, so returning an empty list
        turns a size problem into a correctness problem.
        """
        kept, meta = cap_results([_doc(0, size=10_000)], 10)
        assert len(kept) == 1
        assert meta == {}

    def test_a_zero_or_negative_cap_disables_the_limit(self):
        # TC-RESP-005: an explicit opt-out rather than an accidental empty response.
        docs = [_doc(i) for i in range(5)]
        assert cap_results(docs, 0) == (docs, {})
        assert cap_results(docs, -1) == (docs, {})

    def test_an_empty_list_is_handled(self):
        # TC-RESP-006
        assert cap_results([], 1000) == ([], {})

    def test_unserialisable_values_do_not_raise(self):
        """TC-RESP-007: sizing runs on every read and must never be the failure.

        Sanitisation should have converted BSON types already, but a datetime that
        slipped through must be measured, not raised on.
        """
        from datetime import datetime

        docs = [{"_id": "m1", "ts": datetime.now(UTC)}]
        kept, meta = cap_results(docs, 16_777_216)
        assert kept == docs and meta == {}


class TestFacadeEnforcement:
    """The knob has to be read by the read path, not just exist in a module."""

    @staticmethod
    def _facade(max_bytes: int):
        from agent_memory.memory import AsyncMemory

        facade = AsyncMemory.__new__(AsyncMemory)
        facade.config = MagicMock(max_response_bytes=max_bytes)
        return facade

    async def test_the_recall_envelope_is_capped(self):
        """TC-RESP-010: `limit` bounds the count, not the size.

        Ten episodic turns carrying projected message content, todos, and a
        files-touched array is megabytes — one MCP frame, or one HTTP body, and
        `limit=10` said nothing about it.
        """
        facade = self._facade(2000)
        docs = [_doc(i, size=500) for i in range(20)]

        out = facade._results(docs, "memories")

        assert out["truncated"] is True
        assert out["total_count"] == 20
        assert len(out["results"]) < 20
        # `count` describes the payload, so it stays honest about what was sent.
        assert out["count"] == len(out["results"])

    async def test_an_untruncated_envelope_has_exactly_the_old_keys(self):
        """TC-RESP-011: no new keys on the common path.

        A `truncated: false` on every response would be a compatibility change to
        every client for a case that did not happen.
        """
        facade = self._facade(16_777_216)
        out = facade._results([_doc(0)], "memories")
        assert set(out) == {"results", "count"}

    async def test_every_read_method_goes_through_the_cap(self):
        """TC-RESP-012: five read methods, one enforcement point.

        Enforcing in each method separately is how one of them ends up missing it,
        which is the shape of the original bug — the knob existed in one place and
        was honoured in none.
        """
        import inspect

        from agent_memory.memory import AsyncMemory

        for name in ("recall", "search", "recall_activity", "get_thread",
                     "get_activity_by_correlation"):
            source = inspect.getsource(getattr(AsyncMemory, name))
            assert "self._results(" in source, (
                f"{name} builds its own envelope and bypasses max_response_bytes"
            )
            assert '"count": len(' not in source, (
                f"{name} still hand-rolls the envelope"
            )
