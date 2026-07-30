"""Sample-UI backend: the turn loop, the SSE transport, and the cache axis.

These cover the demo's failure modes rather than its happy path, because the
happy path is visible in rehearsal and these are not:

* **The toggle must be honest.** Memory OFF has to skip recall *and* bypass the
  response cache. If a memory-OFF turn can be served a memory-ON cached answer,
  the screen shows identical answers with the toggle flipped and the talk's
  central claim is false in front of the audience. Verification item 10.
* **The stream must terminate and must not truncate.** A dropped terminator hangs
  the browser; an early exit loses the trailing ``memory{phase:write}`` frames,
  which are the entire point of the memory panel.
* **A memory failure must not cost the answer.** The user already has their
  tokens; a write error is worth a frame, not an exception.

The demo backend lives under ``examples/`` with a hyphenated parent directory, so
it is loaded by putting that directory on ``sys.path`` — the modules use relative
imports and must be imported as the ``server`` package, not by file path.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_DEMO_ROOT = Path(__file__).resolve().parents[2] / "examples" / "memory-ui"
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

pytest.importorskip(
    "sse_starlette",
    reason="demo backend transport; install the 'demo' extra",
)

from server.cache_key import DemoResponseCache  # noqa: E402
from server.history import ConversationHistory  # noqa: E402
from server.prompt import build_call, build_system_text, format_context  # noqa: E402
from server.sse import sse_response  # noqa: E402
from server.turn import (  # noqa: E402
    TurnRunner,
    collapse_stm_ltm_pairs,
    project_episode_hit,
    project_memory_hit,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _cursor(docs):
    cursor = MagicMock()
    cursor.to_list = AsyncMock(return_value=docs)
    return cursor


def _memory(*, answer: str, semantic=None, episodic=None):
    """An AsyncMemory stand-in whose LLM records what it was asked.

    ``llm_calls`` is a real list rather than a MagicMock attribute because
    ``hasattr`` on a MagicMock is unconditionally true — an assertion that the
    model was *not* called has to read a container that only grows when it was.
    """
    memory = MagicMock()
    memory.llm_calls = []

    async def _stream(messages, **kwargs):
        memory.llm_calls.append((messages, kwargs))
        memory.last_messages = messages
        memory.last_kwargs = kwargs
        for chunk in answer.split(" "):
            yield chunk + " "

    memory.providers.llm.chat_stream = _stream
    memory.search = AsyncMock(return_value={"results": semantic or []})
    memory.recall_activity = AsyncMock(return_value={"results": episodic or []})
    memory.add = AsyncMock(return_value={})
    memory.log_activity = AsyncMock(return_value=None)
    return memory


def _cache(hit: dict | None = None):
    cache = MagicMock()
    cache.lookup = AsyncMock(return_value=hit)
    cache.store = AsyncMock(return_value=None)
    return cache


def _runner(memory, cache=None, history=None):
    return TurnRunner(
        memory,
        provider_name="bedrock",
        cache=cache if cache is not None else _cache(),
        history=history or ConversationHistory(),
    )


async def _collect(runner, **kwargs):
    kwargs.setdefault("user_id", "u1")
    kwargs.setdefault("thread_id", "t1")
    kwargs.setdefault("message", "what should I cook?")
    kwargs.setdefault("correlation_id", "cid-1")
    kwargs.setdefault("memory_enabled", True)
    return [f async for f in runner.run(**kwargs)]


def _events(frames, event: str):
    return [f for f in frames if f["event"] == event]


def _payloads(frames, event: str = "memory"):
    return [json.loads(f["data"]) for f in _events(frames, event)]


def _text(frames):
    return "".join(f["data"] for f in _events(frames, "token"))


# ── The toggle ───────────────────────────────────────────────────────


class TestMemoryToggleIsHonest:
    """Verification item 10 — the failure that would happen on stage."""

    async def test_memory_off_never_consults_the_cache(self):
        """The demo-breaking case: a memory-ON answer replayed with memory OFF.

        Bypassing the cache entirely is stricter than keying on the flag, and
        deliberately so — this is the one failure with no mid-talk recovery.
        """
        cache = _cache(hit={"response": "You're allergic to shellfish, so...",
                            "score": 0.99})
        runner = _runner(_memory(answer="I need to know more."), cache=cache)

        frames = await _collect(runner, memory_enabled=False)

        cache.lookup.assert_not_awaited()
        assert "shellfish" not in _text(frames)

    async def test_memory_off_does_not_recall(self):
        memory = _memory(answer="Tell me your preferences.")
        runner = _runner(memory)

        await _collect(runner, memory_enabled=False)

        memory.search.assert_not_awaited()
        memory.recall_activity.assert_not_awaited()

    async def test_memory_off_writes_nothing(self):
        """A memory-OFF pass must leave no trace, or the second demo pass starts
        from state the first one created."""
        memory = _memory(answer="ok")
        cache = _cache()
        runner = _runner(memory, cache=cache)

        await _collect(runner, memory_enabled=False)

        memory.add.assert_not_awaited()
        memory.log_activity.assert_not_awaited()
        cache.store.assert_not_awaited()

    async def test_the_same_prompt_yields_a_different_prompt_with_the_toggle_on(self):
        """The contrast is real at the prompt level: memory ON sends the model
        facts that memory OFF does not, from one identical user message."""
        semantic = [{"tier": "ltm", "summary": "Allergic to shellfish",
                     "importance": 0.9, "score": 0.88}]
        memory_on = _memory(answer="Skip the shrimp.", semantic=semantic)
        memory_off = _memory(answer="Skip the shrimp.", semantic=semantic)

        await _collect(_runner(memory_on), memory_enabled=True)
        await _collect(_runner(memory_off), memory_enabled=False)

        on_system = memory_on.last_kwargs["system"][0]["text"]
        off_system = memory_off.last_kwargs["system"][0]["text"]
        assert "shellfish" in on_system
        assert "shellfish" not in off_system
        assert on_system != off_system


class TestCacheKeyIncludesTheMemoryAxis:
    async def test_lookup_filters_on_both_user_and_memory_mode(self):
        collection = MagicMock()
        collection.aggregate = AsyncMock(return_value=_cursor([]))
        embedding = MagicMock()
        embedding.generate_embedding = AsyncMock(return_value=[0.1, 0.2])
        cache = DemoResponseCache({"demo_response_cache": collection}, embedding)

        await cache.lookup("u1", "q", memory_enabled=True)

        stage = collection.aggregate.call_args.args[0][0]["$vectorSearch"]
        assert stage["filter"] == {
            "user_id": {"$eq": "u1"},
            "memory_enabled": {"$eq": True},
        }

    async def test_a_below_threshold_match_is_a_miss(self):
        """A loose threshold turns a deliberate second ask into a replay, which
        looks like the agent refusing to think."""
        collection = MagicMock()
        collection.aggregate = AsyncMock(
            return_value=_cursor([{"response": "cached", "score": 0.80}])
        )
        embedding = MagicMock()
        embedding.generate_embedding = AsyncMock(return_value=[0.1])
        cache = DemoResponseCache({"demo_response_cache": collection}, embedding)

        assert await cache.lookup("u1", "q", memory_enabled=True) is None

    async def test_a_lookup_failure_is_a_bypass_not_an_error(self):
        collection = MagicMock()
        collection.aggregate = AsyncMock(side_effect=RuntimeError("index missing"))
        embedding = MagicMock()
        embedding.generate_embedding = AsyncMock(return_value=[0.1])
        cache = DemoResponseCache({"demo_response_cache": collection}, embedding)

        assert await cache.lookup("u1", "q", memory_enabled=True) is None

    async def test_an_empty_answer_is_not_cached(self):
        """Caching an empty answer poisons the demo: every later ask replays
        nothing at all."""
        collection = MagicMock()
        collection.insert_one = AsyncMock()
        embedding = MagicMock()
        embedding.generate_embedding = AsyncMock(return_value=[0.1])
        cache = DemoResponseCache({"demo_response_cache": collection}, embedding)

        await cache.store("u1", "q", "   ", memory_enabled=True)

        collection.insert_one.assert_not_awaited()


# ── Frame sequence ───────────────────────────────────────────────────


class TestFrameSequence:
    async def test_recall_frames_precede_the_first_token(self):
        """The panel's claim is causal — recall happened *before* the answer. Out
        of order, the demo shows memory arriving after the fact."""
        memory = _memory(
            answer="Try the risotto.",
            semantic=[{"tier": "ltm", "summary": "Vegetarian", "score": 0.9}],
        )
        frames = await _collect(_runner(memory), memory_enabled=True)

        events = [f["event"] for f in frames]
        first_token = events.index("token")
        recalls = [
            i for i, f in enumerate(frames)
            if f["event"] == "memory" and json.loads(f["data"])["phase"] == "recall"
        ]
        assert recalls
        assert max(recalls) < first_token

    async def test_every_tier_reports_even_when_empty(self):
        """An absent tier and an empty tier look identical to the audience unless
        the empty one says so — that is what makes the OFF pass legible."""
        frames = await _collect(_runner(_memory(answer="hi")), memory_enabled=True)

        recalled = {
            p["tier"] for p in _payloads(frames) if p["phase"] == "recall"
        }
        assert recalled == {"ltm", "stm", "episodic"}

    async def test_write_frames_follow_the_answer(self):
        memory = _memory(answer="Roast the peppers.")
        frames = await _collect(_runner(memory), memory_enabled=True)

        writes = [p for p in _payloads(frames) if p["phase"] == "write"]
        assert [w["tier"] for w in writes] == ["ltm", "episodic"]
        last_token = max(
            i for i, f in enumerate(frames) if f["event"] == "token"
        )
        write_indexes = [
            i for i, f in enumerate(frames)
            if f["event"] == "memory" and json.loads(f["data"])["phase"] == "write"
        ]
        assert min(write_indexes) > last_token

    async def test_a_cache_hit_replays_tokens_and_skips_the_model(self):
        cache = _cache(hit={"response": "Shrimp is out; try the mushroom orzo.",
                            "score": 0.99})
        memory = _memory(answer="SHOULD NOT BE CALLED")
        frames = await _collect(_runner(memory, cache=cache), memory_enabled=True)

        assert _text(frames) == "Shrimp is out; try the mushroom orzo."
        # The saving a cache is claimed to deliver: no inference happened.
        assert memory.llm_calls == []
        assert all(p["cache_hit"] for p in _payloads(frames))

    async def test_token_frames_carry_raw_text_not_json(self):
        """The existing client reads `token` data as text. JSON-encoding it would
        render quotes on screen."""
        frames = await _collect(_runner(_memory(answer="one two")))

        for f in _events(frames, "token"):
            assert not f["data"].startswith('"')


# ── Failure containment ──────────────────────────────────────────────


class TestFailuresDoNotCostTheAnswer:
    async def test_a_recall_failure_degrades_to_no_context(self):
        memory = _memory(answer="Here's an idea.")
        memory.search = AsyncMock(side_effect=RuntimeError("no search index"))
        frames = await _collect(_runner(memory), memory_enabled=True)

        assert _text(frames).strip() == "Here's an idea."

    async def test_one_failed_tier_does_not_sink_the_other(self):
        """Without per-tier guarding, a broken episodic index empties the whole
        panel and the audience cannot see why."""
        memory = _memory(
            answer="ok",
            semantic=[{"tier": "ltm", "summary": "Cooks for six", "score": 0.8}],
        )
        memory.recall_activity = AsyncMock(side_effect=RuntimeError("boom"))
        frames = await _collect(_runner(memory), memory_enabled=True)

        by_tier = {p["tier"]: p for p in _payloads(frames) if p["phase"] == "recall"}
        assert by_tier["ltm"]["hits"]
        assert by_tier["episodic"]["hits"] == []

    async def test_a_write_failure_is_reported_not_raised(self):
        memory = _memory(answer="done")
        memory.add = AsyncMock(side_effect=RuntimeError("write concern"))
        frames = await _collect(_runner(memory), memory_enabled=True)

        ltm = next(
            p for p in _payloads(frames)
            if p["phase"] == "write" and p["tier"] == "ltm"
        )
        assert "write concern" in ltm["error"]
        # The episodic write is still attempted.
        memory.log_activity.assert_awaited_once()


# ── History is not memory ────────────────────────────────────────────


class TestHistoryIsNotMemory:
    async def test_a_new_thread_starts_with_no_transcript(self):
        """The proof behind the cross-thread recall demo: a new thread carries
        nothing forward, so a recalled fact must have come from Atlas."""
        history = ConversationHistory()
        memory = _memory(answer="ok")
        runner = _runner(memory, history=history)

        await _collect(runner, thread_id="t1", message="I hate cilantro")
        await _collect(runner, thread_id="t2", message="what should I cook?")

        sent = [m["content"][0]["text"] for m in memory.last_messages]
        assert "I hate cilantro" not in sent

    def test_history_always_starts_on_a_user_turn(self):
        """Some providers reject a leading assistant message outright."""
        history = ConversationHistory()
        history.append("u", "t", "user", "a")
        history.append("u", "t", "assistant", "b")
        history.append("u", "t", "user", "c")
        history.append("u", "t", "assistant", "d")

        assert history.turns("u", "t", limit=1)[0][0] == "user"

    def test_a_thread_is_bounded(self):
        history = ConversationHistory()
        for i in range(50):
            history.append("u", "t", "user", f"m{i}")

        assert len(history.turns("u", "t", limit=99)) <= 16


# ── Prompt shape ─────────────────────────────────────────────────────


class TestPromptShape:
    def test_absent_and_empty_context_produce_the_same_system_text(self):
        """Otherwise the model can infer from the prompt's shape that something
        was withheld, and says so — narrating the demo's mechanism."""
        assert build_system_text(None) == build_system_text("")

    def test_no_sampling_parameters_are_sent(self):
        """The newest Claude models reject `temperature` outright. The provider
        recovers, but the demo should not depend on that recovery path."""
        _, kwargs = build_call("bedrock", "sys", [("user", "hi")])

        assert set(kwargs["inferenceConfig"]) == {"maxTokens"}

    def test_blank_turns_are_dropped(self):
        """An empty content block is a hard API error, and a model that returned
        nothing is exactly when one would appear."""
        messages, _ = build_call(
            "bedrock", "sys", [("user", "hi"), ("assistant", ""), ("user", "again")]
        )

        assert len(messages) == 2

    def test_context_omits_scores(self):
        """Scores belong on screen, not in the prompt — a model handed a 0.87
        will sometimes quote it back."""
        block = format_context(
            {"ltm": [{"text": "Allergic to shellfish", "score": 0.87,
                      "importance": 0.9}]}
        )

        assert "Allergic to shellfish" in block
        assert "0.87" not in block

    def test_each_provider_gets_its_own_message_shape(self):
        bedrock, bk = build_call("bedrock", "sys", [("user", "hi")])
        anthropic, ak = build_call("anthropic", "sys", [("user", "hi")])
        openai, ok = build_call("openai", "sys", [("user", "hi")])

        assert bedrock[0]["content"] == [{"text": "hi"}]
        assert bk["system"] == [{"text": "sys"}]
        assert anthropic[0]["content"] == "hi"
        assert ak["system"] == "sys"
        # OpenAI has no system parameter; it is the first message instead.
        assert openai[0] == {"role": "system", "content": "sys"}
        assert "system" not in ok


# ── Projections ──────────────────────────────────────────────────────


class TestProjections:
    def test_a_memory_hit_keeps_its_score_and_importance(self):
        """The retail UI this borrows from joined recalled memories into one
        string and dropped the scores — the evidence a memory demo needs."""
        hit = project_memory_hit(
            {"summary": "Cooks for six", "final_score": 0.91,
             "importance": 0.8, "access_count": 3, "tier": "ltm"}
        )

        assert hit["score"] == 0.91
        assert hit["importance"] == 0.8
        assert hit["access_count"] == 3

    def test_an_episode_hit_surfaces_tools_and_files(self):
        hit = project_episode_hit(
            {
                "search_text": "asked about dinner",
                "step": 2,
                "messages": [
                    {"type": "ai", "tool_calls": [{"name": "search_recipes"}]},
                    {"type": "ai", "tool_calls": [{"name": "write_file"}]},
                ],
                "files_touched": [{"path": "menu.md"}, {"not_a_path": 1}],
            }
        )

        assert hit["tools"] == ["search_recipes", "write_file"]
        assert hit["files"] == ["menu.md"]
        assert hit["step"] == 2

    def test_the_rank_label_leads_with_position_not_the_raw_score(self):
        """A perfect top hit scores ~0.016 under RRF with k=60. Displayed raw,
        that reads as a failed match to an audience, and costs a paragraph of
        explanation from the stage."""
        hit = project_memory_hit({"summary": "x", "score": 0.0163934}, 0)

        assert hit["rank"].startswith("#1")
        # The measurement is still reported verbatim; the label is additional.
        assert hit["score"] == 0.0163934

    def test_a_zero_calibrated_score_is_not_mistaken_for_absent(self):
        """`final_score or score` would swallow a legitimate 0.0 and silently
        fall back to the raw fused rank — reporting a different number than the
        one that actually ordered the list."""
        hit = project_memory_hit({"summary": "x", "final_score": 0.0, "score": 0.9})

        assert hit["score"] == 0.0

    def test_a_scoreless_hit_still_gets_a_rank(self):
        """Browse mode lists documents without running a search, so there is no
        score — but the rows still need to be numbered."""
        hit = project_memory_hit({"summary": "x"}, 2)

        assert hit["rank"] == "#3"
        assert hit["score"] is None

    def test_content_is_used_when_no_summary_exists(self):
        """STM documents are unenriched — summary arrives later, from the
        worker. Showing a blank row would read as a recall failure."""
        hit = project_memory_hit({"content": "raw turn text", "tier": "stm"})

        assert hit["text"] == "raw turn text"

    def test_a_stored_refusal_does_not_displace_the_memory(self):
        """REQ-E-121. The worker no longer stores these, but documents written
        before that guard landed still carry them, and the panel is what shows
        them. Preferring `summary` unconditionally put "This text fragment is too
        brief and lacks sufficient context" on screen where the memory belonged."""
        hit = project_memory_hit(
            {
                "content": (
                    "I cook for four most nights and for six when guests come "
                    "over, so I plan around meals that scale without extra work."
                ),
                "summary": "This text fragment is too brief and lacks context.",
                "tier": "ltm",
            }
        )

        assert hit["text"].startswith("I cook for four")

    def test_a_real_summary_is_still_preferred_over_content(self):
        """The summary is the enriched, condensed form — that is what the tier is
        for, and a panel row has limited width."""
        hit = project_memory_hit(
            {
                "content": (
                    "I cook for four most nights and for six when guests come "
                    "over, so I plan around meals that scale without extra work."
                ),
                "summary": "Cooks for four to six; wants meals that scale.",
                "tier": "ltm",
            }
        )

        assert hit["text"] == "Cooks for four to six; wants meals that scale."


# ── STM/LTM twins ────────────────────────────────────────────────────


class TestCollapseStmLtmPairs:
    """``store_stm`` writes two documents per significant human message.

    Both are real and both match the same query, so the raw ``search`` primitive
    returns each fact twice — correct for a search primitive, wrong for a panel
    whose whole job is to make the tiers legible. Left uncollapsed, the demo shows
    "I'm allergic to shellfish" twice with two different importance scores, and the
    obvious audience question ("why is it in there twice?") costs more time than
    the talk has.
    """

    def test_the_short_term_twin_is_suppressed(self):
        docs = [
            {"_id": "ltm1", "source_stm_id": "stm1", "tier": "ltm"},
            {"_id": "stm1", "tier": "stm"},
        ]

        kept = collapse_stm_ltm_pairs(docs)

        assert [d["_id"] for d in kept] == ["ltm1"]

    def test_the_long_term_half_is_the_one_kept(self):
        """Not an arbitrary choice: only the LTM half has the enriched importance
        and the summary, which is the difference the tier exists to show."""
        docs = [
            {"_id": "stm1", "tier": "stm", "importance": 0.5, "summary": None},
            {"_id": "ltm1", "source_stm_id": "stm1", "tier": "ltm",
             "importance": 0.9, "summary": "Allergic to shellfish"},
        ]

        kept = collapse_stm_ltm_pairs(docs)

        assert len(kept) == 1
        assert kept[0]["importance"] == 0.9
        assert kept[0]["summary"] == "Allergic to shellfish"

    def test_an_unpaired_short_term_doc_survives(self):
        """Assistant turns and short human turns get no LTM candidate. Dropping
        every STM row would empty the short-term group the slide names."""
        docs = [
            {"_id": "stm_solo", "tier": "stm"},
            {"_id": "ltm1", "source_stm_id": "stm_other", "tier": "ltm"},
        ]

        kept = collapse_stm_ltm_pairs(docs)

        assert {d["_id"] for d in kept} == {"stm_solo", "ltm1"}

    def test_retrieval_order_is_preserved(self):
        """The panel's ranks come from this list's order, and reordering here
        would misreport what retrieval actually returned."""
        docs = [
            {"_id": "a", "tier": "ltm"},
            {"_id": "stm1", "tier": "stm"},
            {"_id": "b", "tier": "ltm", "source_stm_id": "stm1"},
            {"_id": "c", "tier": "stm"},
        ]

        assert [d["_id"] for d in collapse_stm_ltm_pairs(docs)] == ["a", "b", "c"]

    def test_object_ids_and_strings_compare_equal(self):
        """``_sanitize_doc`` coerces ``ObjectId`` to ``str`` on read, and the two
        id fields do not necessarily arrive in the same representation. Comparing
        them raw would silently never match, and the collapse would be a no-op
        that still passed every other test here."""
        from bson import ObjectId

        oid = ObjectId()
        docs = [
            {"_id": str(oid), "tier": "stm"},
            {"_id": "ltm1", "source_stm_id": oid, "tier": "ltm"},
        ]

        assert [d["_id"] for d in collapse_stm_ltm_pairs(docs)] == ["ltm1"]

    def test_an_empty_result_set_is_not_an_error(self):
        assert collapse_stm_ltm_pairs([]) == []


# ── SSE transport ────────────────────────────────────────────────────


class TestSSETransport:
    async def _drain(self, response):
        return [f async for f in response.body_iterator]

    async def test_correlation_leads_and_done_terminates(self):
        async def drive():
            yield {"event": "token", "data": "hi"}

        frames = await self._drain(sse_response(drive, "cid-9", 5.0))

        assert frames[0] == {"event": "correlation", "data": "cid-9"}
        assert frames[-1]["event"] == "done"

    async def test_a_full_queue_still_terminates(self):
        """The bug this guards: an in-band sentinel enqueued with put_nowait is
        dropped when the queue is full, and the browser hangs on a finished
        stream. 200 frames against a 64-slot queue forces that state.
        """
        async def drive():
            for i in range(200):
                yield {"event": "token", "data": str(i)}

        frames = await asyncio.wait_for(
            self._drain(sse_response(drive, "cid", 30.0)), timeout=10
        )

        assert frames[-1]["event"] == "done"
        assert len([f for f in frames if f["event"] == "token"]) == 200

    async def test_no_trailing_frames_are_lost(self):
        """The producer sets `finished` right after `done`. A consumer that
        checked the flag before draining would cut the write-back frames — the
        ones the memory panel exists to show."""
        async def drive():
            yield {"event": "token", "data": "answer"}
            yield {"event": "memory", "data": '{"phase":"write","tier":"ltm"}'}
            yield {"event": "memory", "data": '{"phase":"write","tier":"episodic"}'}

        frames = await self._drain(sse_response(drive, "cid", 5.0))

        assert [f["event"] for f in frames] == [
            "correlation", "token", "memory", "memory", "done",
        ]

    async def test_a_slow_turn_ends_in_a_timeout_error_frame(self):
        async def drive():
            yield {"event": "token", "data": "start"}
            await asyncio.sleep(5)
            yield {"event": "token", "data": "never"}

        frames = await asyncio.wait_for(
            self._drain(sse_response(drive, "cid", 0.05)), timeout=5
        )

        assert frames[-1] == {"event": "error", "data": "turn_timeout"}

    async def test_a_driver_crash_becomes_an_error_frame_with_the_trace_id(self):
        """A stack trace in the browser is unreadable on stage; a correlation id
        is something you can grep for while still talking."""
        async def drive():
            yield {"event": "token", "data": "partial"}
            raise RuntimeError("bedrock exploded")

        frames = await self._drain(sse_response(drive, "cid-42", 5.0))

        assert frames[-1]["event"] == "error"
        assert "cid-42" in frames[-1]["data"]

    async def test_leading_spaces_survive_the_wire(self):
        """The whitespace contract, locked because getting it wrong is invisible
        in code review and mangles every answer on screen.

        SSE frames a value as ``data: `` + value, and a conforming client strips
        exactly *one* space after the colon. A token of ``" the"`` therefore goes
        out as ``data:  the`` (two spaces) and comes back as ``" the"``. A client
        that instead calls ``.strip()`` on the line — the obvious-looking thing —
        eats the word boundary, and the answer renders as "Roastthepeppers".
        """
        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        chunks = ["Roast", " the", " peppers", " and", " serve", "."]
        app = FastAPI()

        @app.post("/s")
        async def _s():
            async def drive():
                for chunk in chunks:
                    yield {"event": "token", "data": chunk}

            return sse_response(drive, "cid", 5.0)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t"
        ) as client:
            raw = (await client.post("/s")).text

        assert "data:  the" in raw  # two spaces: one padding, one meaningful
        received = "".join(
            line[6:] if line.startswith("data: ") else line[5:]
            for line in raw.split("\r\n")
            if line.startswith("data:")
            and line[5:].lstrip() not in ("cid", "[DONE]")
        )
        assert received == "".join(chunks)

    async def test_the_generator_is_closed_on_exit(self):
        """The driver holds a Bedrock response and a Motor cursor; neither is
        released by garbage collection alone."""
        closed = asyncio.Event()

        async def drive():
            try:
                yield {"event": "token", "data": "x"}
            finally:
                closed.set()

        await self._drain(sse_response(drive, "cid", 5.0))

        assert closed.is_set()
