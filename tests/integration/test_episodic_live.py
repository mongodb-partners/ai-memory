"""Episodic memory against a real Atlas cluster.

Everything about the episodic tier is verifiable under mocks except the one
thing that is genuinely new: ``$rankFusion`` over the ``episodes`` collection.
A mock cannot execute ``$vectorSearch`` or ``$search``, so it cannot catch an
undeclared filter field or a ``string``-typed token field — and both of those
fail by returning nothing, with no error. That failure looks exactly like "no
matching documents," which is why it needs a live test rather than review.

Gated on ``MONGODB_CONNECTION_STRING`` (see conftest). No server involved:
these drive the library directly.

    MONGODB_CONNECTION_STRING="mongodb+srv://..." \\
        uv run pytest tests/integration/test_episodic_live.py -q

Each test uses a UUID-suffixed user and thread id, so a run leaves no state
another run can see and parallel runs cannot collide. A module-scoped fixture
builds one AsyncMemory: index creation is slow, and re-creating it per test
would dominate the runtime without testing anything extra.
"""

import asyncio
import uuid

import pytest
import pytest_asyncio

# asyncio_mode is "auto" in pyproject.toml, so async tests need no mark. The
# loop scope does need one: the `memory` fixture is module-scoped (create() is
# slow), and pytest-asyncio's default function-scoped loop would tear the loop
# out from under it between tests. AsyncMongoClient binds to the loop it was
# created on, so the two scopes have to agree.
pytestmark = [pytest.mark.live_atlas, pytest.mark.asyncio(loop_scope="module")]

# Atlas Search indexes build asynchronously. A fresh collection's index can take
# tens of seconds to reach `queryable`, and a query against a building index
# returns nothing rather than an error.
INDEX_TIMEOUT_SECONDS = 300
SEARCH_SETTLE_SECONDS = 60


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def memory():
    """One live AsyncMemory for the module, torn down at the end."""
    from agent_memory import AsyncMemory, MemoryConfig

    config = MemoryConfig.from_env(
        # Block until the search indexes exist, rather than racing them.
        await_search_indexes=True,
        # Small batches and a short interval so a test's flush is quick.
        episodic_batch_size=5,
        episodic_flush_interval_seconds=0.2,
    )
    mem = await AsyncMemory.create(config)
    try:
        yield mem
    finally:
        await mem.close()


async def _wait_for_queryable(mem, collection_name: str, index_name: str) -> dict:
    """Poll until an Atlas Search index reports queryable, or fail loudly.

    Returns the index's status document. A test that queried a still-building
    index would fail with an empty result set and blame the pipeline.
    """
    col = mem.episodic_service.episodes.database[collection_name]
    deadline = asyncio.get_running_loop().time() + INDEX_TIMEOUT_SECONDS
    last = None
    while asyncio.get_running_loop().time() < deadline:
        cursor = await col.list_search_indexes()
        for idx in await cursor.to_list(None):
            if idx.get("name") == index_name:
                last = idx
                if idx.get("queryable"):
                    return idx
        await asyncio.sleep(5)
    pytest.fail(
        f"{collection_name}.{index_name} never became queryable within "
        f"{INDEX_TIMEOUT_SECONDS}s; last status: {last}"
    )


class TestIndexes:
    """The indexes exist, are queryable, and agree with the provider."""

    async def test_both_search_indexes_become_queryable(self, memory):
        vector = await _wait_for_queryable(
            memory, "episodes", "episodes_vector_index"
        )
        text = await _wait_for_queryable(memory, "episodes", "episodes_fts_index")
        assert vector["queryable"] is True
        assert text["queryable"] is True

    async def test_episodes_dimension_matches_memories(self, memory):
        """A per-collection dimension drift is silent until recall returns nothing.

        Compared against the *resolved* dimension, which is what every writer and
        every index actually uses. ``config.embedding_dimension`` is the wrong
        yardstick on a Voyage deployment: it holds Titan's inherited 1536 while the
        embedder emits 1024. This used to read the config field and pass only
        because ``_create_embedding_provider`` overwrote it in place during
        ``create()`` — the resolution is a published value now, so the assertion
        has to ask for it rather than rely on a side effect.
        """
        db = memory.episodic_service.episodes.database

        async def _dims(collection: str, index: str) -> int:
            cursor = await db[collection].list_search_indexes()
            for idx in await cursor.to_list(None):
                if idx.get("name") == index:
                    fields = idx["latestDefinition"]["fields"]
                    vec = next(f for f in fields if f["type"] == "vector")
                    return vec["numDimensions"]
            pytest.fail(f"{collection}.{index} not found")

        episodes = await _dims("episodes", "episodes_vector_index")
        memories = await _dims("memories", "memories_vector_index")
        assert episodes == memories == memory.providers.embedding_spec.dimension

    async def test_vector_index_declares_every_prefilter_field(self, memory):
        """An undeclared filter field makes the branch return nothing, silently.

        recall_activity pre-filters on all three of these, so this is the check
        that a passing search test isn't passing by accident.
        """
        cursor = await memory.episodic_service.episodes.list_search_indexes()
        idx = next(
            i
            for i in await cursor.to_list(None)
            if i["name"] == "episodes_vector_index"
        )
        declared = {
            f["path"] for f in idx["latestDefinition"]["fields"] if f["type"] == "filter"
        }
        assert {"user_id", "thread_id", "agent_name"} <= declared


class TestWritePath:
    """A logged turn reaches Atlas with the shape the reference documents."""

    async def test_a_final_step_turn_is_stored_and_embedded(self, memory):
        """A tool call mid-turn, then an answer — the shape that gets embedded.

        The tool call has to be on a *non-final* assistant message. is_final_step
        looks at the last assistant message, so hanging tool_calls off the final
        answer makes the whole turn read as mid-turn and skips embedding.
        """
        user, thread = _uid("live-write"), _uid("thread")

        await memory.log_activity(
            user,
            thread,
            [
                {"type": "human", "content": "Book a vegetarian place for Friday"},
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "write_file",
                            "args": {"file_path": "booking.md", "content": "Nopalito"},
                        }
                    ],
                },
                {"type": "tool", "content": "written", "tool_call_id": "call_1"},
                {"type": "ai", "content": "Booked Nopalito at 7pm."},
            ],
            correlation_id="live-trace-write",
        )
        assert await memory.flush_activity(timeout=20.0) is True

        replay = await memory.get_thread(user, thread)
        assert replay["count"] == 1
        doc = replay["results"][0]

        assert doc["user_id"] == user
        assert doc["thread_id"] == thread
        assert doc["conversation_id"] == thread     # defaults to thread_id
        assert doc["agent_name"] == "main"
        assert doc["correlation_id"] == "live-trace-write"
        assert doc["step"] == 0
        assert doc["parent_step"] is None
        assert isinstance(doc["_id"], str)          # coerced, not an ObjectId
        assert len(doc["messages"]) == 4
        assert doc["files_touched"] == [
            {"path": "booking.md", "size": 8, "content_hash": None, "op": "write"}
        ]
        # Reads project the embedding out; the raw document has it.
        assert "embedding" not in doc

        raw = await memory.episodic_service.episodes.find_one(
            {"user_id": user, "thread_id": thread}
        )
        # The resolved dimension, not the declared one — see
        # test_episodes_dimension_matches_memories.
        assert len(raw["embedding"]) == memory.providers.embedding_spec.dimension
        assert raw["search_text"]

    async def test_steps_are_monotonic_within_a_thread(self, memory):
        """Ordering depends on a durable counter and a single consumer task."""
        user, thread = _uid("live-steps"), _uid("thread")

        for i in range(3):
            await memory.log_activity(
                user,
                thread,
                [
                    {"type": "human", "content": f"question {i}"},
                    {"type": "ai", "content": f"answer {i}"},
                ],
            )
        assert await memory.flush_activity(timeout=20.0) is True

        turns = (await memory.get_thread(user, thread, ascending=True))["results"]
        assert [t["step"] for t in turns] == [0, 1, 2]
        assert [t["parent_step"] for t in turns] == [None, 0, 1]

    async def test_a_mid_turn_step_is_stored_without_search_fields(self, memory):
        """episodic_embed_final_steps_only: a tool request has no answer yet."""
        user, thread = _uid("live-midstep"), _uid("thread")

        await memory.log_activity(
            user,
            thread,
            [
                {"type": "human", "content": "What's the weather?"},
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [{"name": "get_weather", "args": {}}],
                },
            ],
        )
        assert await memory.flush_activity(timeout=20.0) is True

        raw = await memory.episodic_service.episodes.find_one(
            {"user_id": user, "thread_id": thread}
        )
        assert raw is not None
        # Omitted together or not at all — never text without a vector.
        assert "search_text" not in raw
        assert "embedding" not in raw

    async def test_a_turn_without_a_user_id_is_never_stored(self, memory):
        """The write-side half of isolation. Discarded, not stored unscoped."""
        thread = _uid("thread-orphan")
        await memory.log_activity(
            "", thread, [{"type": "human", "content": "orphan"}]
        )
        await memory.flush_activity(timeout=5.0)

        count = await memory.episodic_service.episodes.count_documents(
            {"thread_id": thread}
        )
        assert count == 0


class TestHybridRecall:
    """$rankFusion over episodes — the part mocks cannot reach."""

    @pytest_asyncio.fixture(scope="class", loop_scope="module")
    @classmethod
    async def planted(cls, memory):
        """Plant two users' turns, then wait for the search indexes to catch up.

        Class-scoped: the settle wait is the expensive part, and every test in
        this class asks a question about the same planted data.
        """
        await _wait_for_queryable(memory, "episodes", "episodes_vector_index")
        await _wait_for_queryable(memory, "episodes", "episodes_fts_index")

        owner, other = _uid("live-owner"), _uid("live-other")
        thread_a, thread_b = _uid("thread-a"), _uid("thread-b")

        await memory.log_activity(
            owner, thread_a,
            [
                {"type": "human", "content": "Find me a vegetarian restaurant"},
                {"type": "ai", "content": "Nopalito on Broderick — fully vegetarian."},
            ],
            correlation_id="live-trace-recall",
        )
        await memory.log_activity(
            owner, thread_b,
            [
                {"type": "human", "content": "Rebalance the portfolio into bonds"},
                {"type": "ai", "content": "Moved 30% from equities into treasuries."},
            ],
            correlation_id="live-trace-recall",
            agent_name="finance",
        )
        # Same question, different user. This is the isolation control.
        await memory.log_activity(
            other, _uid("thread-c"),
            [
                {"type": "human", "content": "Find me a vegetarian restaurant" },
                {"type": "ai", "content": "Greens in Fort Mason is vegetarian."},
            ],
        )
        assert await memory.flush_activity(timeout=30.0) is True

        # Newly-written documents are not immediately searchable: the index
        # replicates asynchronously. Poll rather than sleeping blind.
        deadline = asyncio.get_running_loop().time() + SEARCH_SETTLE_SECONDS
        while asyncio.get_running_loop().time() < deadline:
            hits = (await memory.recall_activity(owner, "vegetarian restaurant"))["results"]
            if hits:
                break
            await asyncio.sleep(3)
        else:
            pytest.fail(
                f"planted turns were not searchable within {SEARCH_SETTLE_SECONDS}s"
            )

        return {"owner": owner, "other": other, "a": thread_a, "b": thread_b}

    async def test_recall_finds_the_planted_turn(self, memory, planted):
        found = await memory.recall_activity(planted["owner"], "vegetarian restaurant")
        hits = found["results"]
        assert found["count"] == len(hits) > 0
        assert any("Nopalito" in str(h.get("messages")) for h in hits)

    async def test_another_users_query_cannot_reach_it(self, memory, planted):
        """user_id goes into both branches, so isolation is the engine's job."""
        hits = (await memory.recall_activity(planted["other"], "Nopalito Broderick"))["results"]
        assert all(h["user_id"] == planted["other"] for h in hits)
        assert not any("Nopalito" in str(h.get("messages")) for h in hits)

    async def test_thread_filter_narrows_to_one_thread(self, memory, planted):
        """thread_id is a declared filter field — undeclared, this returns []."""
        hits = (await memory.recall_activity(
            planted["owner"], "portfolio bonds", thread_id=planted["b"]
        ))["results"]
        assert hits
        assert all(h["thread_id"] == planted["b"] for h in hits)

    async def test_agent_name_filter_narrows_to_one_agent(self, memory, planted):
        hits = (await memory.recall_activity(
            planted["owner"], "treasuries equities", agent_name="finance"
        ))["results"]
        assert hits
        assert all(h["agent_name"] == "finance" for h in hits)

    async def test_exact_term_recall_proves_the_text_branch_runs(self, memory, planted):
        """A rare proper noun is the text branch's job, not the vector branch's."""
        hits = (await memory.recall_activity(planted["owner"], "Nopalito"))["results"]
        assert hits

    async def test_correlation_lookup_spans_threads(self, memory, planted):
        turns = (await memory.get_activity_by_correlation(
            planted["owner"], "live-trace-recall"
        ))["results"]
        assert {t["thread_id"] for t in turns} == {planted["a"], planted["b"]}


class TestRetention:
    """collMod round-trip on the real TTL index."""

    async def test_ttl_changes_in_place_and_can_be_dropped(self, memory):
        col = memory.episodic_service.episodes
        admin = _uid("live-admin")

        async def _ttl() -> int | None:
            cursor = await col.list_indexes()
            for idx in await cursor.to_list(None):
                if idx.get("name") == "ix_episodes_ttl":
                    return idx.get("expireAfterSeconds")
            return None

        original = await _ttl()
        assert original is not None, "the TTL index should exist after create()"

        try:
            res = await memory.set_activity_retention(admin, ttl_seconds=3600)
            assert res["status"] in ("updated", "created")
            assert await _ttl() == 3600

            # The in-place path: a second change must not drop and rebuild.
            res = await memory.set_activity_retention(admin, ttl_seconds=7200)
            assert res["status"] in ("updated", "created")
            assert await _ttl() == 7200

            # None is meaningful — it removes the TTL and keeps the log forever.
            res = await memory.set_activity_retention(admin, ttl_seconds=None)
            assert res["status"] == "removed"
            assert await _ttl() is None
        finally:
            # Leave the cluster as we found it, even on failure — a dropped TTL
            # index would silently make every later run's data permanent.
            await memory.set_activity_retention(admin, ttl_seconds=original)


class TestWriterHealth:
    """The counters a probe reads, against real writes."""

    async def test_counters_account_for_every_enqueued_turn(self, memory):
        user, thread = _uid("live-stats"), _uid("thread")
        before = memory.activity_stats()

        for i in range(4):
            await memory.log_activity(
                user, thread,
                [
                    {"type": "human", "content": f"q{i}"},
                    {"type": "ai", "content": f"a{i}"},
                ],
            )
        assert await memory.flush_activity(timeout=20.0) is True

        after = memory.activity_stats()

        # The counters are process-wide and `memory` is module-scoped, so other
        # tests in this module contribute to the same totals. Assert the
        # invariant — everything enqueued is accounted for — rather than an exact
        # delta, which would make this test depend on execution order.
        assert after["enqueued"] - before["enqueued"] >= 4
        assert after["written"] - before["written"] >= 4
        assert after["enqueued"] == after["written"] + after["dropped"]

        # These *are* exact: this test's four turns must cost nothing.
        assert after["dropped"] == before["dropped"]
        assert after["write_failures"] == before["write_failures"]
        assert after["embed_failures"] == before["embed_failures"]
        assert after["worker_alive"] is True
        assert after["last_write_ts"] is not None

        # And the writes are actually there, which the counters alone don't prove.
        assert (await memory.get_thread(user, thread))["count"] == 4
