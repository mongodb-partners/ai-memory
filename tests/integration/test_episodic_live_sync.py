"""The blocking ``Memory`` API against a real Atlas cluster.

Separate module from ``test_episodic_live.py`` on purpose. That file carries a
module-level ``pytest.mark.asyncio(loop_scope="module")``, and this test is
deliberately *not* async — ``Memory`` owns its own event loop on a daemon
thread, and the whole point is that it works from a plain synchronous function.
A sync test inheriting an asyncio mark warns rather than exercising that.

Gated on ``MONGODB_CONNECTION_STRING`` (see conftest).
"""

import uuid

import pytest

pytestmark = [pytest.mark.live_atlas]


def test_memory_logs_and_replays_without_an_async_context():
    from agent_memory import Memory, MemoryConfig

    suffix = uuid.uuid4().hex[:10]
    user, thread = f"live-sync-{suffix}", f"thread-{suffix}"
    config = MemoryConfig.from_env(
        # This test does not search, so there is no reason to block on index
        # creation — the write and replay paths do not need a queryable index.
        await_search_indexes=False,
        episodic_flush_interval_seconds=0.2,
    )

    with Memory(config) as mem:
        mem.log_activity(
            user,
            thread,
            [
                {"type": "human", "content": "sync path check"},
                {"type": "ai", "content": "logged from a plain function"},
            ],
        )
        assert mem.flush_activity(timeout=20.0) is True
        replay = mem.get_thread(user, thread)
        assert mem.activity_stats()["written"] >= 1

    # Same envelope as the async facade: {"results": [...], "count": n}.
    assert replay["count"] == 1
    turn = replay["results"][0]
    assert turn["step"] == 0
    assert turn["user_id"] == user
    assert isinstance(turn["_id"], str)     # coerced, not an ObjectId
