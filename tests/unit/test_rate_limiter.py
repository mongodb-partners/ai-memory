"""Tests for RateLimiter — fixed-window, counted atomically.

These tests were rewritten when the limiter was. The previous implementation
counted documents in a window and *then* inserted its own:

    count = await collection.count_documents({...})
    if count >= limit: return False
    await collection.insert_one({...})

The old tests asserted exactly that shape — `count_documents.call_args`,
`insert_one.assert_called_once()` — and every one of them passed while the limiter
did not limit. Between the count and the insert there is no lock, so under
concurrency every request in a burst read the same below-limit count and every one
was admitted. The suite could not see it because each test made one sequential
call, which is the only access pattern the old code handled correctly.

So the headline test here is `TestConcurrentBurst`: N simultaneous callers against
one limit, which is the traffic a rate limiter exists to bound and the traffic the
old tests never generated. The single-call cases are kept too — they are still
worth asserting, they were just never sufficient.

Requirement: REQ-E-145 (a rate limit holds under concurrency).
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from pymongo import ReturnDocument
from pymongo.errors import PyMongoError

from agent_memory.core.config import MCPConfig
from agent_memory.services.rate_limiter import RateLimiter


def _make_config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


def _make_collection(post_increment_count: int = 1):
    """A collection whose `find_one_and_update` returns a fixed post-`$inc` count.

    `post_increment_count` is the value the limiter reads back — i.e. how many
    requests have been made in this window *including* the current one. That
    off-by-one against the old `count_documents` fixtures is the whole semantic
    change: the count now includes the caller, so the comparison is `>` rather
    than `>=`.
    """
    col = MagicMock()
    col.find_one_and_update = AsyncMock(
        return_value={"count": post_increment_count}
    )
    return col


class _AtomicFakeCollection:
    """A single-document counter that increments the way MongoDB does.

    The `$inc` and the read of its result are one indivisible step — the await
    that models network latency happens *after* the increment, never between the
    read and the write. That is precisely the guarantee MongoDB gives for updates
    to a single document, and it is the guarantee the old count-then-insert code
    did not have.
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.calls = 0

    async def find_one_and_update(self, query, update, **kwargs):
        self.calls += 1
        key = repr(query["_id"])
        # Increment first, with no await in between — this is the atomic part.
        self.counts[key] = self.counts.get(key, 0) + update["$inc"]["count"]
        result = self.counts[key]
        # Now yield, as a real round trip would. Anything racy in the caller shows
        # up here, because every other pending caller gets to run at this point.
        await asyncio.sleep(0)
        return {"count": result, **update["$setOnInsert"]}


class TestRateLimiterCheck:
    """check_rate_limit enforces the window limit."""

    async def test_within_limit(self):
        col = _make_collection(post_increment_count=5)
        config = _make_config(rate_limit_enabled=True, rate_limit_max_requests=100)
        limiter = RateLimiter(col, config)

        assert await limiter.check_rate_limit("user1", "store_memory") is True
        # One round trip, not two: the count and the decision are one operation.
        col.find_one_and_update.assert_awaited_once()

    async def test_exceeds_limit(self):
        col = _make_collection(post_increment_count=11)
        config = _make_config(rate_limit_enabled=True, rate_limit_max_requests=10)
        limiter = RateLimiter(col, config)

        assert await limiter.check_rate_limit("user1", "store_memory") is False

    async def test_disabled_always_allows(self):
        col = _make_collection()
        config = _make_config(rate_limit_enabled=False)
        limiter = RateLimiter(col, config)

        assert await limiter.check_rate_limit("user1", "store_memory") is True
        # Disabled means no round trip at all, not an unchecked one.
        col.find_one_and_update.assert_not_called()

    async def test_records_user_and_operation_outside_the_id(self):
        """`user_id` and `operation` are duplicated out of the composite `_id`.

        Not redundancy for its own sake: `wipe_user_data` deletes rate-limit
        records with a plain `{"user_id": ...}` query, and a value that exists
        only inside `_id` is invisible to that. The completeness fix and this one
        have to agree, so it is asserted here rather than assumed.
        """
        col = _make_collection()
        config = _make_config(rate_limit_enabled=True, rate_limit_max_requests=100)
        limiter = RateLimiter(col, config)

        await limiter.check_rate_limit("user1", "recall_memory")

        update = col.find_one_and_update.call_args.args[1]
        on_insert = update["$setOnInsert"]
        assert on_insert["user_id"] == "user1"
        assert on_insert["operation"] == "recall_memory"
        # Feeds the existing TTL index on `timestamp`, which is what expires the
        # bucket — without it the counters accumulate forever.
        assert isinstance(on_insert["timestamp"], datetime)

    async def test_counts_by_incrementing_one_shared_bucket(self):
        col = _make_collection()
        config = _make_config(
            rate_limit_enabled=True,
            rate_limit_max_requests=100,
            rate_limit_window_seconds=3600,
        )
        limiter = RateLimiter(col, config)

        await limiter.check_rate_limit("user1", "store_memory")

        query, update = col.find_one_and_update.call_args.args[:2]
        bucket = query["_id"]
        assert bucket["user_id"] == "user1"
        assert bucket["operation"] == "store_memory"
        assert isinstance(bucket["window_start"], datetime)
        assert update["$inc"] == {"count": 1}
        kwargs = col.find_one_and_update.call_args.kwargs
        assert kwargs["upsert"] is True
        # AFTER, not BEFORE: the decision is made on the post-increment value, so
        # the caller is counted. Reading BEFORE reintroduces an off-by-one that
        # lets one extra request through every window.
        assert kwargs["return_document"] is ReturnDocument.AFTER

    async def test_operations_are_limited_separately(self):
        col = _make_collection()
        config = _make_config(rate_limit_enabled=True, rate_limit_max_requests=100)
        limiter = RateLimiter(col, config)

        await limiter.check_rate_limit("user1", "store_memory")
        await limiter.check_rate_limit("user1", "recall_memory")

        buckets = [
            c.args[0]["_id"] for c in col.find_one_and_update.call_args_list
        ]
        assert buckets[0] != buckets[1]

    async def test_users_are_limited_separately(self):
        col = _make_collection()
        config = _make_config(rate_limit_enabled=True, rate_limit_max_requests=100)
        limiter = RateLimiter(col, config)

        await limiter.check_rate_limit("user1", "store_memory")
        await limiter.check_rate_limit("user2", "store_memory")

        buckets = [
            c.args[0]["_id"] for c in col.find_one_and_update.call_args_list
        ]
        assert buckets[0]["user_id"] == "user1"
        assert buckets[1]["user_id"] == "user2"
        assert buckets[0] != buckets[1]


class TestConcurrentBurst:
    """The case the old implementation got wrong and the old tests could not see."""

    async def test_a_burst_of_n_admits_exactly_the_limit(self):
        """20 simultaneous callers against a limit of 5 ⇒ 5 allowed, 15 refused.

        Under the count-then-insert shape all 20 read the same count of 0 and all
        20 were admitted. This is the assertion that distinguishes a limiter from
        a counter that happens to be consulted.
        """
        col = _AtomicFakeCollection()
        config = _make_config(rate_limit_enabled=True, rate_limit_max_requests=5)
        limiter = RateLimiter(col, config)

        results = await asyncio.gather(
            *(limiter.check_rate_limit("user1", "store_memory") for _ in range(20))
        )

        assert sum(results) == 5
        assert col.calls == 20

    async def test_a_burst_from_two_users_does_not_share_a_budget(self):
        """One user exhausting the limit must not throttle another.

        A bucket keyed on `operation` alone — or a global counter — passes every
        single-caller test above and fails this one.
        """
        col = _AtomicFakeCollection()
        config = _make_config(rate_limit_enabled=True, rate_limit_max_requests=3)
        limiter = RateLimiter(col, config)

        a = await asyncio.gather(
            *(limiter.check_rate_limit("userA", "store_memory") for _ in range(10))
        )
        b = await asyncio.gather(
            *(limiter.check_rate_limit("userB", "store_memory") for _ in range(10))
        )

        assert sum(a) == 3
        assert sum(b) == 3

    async def test_sequential_calls_count_toward_the_same_window(self):
        col = _AtomicFakeCollection()
        config = _make_config(rate_limit_enabled=True, rate_limit_max_requests=2)
        limiter = RateLimiter(col, config)

        assert await limiter.check_rate_limit("user1", "op") is True
        assert await limiter.check_rate_limit("user1", "op") is True
        assert await limiter.check_rate_limit("user1", "op") is False


class TestWindowBoundaries:
    """`_window_start` floors to a shared, epoch-derived boundary."""

    def test_calls_in_the_same_window_share_a_bucket(self):
        config = _make_config(rate_limit_window_seconds=3600)
        limiter = RateLimiter(MagicMock(), config)

        base = datetime(2026, 8, 4, 11, 0, 30, tzinfo=UTC)
        assert limiter._window_start(base) == limiter._window_start(
            base + timedelta(minutes=20)
        )

    def test_calls_in_different_windows_do_not(self):
        config = _make_config(rate_limit_window_seconds=3600)
        limiter = RateLimiter(MagicMock(), config)

        base = datetime(2026, 8, 4, 11, 0, 30, tzinfo=UTC)
        assert limiter._window_start(base) != limiter._window_start(
            base + timedelta(hours=2)
        )

    def test_the_boundary_is_derived_from_the_epoch_not_first_request(self):
        """Every process must agree on where the window starts, uncoordinated.

        Anchoring to first-request time would give each process its own window
        and therefore its own counter document, so a limit of N across K
        processes would admit N*K.
        """
        config = _make_config(rate_limit_window_seconds=3600)
        limiter = RateLimiter(MagicMock(), config)

        start = limiter._window_start(
            datetime(2026, 8, 4, 11, 43, 17, tzinfo=UTC)
        )
        assert start == datetime(2026, 8, 4, 11, 0, 0, tzinfo=UTC)
        assert int(start.timestamp()) % 3600 == 0

    def test_a_zero_window_does_not_divide_by_zero(self):
        config = _make_config(rate_limit_window_seconds=0)
        limiter = RateLimiter(MagicMock(), config)
        # Clamped to 1s rather than raising: a misconfigured window should give a
        # very short one, not crash every request through the limiter.
        assert limiter._window_start(datetime.now(UTC)) is not None


class TestRateLimiterBoundary:
    """The post-increment count includes the caller, so the compare is `>`."""

    async def test_at_exact_limit_is_allowed(self):
        """The 50th request under a limit of 50 is the last allowed one.

        Semantics changed with the shape: the old fixture's `count_documents == 50`
        meant "50 already made, this is the 51st", which was correctly denied. The
        post-increment 50 means "this is the 50th", which must be allowed.
        """
        col = _make_collection(post_increment_count=50)
        config = _make_config(rate_limit_enabled=True, rate_limit_max_requests=50)
        limiter = RateLimiter(col, config)

        assert await limiter.check_rate_limit("user1", "store_memory") is True

    async def test_one_over_the_limit_is_denied(self):
        col = _make_collection(post_increment_count=51)
        config = _make_config(rate_limit_enabled=True, rate_limit_max_requests=50)
        limiter = RateLimiter(col, config)

        assert await limiter.check_rate_limit("user1", "store_memory") is False

    async def test_just_below_limit_is_allowed(self):
        col = _make_collection(post_increment_count=49)
        config = _make_config(rate_limit_enabled=True, rate_limit_max_requests=50)
        limiter = RateLimiter(col, config)

        assert await limiter.check_rate_limit("user1", "store_memory") is True

    async def test_a_zero_limit_refuses_without_a_round_trip(self):
        """Zero means "no requests", not "unlimited".

        Short-circuited, because `$inc` would exceed it on the first call anyway
        and there is no reason to spend a round trip proving that.
        """
        col = _make_collection()
        config = _make_config(rate_limit_enabled=True, rate_limit_max_requests=0)
        limiter = RateLimiter(col, config)

        assert await limiter.check_rate_limit("user1", "store_memory") is False
        col.find_one_and_update.assert_not_called()

    async def test_a_zero_governance_override_also_refuses(self):
        col = _make_collection()
        config = _make_config(rate_limit_enabled=True, rate_limit_max_requests=100)
        limiter = RateLimiter(col, config)

        assert (
            await limiter.check_rate_limit("user1", "store_memory", max_requests=0)
            is False
        )
        col.find_one_and_update.assert_not_called()


class TestRateLimiterFailsOpen:
    """An unavailable counter must not become a total outage."""

    async def test_a_database_error_allows_the_request(self):
        """Fail open, deliberately — and only here.

        A rate limit is a throughput guard, not a policy control. Governance and
        the access check have already run by this point and are the controls that
        must not be bypassed; refusing every request because the *counter* is
        unreachable converts one collection's outage into a full one. The
        asymmetry with `_check_access`, which fails closed, is the point.
        """
        col = MagicMock()
        col.find_one_and_update = AsyncMock(side_effect=PyMongoError("no primary"))
        config = _make_config(rate_limit_enabled=True, rate_limit_max_requests=1)
        limiter = RateLimiter(col, config)

        assert await limiter.check_rate_limit("user1", "store_memory") is True

    async def test_the_failure_is_logged_at_warning(self, caplog):
        col = MagicMock()
        col.find_one_and_update = AsyncMock(side_effect=PyMongoError("no primary"))
        config = _make_config(rate_limit_enabled=True, rate_limit_max_requests=1)
        limiter = RateLimiter(col, config)

        with caplog.at_level("WARNING"):
            await limiter.check_rate_limit("user1", "store_memory")

        # Failing open silently is how an unlimited system looks limited. The
        # operator has to be able to see that the guard is not running.
        assert any("allowing" in r.message.lower() for r in caplog.records)

    async def test_a_missing_document_does_not_crash(self):
        """`find_one_and_update` returning None is treated as the first request."""
        col = MagicMock()
        col.find_one_and_update = AsyncMock(return_value=None)
        config = _make_config(rate_limit_enabled=True, rate_limit_max_requests=5)
        limiter = RateLimiter(col, config)

        assert await limiter.check_rate_limit("user1", "store_memory") is True


class TestRateLimiterGovernanceOverride:
    """TC-E-028: Rate limiter reads limits from the governance profile."""

    async def test_governance_override_allows(self):
        col = _make_collection(post_increment_count=50)
        config = _make_config(rate_limit_enabled=True, rate_limit_max_requests=10)
        limiter = RateLimiter(col, config)

        # 50 in-window would exceed the config default of 10, but the profile
        # allows 100.
        assert (
            await limiter.check_rate_limit("user1", "store_memory", max_requests=100)
            is True
        )

    async def test_governance_override_blocks(self):
        col = _make_collection(post_increment_count=50)
        config = _make_config(rate_limit_enabled=True, rate_limit_max_requests=100)
        limiter = RateLimiter(col, config)

        # Within the config default, over the profile's 20.
        assert (
            await limiter.check_rate_limit("user1", "store_memory", max_requests=20)
            is False
        )

    async def test_no_override_uses_config_default(self):
        col = _make_collection(post_increment_count=50)
        config = _make_config(rate_limit_enabled=True, rate_limit_max_requests=100)
        limiter = RateLimiter(col, config)

        assert await limiter.check_rate_limit("user1", "store_memory") is True
