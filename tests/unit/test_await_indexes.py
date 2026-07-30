"""create() can block until search indexes are queryable (live-test fix #1).

A short-lived library/script caller (`async with AsyncMemory.create(cfg) as m`)
otherwise exits before the background index task finishes, so search/recall
silently return nothing. `await_search_indexes=True` makes create() await index
creation; the default (False) keeps the background behaviour for long-running
servers.
"""

from unittest.mock import AsyncMock, MagicMock

from agent_memory.config import MemoryConfig


def _config(**overrides) -> MemoryConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MemoryConfig(**defaults, _env_file=None)


class TestAwaitSearchIndexesFlag:
    def test_flag_defaults_false(self):
        assert _config().await_search_indexes is False

    def test_flag_settable(self):
        assert _config(await_search_indexes=True).await_search_indexes is True

    def test_flag_from_env(self, monkeypatch):
        monkeypatch.setenv("MONGODB_CONNECTION_STRING", "mongodb://h:27017")
        monkeypatch.setenv("AWAIT_SEARCH_INDEXES", "true")
        assert MemoryConfig.from_env(_env_file=None).await_search_indexes is True


class TestCreateAwaitsIndexes:
    """create() awaits (not backgrounds) index build when the flag is set."""

    async def test_await_true_calls_ensure_and_no_pending_task(self, monkeypatch):
        import agent_memory.memory as mem

        ensure_search = AsyncMock()
        self = mem.AsyncMemory.__new__(mem.AsyncMemory)
        await self._provision_search_indexes(MagicMock(), _config(await_search_indexes=True), ensure_search)
        ensure_search.assert_awaited_once()
        # nothing left running in the background
        assert getattr(self, "_search_index_task", None) is None

    async def test_await_false_backgrounds_the_task(self, monkeypatch):
        import agent_memory.memory as mem

        ensure_search = AsyncMock()
        self = mem.AsyncMemory.__new__(mem.AsyncMemory)
        await self._provision_search_indexes(MagicMock(), _config(await_search_indexes=False), ensure_search)
        # a background task was scheduled (not yet awaited)
        assert self._search_index_task is not None
        await self._search_index_task  # let it finish
        ensure_search.assert_awaited_once()


class TestTheIndexIsBuiltForTheVectorsThatWillGoIntoIt:
    """`numDimensions` must match what the embedder emits, not what config declares.

    On a Voyage deployment those differ: `embedding_dimension` inherits Titan's
    1536 while the Atlas gateway's models emit 1024. Provisioning from the declared
    value builds a 1536-dim index, and Atlas then accepts every 1024-dim vector
    written into it and returns none of them from `$vectorSearch`. Nothing raises —
    recall simply goes empty, and every memory stored until someone notices has to
    be re-embedded.

    This used to work only by accident of ordering: `_create_embedding_provider`
    overwrote `config.embedding_dimension` in place, so a later read of it happened
    to be right. Once that write-back was removed the read became silently wrong,
    and nothing in the suite noticed — these tests are what noticed.
    """

    def _facade(self, dimension: int):
        """A facade wired the way `create()` leaves it, for one dimension."""
        import agent_memory.memory as mem

        self_ = mem.AsyncMemory.__new__(mem.AsyncMemory)
        self_._embedding_dimension = dimension
        return self_

    async def test_the_awaited_path_uses_the_resolved_dimension(self):
        ensure = AsyncMock()
        facade = self._facade(1024)
        config = _config(await_search_indexes=True)
        assert config.embedding_dimension == 1536, "precondition: declared != resolved"

        await facade._provision_search_indexes(MagicMock(), config, ensure)

        assert ensure.await_args.kwargs["embedding_dimension"] == 1024, (
            "the index was provisioned at the declared dimension; a 1024-dim "
            "embedder's vectors would be accepted into it and never returned"
        )

    async def test_the_backgrounded_path_uses_the_resolved_dimension(self):
        """Asserted separately because it is a different call site.

        The two paths pass the dimension differently — one by keyword, one
        positionally — so a fix applied to only one of them is a real possibility.
        """
        ensure = AsyncMock()
        facade = self._facade(1024)

        await facade._provision_search_indexes(
            MagicMock(), _config(await_search_indexes=False), ensure
        )
        await facade._search_index_task

        assert ensure.await_args.kwargs["embedding_dimension"] == 1024

    async def test_a_hand_built_facade_falls_back_to_the_declared_value(self):
        """`create()` always sets `_embedding_dimension`; a facade built directly
        by a test or an embedder does not. The declared value is the best answer
        available there, and an AttributeError on the provisioning path would be a
        worse one."""
        import agent_memory.memory as mem

        ensure = AsyncMock()
        facade = mem.AsyncMemory.__new__(mem.AsyncMemory)

        await facade._provision_search_indexes(
            MagicMock(), _config(await_search_indexes=True, embedding_dimension=768),
            ensure,
        )

        assert ensure.await_args.kwargs["embedding_dimension"] == 768
