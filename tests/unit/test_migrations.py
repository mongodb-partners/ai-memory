"""Tests for migrations.py (ensure_indexes, ensure_search_indexes)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory.core.collections import STANDARD_INDEXES, SEARCH_INDEXES


class TestEnsureIndexes:
    """REQ-DB-001: Standard indexes created idempotently on startup."""

    async def test_creates_all_standard_indexes(self):
        """Each STANDARD_INDEXES entry results in a create_index call."""
        from agent_memory.core.migrations import ensure_indexes

        mock_db = MagicMock()
        collections = {}

        def get_col(name):
            if name not in collections:
                col = MagicMock()
                col.create_index = AsyncMock(return_value="ok")
                collections[name] = col
            return collections[name]

        mock_db.__getitem__ = MagicMock(side_effect=get_col)

        await ensure_indexes(mock_db)

        total_calls = sum(
            col.create_index.call_count for col in collections.values()
        )
        assert total_calls == len(STANDARD_INDEXES)

    async def test_idempotent_no_error_on_existing(self):
        """Calling ensure_indexes twice should not raise."""
        from agent_memory.core.migrations import ensure_indexes

        mock_db = MagicMock()
        col = MagicMock()
        col.create_index = AsyncMock(return_value="ok")
        mock_db.__getitem__ = MagicMock(return_value=col)

        await ensure_indexes(mock_db)
        await ensure_indexes(mock_db)
        # Should succeed without exception


class TestEnsureIndexesConflict:
    """ensure_indexes handles OperationFailure code 86 (index conflict)."""

    async def test_conflict_drops_and_recreates(self):
        from agent_memory.core.migrations import ensure_indexes
        from pymongo.errors import OperationFailure

        mock_db = MagicMock()
        col = MagicMock()
        # First call raises code 86, subsequent calls succeed
        col.create_index = AsyncMock(
            side_effect=[OperationFailure("conflict", code=86)] +
                        [AsyncMock(return_value="ok")] * (len(STANDARD_INDEXES) * 2)
        )
        col.drop_index = AsyncMock()
        mock_db.__getitem__ = MagicMock(return_value=col)

        await ensure_indexes(mock_db)
        col.drop_index.assert_called_once()

    async def test_conflict_recreate_failure_logs(self):
        from agent_memory.core.migrations import ensure_indexes
        from pymongo.errors import OperationFailure

        mock_db = MagicMock()
        col = MagicMock()
        col.create_index = AsyncMock(
            side_effect=OperationFailure("conflict", code=86)
        )
        col.drop_index = AsyncMock(side_effect=Exception("drop failed"))
        mock_db.__getitem__ = MagicMock(return_value=col)

        # Should not raise — logs the error
        await ensure_indexes(mock_db)

    async def test_non_conflict_operation_failure_logs(self):
        from agent_memory.core.migrations import ensure_indexes
        from pymongo.errors import OperationFailure

        mock_db = MagicMock()
        col = MagicMock()
        col.create_index = AsyncMock(
            side_effect=OperationFailure("other error", code=42)
        )
        mock_db.__getitem__ = MagicMock(return_value=col)

        await ensure_indexes(mock_db)


class TestEnsureSearchIndexes:
    """REQ-DB-002..004: Atlas Search indexes created in background."""

    async def test_creates_search_indexes_when_not_existing(self):
        """Each SEARCH_INDEXES entry results in create_search_index if not found."""
        from agent_memory.core.migrations import ensure_search_indexes

        mock_db = MagicMock()
        collections = {}

        async def empty_list_search(*args, **kwargs):
            # Simulate no existing indexes
            return AsyncMock(__aiter__=lambda self: self, __anext__=_stop_aiter)()

        async def _stop_aiter(self):
            raise StopAsyncIteration

        def get_col(name):
            if name not in collections:
                col = MagicMock()
                col.list_search_indexes = AsyncMock(return_value=_empty_async_iter())
                col.create_search_index = AsyncMock(return_value="idx_name")
                collections[name] = col
            return collections[name]

        mock_db.__getitem__ = MagicMock(side_effect=get_col)

        with patch("agent_memory.core.migrations._wait_for_search_index",
                    new_callable=AsyncMock, return_value=True):
            await ensure_search_indexes(mock_db)

        total_calls = sum(
            col.create_search_index.call_count for col in collections.values()
        )
        assert total_calls == len(SEARCH_INDEXES)

    async def test_skips_existing_search_index(self):
        """REQ-DB-003: an existing index is never re-created.

        It may be *updated* — see `TestExistingIndexesAreReconciled` — but
        creation is for absent indexes only. This existing index reports no
        definition at all, which cannot be shown to match, so reconciliation
        issues an update; the assertion here is only about creation.
        """
        from agent_memory.core.migrations import ensure_search_indexes

        mock_db = MagicMock()
        col = MagicMock()

        # Return an existing index whose name matches whatever is queried
        def make_existing_iter(index_name):
            return _async_iter_of([{"name": index_name, "queryable": True}])

        col.list_search_indexes = AsyncMock(side_effect=make_existing_iter)
        col.create_search_index = AsyncMock()
        col.update_search_index = AsyncMock()
        col.drop_search_index = AsyncMock()
        mock_db.__getitem__ = MagicMock(return_value=col)

        with patch("agent_memory.core.migrations._wait_for_search_index",
                    new_callable=AsyncMock, return_value=True):
            await ensure_search_indexes(mock_db)

        col.create_search_index.assert_not_called()
        col.drop_search_index.assert_not_called()

    async def test_handles_non_atlas_gracefully(self):
        """REQ-DB-004: Non-Atlas deployment logs warning, doesn't raise."""
        from agent_memory.core.migrations import ensure_search_indexes
        from pymongo.errors import OperationFailure

        mock_db = MagicMock()
        col = MagicMock()
        col.list_search_indexes = AsyncMock(
            side_effect=OperationFailure("not supported", code=None)
        )
        mock_db.__getitem__ = MagicMock(return_value=col)

        # Should not raise
        await ensure_search_indexes(mock_db)

    async def test_non_atlas_skips_remaining_indexes(self):
        """After first index fails with OperationFailure, remaining indexes are skipped via break."""
        from agent_memory.core.migrations import ensure_search_indexes
        from pymongo.errors import OperationFailure

        mock_db = MagicMock()
        cols = {}
        call_count = 0

        def get_col(name):
            nonlocal call_count
            if name not in cols:
                col = MagicMock()
                col.list_search_indexes = AsyncMock(
                    side_effect=OperationFailure("not supported", code=None)
                )
                col.create_search_index = AsyncMock()
                cols[name] = col
            call_count += 1
            return cols[name]

        mock_db.__getitem__ = MagicMock(side_effect=get_col)

        await ensure_search_indexes(mock_db)

        # Only the first collection should have list_search_indexes called;
        # the rest should be skipped by the atlas_available break
        total_list_calls = sum(
            c.list_search_indexes.call_count for c in cols.values()
        )
        assert total_list_calls == 1
        # No create calls at all
        total_create_calls = sum(
            c.create_search_index.call_count for c in cols.values()
        )
        assert total_create_calls == 0

    async def test_dimension_mismatch_drops_and_recreates(self):
        """Vector index with wrong dims is dropped and recreated.

        ``count_documents=0`` is the precondition, not incidental setup: an empty
        collection has no vectors to strand, so this is the ordinary case of an
        index left over from a previous config. A collection with vectors in it
        refuses instead — see ``TestADimensionChangeWillNotStrandStoredVectors``.
        """
        from agent_memory.core.migrations import ensure_search_indexes

        mock_db = MagicMock()
        col = MagicMock()

        # Existing index has 1536 dims, but we request 1024
        def make_existing_iter(index_name):
            return _async_iter_of([{
                "name": index_name,
                "queryable": True,
                "latestDefinition": {
                    "fields": [{"type": "vector", "path": "embedding", "numDimensions": 1536}]
                },
            }])

        col.list_search_indexes = AsyncMock(side_effect=make_existing_iter)
        col.drop_search_index = AsyncMock()
        col.create_search_index = AsyncMock(return_value="idx_name")
        col.count_documents = AsyncMock(return_value=0)
        mock_db.__getitem__ = MagicMock(return_value=col)

        with patch("agent_memory.core.migrations._wait_for_search_index_dropped",
                    new_callable=AsyncMock), \
             patch("agent_memory.core.migrations._wait_for_search_index",
                    new_callable=AsyncMock, return_value=True):
            await ensure_search_indexes(mock_db, embedding_dimension=1024)

        # Should drop and recreate vector indexes
        assert col.drop_search_index.call_count >= 1
        assert col.create_search_index.call_count >= 1

    async def test_search_index_not_queryable_within_timeout(self):
        """Logs warning when index doesn't become queryable."""
        from agent_memory.core.migrations import ensure_search_indexes

        mock_db = MagicMock()
        collections = {}

        def get_col(name):
            if name not in collections:
                col = MagicMock()
                col.list_search_indexes = AsyncMock(return_value=_empty_async_iter())
                col.create_search_index = AsyncMock(return_value="idx_name")
                collections[name] = col
            return collections[name]

        mock_db.__getitem__ = MagicMock(side_effect=get_col)

        with patch("agent_memory.core.migrations._wait_for_search_index",
                    new_callable=AsyncMock, return_value=False):
            await ensure_search_indexes(mock_db)  # Should not raise

    async def test_create_search_index_operation_failure(self):
        """OperationFailure on create_search_index is handled."""
        from agent_memory.core.migrations import ensure_search_indexes
        from pymongo.errors import OperationFailure

        mock_db = MagicMock()
        col = MagicMock()
        col.list_search_indexes = AsyncMock(return_value=_empty_async_iter())
        col.create_search_index = AsyncMock(
            side_effect=OperationFailure("already exists", code=68)
        )
        mock_db.__getitem__ = MagicMock(return_value=col)

        await ensure_search_indexes(mock_db)  # Should not raise

    async def test_create_search_index_unexpected_exception(self):
        """Unexpected exception on create_search_index is handled."""
        from agent_memory.core.migrations import ensure_search_indexes

        mock_db = MagicMock()
        col = MagicMock()
        col.list_search_indexes = AsyncMock(return_value=_empty_async_iter())
        col.create_search_index = AsyncMock(side_effect=RuntimeError("boom"))
        mock_db.__getitem__ = MagicMock(return_value=col)

        await ensure_search_indexes(mock_db)  # Should not raise


class TestExistingIndexesAreReconciled:
    """An index that already exists is brought into line with the shipped definition.

    It used to be left alone. The vector branch compared ``numDimensions`` and
    nothing else; the full-text branch compared nothing at all and ``continue``d.
    So an index created by an earlier version kept that version's definition for
    the life of the cluster, and only a *fresh* deployment ever ran the current
    schema.

    That makes every definition change undeliverable, and undeliverable in the
    quietest way available: a filter path missing from an index does not raise —
    the branch matches nothing. The change that added ``memory_type`` and ``tags``
    as filter fields would have passed every test, because tests start from an
    empty cluster, and done nothing at all on the deployment that needed it.
    """

    @staticmethod
    def _shipped(index_name: str, dims: int = 1024) -> list[dict]:
        """The real shipped definition, so these tests move with the schema."""
        from agent_memory.core.collections import get_search_indexes
        return [i for i in get_search_indexes(dims) if i["name"] == index_name]

    @staticmethod
    def _collection(existing: dict, embedded_count: int = 0):
        col = MagicMock()
        col.list_search_indexes = AsyncMock(
            side_effect=lambda name: _async_iter_of([{**existing, "name": name}])
        )
        col.create_search_index = AsyncMock(return_value="idx")
        col.update_search_index = AsyncMock()
        col.drop_search_index = AsyncMock()
        # Empty by default: these tests are about *definition* reconciliation, and
        # a dimension rebuild is only permitted when there are no vectors to
        # strand. Stranding is its own concern —
        # `TestADimensionChangeWillNotStrandStoredVectors`.
        col.count_documents = AsyncMock(return_value=embedded_count)
        return col

    @staticmethod
    def _db(col):
        db = MagicMock()
        db.__getitem__ = MagicMock(return_value=col)
        return db

    async def _reconcile(
        self,
        index_name: str,
        existing: dict,
        dims: int = 1024,
        embedded_count: int = 0,
        allow_dimension_change: bool = False,
    ):
        from agent_memory.core.migrations import ensure_search_indexes
        shipped = self._shipped(index_name, dims)
        assert shipped, f"no shipped definition named {index_name!r}"
        col = self._collection(existing, embedded_count=embedded_count)
        with patch("agent_memory.core.migrations.get_search_indexes",
                   return_value=shipped), \
             patch("agent_memory.core.migrations._wait_for_search_index",
                   new_callable=AsyncMock, return_value=True), \
             patch("agent_memory.core.migrations._wait_for_search_index_dropped",
                   new_callable=AsyncMock):
            await ensure_search_indexes(
                self._db(col),
                embedding_dimension=dims,
                allow_dimension_change=allow_dimension_change,
            )
        return col, shipped[0]["definition"]

    async def test_an_existing_full_text_index_receives_the_new_definition(self):
        """The reported bug, at its narrowest: the FTS branch used to `continue`."""
        stale = {
            "queryable": True,
            "latestDefinition": {
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "content": {"type": "string"},
                        "summary": {"type": "string"},
                        "user_id": {"type": "token"},
                        "tier": {"type": "token"},
                        "is_deleted": {"type": "token"},
                        # No `memory_type`, no `tags` — an index from before the
                        # filter fields were added.
                    },
                }
            },
        }
        col, wanted = await self._reconcile("memories_fts_index", stale)

        col.update_search_index.assert_awaited_once()
        name, definition = col.update_search_index.await_args.args[:2]
        assert name == "memories_fts_index"
        assert definition == wanted, "the update did not carry the shipped definition"

    async def test_a_stale_vector_index_is_updated_not_dropped(self):
        """A new filter field does not justify taking vector search offline.

        Atlas rebuilds an updated index in the background and keeps serving the
        old definition meanwhile. A drop-and-recreate would mean minutes with no
        vector search at all, on every deployment, for a change that
        ``update_search_index`` applies in place.
        """
        stale = {
            "queryable": True,
            "latestDefinition": {
                "fields": [
                    {"type": "vector", "path": "embedding",
                     "numDimensions": 1024, "similarity": "cosine"},
                    {"type": "filter", "path": "user_id"},
                    {"type": "filter", "path": "tier"},
                    {"type": "filter", "path": "deleted_at"},
                ]
            },
        }
        col, wanted = await self._reconcile("memories_vector_index", stale)

        col.update_search_index.assert_awaited_once()
        assert col.update_search_index.await_args.args[1] == wanted
        col.drop_search_index.assert_not_called()
        col.create_search_index.assert_not_called()

    async def test_the_update_carries_the_paths_that_were_missing(self):
        """Not just "an update happened" — the right fields arrive.

        An update that pushed the *live* definition back would satisfy the
        assertion above and fix nothing.
        """
        stale = {
            "queryable": True,
            "latestDefinition": {
                "fields": [
                    {"type": "vector", "path": "embedding",
                     "numDimensions": 1024, "similarity": "cosine"},
                    {"type": "filter", "path": "user_id"},
                ]
            },
        }
        col, _ = await self._reconcile("memories_vector_index", stale)

        pushed = col.update_search_index.await_args.args[1]
        paths = {f["path"] for f in pushed["fields"] if f["type"] == "filter"}
        assert {"memory_type", "tags"} <= paths

    async def test_an_up_to_date_index_is_left_completely_alone(self):
        """Including when Atlas has echoed the definition back enriched.

        Atlas adds its own defaults — `quantization` on a vector field, an
        analyzer on a string mapping, `storedSource`, whatever the cluster's
        version contributes. Comparing for equality would find a difference on a
        cluster that is perfectly current and reissue the update on *every*
        startup, rebuilding the whole index each time. Hence the subset check.
        """
        shipped = self._shipped("memories_vector_index")[0]["definition"]
        enriched = {
            "fields": [
                # Reordered as well as enriched: field order is not meaningful and
                # Atlas does not preserve ours.
                *({**f, "quantization": "none"} if f["type"] == "vector" else f
                  for f in reversed(shipped["fields"])),
            ],
            "numPartitions": 1,
        }
        col, _ = await self._reconcile(
            "memories_vector_index", {"queryable": True, "latestDefinition": enriched}
        )

        col.update_search_index.assert_not_called()
        col.drop_search_index.assert_not_called()
        col.create_search_index.assert_not_called()

    async def test_an_up_to_date_full_text_index_is_left_alone(self):
        shipped = self._shipped("memories_fts_index")[0]["definition"]
        fields = shipped["mappings"]["fields"]
        enriched = {
            "mappings": {
                "dynamic": False,
                "fields": {
                    name: ({**spec, "analyzer": "lucene.standard"}
                           if spec["type"] == "string" else spec)
                    for name, spec in fields.items()
                },
            },
            "storedSource": False,
        }
        col, _ = await self._reconcile(
            "memories_fts_index", {"queryable": True, "latestDefinition": enriched}
        )

        col.update_search_index.assert_not_called()
        col.drop_search_index.assert_not_called()

    async def test_a_dimension_change_still_drops_and_recreates(self):
        """The one change Atlas will not apply in place.

        ``numDimensions`` is not editable, so an update cannot deliver it.

        This test used to justify the drop with "the stored vectors are the wrong
        width regardless — there is nothing to preserve", and that reasoning was
        wrong in the case that matters. The vectors are not deleted by the rebuild;
        they survive at the old width, unsearchable, and re-embedding them needs
        the provider config that has just been replaced. So the drop is conditional
        now, and this test states the condition it relies on: an empty collection,
        which is the case where the old justification is actually true.
        """
        wrong_width = {
            "queryable": True,
            "latestDefinition": {
                "fields": [
                    {"type": "vector", "path": "embedding", "numDimensions": 1536},
                ]
            },
        }
        col, _ = await self._reconcile(
            "memories_vector_index", wrong_width, dims=1024, embedded_count=0
        )

        col.drop_search_index.assert_awaited_once_with("memories_vector_index")
        col.create_search_index.assert_awaited_once()
        col.update_search_index.assert_not_called()

    async def test_a_failed_update_leaves_the_working_index_in_place(self):
        """A rejected update is not a reason to drop an index that serves queries.

        The old definition still answers everything except whatever the new one
        added. Dropping on failure would convert a partial degradation into a
        total outage, on startup, unattended.
        """
        from agent_memory.core.migrations import ensure_search_indexes
        from pymongo.errors import OperationFailure

        shipped = self._shipped("memories_fts_index")
        col = self._collection({"queryable": True, "latestDefinition": {
            "mappings": {"dynamic": False, "fields": {"content": {"type": "string"}}}}})
        col.update_search_index = AsyncMock(
            side_effect=OperationFailure("index is building", code=None))

        with patch("agent_memory.core.migrations.get_search_indexes",
                   return_value=shipped):
            await ensure_search_indexes(self._db(col), embedding_dimension=1024)

        col.drop_search_index.assert_not_called()
        col.create_search_index.assert_not_called()

    async def test_an_unexpected_update_error_is_also_survivable(self):
        """Stage 2 runs as a background task; nothing here may escape."""
        from agent_memory.core.migrations import ensure_search_indexes

        shipped = self._shipped("memories_fts_index")
        col = self._collection({"queryable": True, "latestDefinition": {
            "mappings": {"dynamic": False, "fields": {}}}})
        col.update_search_index = AsyncMock(side_effect=RuntimeError("boom"))

        with patch("agent_memory.core.migrations.get_search_indexes",
                   return_value=shipped):
            await ensure_search_indexes(self._db(col), embedding_dimension=1024)

        col.drop_search_index.assert_not_called()

    async def test_reconciliation_continues_to_the_next_index(self):
        """One index needing an update must not stop the others being checked."""
        from agent_memory.core.migrations import ensure_search_indexes
        from agent_memory.core.collections import get_search_indexes

        shipped = get_search_indexes(1024)
        col = self._collection({"queryable": True, "latestDefinition": {
            "mappings": {"dynamic": False, "fields": {}}}})

        with patch("agent_memory.core.migrations.get_search_indexes",
                   return_value=shipped):
            await ensure_search_indexes(self._db(col), embedding_dimension=1024)

        # Every shipped index was inspected, and each stale one updated.
        assert col.list_search_indexes.await_count == len(shipped)
        updated = {c.args[0] for c in col.update_search_index.await_args_list}
        assert updated == {i["name"] for i in shipped}


class TestADimensionChangeWillNotStrandStoredVectors:
    """Rebuilding a vector index at a new width orphans the vectors under it.

    The rebuild does not touch the documents, which is what makes it dangerous
    rather than what makes it safe. Every vector already stored keeps the old
    width, and the new index returns none of them from ``$vectorSearch`` — no
    exception, no changed document count, ``find`` still showing every memory.
    Recall goes empty for the whole history while working perfectly for anything
    written afterwards, so it reads as "there are no memories about that".

    Recovery means re-embedding every document with the *previous* provider, which
    is the config the operator has just replaced. That asymmetry is the argument
    for refusing: leaving the old index in place costs indexing of new writes,
    which is recoverable by fixing the config; rebuilding costs the history, which
    is not.

    The guard lives inside reconciliation rather than only at startup because
    stage 2 usually runs as a background task whose exceptions are logged and
    discarded — a check that only raised upstream would be one the default path
    routes around.
    """

    @staticmethod
    def _collection(existing_dims: int, embedded_count):
        col = MagicMock()
        col.list_search_indexes = AsyncMock(
            side_effect=lambda name: _async_iter_of([{
                "name": name,
                "queryable": True,
                "latestDefinition": {"fields": [
                    {"type": "vector", "path": "embedding",
                     "numDimensions": existing_dims},
                ]},
            }])
        )
        col.drop_search_index = AsyncMock()
        col.create_search_index = AsyncMock(return_value="idx")
        col.update_search_index = AsyncMock()
        if isinstance(embedded_count, Exception):
            col.count_documents = AsyncMock(side_effect=embedded_count)
        else:
            col.count_documents = AsyncMock(return_value=embedded_count)
        return col

    async def _reconcile(self, col, dims=1024, allow=False):
        from agent_memory.core.migrations import ensure_search_indexes
        from agent_memory.core.collections import get_search_indexes

        shipped = [i for i in get_search_indexes(dims)
                   if i["name"] == "memories_vector_index"]
        db = MagicMock()
        db.__getitem__ = MagicMock(return_value=col)
        with patch("agent_memory.core.migrations.get_search_indexes",
                   return_value=shipped), \
             patch("agent_memory.core.migrations._wait_for_search_index",
                   new_callable=AsyncMock, return_value=True), \
             patch("agent_memory.core.migrations._wait_for_search_index_dropped",
                   new_callable=AsyncMock):
            await ensure_search_indexes(
                db, embedding_dimension=dims, allow_dimension_change=allow
            )

    async def test_an_index_over_stored_vectors_is_not_rebuilt(self):
        col = self._collection(existing_dims=1536, embedded_count=42)

        await self._reconcile(col)

        col.drop_search_index.assert_not_called()
        col.create_search_index.assert_not_called()

    async def test_the_existing_index_is_left_serving_queries(self):
        """Stated separately from "did not drop".

        Declining the rebuild has to leave the old index intact and usable. An
        implementation that refused the drop but pushed an incompatible
        ``update_search_index`` would satisfy the assertion above and still break
        the index it was protecting.
        """
        col = self._collection(existing_dims=1536, embedded_count=42)

        await self._reconcile(col)

        col.update_search_index.assert_not_called()

    async def test_an_empty_collection_is_rebuilt_normally(self):
        """The paired positive case, so refusing cannot masquerade as the fix.

        This is the ordinary first-run shape — an index left by a previous config
        over a collection with nothing in it. Nothing is at risk, and a guard that
        blocked this would break every legitimate dimension change instead of the
        destructive ones.
        """
        col = self._collection(existing_dims=1536, embedded_count=0)

        await self._reconcile(col)

        col.drop_search_index.assert_awaited_once()
        col.create_search_index.assert_awaited_once()

    async def test_the_operator_can_opt_in(self):
        """Someone who has read the refusal and means it is not blocked."""
        col = self._collection(existing_dims=1536, embedded_count=42)

        await self._reconcile(col, allow=True)

        col.drop_search_index.assert_awaited_once()
        col.create_search_index.assert_awaited_once()

    async def test_an_uncountable_collection_is_treated_as_non_empty(self):
        """"We could not count" is not "there is nothing there".

        Defaulting to empty would convert a transient count failure into the
        silent data loss the guard exists to prevent, so the unknown case takes
        the conservative branch.
        """
        col = self._collection(
            existing_dims=1536, embedded_count=RuntimeError("count failed")
        )

        await self._reconcile(col)

        col.drop_search_index.assert_not_called()

    async def test_only_embedded_documents_count(self):
        """Counting rows rather than vectors would refuse for no reason.

        A collection can hold documents that carry no ``embedding`` at all — the
        counter and metadata shapes, or memories whose enrichment has not run.
        Those have no vector to strand.
        """
        col = self._collection(existing_dims=1536, embedded_count=0)

        await self._reconcile(col)

        assert col.count_documents.await_args.args[0] == {
            "embedding": {"$ne": None}
        }

    async def test_a_matching_dimension_never_asks_the_question(self):
        """No count, no refusal, no log — an unchanged dimension is not this case."""
        col = self._collection(existing_dims=1024, embedded_count=42)

        await self._reconcile(col, dims=1024)

        col.count_documents.assert_not_called()
        col.drop_search_index.assert_not_called()

    async def test_the_refusal_is_logged_loudly_enough_to_find(self, caplog):
        """The one channel that reaches an operator here.

        Raising is not available: on the default path this runs inside a task whose
        exception is caught and logged as "non-fatal". So the message has to name
        the index, both dimensions, and the way out, at a level that is not
        filtered out of production logs.
        """
        import logging

        col = self._collection(existing_dims=1536, embedded_count=42)

        with caplog.at_level(logging.ERROR, logger="agent_memory.core.migrations"):
            await self._reconcile(col)

        assert caplog.records, "the refusal was silent"
        message = caplog.records[-1].getMessage()
        assert "memories_vector_index" in message
        assert "1536" in message and "1024" in message
        assert "allow_embedding_dimension_change" in message


class TestFindStrandingDimensionChanges:
    """The startup preflight: report, so the caller can refuse before booting.

    Separate from the in-reconciliation guard because they answer to different
    needs. Reconciliation must survive anything and keep going, so it declines the
    one destructive step and logs. Startup can and should stop — a process that
    boots and serves empty recall for its whole history is worse than one that
    does not boot — so it needs the findings as data, not as a log line.
    """

    @staticmethod
    def _db(**by_collection):
        db = MagicMock()
        db.__getitem__ = MagicMock(side_effect=lambda name: by_collection[name])
        return db

    @staticmethod
    def _collection(existing_dims: int | None, embedded_count=0):
        col = MagicMock()
        if existing_dims is None:
            col.list_search_indexes = AsyncMock(
                side_effect=lambda name: _empty_async_iter()
            )
        else:
            col.list_search_indexes = AsyncMock(
                side_effect=lambda name: _async_iter_of([{
                    "name": name,
                    "latestDefinition": {"fields": [
                        {"type": "vector", "path": "embedding",
                         "numDimensions": existing_dims},
                    ]},
                }])
            )
        col.count_documents = AsyncMock(return_value=embedded_count)
        return col

    def _all(self, existing_dims, embedded_count=0):
        """Every collection with a vector index, wired the same way."""
        from agent_memory.core.collections import EPISODES, MEMORIES, SEMANTIC_CACHE

        return {
            name: self._collection(existing_dims, embedded_count)
            for name in (MEMORIES, EPISODES, SEMANTIC_CACHE)
        }

    async def test_a_change_over_stored_vectors_is_reported(self):
        from agent_memory.core.migrations import find_stranding_dimension_changes

        findings = await find_stranding_dimension_changes(
            self._db(**self._all(1536, embedded_count=7)), 1024
        )

        assert findings, "the stranding went unreported"
        assert {f.collection for f in findings} == {
            "memories", "episodes", "semantic_cache"
        }
        one = findings[0]
        assert (one.existing_dimension, one.wanted_dimension) == (1536, 1024)
        assert one.document_count == 7

    async def test_an_empty_collection_is_not_a_finding(self):
        """Changing the dimension with nothing stored is the normal first run."""
        from agent_memory.core.migrations import find_stranding_dimension_changes

        findings = await find_stranding_dimension_changes(
            self._db(**self._all(1536, embedded_count=0)), 1024
        )

        assert findings == []

    async def test_an_unchanged_dimension_is_not_a_finding(self):
        from agent_memory.core.migrations import find_stranding_dimension_changes

        findings = await find_stranding_dimension_changes(
            self._db(**self._all(1024, embedded_count=7)), 1024
        )

        assert findings == []

    async def test_an_absent_index_is_not_a_finding(self):
        """Nothing to drop, so nothing to strand — a fresh cluster."""
        from agent_memory.core.migrations import find_stranding_dimension_changes

        findings = await find_stranding_dimension_changes(
            self._db(**self._all(None)), 1024
        )

        assert findings == []

    async def test_a_non_atlas_deployment_is_not_a_finding(self):
        """``list_search_indexes`` raising ``OperationFailure`` means no Atlas
        Search at all. There are no vector indexes to rebuild, so refusing to
        start would block a deployment that was never at risk."""
        from pymongo.errors import OperationFailure

        from agent_memory.core.migrations import find_stranding_dimension_changes

        cols = self._all(1536, embedded_count=7)
        for col in cols.values():
            col.list_search_indexes = AsyncMock(
                side_effect=OperationFailure("no search")
            )

        assert await find_stranding_dimension_changes(self._db(**cols), 1024) == []

    async def test_an_uncountable_collection_is_still_reported(self):
        """The dimension change is established; only its size is unknown.

        Dropping the finding here would let a transient count failure wave through
        exactly the migration this exists to stop.
        """
        from agent_memory.core.migrations import find_stranding_dimension_changes

        cols = self._all(1536)
        for col in cols.values():
            col.count_documents = AsyncMock(side_effect=RuntimeError("nope"))

        findings = await find_stranding_dimension_changes(self._db(**cols), 1024)

        assert findings, "an uncountable collection was treated as empty"
        assert findings[0].document_count == -1

    async def test_the_error_names_what_to_do_about_it(self):
        from agent_memory.core.migrations import (
            StrandedVectors,
            stranding_error,
        )

        err = stranding_error([StrandedVectors(
            collection="memories", index_name="memories_vector_index",
            existing_dimension=1536, wanted_dimension=1024, document_count=7,
        )])

        message = str(err)
        assert "memories_vector_index" in message
        assert "1536" in message and "1024" in message
        assert "7 document" in message
        # The four ways out, so an operator is not left with only the refusal.
        assert "Re-embed" in message
        assert "allow_embedding_dimension_change=true" in message
        assert "ALLOW_EMBEDDING_DIMENSION_CHANGE" in message, (
            "the env var name is how this is actually set in a deployment"
        )


class TestDefinitionMatches:
    """``_definition_matches`` answers "is anything we require missing?"

    Not "are these equal". See the docstring on the helper, and
    ``test_an_up_to_date_index_is_left_completely_alone`` for why.
    """

    def test_identical_definitions_match(self):
        from agent_memory.core.migrations import _definition_matches
        defn = {"mappings": {"dynamic": False, "fields": {"a": {"type": "token"}}}}
        assert _definition_matches({"latestDefinition": defn}, defn)

    def test_extra_keys_in_the_live_definition_still_match(self):
        from agent_memory.core.migrations import _definition_matches
        wanted = {"mappings": {"fields": {"a": {"type": "string"}}}}
        live = {"mappings": {"dynamic": False, "storedSource": True,
                             "fields": {"a": {"type": "string",
                                              "analyzer": "lucene.standard"}}}}
        assert _definition_matches({"latestDefinition": live}, wanted)

    def test_a_missing_field_does_not_match(self):
        from agent_memory.core.migrations import _definition_matches
        wanted = {"mappings": {"fields": {"a": {"type": "token"},
                                          "tags": {"type": "token"}}}}
        live = {"mappings": {"fields": {"a": {"type": "token"}}}}
        assert not _definition_matches({"latestDefinition": live}, wanted)

    def test_a_changed_value_does_not_match(self):
        """`string` where we now want `token` — an analyzed field cannot back
        an exact `equals`, so this is exactly the case that must be caught."""
        from agent_memory.core.migrations import _definition_matches
        wanted = {"mappings": {"fields": {"tags": {"type": "token"}}}}
        live = {"mappings": {"fields": {"tags": {"type": "string"}}}}
        assert not _definition_matches({"latestDefinition": live}, wanted)

    def test_vector_fields_are_compared_by_type_and_path_not_position(self):
        from agent_memory.core.migrations import _definition_matches
        wanted = {"fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 1024},
            {"type": "filter", "path": "user_id"},
            {"type": "filter", "path": "tags"},
        ]}
        live = {"fields": [
            {"type": "filter", "path": "tags"},
            {"type": "filter", "path": "user_id"},
            {"type": "vector", "path": "embedding", "numDimensions": 1024,
             "similarity": "cosine", "quantization": "none"},
        ]}
        assert _definition_matches({"latestDefinition": live}, wanted)

    def test_a_missing_filter_path_does_not_match(self):
        from agent_memory.core.migrations import _definition_matches
        wanted = {"fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 1024},
            {"type": "filter", "path": "memory_type"},
        ]}
        live = {"fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 1024},
        ]}
        assert not _definition_matches({"latestDefinition": live}, wanted)

    def test_definition_is_used_when_latest_is_absent(self):
        from agent_memory.core.migrations import _definition_matches
        defn = {"fields": [{"type": "filter", "path": "user_id"}]}
        assert _definition_matches({"definition": defn}, defn)

    def test_latest_definition_wins_over_definition(self):
        """After an update, `latestDefinition` is the new one and `definition`
        the one still serving queries. Reading the stale one would reissue the
        same update on every startup during the rebuild."""
        from agent_memory.core.migrations import _definition_matches
        wanted = {"fields": [{"type": "filter", "path": "tags"}]}
        info = {
            "latestDefinition": {"fields": [{"type": "filter", "path": "tags"}]},
            "definition": {"fields": []},
        }
        assert _definition_matches(info, wanted)

    def test_an_unreadable_definition_does_not_match(self):
        """An index we cannot read must stay reconcilable. `update_search_index`
        is idempotent, so issuing one is the safe answer; declaring a match would
        pin that index at its current definition forever."""
        from agent_memory.core.migrations import _definition_matches
        assert not _definition_matches({}, {"fields": []})
        assert not _definition_matches({"latestDefinition": {}}, {"fields": []})

    def test_the_shipped_definitions_match_themselves(self):
        """Whatever we ship must reconcile as up to date against itself —
        otherwise every startup issues an update and rebuilds every index."""
        from agent_memory.core.collections import get_search_indexes
        from agent_memory.core.migrations import _definition_matches
        for idx in get_search_indexes(1024):
            assert _definition_matches(
                {"latestDefinition": idx["definition"]}, idx["definition"]
            ), f"{idx['name']} does not match itself"


class TestIsSubset:
    """The recursive comparison underneath ``_definition_matches``."""

    def test_scalars(self):
        from agent_memory.core.migrations import _is_subset
        assert _is_subset("token", "token")
        assert not _is_subset("token", "string")
        assert _is_subset(1024, 1024)

    def test_nested_dicts(self):
        from agent_memory.core.migrations import _is_subset
        assert _is_subset({"a": {"b": 1}}, {"a": {"b": 1, "c": 2}, "d": 3})
        assert not _is_subset({"a": {"b": 1}}, {"a": {"c": 2}})

    def test_a_dict_wanted_against_a_scalar_live(self):
        from agent_memory.core.migrations import _is_subset
        assert not _is_subset({"type": "token"}, "token")

    def test_lists_compare_by_equality(self):
        """Lists other than vector `fields` (e.g. a `synonyms` array) are not
        reordered by Atlas, so equality is the right test and a spurious update
        is the cost of being wrong."""
        from agent_memory.core.migrations import _is_subset
        assert _is_subset([1, 2], [1, 2])
        assert not _is_subset([1, 2], [2, 1])


class TestGetExistingDims:
    """_get_existing_dims extracts numDimensions from index info."""

    def test_extracts_from_latest_definition(self):
        from agent_memory.core.migrations import _get_existing_dims
        info = {"latestDefinition": {"fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 1024}
        ]}}
        assert _get_existing_dims(info) == 1024

    def test_extracts_from_definition_fallback(self):
        from agent_memory.core.migrations import _get_existing_dims
        info = {"definition": {"fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 1536}
        ]}}
        assert _get_existing_dims(info) == 1536

    def test_returns_none_for_non_vector(self):
        from agent_memory.core.migrations import _get_existing_dims
        info = {"latestDefinition": {"fields": [{"type": "filter", "path": "user_id"}]}}
        assert _get_existing_dims(info) is None

    def test_returns_none_for_empty(self):
        from agent_memory.core.migrations import _get_existing_dims
        assert _get_existing_dims({}) is None


class TestWaitForSearchIndexDropped:
    """_wait_for_search_index_dropped polls until index is gone."""

    async def test_returns_when_index_gone(self):
        from agent_memory.core.migrations import _wait_for_search_index_dropped
        col = MagicMock()
        col.list_search_indexes = AsyncMock(return_value=_empty_async_iter())
        with patch("agent_memory.core.migrations._SEARCH_INDEX_POLL_INTERVAL", 0):
            await _wait_for_search_index_dropped(col, "test_idx", timeout=5)

    async def test_returns_on_exception(self):
        from agent_memory.core.migrations import _wait_for_search_index_dropped
        col = MagicMock()
        col.list_search_indexes = AsyncMock(side_effect=Exception("fail"))
        with patch("agent_memory.core.migrations._SEARCH_INDEX_POLL_INTERVAL", 0):
            await _wait_for_search_index_dropped(col, "test_idx", timeout=5)

    async def test_timeout_logs_warning(self):
        from agent_memory.core.migrations import _wait_for_search_index_dropped
        col = MagicMock()
        col.list_search_indexes = AsyncMock(
            return_value=_async_iter_of([{"name": "still_here"}])
        )
        with patch("agent_memory.core.migrations._SEARCH_INDEX_POLL_INTERVAL", 0):
            await _wait_for_search_index_dropped(col, "test_idx", timeout=0)

    async def test_polls_then_finds_gone(self):
        """Index still present on first poll, gone on second — covers loop body."""
        from agent_memory.core.migrations import _wait_for_search_index_dropped
        col = MagicMock()
        # First call: index still present; second call: gone
        col.list_search_indexes = AsyncMock(side_effect=[
            _async_iter_of([{"name": "test_idx"}]),
            _empty_async_iter(),
        ])
        with patch("agent_memory.core.migrations._SEARCH_INDEX_POLL_INTERVAL", 0):
            await _wait_for_search_index_dropped(col, "test_idx", timeout=10)


class TestWaitForSearchIndex:
    """_wait_for_search_index polls until index is queryable."""

    async def test_returns_true_when_queryable(self):
        from agent_memory.core.migrations import _wait_for_search_index
        col = MagicMock()
        col.list_search_indexes = AsyncMock(
            return_value=_async_iter_of([{"queryable": True}])
        )
        with patch("agent_memory.core.migrations._SEARCH_INDEX_POLL_INTERVAL", 0):
            result = await _wait_for_search_index(col, "test_idx", timeout=5)
        assert result is True

    async def test_returns_false_on_timeout(self):
        from agent_memory.core.migrations import _wait_for_search_index
        col = MagicMock()
        col.list_search_indexes = AsyncMock(
            return_value=_async_iter_of([{"queryable": False}])
        )
        with patch("agent_memory.core.migrations._SEARCH_INDEX_POLL_INTERVAL", 0):
            result = await _wait_for_search_index(col, "test_idx", timeout=0)
        assert result is False

    async def test_handles_exception_during_poll(self):
        from agent_memory.core.migrations import _wait_for_search_index
        col = MagicMock()
        col.list_search_indexes = AsyncMock(side_effect=Exception("network"))
        with patch("agent_memory.core.migrations._SEARCH_INDEX_POLL_INTERVAL", 0):
            result = await _wait_for_search_index(col, "test_idx", timeout=0)
        assert result is False

    async def test_exception_then_queryable(self):
        """Exception on first poll, queryable on second — covers loop body + exception pass."""
        from agent_memory.core.migrations import _wait_for_search_index
        col = MagicMock()
        col.list_search_indexes = AsyncMock(side_effect=[
            Exception("transient"),
            _async_iter_of([{"queryable": True}]),
        ])
        with patch("agent_memory.core.migrations._SEARCH_INDEX_POLL_INTERVAL", 0):
            result = await _wait_for_search_index(col, "test_idx", timeout=10)
        assert result is True

    async def test_not_queryable_then_queryable(self):
        """Not queryable on first poll, queryable on second — covers normal loop body."""
        from agent_memory.core.migrations import _wait_for_search_index
        col = MagicMock()
        col.list_search_indexes = AsyncMock(side_effect=[
            _async_iter_of([{"queryable": False}]),
            _async_iter_of([{"queryable": True}]),
        ])
        with patch("agent_memory.core.migrations._SEARCH_INDEX_POLL_INTERVAL", 0):
            result = await _wait_for_search_index(col, "test_idx", timeout=10)
        assert result is True


# ─── Helpers for async iteration mocking ──────────────────────

class _AsyncIter:
    """Async iterator helper for mocking."""
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


def _async_iter_of(items):
    return _AsyncIter(items)


def _empty_async_iter():
    return _AsyncIter([])
