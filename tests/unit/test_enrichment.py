"""Tests for EnrichmentWorker."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId

import pytest

from agent_memory.core.config import MCPConfig
from agent_memory.services.enrichment import EnrichmentWorker


def _make_config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


def _make_providers():
    providers = MagicMock()
    providers.llm = AsyncMock()
    providers.llm.assess_importance = AsyncMock(return_value=0.7)
    providers.llm.generate_summary = AsyncMock(return_value="A test summary")
    providers.embedding = AsyncMock()
    providers.embedding.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
    return providers


def _make_memory_service():
    svc = AsyncMock()
    svc.evolve_memory = AsyncMock(return_value="created")
    return svc


# Long enough to clear MIN_SUMMARIZABLE_CHARS. Below that threshold `_summarize`
# returns None without calling the model, so a short fixture would assert nothing
# about summarization while appearing to. See REQ-E-121.
LONG_CONTENT = (
    "A test memory that needs enrichment, written long enough to be worth "
    "summarizing rather than skipped as its own best summary."
)


def _make_pending_memory():
    return {
        "_id": ObjectId(),
        "user_id": "user1",
        "content": LONG_CONTENT,
        "enrichment_status": "pending",
        "enrichment_retries": 0,
        "embedding": [0.1] * 1536,
    }


class TestEnrichmentWorkerProcessBatch:
    """TC-040: Worker finds and processes pending memories."""

    async def test_process_batch_updates_memories(self):
        col = MagicMock()
        col.update_one = AsyncMock()
        config = _make_config(enrichment_batch_size=10)
        providers = _make_providers()
        memory_svc = _make_memory_service()

        memory = _make_pending_memory()
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[memory])
        col.find.return_value = mock_cursor

        worker = EnrichmentWorker(col, config, providers, memory_svc)
        count = await worker.process_batch()

        assert count == 1
        col.update_one.assert_called_once()
        update_call = col.update_one.call_args
        update_set = update_call[0][1]["$set"]
        assert update_set["enrichment_status"] == "complete"
        assert update_set["importance"] == 0.7
        assert update_set["summary"] == "A test summary"


def _make_col_with_cursor(memories: list[dict]):
    """Create a MagicMock collection with find() returning a cursor."""
    col = MagicMock()
    col.update_one = AsyncMock()
    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(return_value=memories)
    col.find.return_value = mock_cursor
    return col


class TestEnrichmentWorkerFailure:
    """TC-041: Worker handles LLM failures."""

    async def test_failure_increments_retries(self):
        memory = _make_pending_memory()
        col = _make_col_with_cursor([memory])
        config = _make_config(enrichment_max_retries=3)
        providers = _make_providers()
        providers.llm.assess_importance = AsyncMock(side_effect=Exception("LLM down"))
        memory_svc = _make_memory_service()

        worker = EnrichmentWorker(col, config, providers, memory_svc)
        await worker.process_batch()

        update_call = col.update_one.call_args
        update_set = update_call[0][1]["$set"]
        assert update_set["enrichment_retries"] == 1

    async def test_max_retries_marks_failed(self):
        """REQ-027: Set failed after max retries."""
        memory = _make_pending_memory()
        memory["enrichment_retries"] = 2
        col = _make_col_with_cursor([memory])
        config = _make_config(enrichment_max_retries=3)
        providers = _make_providers()
        providers.llm.assess_importance = AsyncMock(side_effect=Exception("LLM down"))
        memory_svc = _make_memory_service()

        worker = EnrichmentWorker(col, config, providers, memory_svc)
        await worker.process_batch()

        update_call = col.update_one.call_args
        update_set = update_call[0][1]["$set"]
        assert update_set["enrichment_status"] == "failed"


class TestEnrichmentWorkerSemaphore:
    """TC-042: Concurrency limited by semaphore."""

    async def test_semaphore_limits_concurrency(self):
        col = MagicMock()
        config = _make_config(enrichment_concurrency=2, enrichment_batch_size=5)
        providers = _make_providers()
        memory_svc = _make_memory_service()

        worker = EnrichmentWorker(col, config, providers, memory_svc)
        assert worker._semaphore._value == 2


class TestEnrichmentWorkerEvolution:
    """TC-043: Worker triggers memory evolution check."""

    async def test_evolution_called_on_success(self):
        memory = _make_pending_memory()
        col = _make_col_with_cursor([memory])
        config = _make_config()
        providers = _make_providers()
        memory_svc = _make_memory_service()

        worker = EnrichmentWorker(col, config, providers, memory_svc)
        await worker.process_batch()

        memory_svc.evolve_memory.assert_called_once_with(
            "user1",
            memory["content"],
            memory["embedding"],
        )


class TestEnrichmentWorkerEmptyQueue:
    """TC-044: No pending memories is a no-op."""

    async def test_empty_queue_returns_zero(self):
        col = _make_col_with_cursor([])
        config = _make_config()
        providers = _make_providers()
        memory_svc = _make_memory_service()

        worker = EnrichmentWorker(col, config, providers, memory_svc)
        count = await worker.process_batch()

        assert count == 0
        col.update_one.assert_not_called()


class TestEnrichmentWorkerSummaryGuard:
    """REQ-E-121: enrichment must not store a refusal as a summary.

    The same defect as in ``ConsolidationWorker._compress_stm``, on the path the
    sample UI actually exercises — every LTM candidate queued by ``store_stm``
    comes through here. Importance still has to land; dropping the summary must
    not take the rest of the enrichment with it.
    """

    async def test_a_refusal_leaves_the_field_unset(self):
        memory = _make_pending_memory()
        col = _make_col_with_cursor([memory])
        providers = _make_providers()
        providers.llm.generate_summary = AsyncMock(
            return_value="I don't see the original text that needs to be summarized."
        )

        worker = EnrichmentWorker(col, _make_config(), providers, _make_memory_service())
        await worker.process_batch()

        update_set = col.update_one.call_args[0][1]["$set"]
        assert "summary" not in update_set
        # Enrichment is not abandoned — a memory with no summary is still enriched.
        assert update_set["enrichment_status"] == "complete"
        assert update_set["importance"] == 0.7

    async def test_short_content_skips_the_call_entirely(self):
        memory = _make_pending_memory()
        memory["content"] = "No alcohol in pairings."
        col = _make_col_with_cursor([memory])
        providers = _make_providers()

        worker = EnrichmentWorker(col, _make_config(), providers, _make_memory_service())
        await worker.process_batch()

        providers.llm.generate_summary.assert_not_called()
        update_set = col.update_one.call_args[0][1]["$set"]
        assert "summary" not in update_set
        # Importance is still assessed: short does not mean unimportant. "No
        # alcohol" is exactly the kind of short, high-importance constraint the
        # memory tier exists to keep.
        assert update_set["importance"] == 0.7

    async def test_a_real_summary_is_stored(self):
        memory = _make_pending_memory()
        col = _make_col_with_cursor([memory])
        providers = _make_providers()

        worker = EnrichmentWorker(col, _make_config(), providers, _make_memory_service())
        await worker.process_batch()

        assert col.update_one.call_args[0][1]["$set"]["summary"] == "A test summary"


class TestEnrichmentWorkerMergePending:
    """REQ-E-005: Enrichment worker handles merge_pending memories."""

    async def test_merge_pending_calls_llm_merge(self):
        """REQ-E-005: Worker merges content via LLM for merge_pending memories."""
        merge_target_id = ObjectId()
        merge_memory = {
            "_id": ObjectId(),
            "user_id": "user1",
            "content": "new content to merge",
            "enrichment_status": "merge_pending",
            "enrichment_retries": 0,
            "embedding": [0.1] * 1536,
            "merge_target_id": merge_target_id,
        }

        col = MagicMock()
        col.update_one = AsyncMock()
        # find() returns the merge_pending memory
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[merge_memory])
        col.find.return_value = mock_cursor
        # find_one() returns the merge target
        col.find_one = AsyncMock(return_value={
            "_id": merge_target_id,
            "content": "existing LTM content",
            "importance": 0.6,
        })

        config = _make_config()
        providers = _make_providers()
        # `complete`, not `chat`: the worker must not build the message itself.
        # See REQ-E-120 — this assertion used to be on `chat`, which let the
        # worker send an OpenAI-shaped message to every provider. Bedrock (the
        # default) rejected it and the merge silently never happened.
        providers.llm.complete = AsyncMock(
            return_value="merged content combining both"
        )
        memory_svc = _make_memory_service()

        worker = EnrichmentWorker(col, config, providers, memory_svc)
        await worker.process_batch()

        # Should have called LLM to merge content
        providers.llm.complete.assert_called_once()
        # Should update the memory with merged content and set status to complete
        update_calls = col.update_one.call_args_list
        assert len(update_calls) >= 2  # One for merged memory, one for soft-delete target
        # Find the update that sets enrichment_status to "complete"
        complete_found = False
        for call in update_calls:
            update_arg = call[0][1]
            if "$set" in update_arg and update_arg["$set"].get("enrichment_status") == "complete":
                assert update_arg["$set"]["content"] == "merged content combining both"
                complete_found = True
                break
        assert complete_found, "Should update memory with merged content and status=complete"

    async def test_merge_pending_soft_deletes_target(self):
        """REQ-E-005: After merging, the target memory should be soft-deleted."""
        merge_target_id = ObjectId()
        merge_memory = {
            "_id": ObjectId(),
            "user_id": "user1",
            "content": "new content",
            "enrichment_status": "merge_pending",
            "enrichment_retries": 0,
            "embedding": [0.1] * 1536,
            "merge_target_id": merge_target_id,
        }

        col = MagicMock()
        col.update_one = AsyncMock()
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[merge_memory])
        col.find.return_value = mock_cursor
        col.find_one = AsyncMock(return_value={
            "_id": merge_target_id,
            "content": "existing content",
            "importance": 0.6,
        })

        config = _make_config()
        providers = _make_providers()
        providers.llm.chat = AsyncMock(return_value="merged content")
        memory_svc = _make_memory_service()

        worker = EnrichmentWorker(col, config, providers, memory_svc)
        await worker.process_batch()

        # One of the update_one calls should soft-delete the merge target
        update_calls = col.update_one.call_args_list
        target_delete_found = False
        for call in update_calls:
            filter_arg = call[0][0]
            update_arg = call[0][1]
            if filter_arg.get("_id") == merge_target_id:
                if "$set" in update_arg and update_arg["$set"].get("is_deleted") is True:
                    target_delete_found = True
                    break
        assert target_delete_found, "Merge target should be soft-deleted after merge"


class TestEnrichmentWorkerRunLoop:
    """run() loop and stop() control."""

    async def test_run_processes_and_stops(self):
        col = _make_col_with_cursor([])
        config = _make_config(enrichment_interval_seconds=0)
        providers = _make_providers()
        memory_svc = _make_memory_service()

        worker = EnrichmentWorker(col, config, providers, memory_svc)
        # Stop after first iteration
        async def run_and_stop():
            await asyncio.sleep(0.05)
            worker.stop()
        asyncio.get_event_loop().create_task(run_and_stop())
        await worker.run()
        assert worker._running is False

    async def test_run_handles_cancelled_error(self):
        col = _make_col_with_cursor([])
        config = _make_config(enrichment_interval_seconds=0)
        providers = _make_providers()
        memory_svc = _make_memory_service()

        worker = EnrichmentWorker(col, config, providers, memory_svc)

        async def cancel_soon():
            await asyncio.sleep(0.05)
            task.cancel()

        task = asyncio.get_event_loop().create_task(worker.run())
        asyncio.get_event_loop().create_task(cancel_soon())

        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_run_breaks_on_cancelled_error_from_process_batch(self):
        """CancelledError raised inside process_batch triggers the break path."""
        col = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(side_effect=asyncio.CancelledError)
        col.find.return_value = mock_cursor
        config = _make_config(enrichment_interval_seconds=0)
        providers = _make_providers()
        memory_svc = _make_memory_service()

        worker = EnrichmentWorker(col, config, providers, memory_svc)
        await worker.run()
        # run() exited cleanly via the CancelledError break path
        # _running remains True because stop() was not called — the break
        # only exits the while loop without toggling the flag.
        assert worker._running is True

    async def test_run_handles_exception_in_batch(self):
        col = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(side_effect=Exception("db error"))
        col.find.return_value = mock_cursor
        config = _make_config(enrichment_interval_seconds=0)
        providers = _make_providers()
        memory_svc = _make_memory_service()

        worker = EnrichmentWorker(col, config, providers, memory_svc)
        async def stop_soon():
            await asyncio.sleep(0.05)
            worker.stop()
        asyncio.get_event_loop().create_task(stop_soon())
        await worker.run()  # Should not raise

    async def test_stop_sets_running_false(self):
        col = _make_col_with_cursor([])
        config = _make_config()
        providers = _make_providers()
        memory_svc = _make_memory_service()

        worker = EnrichmentWorker(col, config, providers, memory_svc)
        worker._running = True
        worker.stop()
        assert worker._running is False


class TestEnrichmentWorkerMergeTargetNotFound:
    """merge_pending with missing target marks as complete."""

    async def test_merge_target_deleted_marks_complete(self):
        merge_memory = {
            "_id": ObjectId(),
            "user_id": "user1",
            "content": "new content",
            "enrichment_status": "merge_pending",
            "enrichment_retries": 0,
            "embedding": [0.1] * 1536,
            "merge_target_id": ObjectId(),
        }

        col = MagicMock()
        col.update_one = AsyncMock()
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[merge_memory])
        col.find.return_value = mock_cursor
        col.find_one = AsyncMock(return_value=None)  # Target deleted

        config = _make_config()
        providers = _make_providers()
        memory_svc = _make_memory_service()

        worker = EnrichmentWorker(col, config, providers, memory_svc)
        await worker.process_batch()

        # Should mark as complete without calling LLM
        update_call = col.update_one.call_args
        assert update_call[0][1]["$set"]["enrichment_status"] == "complete"
        providers.llm.chat.assert_not_called()
