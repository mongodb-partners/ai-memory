"""Tests for EnrichmentWorker."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch
from bson import ObjectId

import pytest

from agent_memory.core.config import MCPConfig
from agent_memory.providers.base import LLMProvider
from agent_memory.services.enrichment import EnrichmentWorker


def _make_config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


def _make_providers():
    """Mock providers whose LLM presents the real method set.

    `spec=LLMProvider` catches a *renamed or removed* method: the mock raises
    AttributeError for anything not on the ABC. It does **not** enforce
    signatures — verified on 3.11.13 that `AsyncMock(spec=LLMProvider.assess_importance)`
    accepts bogus keywords and extra positional arguments. Signature drift is
    caught by `test_provider_prompt_contract.py`, which inspects the real
    implementations; that is the test that would have caught the `prompt=`
    TypeError on OpenAI and Anthropic.
    """
    providers = MagicMock()
    providers.llm = AsyncMock(spec=LLMProvider)
    providers.llm.assess_importance = AsyncMock(
        spec=LLMProvider.assess_importance, return_value=0.7
    )
    providers.llm.generate_summary = AsyncMock(
        spec=LLMProvider.generate_summary, return_value="A test summary"
    )
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
        col.find_one_and_update = AsyncMock(side_effect=[memory, None])

        worker = EnrichmentWorker(col, config, providers, memory_svc)
        count = await worker.process_batch()

        assert count == 1
        col.update_one.assert_called_once()
        update_call = col.update_one.call_args
        update_set = update_call[0][1]["$set"]
        assert update_set["enrichment_status"] == "complete"
        assert update_set["importance"] == 0.7
        assert update_set["summary"] == "A test summary"


def _make_col_with_queue(memories: list[dict]):
    """A collection whose claim call hands out `memories`, then reports empty.

    The worker claims one document per `find_one_and_update` and stops at the
    first `None`, so the queue is the list followed by a sentinel. A mock that
    returned the same document forever would loop to `enrichment_batch_size`
    and enrich one memory fifty times.
    """
    col = MagicMock()
    col.update_one = AsyncMock()
    col.find_one_and_update = AsyncMock(side_effect=[*memories, None])
    return col


#: The name this helper had when the worker selected work with `find`. Kept as an
#: alias so the diff that introduced claiming does not also rewrite forty call
#: sites — those tests are the evidence that claiming changed nothing else.
_make_col_with_cursor = _make_col_with_queue


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

        # `exclude_id` is part of the contract, not an optional extra. The worker
        # runs after the document is stored, so without it the evolution search's
        # top hit is this memory itself at similarity ~1.0 — above
        # reinforce_threshold by construction — and every enrichment pass
        # "reinforced" its own document instead of finding real duplicates.
        memory_svc.evolve_memory.assert_called_once_with(
            "user1",
            memory["content"],
            memory["embedding"],
            exclude_id=memory["_id"],
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
        # the claim hands out the merge_pending memory, then the queue is empty
        col.find_one_and_update = AsyncMock(side_effect=[merge_memory, None])
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
        col.find_one_and_update = AsyncMock(side_effect=[merge_memory, None])
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


class TestAMergeIsNotCommittedWithAnUnusableVector:
    """A wrong-width re-embed retries the merge rather than writing it.

    The re-embed exists so the merged document does not search as its pre-merge
    half. A vector of the wrong width is worse than that: Atlas accepts it and
    ``$vectorSearch`` then returns the document for nothing at all. Both halves of
    the merge — the content write and the target's soft-delete — must be skipped,
    which is what raising before either one achieves.
    """

    @staticmethod
    def _merge_setup(embedding_width):
        from agent_memory.providers.manager import ResolvedEmbedding

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
        col.find_one_and_update = AsyncMock(side_effect=[merge_memory, None])
        col.find_one = AsyncMock(return_value={
            "_id": merge_target_id,
            "content": "existing content",
            "importance": 0.6,
        })
        providers = _make_providers()
        providers.llm.complete = AsyncMock(return_value="merged content")
        providers.embedding.generate_embedding = AsyncMock(
            return_value=[0.1] * embedding_width
        )
        providers.embedding_spec = ResolvedEmbedding(model="m", dimension=1536)
        return col, providers, merge_target_id

    async def test_a_wrong_width_leaves_the_merge_pending(self):
        col, providers, target_id = self._merge_setup(1024)

        worker = EnrichmentWorker(
            col, _make_config(), providers, _make_memory_service()
        )
        await worker.process_batch()

        # `_enrich_memory` catches, counts a retry, and keeps `merge_pending` — so
        # the only write is the retry bookkeeping. Neither the merged content nor
        # the target's soft-delete happened.
        for call in col.update_one.call_args_list:
            set_arg = call[0][1].get("$set", {})
            assert set_arg.get("enrichment_status") != "complete"
            assert "content" not in set_arg
            assert call[0][0].get("_id") != target_id
        statuses = [
            c[0][1].get("$set", {}).get("enrichment_status")
            for c in col.update_one.call_args_list
        ]
        assert "merge_pending" in statuses

    async def test_the_target_survives_so_the_retry_can_use_it(self):
        # The soft-delete is the irreversible half. If it ran while the content
        # write did not, the retry would find no target and mark the merge complete
        # with the pre-merge content — losing the target's content for good.
        col, providers, target_id = self._merge_setup(1024)
        worker = EnrichmentWorker(
            col, _make_config(), providers, _make_memory_service()
        )
        await worker.process_batch()

        deletes = [
            c for c in col.update_one.call_args_list
            if c[0][1].get("$set", {}).get("is_deleted") is True
        ]
        assert deletes == []

    async def test_the_right_width_still_commits_the_merge(self):
        # Paired case: the guard must not turn every merge into a retry loop.
        col, providers, target_id = self._merge_setup(1536)
        worker = EnrichmentWorker(
            col, _make_config(), providers, _make_memory_service()
        )
        await worker.process_batch()

        completed = [
            c for c in col.update_one.call_args_list
            if c[0][1].get("$set", {}).get("enrichment_status") == "complete"
        ]
        assert len(completed) == 1
        assert completed[0][0][1]["$set"]["content"] == "merged content"
        assert completed[0][0][1]["$set"]["embedding"] == [0.1] * 1536


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
        col.find_one_and_update = AsyncMock(side_effect=asyncio.CancelledError)
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
        col.find_one_and_update = AsyncMock(side_effect=Exception("db error"))
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
        col.find_one_and_update = AsyncMock(side_effect=[merge_memory, None])
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


class TestScorerInjection:
    """REQ-E-160, REQ-E-172. The worker delegates importance to a scorer."""

    def test_defaults_to_an_llm_scorer(self):
        """Every existing call site omits `scorer`. Defaulting here is what makes
        'an upgrade changes nothing' true without editing twenty constructions."""
        from agent_memory.services.importance import LLMScorer

        worker = EnrichmentWorker(
            MagicMock(), _make_config(), _make_providers(), _make_memory_service()
        )
        assert isinstance(worker.scorer, LLMScorer)

    def test_default_scorer_wraps_the_configured_llm(self):
        providers = _make_providers()
        worker = EnrichmentWorker(
            MagicMock(), _make_config(), providers, _make_memory_service()
        )
        assert worker.scorer._llm is providers.llm

    def test_default_scorer_uses_the_worker_prompt_getter(self):
        """The prompt library moved behind the scorer. If it is not wired, a
        deployment with a customized importance prompt silently reverts to the
        provider's built-in template — same scores-look-fine failure mode."""
        worker = EnrichmentWorker(
            MagicMock(), _make_config(), _make_providers(), _make_memory_service()
        )
        assert worker.scorer._prompt_getter == worker._get_prompt

    async def test_injected_scorer_is_used(self):
        from agent_memory.services.importance import ImportanceScorer

        scorer = create_autospec(ImportanceScorer, instance=True)
        scorer.score.return_value = 0.42
        col = MagicMock()
        col.update_one = AsyncMock()
        worker = EnrichmentWorker(
            col, _make_config(), _make_providers(), _make_memory_service(),
            scorer=scorer,
        )
        await worker._process_standard_enrichment(_make_pending_memory())
        assert col.update_one.call_args[0][1]["$set"]["importance"] == 0.42

    async def test_injected_scorer_receives_content_and_embedding(self):
        from agent_memory.services.importance import ImportanceScorer

        scorer = create_autospec(ImportanceScorer, instance=True)
        scorer.score.return_value = 0.42
        col = MagicMock()
        col.update_one = AsyncMock()
        memory = _make_pending_memory()
        worker = EnrichmentWorker(
            col, _make_config(), _make_providers(), _make_memory_service(),
            scorer=scorer,
        )
        await worker._process_standard_enrichment(memory)
        args, kwargs = scorer.score.call_args
        assert args[0] == memory["content"]
        assert args[1] == memory["embedding"]

    async def test_injected_scorer_replaces_the_llm_importance_call(self):
        """The reason the feature exists. If `assess_importance` still fires, the
        local path costs a token round trip and saves nothing."""
        from agent_memory.services.importance import ImportanceScorer

        scorer = create_autospec(ImportanceScorer, instance=True)
        scorer.score.return_value = 0.42
        providers = _make_providers()
        col = MagicMock()
        col.update_one = AsyncMock()
        worker = EnrichmentWorker(
            col, _make_config(), providers, _make_memory_service(), scorer=scorer,
        )
        await worker._process_standard_enrichment(_make_pending_memory())
        providers.llm.assess_importance.assert_not_awaited()

    async def test_summary_still_uses_the_llm(self):
        """Only scoring is swappable. Summarization is generation and stays on the
        LLM — a linear model cannot write a summary."""
        from agent_memory.services.importance import ImportanceScorer

        scorer = create_autospec(ImportanceScorer, instance=True)
        scorer.score.return_value = 0.42
        providers = _make_providers()
        col = MagicMock()
        col.update_one = AsyncMock()
        worker = EnrichmentWorker(
            col, _make_config(), providers, _make_memory_service(), scorer=scorer,
        )
        await worker._process_standard_enrichment(_make_pending_memory())
        providers.llm.generate_summary.assert_awaited()

    async def test_scorer_failure_leaves_the_memory_retryable(self):
        """A scorer that raises must go down the existing retry path rather than
        writing a wrong importance."""
        from agent_memory.services.importance import ImportanceScorer

        scorer = create_autospec(ImportanceScorer, instance=True)
        scorer.score.side_effect = RuntimeError("artifact went away")
        col = MagicMock()
        col.update_one = AsyncMock()
        worker = EnrichmentWorker(
            col, _make_config(), _make_providers(), _make_memory_service(),
            scorer=scorer,
        )
        await worker._enrich_memory(_make_pending_memory())
        update = col.update_one.call_args[0][1]["$set"]
        assert update["enrichment_status"] == "pending"
        assert update["enrichment_retries"] == 1


class TestLLMPathUnchanged:
    """The safety property, asserted against the default construction.

    These read `await_args` rather than calling `assert_awaited_once_with`. The
    fixture specs the mock on the *unbound* `LLMProvider.assess_importance`, whose
    first parameter is `self`, so the matcher binds the content string to `self`
    and reports a mismatch on calls that are in fact identical. Another way the
    `spec=` in `_make_providers` is weaker than it looks — see its docstring.
    """

    async def test_still_calls_assess_importance_with_the_library_prompt(self):
        col = MagicMock()
        col.update_one = AsyncMock()
        providers = _make_providers()
        library = MagicMock()
        library.get_prompt = AsyncMock(return_value="Rate: {content}")
        worker = EnrichmentWorker(
            col, _make_config(), providers, _make_memory_service(),
            prompt_library=library,
        )
        await worker._process_standard_enrichment(_make_pending_memory())
        assert providers.llm.assess_importance.await_count == 1
        args, kwargs = providers.llm.assess_importance.await_args
        assert args == (LONG_CONTENT,)
        assert kwargs == {"prompt": "Rate: {content}"}

    async def test_omits_prompt_when_the_library_has_none(self):
        """No `prompt` kwarg at all, not `prompt=None`. The providers' default
        templates are keyed on the kwarg being absent."""
        col = MagicMock()
        col.update_one = AsyncMock()
        providers = _make_providers()
        worker = EnrichmentWorker(
            col, _make_config(), providers, _make_memory_service()
        )
        await worker._process_standard_enrichment(_make_pending_memory())
        assert providers.llm.assess_importance.await_count == 1
        args, kwargs = providers.llm.assess_importance.await_args
        assert args == (LONG_CONTENT,)
        assert kwargs == {}


# ── Atomic claiming ─────────────────────────────────────────────────────────────


class _ClaimableCollection:
    """A collection that models `find_one_and_update` as a real atomic operation.

    Written as a fake rather than a mock because what is under test *is* the
    atomicity. A `MagicMock` returns whatever it was told to return, so two
    workers polling it both "win" every document and the test passes whether or
    not the production code claims anything — which is precisely the bug.

    Here, the matched document is mutated in place before the call returns, so a
    second caller sees the claim the first one made. That is the property MongoDB
    guarantees for a single-document update and the only one this fix relies on.
    """

    def __init__(self, docs: list[dict], now: datetime | None = None) -> None:
        self.docs = docs
        self.now = now or datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
        self.claim_filters: list[dict] = []
        self.updates: list[tuple[dict, dict]] = []

    def _matches(self, doc: dict, query: dict) -> bool:
        for key, cond in query.items():
            if key == "$or":
                if not any(self._matches(doc, branch) for branch in cond):
                    return False
                continue
            value = doc.get(key)
            if isinstance(cond, dict):
                for op, operand in cond.items():
                    if op == "$in" and value not in operand:
                        return False
                    if op == "$eq" and value != operand:
                        return False
                    if op == "$lt" and not (value is not None and value < operand):
                        return False
                    if op == "$exists" and (value is not None) != operand:
                        return False
            elif value != cond:
                return False
        return True

    async def find_one_and_update(
        self, query: dict, update: dict, sort=None, return_document=None
    ):
        self.claim_filters.append(query)
        candidates = [d for d in self.docs if self._matches(d, query)]
        if sort:
            for key, direction in reversed(sort):
                candidates.sort(
                    key=lambda d: d.get(key) or 0, reverse=direction < 0
                )
        if not candidates:
            return None
        doc = candidates[0]
        before = dict(doc)
        doc.update(update.get("$set", {}))
        return before

    async def update_one(self, query: dict, update: dict) -> None:
        self.updates.append((query, update))
        for doc in self.docs:
            if self._matches(doc, query):
                doc.update(update.get("$set", {}))
                for field in update.get("$unset", {}):
                    doc.pop(field, None)
                break

    async def find_one(self, query: dict):
        for doc in self.docs:
            if self._matches(doc, query):
                return doc
        return None

    def find(self, query: dict, sort=None, limit=None):
        """Non-atomic selection — deliberately supported though production no
        longer calls it.

        Without this, reverting the worker to a plain `find` would fail the tests
        below with `AttributeError` rather than on their assertions, which proves
        only that the method name changed. With it, the concurrency test fails
        because both workers really do win the same document — the actual defect.
        """
        matched = [d for d in self.docs if self._matches(d, query)]
        if sort:
            for key, direction in reversed(sort):
                matched.sort(key=lambda d: d.get(key) or 0, reverse=direction < 0)
        if limit:
            matched = matched[:limit]

        class _Cursor:
            async def to_list(self, _n=None):
                return matched

        return _Cursor()


def _claimable(**overrides) -> dict:
    doc = _make_pending_memory()
    doc.update(overrides)
    return doc


class TestWorkIsClaimedBeforeItIsDone:
    """Two workers on one queue must not both enrich the same memory.

    `TRANSPORT=rest` and `TRANSPORT=mcp` are separate processes, each running a
    worker, and running both is the documented way to serve both shells. Before
    claiming, they polled the same collection with a plain `find` and both got the
    same documents: two LLM bills per memory, two evolution passes over the same
    content, and a last-writer-wins race on the resulting importance and summary.
    """

    async def test_two_workers_do_not_claim_the_same_memory(self):
        col = _ClaimableCollection([_claimable(), _claimable()])
        config = _make_config(enrichment_batch_size=10)
        worker_a = EnrichmentWorker(col, config, _make_providers(), _make_memory_service())
        worker_b = EnrichmentWorker(col, config, _make_providers(), _make_memory_service())

        # Claims only — not full enrichment — so the assertion is about who won
        # each document rather than about what enrichment then did with it.
        claims_a = [await worker_a._claim_one() for _ in range(2)]
        claims_b = [await worker_b._claim_one() for _ in range(2)]

        won_a = {c["_id"] for c in claims_a if c}
        won_b = {c["_id"] for c in claims_b if c}
        assert len(won_a) == 2, "first worker should claim both available memories"
        assert won_b == set(), (
            "second worker claimed memories the first already owns — this is the "
            "duplicate-enrichment bug the claim exists to prevent"
        )

    async def test_a_claimed_memory_is_not_reclaimed_within_the_lease(self):
        doc = _claimable()
        col = _ClaimableCollection([doc])
        worker = EnrichmentWorker(
            col, _make_config(), _make_providers(), _make_memory_service()
        )
        assert await worker._claim_one() is not None
        assert await worker._claim_one() is None

    async def test_the_claim_stamps_the_document(self):
        doc = _claimable()
        col = _ClaimableCollection([doc])
        worker = EnrichmentWorker(
            col, _make_config(), _make_providers(), _make_memory_service()
        )
        await worker._claim_one()
        assert isinstance(doc["enrichment_claimed_at"], datetime)

    async def test_the_claim_returns_the_document_as_it_was(self):
        """`ReturnDocument.BEFORE`: the caller needs the memory's content, and the
        claim's own write is the only difference between the two versions."""
        doc = _claimable()
        col = _ClaimableCollection([doc])
        worker = EnrichmentWorker(
            col, _make_config(), _make_providers(), _make_memory_service()
        )
        claimed = await worker._claim_one()
        assert claimed["content"] == LONG_CONTENT
        assert claimed["enrichment_status"] == "pending"

    async def test_process_batch_stops_at_the_first_empty_claim(self):
        """The batch size is a ceiling, not a target. A queue of two must not cost
        fifty round trips."""
        col = _ClaimableCollection([_claimable(), _claimable()])
        worker = EnrichmentWorker(
            col, _make_config(enrichment_batch_size=50),
            _make_providers(), _make_memory_service(),
        )
        count = await worker.process_batch()
        assert count == 2
        assert len(col.claim_filters) == 3, (
            "expected two winning claims plus one that finds the queue empty"
        )

    async def test_an_empty_queue_costs_one_claim(self):
        col = _ClaimableCollection([])
        worker = EnrichmentWorker(
            col, _make_config(enrichment_batch_size=50),
            _make_providers(), _make_memory_service(),
        )
        assert await worker.process_batch() == 0
        assert len(col.claim_filters) == 1

    async def test_merge_pending_is_claimed_too(self):
        """Both queue states go through the same claim. `merge_pending` is the more
        expensive one to duplicate — it writes merged content and soft-deletes a
        target, so two workers racing it can delete two memories for one merge."""
        col = _ClaimableCollection([_claimable(enrichment_status="merge_pending")])
        worker = EnrichmentWorker(
            col, _make_config(), _make_providers(), _make_memory_service()
        )
        claimed = await worker._claim_one()
        assert claimed is not None
        assert claimed["enrichment_status"] == "merge_pending"

    async def test_terminal_states_are_not_claimed(self):
        col = _ClaimableCollection([
            _claimable(enrichment_status="complete"),
            _claimable(enrichment_status="failed"),
            _claimable(enrichment_status="not_applicable"),
        ])
        worker = EnrichmentWorker(
            col, _make_config(), _make_providers(), _make_memory_service()
        )
        assert await worker._claim_one() is None


class TestAStrandedClaimIsRecoverable:
    """A lease, not a lock.

    A worker that is SIGKILLed mid-LLM-call leaves its claim on the document. With
    a plain lock that memory is never enriched again — it sits in `pending`
    forever, invisible except as a number in the admin status breakdown. The
    expiry is what makes losing a worker a delay rather than data left unenriched.
    """

    async def test_an_expired_lease_can_be_taken_over(self):
        stale = datetime.now(timezone.utc) - timedelta(seconds=601)
        col = _ClaimableCollection([_claimable(enrichment_claimed_at=stale)])
        worker = EnrichmentWorker(
            col, _make_config(enrichment_lease_seconds=300),
            _make_providers(), _make_memory_service(),
        )
        assert await worker._claim_one() is not None, (
            "a memory whose owner died is stranded forever unless the claim expires"
        )

    async def test_a_live_lease_is_left_alone(self):
        fresh = datetime.now(timezone.utc) - timedelta(seconds=10)
        col = _ClaimableCollection([_claimable(enrichment_claimed_at=fresh)])
        worker = EnrichmentWorker(
            col, _make_config(enrichment_lease_seconds=300),
            _make_providers(), _make_memory_service(),
        )
        assert await worker._claim_one() is None

    async def test_the_lease_length_is_configurable(self):
        """An operator whose provider is slow needs a longer lease; the alternative
        is a second worker starting work the first has not finished."""
        claimed_at = datetime.now(timezone.utc) - timedelta(seconds=400)
        long_lease = _ClaimableCollection([_claimable(enrichment_claimed_at=claimed_at)])
        short_lease = _ClaimableCollection([_claimable(enrichment_claimed_at=claimed_at)])

        patient = EnrichmentWorker(
            long_lease, _make_config(enrichment_lease_seconds=900),
            _make_providers(), _make_memory_service(),
        )
        impatient = EnrichmentWorker(
            short_lease, _make_config(enrichment_lease_seconds=60),
            _make_providers(), _make_memory_service(),
        )
        assert await patient._claim_one() is None
        assert await impatient._claim_one() is not None

    async def test_a_document_predating_the_field_is_claimable(self):
        """Every memory written before this change, and every one written by
        `store_ltm`, has no `enrichment_claimed_at` at all. Treating a missing
        field as "claimed" would freeze the existing queue on upgrade."""
        doc = _claimable()
        doc.pop("enrichment_claimed_at", None)
        col = _ClaimableCollection([doc])
        worker = EnrichmentWorker(
            col, _make_config(), _make_providers(), _make_memory_service()
        )
        assert await worker._claim_one() is not None


class TestTheClaimIsReleasedWhenWorkEnds:
    """A finished document must not keep its lease.

    `consolidation` puts memories back into `pending` to be re-enriched. A
    leftover claim would make that re-enrichment wait out the lease for no
    reason, and on failure it would delay the retry the counter was just
    incremented for.
    """

    async def test_success_releases_the_claim(self):
        doc = _claimable()
        col = _ClaimableCollection([doc])
        worker = EnrichmentWorker(
            col, _make_config(), _make_providers(), _make_memory_service()
        )
        await worker.process_batch()
        assert doc["enrichment_status"] == "complete"
        assert "enrichment_claimed_at" not in doc

    async def test_failure_releases_the_claim_so_the_retry_is_immediate(self):
        doc = _claimable()
        col = _ClaimableCollection([doc])
        providers = _make_providers()
        providers.llm.assess_importance = AsyncMock(side_effect=Exception("LLM down"))
        worker = EnrichmentWorker(
            col, _make_config(enrichment_max_retries=3), providers,
            _make_memory_service(),
        )
        await worker.process_batch()
        assert doc["enrichment_status"] == "pending"
        assert doc["enrichment_retries"] == 1
        assert "enrichment_claimed_at" not in doc, (
            "the memory is retryable but would wait a full lease to be picked up"
        )

    async def test_the_retry_is_actually_claimable_again(self):
        """The end-to-end version of the previous test: fail once, then confirm a
        fresh poll picks the same memory back up rather than skipping it."""
        doc = _claimable()
        col = _ClaimableCollection([doc])
        providers = _make_providers()
        providers.llm.assess_importance = AsyncMock(side_effect=Exception("LLM down"))
        worker = EnrichmentWorker(
            col, _make_config(enrichment_max_retries=3), providers,
            _make_memory_service(),
        )
        await worker.process_batch()
        assert await worker._claim_one() is not None

    async def test_exhausted_retries_release_the_claim_too(self):
        doc = _claimable(enrichment_retries=2)
        col = _ClaimableCollection([doc])
        providers = _make_providers()
        providers.llm.assess_importance = AsyncMock(side_effect=Exception("LLM down"))
        worker = EnrichmentWorker(
            col, _make_config(enrichment_max_retries=3), providers,
            _make_memory_service(),
        )
        await worker.process_batch()
        assert doc["enrichment_status"] == "failed"
        assert "enrichment_claimed_at" not in doc

    async def test_every_terminal_write_unsets_the_claim(self):
        """A structural check over the source, not a behavioural one.

        There are four places the worker writes a final `enrichment_status`, and a
        fifth added later would silently strand a lease. Each is expected to carry
        an `$unset` of the claim field.
        """
        import re
        from pathlib import Path

        source = Path(
            EnrichmentWorker.__module__.replace(".", "/") + ".py"
        ).read_text()
        status_writes = re.findall(
            r'"enrichment_status":\s*(?:status|"[a-z_]+")', source
        )
        # One per terminal write plus the claim filter's `$in`, which is a read.
        unsets = source.count('"$unset": {"enrichment_claimed_at": ""}')
        assert unsets == len(status_writes), (
            f"{len(status_writes)} writes set enrichment_status but only {unsets} "
            "release the claim; a write that keeps the lease strands the document "
            "for a full lease period after it is already finished"
        )


class TestEvolutionsDecisionSurvivesTheFinalWrite:
    """Enrichment ends with a write; that write must not undo what evolution did.

    `_process_standard_enrichment` calls `evolve_memory` and then sets
    `enrichment_status: complete`. When evolution has just queued a merge — setting
    the status to `merge_pending` on this very document — the unconditional
    `complete` overwrote it, leaving a document with a `merge_target_id` that no
    worker would ever act on. The merge was silently dropped and the duplicate
    stayed live.
    """

    def _worker(self, outcome, col=None):
        col = col or _ClaimableCollection([_claimable()])
        memory_svc = _make_memory_service()
        memory_svc.evolve_memory = AsyncMock(return_value=outcome)
        return EnrichmentWorker(
            col, _make_config(), _make_providers(), memory_svc
        ), col

    async def test_a_queued_merge_is_not_marked_complete(self):
        doc = _claimable()
        worker, col = self._worker("merge_queued", _ClaimableCollection([doc]))
        # `evolve_memory` is mocked, so emulate the status it would have written.
        doc["enrichment_status"] = "merge_pending"

        await worker._process_standard_enrichment(dict(doc, enrichment_status="pending"))

        assert doc["enrichment_status"] == "merge_pending", (
            "the final write overwrote the queued merge with 'complete'; the "
            "document keeps a merge_target_id no worker will act on"
        )

    async def test_the_importance_is_still_recorded_for_a_queued_merge(self):
        """Skipping the status must not skip the work that was actually done."""
        doc = _claimable()
        worker, col = self._worker("merge_queued", _ClaimableCollection([doc]))

        await worker._process_standard_enrichment(dict(doc))

        assert doc["importance"] == 0.7
        assert doc["summary"] == "A test summary"

    async def test_an_ordinary_enrichment_still_completes(self):
        doc = _claimable()
        worker, col = self._worker("created", _ClaimableCollection([doc]))

        await worker._process_standard_enrichment(dict(doc))

        assert doc["enrichment_status"] == "complete"

    async def test_a_reinforced_memory_completes_too(self):
        """On `reinforced` evolution soft-deleted this document. `complete` is the
        right status for it — what matters is that the write sets only the three
        fields it owns and does not resurrect it."""
        doc = _claimable()
        col = _ClaimableCollection([doc])
        worker, _ = self._worker("reinforced", col)
        # The state evolution leaves behind.
        doc.update(deleted_at=datetime.now(timezone.utc), is_deleted=True)

        await worker._process_standard_enrichment(dict(doc))

        assert doc["enrichment_status"] == "complete"
        assert doc["deleted_at"] is not None, "the final write undid the retirement"
        assert doc["is_deleted"] is True
