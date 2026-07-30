"""Tests for ConsolidationWorker."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

from agent_memory.core.config import MCPConfig
from agent_memory.services.consolidation import ConsolidationWorker


def _make_config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


def _make_providers():
    providers = MagicMock()
    providers.llm = AsyncMock()
    providers.llm.generate_summary = AsyncMock(return_value="compressed summary")
    return providers


# Long enough to clear MIN_SUMMARIZABLE_CHARS. Below that threshold `_compress_stm`
# skips the memory entirely, so a short fixture would make the tests below pass
# without ever reaching the LLM — green for the wrong reason.
LONG_CONTENT = (
    "I cook for four most nights and for six when guests come over, so I plan "
    "around sheet-pan roasts and one-pot pasta that scale without extra work."
)


def _make_collection():
    col = MagicMock()
    col.find = MagicMock()
    col.update_one = AsyncMock()
    col.update_many = AsyncMock(return_value=MagicMock(modified_count=0))
    return col


class TestCompressSTM:
    """_compress_stm finds and summarizes old STM memories."""

    async def test_compress_finds_old_stm(self):
        col = _make_collection()
        config = _make_config(stm_compression_age_hours=24)
        providers = _make_providers()
        worker = ConsolidationWorker(col, config, providers)

        old_memory = {
            "_id": ObjectId(),
            "content": LONG_CONTENT,
            "tier": "stm",
        }
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[old_memory])
        col.find = MagicMock(return_value=mock_cursor)

        count = await worker._compress_stm()

        assert count == 1
        providers.llm.generate_summary.assert_called_once_with(LONG_CONTENT)
        col.update_one.assert_called_once()
        update_set = col.update_one.call_args[0][1]["$set"]
        assert update_set["summary"] == "compressed summary"

    async def test_compress_skips_when_no_old_stm(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        worker = ConsolidationWorker(col, config, providers)

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        col.find = MagicMock(return_value=mock_cursor)

        count = await worker._compress_stm()

        assert count == 0
        providers.llm.generate_summary.assert_not_called()

    async def test_compress_handles_llm_failure(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        providers.llm.generate_summary = AsyncMock(side_effect=RuntimeError("LLM down"))
        worker = ConsolidationWorker(col, config, providers)

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[
            {"_id": ObjectId(), "content": LONG_CONTENT, "tier": "stm"},
        ])
        col.find = MagicMock(return_value=mock_cursor)

        count = await worker._compress_stm()
        assert count == 0


class TestCompressSTMRejectsNonSummaries:
    """REQ-E-121: a reply is not a summary just because the call succeeded.

    ``generate_summary`` on a short conversational turn returns a well-formed
    string saying the model cannot do it. Storing that string put "This text
    fragment is too brief and lacks sufficient context" into the sample UI's
    memory panel, where the memory should have been — visible on a booth screen,
    and invisible to every test, because nothing raised and the field was set.
    """

    async def _run(self, content: str, reply: str):
        col = _make_collection()
        providers = _make_providers()
        providers.llm.generate_summary = AsyncMock(return_value=reply)
        worker = ConsolidationWorker(col, _make_config(), providers)

        cursor = AsyncMock()
        cursor.to_list = AsyncMock(
            return_value=[{"_id": ObjectId(), "content": content, "tier": "stm"}]
        )
        col.find = MagicMock(return_value=cursor)

        count = await worker._compress_stm()
        return count, col, providers

    @pytest.mark.parametrize(
        "reply",
        [
            "I don't see the original text that needs to be summarized.",
            "This text fragment is too brief and lacks sufficient context.",
            "I'll help summarize this text, but it appears to be incomplete.",
            "Please provide the text you would like me to summarize.",
            "",
            "   ",
        ],
    )
    async def test_a_refusal_is_not_stored(self, reply):
        count, col, _ = await self._run(LONG_CONTENT, reply)

        assert count == 0
        # Not stored *at all* — leaving the field unset is what makes readers
        # fall back to `content`, which is always the memory itself.
        col.update_one.assert_not_called()

    async def test_a_summary_longer_than_its_source_is_rejected(self):
        """Compression that expands is not compression."""
        count, col, _ = await self._run(
            "Short.", "A detailed elaboration of the single word above, at length."
        )

        assert count == 0
        col.update_one.assert_not_called()

    async def test_content_too_short_is_never_sent_to_the_model(self):
        """The skip is before the call, not after: no point paying for a refusal."""
        count, col, providers = await self._run(
            "No alcohol in pairings.", "compressed summary"
        )

        assert count == 0
        providers.llm.generate_summary.assert_not_called()
        col.update_one.assert_not_called()

    async def test_a_real_summary_still_lands(self):
        """The guard must not swallow the working case."""
        count, col, _ = await self._run(LONG_CONTENT, "Cooks for four to six.")

        assert count == 1
        assert col.update_one.call_args[0][1]["$set"]["summary"] == (
            "Cooks for four to six."
        )


class TestForgetLowImportance:
    """_forget_low_importance soft-deletes low-scoring memories."""

    async def test_forget_deletes_low_importance(self):
        col = _make_collection()
        config = _make_config(forgetting_score_threshold=0.1)
        providers = _make_providers()
        worker = ConsolidationWorker(col, config, providers)

        col.update_many = AsyncMock(return_value=MagicMock(modified_count=3))

        count = await worker._forget_low_importance()

        assert count == 3
        query = col.update_many.call_args[0][0]
        assert query["importance"]["$lt"] == 0.1
        assert query["tier"] == "ltm"
        assert query["deleted_at"] is None

    async def test_forget_skips_when_none_qualify(self):
        col = _make_collection()
        config = _make_config(forgetting_score_threshold=0.1)
        providers = _make_providers()
        worker = ConsolidationWorker(col, config, providers)

        col.update_many = AsyncMock(return_value=MagicMock(modified_count=0))

        count = await worker._forget_low_importance()
        assert count == 0


class TestPromoteToLTM:
    """_promote_to_ltm promotes qualifying STM to LTM."""

    async def test_promote_criteria_met(self):
        col = _make_collection()
        config = _make_config(
            promotion_importance_threshold=0.6,
            promotion_access_threshold=2,
            promotion_age_minutes=60,
        )
        providers = _make_providers()
        worker = ConsolidationWorker(col, config, providers)

        candidate = {
            "_id": ObjectId(),
            "tier": "stm",
            "importance": 0.8,
            "access_count": 5,
        }
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[candidate])
        col.find = MagicMock(return_value=mock_cursor)

        count = await worker._promote_to_ltm()

        assert count == 1
        col.update_one.assert_called_once()
        update_set = col.update_one.call_args[0][1]["$set"]
        assert update_set["tier"] == "ltm"
        assert update_set["retention_tier"] == "standard"
        assert update_set["enrichment_status"] == "pending"

    async def test_promote_skips_when_no_candidates(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        worker = ConsolidationWorker(col, config, providers)

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        col.find = MagicMock(return_value=mock_cursor)

        count = await worker._promote_to_ltm()
        assert count == 0

    async def test_promote_checks_all_criteria(self):
        """Find query includes importance, access_count, and age filters."""
        col = _make_collection()
        config = _make_config(
            promotion_importance_threshold=0.7,
            promotion_access_threshold=3,
            promotion_age_minutes=30,
        )
        providers = _make_providers()
        worker = ConsolidationWorker(col, config, providers)

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        col.find = MagicMock(return_value=mock_cursor)

        await worker._promote_to_ltm()

        find_query = col.find.call_args[0][0]
        assert find_query["tier"] == "stm"
        assert find_query["importance"]["$gte"] == 0.7
        assert find_query["access_count"]["$gte"] == 3
        assert "$lt" in find_query["created_at"]


class TestConsolidateStats:
    """consolidate() returns stats from all operations."""

    async def test_consolidate_returns_stats(self):
        col = _make_collection()
        config = _make_config()
        providers = _make_providers()
        worker = ConsolidationWorker(col, config, providers)

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        col.find = MagicMock(return_value=mock_cursor)
        col.update_many = AsyncMock(return_value=MagicMock(modified_count=0))

        stats = await worker.consolidate()

        assert "compressed" in stats
        assert "forgotten" in stats
        assert "promoted" in stats


class TestConsolidationWorkerLifecycle:
    """Worker run/stop lifecycle."""

    async def test_run_and_stop(self):
        col = _make_collection()
        config = _make_config(consolidation_interval_hours=24)
        providers = _make_providers()
        worker = ConsolidationWorker(col, config, providers)

        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        col.find = MagicMock(return_value=mock_cursor)
        col.update_many = AsyncMock(return_value=MagicMock(modified_count=0))

        task = asyncio.create_task(worker.run())
        # Let the first consolidation cycle complete
        await asyncio.sleep(0.05)
        worker.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert worker._running is False

    async def test_run_handles_exception_gracefully(self):
        col = _make_collection()
        config = _make_config(consolidation_interval_hours=24)
        providers = _make_providers()
        worker = ConsolidationWorker(col, config, providers)

        # Make consolidate raise an error
        col.find = MagicMock(side_effect=RuntimeError("db error"))

        task = asyncio.create_task(worker.run())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Worker should have survived the error
        assert worker._running is True
