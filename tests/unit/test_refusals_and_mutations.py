"""The four remaining High findings, which share a shape: a silent wrong answer.

None of these raise. Each one succeeds, writes a plausible-looking document or
returns a plausible-looking result, and is wrong in a way that only shows up much
later — which is exactly why a green suite kept passing over them.

- **Audit gaps.** A denied or throttled call left no audit record at all. The two
  events an audit log exists to capture were the only two it could not show.
- **Regex injection.** ``invalidate(pattern=...)`` interpolated caller input into
  ``$regex``, so ``.*`` cleared the whole cache while asking for one entry.
- **The merge cluster.** The merge fetched its target without a ``user_id`` filter,
  then rewrote ``content`` while leaving the pre-merge ``embedding`` in place — a
  document that reads as merged and searches as its own earlier half.
- **Self-reinforcement.** ``evolve_memory`` ran after the document was stored, so
  its top hit was itself at similarity ~1.0; every enrichment pass "reinforced"
  its own memory and never reached the real duplicates below it.

Requirements: REQ-E-146 (refusals are audited), REQ-E-147 (caller input is not a
query language), REQ-E-148 (content and embedding never disagree), REQ-E-149
(evolution does not consider the memory being evolved).
"""

import re
from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

from agent_memory.config import MemoryConfig
from agent_memory.exceptions import AccessError, RateLimitError
from agent_memory.memory import AsyncMemory
from agent_memory.services.cache import CacheService
from agent_memory.services.enrichment import EnrichmentWorker
from agent_memory.services.memory import MemoryService


def _config(**overrides) -> MemoryConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MemoryConfig(**defaults, _env_file=None)


def _facade(config=None):
    m = AsyncMemory.__new__(AsyncMemory)
    m.config = config or _config()
    m.memory_service = AsyncMock()
    m.cache_service = AsyncMock()
    m.decision_service = AsyncMock()
    m.admin_service = AsyncMock()
    m.audit_service = AsyncMock()
    m.episodic_service = AsyncMock()
    m.episodic_service.log_activity = MagicMock(return_value=True)
    m.governance_service = None
    m.rate_limiter = None
    m.providers = MagicMock()
    m._workers = []
    return m


def _cursor(docs):
    c = MagicMock()
    c.to_list = AsyncMock(return_value=docs)
    return c


# ── Refusals are audited ─────────────────────────────────────────────────────


class TestRefusalsAreAudited:
    """A denial and a throttle each leave a record, with distinct statuses."""

    async def test_a_denied_operation_is_audited(self):
        m = _facade()
        m.governance_service = AsyncMock()
        m.governance_service.check_allowed = AsyncMock(return_value=False)

        with pytest.raises(AccessError):
            await m.add("u1", "c1", [{"role": "human", "content": "hi"}])

        # The finding was that this call count was zero: `_check_access` ran ahead
        # of the audit block, so the refusal returned before anything was written.
        m.audit_service.log.assert_awaited_once()
        assert m.audit_service.log.await_args.args[3] == "denied"

    async def test_a_throttled_operation_is_audited_as_throttled(self):
        m = _facade()
        m.rate_limiter = AsyncMock()
        m.rate_limiter.check_rate_limit = AsyncMock(return_value=False)

        with pytest.raises(RateLimitError):
            await m.add("u1", "c1", [{"role": "human", "content": "hi"}])

        m.audit_service.log.assert_awaited_once()
        # Not "denied". RateLimitError subclasses AccessError, so an isinstance
        # chain that tests the base first labels every throttle as a denial — and
        # sends an operator looking for an authorisation problem that is really a
        # traffic one.
        assert m.audit_service.log.await_args.args[3] == "throttled"

    async def test_a_service_fault_is_audited_as_error(self):
        m = _facade()
        m.memory_service.store_stm = AsyncMock(side_effect=RuntimeError("atlas down"))

        with pytest.raises(RuntimeError):
            await m.add("u1", "c1", [{"role": "human", "content": "hi"}])

        assert m.audit_service.log.await_args.args[3] == "error"

    async def test_a_refusal_is_audited_exactly_once(self):
        """One event, one record.

        The first version of this fix used a nested try: the inner handler wrote
        "denied", then the outer `except Exception` caught the re-raise and wrote
        the same exception again as "error". Two records for one event, the second
        contradicting the first.
        """
        m = _facade()
        m.governance_service = AsyncMock()
        m.governance_service.check_allowed = AsyncMock(return_value=False)

        with pytest.raises(AccessError):
            await m.add("u1", "c1", [{"role": "human", "content": "hi"}])

        assert m.audit_service.log.await_count == 1

    async def test_the_denied_record_carries_the_operation_not_a_generic_label(self):
        m = _facade()
        m.governance_service = AsyncMock()
        m.governance_service.check_allowed = AsyncMock(return_value=False)

        with pytest.raises(AccessError):
            await m.recall("u1", "query")

        args = m.audit_service.log.await_args.args
        assert args[0] == "u1"
        assert args[2] == "recall_memory"

    async def test_a_successful_operation_still_audits_success(self):
        """The fix must not have turned the success path into a refusal path."""
        m = _facade()
        m.memory_service.store_stm = AsyncMock(return_value=["id1"])

        await m.add("u1", "c1", [{"role": "human", "content": "hi"}])

        assert m.audit_service.log.await_args.args[3] == "success"

    async def test_log_activity_audits_a_denial_despite_batching_successes(self):
        """`log_activity` bypasses `_run` by design, so it needed the fix twice.

        Skipping the per-call audit is a volume decision about the *success* path —
        a turn log is high-volume and routing it through the audit buffer costs
        more writes than the agent itself. A denial is rare and security-relevant,
        so the volume argument does not apply to it.
        """
        m = _facade()
        m.governance_service = AsyncMock()
        m.governance_service.check_allowed = AsyncMock(return_value=False)

        with pytest.raises(AccessError):
            await m.log_activity("u1", "t1", [])

        m.audit_service.log.assert_awaited_once()
        assert m.audit_service.log.await_args.args[3] == "denied"
        # And the turn did not get enqueued anyway.
        m.episodic_service.log_activity.assert_not_called()

    async def test_log_activity_audits_a_throttle_as_throttled(self):
        m = _facade()
        m.rate_limiter = AsyncMock()
        m.rate_limiter.check_rate_limit = AsyncMock(return_value=False)

        with pytest.raises(RateLimitError):
            await m.log_activity("u1", "t1", [])

        assert m.audit_service.log.await_args.args[3] == "throttled"

    async def test_log_activity_success_writes_no_per_call_audit(self):
        """The batching decision is preserved — this is what the fix must not break."""
        m = _facade()

        await m.log_activity("u1", "t1", [])

        m.audit_service.log.assert_not_awaited()
        m.episodic_service.log_activity.assert_called_once()


# ── Caller input is not a query language ─────────────────────────────────────


class TestCacheInvalidateEscapesItsPattern:
    """`pattern` is a literal substring, not a regular expression."""

    def _service(self):
        col = MagicMock()
        col.delete_many = AsyncMock(return_value=MagicMock(deleted_count=1))
        return CacheService(col, _config(), AsyncMock()), col

    async def test_a_wildcard_is_matched_literally(self):
        """`.*` deletes the entry containing ".*", not the whole cache.

        This is the finding at its sharpest: the caller asked for one entry and
        got everything, with no error and no way to tell from the return value.
        """
        svc, col = self._service()

        await svc.invalidate("u1", pattern=".*")

        sent = col.delete_many.await_args.args[0]
        assert sent["query"]["$regex"] == re.escape(".*")
        # The escaped form cannot match an arbitrary string.
        assert not re.search(sent["query"]["$regex"], "unrelated cached query")

    async def test_regex_metacharacters_are_escaped(self):
        svc, col = self._service()

        await svc.invalidate("u1", pattern="price? (usd|eur) [a-z]+$")

        sent = col.delete_many.await_args.args[0]
        pattern = sent["query"]["$regex"]
        for metachar in ("?", "(", ")", "[", "]", "|", "$", "+"):
            assert f"\\{metachar}" in pattern or metachar not in "price? (usd|eur) [a-z]+$"

    async def test_a_backtracking_bomb_is_defused(self):
        """A catastrophic pattern is a DoS against the cluster, not this collection.

        `$regex` is evaluated server-side against every cached query for the user,
        so `(a+)+$` against a few thousand documents is a way to spend a primary's
        CPU from an untrusted REST body.
        """
        svc, col = self._service()

        await svc.invalidate("u1", pattern="(a+)+$")

        pattern = col.delete_many.await_args.args[0]["query"]["$regex"]
        # No unescaped quantifier or group survives, so there is nothing to
        # backtrack over.
        assert "+" not in pattern.replace("\\+", "")
        assert "(" not in pattern.replace("\\(", "")

    async def test_an_ordinary_substring_still_works(self):
        """Escaping must not break the legitimate use — this is the regression test."""
        svc, col = self._service()

        await svc.invalidate("u1", pattern="shellfish")

        pattern = col.delete_many.await_args.args[0]["query"]["$regex"]
        assert re.search(pattern, "does the user avoid shellfish")

    async def test_the_user_filter_is_still_present(self):
        svc, col = self._service()

        await svc.invalidate("u1", pattern="anything")

        assert col.delete_many.await_args.args[0]["user_id"] == "u1"

    async def test_invalidate_all_remains_the_explicit_way_to_clear(self):
        """Callers who want "everything" have an honest way to ask.

        This is why escaping is the right fix rather than adding a "safe regex"
        mode: the capability was never missing, only reachable by accident.
        """
        svc, col = self._service()

        await svc.invalidate("u1", invalidate_all=True)

        sent = col.delete_many.await_args.args[0]
        assert sent == {"user_id": "u1"}
        assert "$regex" not in str(sent)

    async def test_no_pattern_and_no_flag_deletes_nothing(self):
        svc, col = self._service()

        assert await svc.invalidate("u1") == 0
        col.delete_many.assert_not_awaited()


# ── The merge cluster ────────────────────────────────────────────────────────


def _merge_worker(target_doc):
    col = MagicMock()
    col.update_one = AsyncMock()
    col.find_one = AsyncMock(return_value=target_doc)

    providers = MagicMock()
    providers.llm.complete = AsyncMock(return_value="merged content")
    providers.embedding.generate_embedding = AsyncMock(return_value=[0.9] * 8)

    worker = EnrichmentWorker(col, _config(enrichment_concurrency=1), providers,
                              MagicMock())
    worker.prompt_library = None
    return worker, col, providers


class TestMergeIsUserScoped:
    """The stored `merge_target_id` is not proof of ownership."""

    async def test_the_target_fetch_filters_on_user_id(self):
        """`{"_id": id}` alone read a stored id as a capability.

        Defence in depth today — `evolve_memory` writes that id from a
        user-filtered search — but it is one corrupted field away from reading
        another tenant's memory into this one and soft-deleting the victim's
        record. The correct filter costs nothing.
        """
        worker, col, _ = _merge_worker(
            {"_id": "target", "user_id": "u1", "content": "existing",
             "importance": 0.6}
        )

        await worker._process_merge(
            {"_id": "new", "user_id": "u1", "content": "incoming",
             "merge_target_id": "target"}
        )

        sent = col.find_one.await_args.args[0]
        assert sent["user_id"] == "u1"
        assert sent["_id"] == "target"

    async def test_the_target_fetch_excludes_deleted_documents(self):
        """Merging a deleted target resurrects content the user asked to remove."""
        worker, col, _ = _merge_worker(
            {"_id": "target", "user_id": "u1", "content": "existing"}
        )

        await worker._process_merge(
            {"_id": "new", "user_id": "u1", "content": "incoming",
             "merge_target_id": "target"}
        )

        assert col.find_one.await_args.args[0]["deleted_at"] is None

    async def test_a_missing_target_completes_without_merging(self):
        """The filter now excludes more, so this path is reached more often."""
        worker, col, providers = _merge_worker(None)

        await worker._process_merge(
            {"_id": "new", "user_id": "u1", "content": "incoming",
             "merge_target_id": "gone"}
        )

        providers.llm.complete.assert_not_awaited()
        update = col.update_one.await_args.args[1]["$set"]
        assert update["enrichment_status"] == "complete"

    async def test_the_soft_delete_is_also_user_scoped(self):
        worker, col, _ = _merge_worker(
            {"_id": "target", "user_id": "u1", "content": "existing"}
        )

        await worker._process_merge(
            {"_id": "new", "user_id": "u1", "content": "incoming",
             "merge_target_id": "target"}
        )

        delete_filter = col.update_one.await_args_list[-1].args[0]
        assert delete_filter["_id"] == "target"
        assert delete_filter["user_id"] == "u1"


class TestMergeReEmbeds:
    """Rewriting `content` without re-embedding makes the document unfindable."""

    async def test_the_merged_content_is_re_embedded(self):
        worker, _, providers = _merge_worker(
            {"_id": "target", "user_id": "u1", "content": "existing",
             "importance": 0.6}
        )

        await worker._process_merge(
            {"_id": "new", "user_id": "u1", "content": "incoming",
             "merge_target_id": "target", "embedding": [0.1] * 8}
        )

        providers.embedding.generate_embedding.assert_awaited_once_with(
            "merged content"
        )

    async def test_the_new_embedding_is_written_with_the_new_content(self):
        """The pair must be updated together or not at all.

        Leaving the old vector produced a document that *reads* as the merged
        memory and *searches* as its pre-merge half — so the information the merge
        existed to preserve became unretrievable, silently. It looks correct in
        Compass and simply never comes back for the queries it should answer.
        """
        worker, col, _ = _merge_worker(
            {"_id": "target", "user_id": "u1", "content": "existing",
             "importance": 0.6}
        )

        await worker._process_merge(
            {"_id": "new", "user_id": "u1", "content": "incoming",
             "merge_target_id": "target", "embedding": [0.1] * 8}
        )

        update = col.update_one.await_args_list[0].args[1]["$set"]
        assert update["content"] == "merged content"
        assert update["embedding"] == [0.9] * 8

    async def test_an_embedding_failure_leaves_the_merge_unwritten(self):
        """Better a merge not yet done than a content/embedding pair that disagrees.

        The re-embed is ordered before the write and allowed to raise;
        `_enrich_memory` counts a retry and leaves the status at `merge_pending`,
        so the merge is attempted again rather than committed half-done.
        """
        worker, col, providers = _merge_worker(
            {"_id": "target", "user_id": "u1", "content": "existing"}
        )
        providers.embedding.generate_embedding = AsyncMock(
            side_effect=RuntimeError("voyage down")
        )

        with pytest.raises(RuntimeError):
            await worker._process_merge(
                {"_id": "new", "user_id": "u1", "content": "incoming",
                 "merge_target_id": "target"}
            )

        # Neither the merged write nor the target's soft-delete happened.
        col.update_one.assert_not_awaited()

    async def test_the_retry_keeps_merge_pending_rather_than_completing(self):
        col = MagicMock()
        col.update_one = AsyncMock()
        col.find_one = AsyncMock(
            return_value={"_id": "target", "user_id": "u1", "content": "existing"}
        )
        providers = MagicMock()
        providers.llm.complete = AsyncMock(return_value="merged")
        providers.embedding.generate_embedding = AsyncMock(
            side_effect=RuntimeError("voyage down")
        )
        worker = EnrichmentWorker(
            col, _config(enrichment_concurrency=1, enrichment_max_retries=3),
            providers, MagicMock(),
        )
        worker.prompt_library = None

        await worker._enrich_memory(
            {"_id": "new", "user_id": "u1", "content": "incoming",
             "merge_target_id": "target", "enrichment_status": "merge_pending",
             "enrichment_retries": 0}
        )

        update = col.update_one.await_args.args[1]["$set"]
        assert update["enrichment_status"] == "merge_pending"
        assert update["enrichment_retries"] == 1

    async def test_the_write_precedes_the_soft_delete(self):
        """Order matters, since the two writes are not transactional.

        If the second write is lost the target stays live beside a merged copy —
        duplicate content, which the next evolution pass detects and merges again.
        The reverse order loses the merge and keeps the deletion, which is not
        recoverable.
        """
        worker, col, _ = _merge_worker(
            {"_id": "target", "user_id": "u1", "content": "existing"}
        )

        await worker._process_merge(
            {"_id": "new", "user_id": "u1", "content": "incoming",
             "merge_target_id": "target"}
        )

        first, second = col.update_one.await_args_list[:2]
        assert first.args[0]["_id"] == "new"
        assert second.args[0]["_id"] == "target"


# ── Evolution excludes the memory being evolved ──────────────────────────────


class TestEvolutionExcludesItself:
    """`evolve_memory` must not consider the document it was called for."""

    def _service(self, hits):
        col = MagicMock()
        col.aggregate = AsyncMock(return_value=_cursor(hits))
        col.update_one = AsyncMock()
        col.insert_one = AsyncMock()
        return MemoryService(col, _config(), MagicMock()), col

    async def test_the_pipeline_excludes_the_given_id(self):
        own_id = ObjectId()
        svc, col = self._service([])

        await svc.evolve_memory("u1", "content", [0.1] * 8, exclude_id=own_id)

        pipeline = col.aggregate.await_args.args[0]
        matches = [s for s in pipeline if "$match" in s]
        assert matches, "no exclusion stage in the pipeline"
        assert matches[0]["$match"] == {"_id": {"$ne": own_id}}

    async def test_the_exclusion_is_a_match_stage_not_a_vector_filter(self):
        """`_id` is not a declared filter field, so a vectorSearch filter is ignored.

        Putting it there would look like a fix, pass a naive assertion on the
        pipeline, and change nothing at all — Atlas silently drops filter clauses
        on undeclared paths.
        """
        own_id = ObjectId()
        svc, col = self._service([])

        await svc.evolve_memory("u1", "content", [0.1] * 8, exclude_id=own_id)

        vs = col.aggregate.await_args.args[0][0]["$vectorSearch"]
        assert "_id" not in vs["filter"]

    async def test_it_asks_for_one_extra_candidate_when_excluding(self):
        """Excluding the self-match must not cost a real candidate."""
        svc, col = self._service([])

        await svc.evolve_memory("u1", "c", [0.1] * 8, exclude_id=ObjectId())
        with_exclusion = col.aggregate.await_args.args[0][0]["$vectorSearch"]["limit"]

        await svc.evolve_memory("u1", "c", [0.1] * 8)
        without = col.aggregate.await_args.args[0][0]["$vectorSearch"]["limit"]

        assert with_exclusion == without + 1

    async def test_a_real_duplicate_is_still_reached(self):
        """The behavioural consequence: the second-ranked hit gets its decision.

        Before the fix the self-match at ~1.0 consumed the decision every time, so
        genuine near-duplicates were never merged and the memory reinforced itself
        instead. The exclusion is what lets this document be seen at all.
        """
        duplicate_id = ObjectId()
        svc, col = self._service(
            [{"_id": duplicate_id, "score": 0.995, "importance": 0.5}]
        )
        svc.config.reinforce_threshold = 0.98

        result = await svc.evolve_memory(
            "u1", "content", [0.1] * 8, exclude_id=ObjectId()
        )

        assert result == "reinforced"
        # The *first* write, not the last: reinforcement now also retires the
        # document being evolved, so there are two. Asserting on `await_args`
        # (the most recent call) would check the retirement and say nothing about
        # whether the duplicate was reached at all.
        reinforcing = col.update_one.await_args_list[0]
        assert reinforcing.args[0]["_id"] == duplicate_id

    async def test_without_the_exclusion_nothing_changes_for_other_callers(self):
        """`exclude_id` is optional, so the public signature stays compatible."""
        svc, col = self._service([])

        result = await svc.evolve_memory("u1", "content", [0.1] * 8)

        assert result == "created"
        assert not [s for s in col.aggregate.await_args.args[0] if "$match" in s]
