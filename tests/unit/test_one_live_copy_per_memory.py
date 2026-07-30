"""Evolution must leave exactly one live document per piece of content.

``evolve_memory`` decides that a new memory duplicates an existing one, and then
did nothing about the new memory. Both branches got this wrong:

* **reinforced** — the older memory's importance went up and the newer one was
  left live and searchable. The user ended up with two near-identical memories,
  one of which had just been declared redundant. Search returned both, they
  competed for the same ``numCandidates`` budget, and the next enrichment pass
  found each as the other's duplicate.
* **merge_queued** — a *third* document was inserted carrying the same content
  with ``enrichment_status: merge_pending``, while the original stayed live beside
  it. One ``add()`` produced three documents for one memory; the merge worker then
  folded the target into the inserted copy and left the original untouched.

Neither raised, and both left documents that look correct in isolation. The only
symptom was a memory store that grew redundant copies of whatever the user
repeated — precisely the thing deduplication exists to prevent.

Asserted against a fake collection that holds documents, so the assertions are
about the resulting *state* — how many live copies of this content exist — rather
than about which calls were made. A mock would let a fix that issues the right
update against the wrong ``_id`` pass.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from bson import ObjectId

from agent_memory.config import MemoryConfig
from agent_memory.services.memory import MemoryService

CONTENT = "The user prefers dark mode in every editor they use."


def _config(**overrides) -> MemoryConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MemoryConfig(**defaults, _env_file=None)


class _Memories:
    """A collection that keeps documents, so "how many are live" is answerable.

    ``aggregate`` is scripted rather than implemented — vector search cannot be
    simulated and is not what these tests are about. Everything else is real
    enough to answer the only question that matters here: after evolution, which
    documents does a search see?
    """

    def __init__(self, docs: list[dict], hits: list[dict] | None = None) -> None:
        self.docs = docs
        self.hits = hits or []
        self.inserted: list[dict] = []

    async def aggregate(self, pipeline):
        # Honour the exclusion stage the real pipeline appends, so a test that
        # forgets `exclude_id` cannot accidentally pass.
        hits = self.hits
        for stage in pipeline:
            match = stage.get("$match", {}).get("_id", {})
            if "$ne" in match:
                hits = [h for h in hits if h["_id"] != match["$ne"]]
        cursor = MagicMock()
        cursor.to_list = AsyncMock(return_value=hits)
        return cursor

    def _get(self, query: dict) -> dict | None:
        for doc in self.docs:
            if doc["_id"] != query.get("_id"):
                continue
            if "deleted_at" in query and doc.get("deleted_at") != query["deleted_at"]:
                continue
            return doc
        return None

    async def update_one(self, query: dict, update: dict) -> None:
        doc = self._get(query)
        if doc is None:
            return
        doc.update(update.get("$set", {}))
        for field in update.get("$unset", {}):
            doc.pop(field, None)
        for field, amount in update.get("$inc", {}).items():
            doc[field] = doc.get(field, 0) + amount

    async def insert_one(self, doc: dict) -> None:
        doc.setdefault("_id", ObjectId())
        self.docs.append(doc)
        self.inserted.append(doc)

    def live(self, content: str) -> list[dict]:
        """Every document a search could return for this content."""
        return [
            d for d in self.docs
            if d.get("content") == content and d.get("deleted_at") is None
        ]


def _stored(content: str = CONTENT, **overrides) -> dict:
    doc = {
        "_id": ObjectId(),
        "user_id": "u1",
        "tier": "ltm",
        "content": content,
        "embedding": [0.1] * 8,
        "importance": 0.5,
        "enrichment_status": "pending",
        "enrichment_retries": 0,
        "created_at": datetime.now(UTC),
        "deleted_at": None,
        "is_deleted": False,
    }
    doc.update(overrides)
    return doc


def _service(col, **config_overrides):
    return MemoryService(col, _config(**config_overrides), MagicMock())


class TestReinforcementRetiresTheDuplicate:
    async def test_only_one_copy_stays_live(self):
        existing = _stored(enrichment_status="complete")
        candidate = _stored()
        col = _Memories(
            [existing, candidate],
            hits=[{"_id": existing["_id"], "score": 0.995, "importance": 0.5}],
        )
        svc = _service(col, reinforce_threshold=0.98)

        result = await svc.evolve_memory(
            "u1", CONTENT, [0.1] * 8, exclude_id=candidate["_id"]
        )

        assert result == "reinforced"
        live = col.live(CONTENT)
        assert len(live) == 1, (
            f"{len(live)} live copies of the same content after reinforcement; "
            "the memory declared redundant is still searchable"
        )
        assert live[0]["_id"] == existing["_id"], (
            "the wrong document survived — the reinforced memory is the one to keep"
        )

    async def test_the_kept_memory_is_still_reinforced(self):
        """Retiring the duplicate must not replace the reinforcement."""
        existing = _stored(importance=0.5)
        candidate = _stored()
        col = _Memories(
            [existing, candidate],
            hits=[{"_id": existing["_id"], "score": 0.995, "importance": 0.5}],
        )
        svc = _service(col, reinforce_threshold=0.98)

        await svc.evolve_memory("u1", CONTENT, [0.1] * 8, exclude_id=candidate["_id"])

        assert existing["importance"] > 0.5
        assert existing["access_count"] == 1

    async def test_the_retirement_records_what_absorbed_it(self):
        """Without `duplicate_of` an operator sees memories quietly disappear."""
        existing = _stored()
        candidate = _stored()
        col = _Memories(
            [existing, candidate],
            hits=[{"_id": existing["_id"], "score": 0.995, "importance": 0.5}],
        )
        svc = _service(col, reinforce_threshold=0.98)

        await svc.evolve_memory("u1", CONTENT, [0.1] * 8, exclude_id=candidate["_id"])

        assert candidate["duplicate_of"] == existing["_id"]
        assert candidate["is_deleted"] is True

    async def test_the_retired_memory_is_not_enriched_again(self):
        """Left in `pending` it would be claimed and spend an LLM call on content
        that is no longer searchable."""
        existing = _stored()
        candidate = _stored()
        col = _Memories(
            [existing, candidate],
            hits=[{"_id": existing["_id"], "score": 0.995, "importance": 0.5}],
        )
        svc = _service(col, reinforce_threshold=0.98)

        await svc.evolve_memory("u1", CONTENT, [0.1] * 8, exclude_id=candidate["_id"])

        assert candidate["enrichment_status"] == "complete"

    async def test_a_soft_delete_rather_than_a_removal(self):
        """Every other deletion in this service is soft, and a threshold tuned too
        loosely should be recoverable rather than destructive."""
        existing = _stored()
        candidate = _stored()
        col = _Memories(
            [existing, candidate],
            hits=[{"_id": existing["_id"], "score": 0.995, "importance": 0.5}],
        )
        svc = _service(col, reinforce_threshold=0.98)

        await svc.evolve_memory("u1", CONTENT, [0.1] * 8, exclude_id=candidate["_id"])

        assert candidate in col.docs, "the document was removed rather than retired"
        assert isinstance(candidate["deleted_at"], datetime)

    async def test_an_already_deleted_candidate_keeps_its_timestamp(self):
        """The user may have deleted it themselves between the search and here."""
        earlier = datetime(2026, 1, 1, tzinfo=UTC)
        existing = _stored()
        candidate = _stored(deleted_at=earlier, is_deleted=True)
        col = _Memories(
            [existing, candidate],
            hits=[{"_id": existing["_id"], "score": 0.995, "importance": 0.5}],
        )
        svc = _service(col, reinforce_threshold=0.98)

        await svc.evolve_memory("u1", CONTENT, [0.1] * 8, exclude_id=candidate["_id"])

        assert candidate["deleted_at"] == earlier

    async def test_a_caller_with_no_document_is_unaffected(self):
        """`evolve_memory` is public and `exclude_id` is optional. A caller
        evolving raw content has nothing to retire, and must not have some other
        document retired on its behalf."""
        existing = _stored()
        col = _Memories(
            [existing],
            hits=[{"_id": existing["_id"], "score": 0.995, "importance": 0.5}],
        )
        svc = _service(col, reinforce_threshold=0.98)

        result = await svc.evolve_memory("u1", CONTENT, [0.1] * 8)

        assert result == "reinforced"
        assert existing["deleted_at"] is None


class TestAQueuedMergeUsesTheDocumentItAlreadyHas:
    async def test_no_third_document_is_created(self):
        existing = _stored("older phrasing of the same fact")
        candidate = _stored()
        col = _Memories(
            [existing, candidate],
            hits=[{"_id": existing["_id"], "score": 0.90, "importance": 0.5}],
        )
        svc = _service(col, reinforce_threshold=0.98, merge_threshold=0.85)

        result = await svc.evolve_memory(
            "u1", CONTENT, [0.1] * 8, exclude_id=candidate["_id"]
        )

        assert result == "merge_queued"
        assert col.inserted == [], (
            "a merge candidate was inserted while the document being evolved was "
            "left live — three documents for one memory"
        )
        assert len(col.live(CONTENT)) == 1

    async def test_the_existing_document_carries_the_merge(self):
        existing = _stored("older phrasing")
        candidate = _stored()
        col = _Memories(
            [existing, candidate],
            hits=[{"_id": existing["_id"], "score": 0.90, "importance": 0.5}],
        )
        svc = _service(col, reinforce_threshold=0.98, merge_threshold=0.85)

        await svc.evolve_memory("u1", CONTENT, [0.1] * 8, exclude_id=candidate["_id"])

        assert candidate["enrichment_status"] == "merge_pending"
        assert candidate["merge_target_id"] == existing["_id"]

    async def test_the_merge_starts_with_a_full_retry_budget(self):
        """The merge is new work, not a continuation. Carrying the enrichment's
        retry count over would let a memory that took two attempts to summarize
        reach the merge with one attempt left."""
        existing = _stored("older phrasing")
        candidate = _stored(enrichment_retries=2)
        col = _Memories(
            [existing, candidate],
            hits=[{"_id": existing["_id"], "score": 0.90, "importance": 0.5}],
        )
        svc = _service(col, reinforce_threshold=0.98, merge_threshold=0.85)

        await svc.evolve_memory("u1", CONTENT, [0.1] * 8, exclude_id=candidate["_id"])

        assert "enrichment_retries" not in candidate

    async def test_the_claim_is_released_so_the_merge_can_be_picked_up(self):
        """The enrichment worker holds a claim on this document while calling
        `evolve_memory`. Leaving it in place would make the queued merge wait out
        the lease before any worker could take it."""
        existing = _stored("older phrasing")
        candidate = _stored(enrichment_claimed_at=datetime.now(UTC))
        col = _Memories(
            [existing, candidate],
            hits=[{"_id": existing["_id"], "score": 0.90, "importance": 0.5}],
        )
        svc = _service(col, reinforce_threshold=0.98, merge_threshold=0.85)

        await svc.evolve_memory("u1", CONTENT, [0.1] * 8, exclude_id=candidate["_id"])

        assert "enrichment_claimed_at" not in candidate

    async def test_a_caller_with_no_document_still_gets_one_inserted(self):
        """The original behaviour, kept for the path that has nothing to convert:
        without an insert the merge would have no document at all."""
        existing = _stored("older phrasing")
        col = _Memories(
            [existing],
            hits=[{"_id": existing["_id"], "score": 0.90, "importance": 0.5}],
        )
        svc = _service(col, reinforce_threshold=0.98, merge_threshold=0.85)

        result = await svc.evolve_memory("u1", CONTENT, [0.1] * 8)

        assert result == "merge_queued"
        assert len(col.inserted) == 1
        assert col.inserted[0]["enrichment_status"] == "merge_pending"
        assert col.inserted[0]["merge_target_id"] == existing["_id"]


class TestCreatedLeavesTheDocumentAlone:
    async def test_a_novel_memory_is_untouched(self):
        candidate = _stored()
        col = _Memories([candidate], hits=[])
        svc = _service(col)

        result = await svc.evolve_memory(
            "u1", CONTENT, [0.1] * 8, exclude_id=candidate["_id"]
        )

        assert result == "created"
        assert candidate["deleted_at"] is None
        assert candidate["enrichment_status"] == "pending"

    async def test_a_distant_match_is_not_treated_as_a_duplicate(self):
        existing = _stored("something else entirely")
        candidate = _stored()
        col = _Memories(
            [existing, candidate],
            hits=[{"_id": existing["_id"], "score": 0.40, "importance": 0.5}],
        )
        svc = _service(col, reinforce_threshold=0.98, merge_threshold=0.85)

        result = await svc.evolve_memory(
            "u1", CONTENT, [0.1] * 8, exclude_id=candidate["_id"]
        )

        assert result == "created"
        assert candidate["deleted_at"] is None
        assert col.inserted == []
