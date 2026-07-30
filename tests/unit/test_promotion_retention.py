"""Promotion must move the expiry, not just the label.

STM→LTM promotion changed three fields — `tier`, `retention_tier`,
`enrichment_status` — and left `expires_at` holding the short-term value the
document was born with, ~24 hours out. The TTL index on that field does not read
`tier`. So a memory that had satisfied every promotion criterion, was labelled
long-term, and was serving recall got deleted the next day.

Nothing about the failure was visible from the code path that caused it. There is
no error, no log line, and the document looks correct in Compass right up until it
is gone — the only symptom is long-term memory that quietly does not accumulate.

Measured against the demo cluster before the fix: of 12 LTM documents, the 5 that
consolidation had promoted in place were all 23.6 hours from deletion, while the 7
written directly as LTM candidates by `store_stm` carried the full 90 days. Same
tier, same `retention_tier: "standard"`, two different lifetimes.

REQ-E-142 (promotion re-stamps expiry to the destination tier).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory.core.config import MCPConfig
from agent_memory.services.consolidation import ConsolidationWorker
from agent_memory.services.memory import PROMOTED_RETENTION_TIER, retention_ttl


def _config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


def _cursor(docs: list[dict]):
    cursor = MagicMock()
    cursor.to_list = AsyncMock(return_value=docs)
    return cursor


def _candidate(**overrides) -> dict:
    """An STM document that satisfies every promotion predicate."""
    born = datetime.now(timezone.utc) - timedelta(days=9)
    doc = {
        "_id": "stm-1",
        "user_id": "u1",
        "tier": "stm",
        "retention_tier": "ephemeral",
        "content": "I'm allergic to shellfish",
        "importance": 0.9,
        "access_count": 5,
        "created_at": born,
        # The short-term expiry, which is the field under test.
        "expires_at": born + timedelta(hours=24),
        "deleted_at": None,
    }
    doc.update(overrides)
    return doc


def _worker(config: MCPConfig, candidates: list[dict]):
    """A worker whose promotion query returns `candidates`.

    `find` answers every call with the same candidate set rather than a positional
    `side_effect` list: these tests call `_promote_to_ltm()` directly, so any
    ordering assumption about which `find` belongs to which consolidation step is
    a property of the test rather than of the code under test.
    """
    memories = MagicMock()
    memories.find = MagicMock(return_value=_cursor(candidates))
    memories.update_many = AsyncMock(return_value=MagicMock(modified_count=0))
    memories.update_one = AsyncMock(return_value=MagicMock(modified_count=1))
    providers = MagicMock()
    return ConsolidationWorker(memories, config, providers), memories


def _promotion_set(memories) -> dict:
    """The `$set` document from the promotion update."""
    assert memories.update_one.await_count == 1, "expected exactly one promotion write"
    _filter, update = memories.update_one.await_args.args
    return update["$set"]


class TestPromotionRestampsTheExpiry:
    @pytest.mark.asyncio
    async def test_expires_at_is_rewritten(self) -> None:
        """The single assertion that would have caught the original bug."""
        config = _config()
        candidate = _candidate()
        worker, memories = _worker(config, [candidate])

        await worker._promote_to_ltm()

        assigned = _promotion_set(memories)
        assert "expires_at" in assigned, (
            "promotion left expires_at untouched; the document keeps its "
            "short-term TTL and the index deletes it on the STM schedule"
        )
        assert assigned["expires_at"] != candidate["expires_at"]

    @pytest.mark.asyncio
    async def test_the_new_expiry_matches_the_destination_tier(self) -> None:
        config = _config()
        worker, memories = _worker(config, [_candidate()])

        await worker._promote_to_ltm()

        assigned = _promotion_set(memories)
        expected = retention_ttl(config, PROMOTED_RETENTION_TIER)
        actual = assigned["expires_at"] - assigned["updated_at"]
        assert actual == expected, (
            f"expiry window is {actual}, but retention_tier "
            f"{PROMOTED_RETENTION_TIER!r} means {expected}"
        )

    @pytest.mark.asyncio
    async def test_expiry_and_retention_tier_agree(self) -> None:
        """The two fields are one fact; a document where they disagree is the bug."""
        config = _config()
        worker, memories = _worker(config, [_candidate()])

        await worker._promote_to_ltm()

        assigned = _promotion_set(memories)
        window = assigned["expires_at"] - assigned["updated_at"]
        assert window == retention_ttl(config, assigned["retention_tier"])

    @pytest.mark.asyncio
    async def test_the_promoted_memory_outlives_the_stm_window(self) -> None:
        """States the consequence directly, in the terms the failure appeared in.

        The demo cluster's promoted documents sat 23.6 hours from deletion. This
        fails if a promoted memory would not survive past the STM TTL.
        """
        config = _config(stm_ttl_hours=24, ltm_retention_standard_days=90)
        worker, memories = _worker(config, [_candidate()])

        await worker._promote_to_ltm()

        assigned = _promotion_set(memories)
        window = assigned["expires_at"] - assigned["updated_at"]
        assert window > timedelta(hours=config.stm_ttl_hours)
        assert window == timedelta(days=90)

    @pytest.mark.asyncio
    async def test_a_longer_configured_retention_is_honoured(self) -> None:
        """Guards against the fix being hardcoded to 90 days."""
        config = _config(ltm_retention_standard_days=365)
        worker, memories = _worker(config, [_candidate()])

        await worker._promote_to_ltm()

        assigned = _promotion_set(memories)
        assert assigned["expires_at"] - assigned["updated_at"] == timedelta(days=365)

    @pytest.mark.asyncio
    async def test_the_tier_fields_still_change(self) -> None:
        """The original behaviour is preserved — this is an addition."""
        worker, memories = _worker(_config(), [_candidate()])

        await worker._promote_to_ltm()

        assigned = _promotion_set(memories)
        assert assigned["tier"] == "ltm"
        assert assigned["retention_tier"] == PROMOTED_RETENTION_TIER
        assert assigned["enrichment_status"] == "pending"


class TestRetentionTtlIsOneTable:
    """The mapping is shared, so promotion and `store_stm` cannot diverge.

    Consolidation previously hardcoded the string `"standard"` with no access to
    the TTL table at all, which is why it could set a tier it had no way to
    translate into a date.
    """

    def test_every_named_tier_resolves(self) -> None:
        config = _config()
        for tier in ("critical", "reference", "standard", "temporary", "ephemeral"):
            assert retention_ttl(config, tier) > timedelta(0)

    def test_ephemeral_is_the_stm_window(self) -> None:
        config = _config(stm_ttl_hours=24)
        assert retention_ttl(config, "ephemeral") == timedelta(hours=24)

    def test_unknown_tiers_fall_back_to_standard(self) -> None:
        config = _config()
        assert retention_ttl(config, "nonsense") == retention_ttl(config, "standard")

    def test_the_promoted_tier_is_not_ephemeral(self) -> None:
        """A promoted memory landing in the STM window is the bug, restated."""
        config = _config()
        assert retention_ttl(config, PROMOTED_RETENTION_TIER) > retention_ttl(
            config, "ephemeral"
        )

    def test_memory_service_uses_the_shared_table(self) -> None:
        """`MemoryService._retention_ttl` must not be a second implementation."""
        from agent_memory.services.memory import MemoryService

        config = _config()
        service = MemoryService(MagicMock(), config, MagicMock())
        for tier in ("standard", "ephemeral", "critical"):
            assert service._retention_ttl(tier) == retention_ttl(config, tier)
