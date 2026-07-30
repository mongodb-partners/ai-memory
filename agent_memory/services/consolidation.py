"""Background consolidation worker for memory lifecycle management.

Handles STM compression, low-importance forgetting, and STM→LTM promotion.
Modeled after EnrichmentWorker — runs as an asyncio task within the server.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from agent_memory.core.config import MCPConfig
from agent_memory.providers.base import MIN_SUMMARIZABLE_CHARS, is_usable_summary

logger = logging.getLogger(__name__)


class ConsolidationWorker:
    """Periodic background task that runs memory consolidation operations.

    Operations:
    1. Compress old STM — summarize & archive STM older than config threshold
    2. Forget low-importance — soft-delete memories below forgetting threshold
    3. Promote to LTM — promote STM meeting importance/access/age criteria
    """

    def __init__(self, memories_collection, config: MCPConfig, providers) -> None:
        self.memories = memories_collection
        self.config = config
        self.providers = providers
        self._running = False

    async def run(self) -> None:
        """Main loop — runs consolidation at configured interval."""
        self._running = True
        while self._running:
            try:
                await self.consolidate()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Consolidation worker error")
            await asyncio.sleep(self.config.consolidation_interval_hours * 3600)

    def stop(self) -> None:
        self._running = False

    async def consolidate(self) -> dict:
        """Run all consolidation operations and return stats."""
        compressed = await self._compress_stm()
        forgotten = await self._forget_low_importance()
        promoted = await self._promote_to_ltm()
        stats = {
            "compressed": compressed,
            "forgotten": forgotten,
            "promoted": promoted,
        }
        logger.info("Consolidation complete: %s", stats)
        return stats

    async def _compress_stm(self) -> int:
        """Summarize and archive STM older than stm_compression_age_hours."""
        cutoff = datetime.now(UTC) - timedelta(
            hours=self.config.stm_compression_age_hours
        )
        cursor = self.memories.find(
            {
                "tier": "stm",
                "deleted_at": None,
                "created_at": {"$lt": cutoff},
                "summary": None,
            },
            limit=self.config.enrichment_batch_size,
        )
        old_stm = await cursor.to_list(None)

        if not old_stm:
            return 0

        count = 0
        for memory in old_stm:
            content = memory.get("content") or ""
            # A single conversational turn is already as short as its summary
            # would be. Asking anyway wastes a model call and invites a refusal.
            if len(content) < MIN_SUMMARIZABLE_CHARS:
                continue
            try:
                summary = await self.providers.llm.generate_summary(content)
                if not is_usable_summary(summary, content):
                    # Leave `summary` unset so every reader falls back to
                    # `content`. Storing the reply would put "I don't see the
                    # original text that needs to be summarized" wherever the
                    # summary is preferred over the content — which is most
                    # places, including the sample UI's memory panel.
                    logger.debug(
                        "Discarded non-summary reply for STM %s", memory["_id"]
                    )
                    continue
                await self.memories.update_one(
                    {"_id": memory["_id"]},
                    {
                        "$set": {
                            "summary": summary,
                            "updated_at": datetime.now(UTC),
                        }
                    },
                )
                count += 1
            except Exception:
                logger.exception("Failed to compress STM %s", memory["_id"])
        return count

    async def _forget_low_importance(self) -> int:
        """Soft-delete memories with importance below forgetting_score_threshold."""
        now = datetime.now(UTC)
        result = await self.memories.update_many(
            {
                "deleted_at": None,
                "importance": {"$lt": self.config.forgetting_score_threshold},
                "tier": "ltm",
                "enrichment_status": "complete",
            },
            {
                "$set": {
                    "deleted_at": now,
                    "is_deleted": True,
                    "updated_at": now,
                }
            },
        )
        return result.modified_count

    async def _promote_to_ltm(self) -> int:
        """Promote STM memories meeting importance/access/age thresholds to LTM."""
        age_cutoff = datetime.now(UTC) - timedelta(
            minutes=self.config.promotion_age_minutes
        )
        cursor = self.memories.find(
            {
                "tier": "stm",
                "deleted_at": None,
                "importance": {"$gte": self.config.promotion_importance_threshold},
                "access_count": {"$gte": self.config.promotion_access_threshold},
                "created_at": {"$lt": age_cutoff},
            },
            limit=self.config.enrichment_batch_size,
        )
        candidates = await cursor.to_list(None)

        if not candidates:
            return 0

        count = 0
        now = datetime.now(UTC)
        for memory in candidates:
            try:
                await self.memories.update_one(
                    {"_id": memory["_id"]},
                    {
                        "$set": {
                            "tier": "ltm",
                            "retention_tier": "standard",
                            "enrichment_status": "pending",
                            "updated_at": now,
                        }
                    },
                )
                count += 1
            except Exception:
                logger.exception("Failed to promote STM %s", memory["_id"])
        return count
