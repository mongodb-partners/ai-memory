"""Background enrichment worker for LTM memory quality improvement."""

import asyncio
import logging
from datetime import datetime, timezone

from agent_memory.core.config import MCPConfig

logger = logging.getLogger(__name__)


class EnrichmentWorker:
    """Background task polling for pending enrichments and processing via LLM.

    Runs as an asyncio task within the FastMCP server process.
    Uses a semaphore to limit concurrent LLM calls.
    """

    def __init__(self, memories_collection, config: MCPConfig, providers, memory_service, prompt_library=None) -> None:
        self.memories = memories_collection
        self.config = config
        self.providers = providers
        self.memory_service = memory_service
        self.prompt_library = prompt_library
        self._semaphore = asyncio.Semaphore(config.enrichment_concurrency)
        self._running = False

    async def run(self) -> None:
        """Main loop — poll and process pending enrichments."""
        self._running = True
        while self._running:
            try:
                await self.process_batch()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Enrichment worker error")
            await asyncio.sleep(self.config.enrichment_interval_seconds)

    def stop(self) -> None:
        self._running = False

    async def process_batch(self) -> int:
        """Find and process one batch of pending/merge_pending memories. Returns count processed."""
        cursor = self.memories.find(
            {"enrichment_status": {"$in": ["pending", "merge_pending"]}},
            sort=[("created_at", 1)],
            limit=self.config.enrichment_batch_size,
        )
        pending = await cursor.to_list(None)

        if not pending:
            return 0

        tasks = [self._enrich_with_semaphore(memory) for memory in pending]
        await asyncio.gather(*tasks, return_exceptions=True)

        return len(pending)

    async def _enrich_with_semaphore(self, memory: dict) -> None:
        async with self._semaphore:
            await self._enrich_memory(memory)

    async def _enrich_memory(self, memory: dict) -> None:
        """Enrich a single memory: importance, summary, evolution check.

        For merge_pending memories, merges content with the target via LLM
        and soft-deletes the target.
        """
        memory_id = memory["_id"]
        retries = memory.get("enrichment_retries", 0)

        try:
            if memory.get("enrichment_status") == "merge_pending":
                await self._process_merge(memory)
            else:
                await self._process_standard_enrichment(memory)

        except Exception:
            logger.exception("Failed to enrich memory %s", memory_id)
            new_retries = retries + 1
            original_status = memory.get("enrichment_status", "pending")
            if new_retries >= self.config.enrichment_max_retries:
                status = "failed"
            else:
                status = original_status  # Keep merge_pending or pending
            await self.memories.update_one(
                {"_id": memory_id},
                {
                    "$set": {
                        "enrichment_status": status,
                        "enrichment_retries": new_retries,
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
            )

    async def _get_prompt(self, name: str) -> str | None:
        """Get a prompt template from the library, or None if unavailable."""
        if self.prompt_library is not None:
            try:
                return await self.prompt_library.get_prompt(name)
            except Exception:
                logger.debug("Failed to get prompt '%s' from library, using default", name)
        return None

    async def _process_standard_enrichment(self, memory: dict) -> None:
        """Standard enrichment: importance, summary, evolution check."""
        memory_id = memory["_id"]

        importance_prompt = await self._get_prompt("importance_assessment")
        if importance_prompt:
            importance = await self.providers.llm.assess_importance(
                memory["content"], prompt=importance_prompt,
            )
        else:
            importance = await self.providers.llm.assess_importance(memory["content"])

        summary_prompt = await self._get_prompt("summary_generation")
        if summary_prompt:
            summary = await self.providers.llm.generate_summary(
                memory["content"], prompt=summary_prompt,
            )
        else:
            summary = await self.providers.llm.generate_summary(memory["content"])

        # Memory evolution check
        await self.memory_service.evolve_memory(
            memory["user_id"],
            memory["content"],
            memory["embedding"],
        )

        await self.memories.update_one(
            {"_id": memory_id},
            {
                "$set": {
                    "enrichment_status": "complete",
                    "importance": importance,
                    "summary": summary,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

    async def _process_merge(self, memory: dict) -> None:
        """Merge memory with its target via LLM, then soft-delete the target."""
        memory_id = memory["_id"]
        merge_target_id = memory.get("merge_target_id")

        # Fetch the merge target
        target = await self.memories.find_one({"_id": merge_target_id})
        if target is None:
            # Target was already deleted — just mark as complete
            await self.memories.update_one(
                {"_id": memory_id},
                {
                    "$set": {
                        "enrichment_status": "complete",
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
            )
            return

        # Ask LLM to merge the two pieces of content
        merge_prompt_template = await self._get_prompt("merge_prompt")
        if merge_prompt_template:
            merge_text = merge_prompt_template.format(
                memory_1=target["content"], memory_2=memory["content"],
            )
        else:
            merge_text = (
                "Merge these two related memory entries into a single, "
                "coherent memory. Preserve all important details.\n\n"
                f"Memory 1: {target['content']}\n\n"
                f"Memory 2: {memory['content']}"
            )
        merged_content = await self.providers.llm.chat(
            messages=[{"role": "user", "content": merge_text}],
        )

        now = datetime.now(timezone.utc)

        # Update the new memory with merged content
        await self.memories.update_one(
            {"_id": memory_id},
            {
                "$set": {
                    "enrichment_status": "complete",
                    "content": merged_content,
                    "importance": max(
                        target.get("importance", 0.5),
                        memory.get("importance", 0.5),
                    ),
                    "updated_at": now,
                }
            },
        )

        # Soft-delete the merge target
        await self.memories.update_one(
            {"_id": merge_target_id},
            {
                "$set": {
                    "deleted_at": now,
                    "is_deleted": True,
                    "updated_at": now,
                }
            },
        )
