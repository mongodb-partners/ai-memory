"""Background enrichment worker for LTM memory quality improvement."""

import asyncio
import logging
from datetime import UTC, datetime

from agent_memory.core.config import MCPConfig
from agent_memory.providers.base import MIN_SUMMARIZABLE_CHARS, is_usable_summary
from agent_memory.services.importance import LLMScorer

logger = logging.getLogger(__name__)


class EnrichmentWorker:
    """Background task polling for pending enrichments and processing via LLM.

    Runs as an asyncio task within the FastMCP server process.
    Uses a semaphore to limit concurrent LLM calls.
    """

    def __init__(
        self,
        memories_collection,
        config: MCPConfig,
        providers,
        memory_service,
        prompt_library=None,
        scorer=None,
    ) -> None:
        self.memories = memories_collection
        self.config = config
        self.providers = providers
        self.memory_service = memory_service
        self.prompt_library = prompt_library
        # Default to the LLM path so every existing construction — including the
        # twenty in the test suite that pass four positional arguments — keeps
        # today's behaviour exactly. Those call sites are the regression suite for
        # "an upgrade changes nothing"; rewriting them as part of this change would
        # mean the property is only asserted by tests edited in the same commit.
        self.scorer = scorer or LLMScorer(
            providers.llm, prompt_getter=self._get_prompt
        )
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
                        "updated_at": datetime.now(UTC),
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

        # One call, whichever scorer is configured. The prompt lookup that used to
        # branch here now lives in `LLMScorer` — the branch could not stay, because
        # the local scorer has no prompt.
        #
        # `.get("embedding")` rather than `["embedding"]`: the scorer must not be
        # the thing that raises on a malformed document. The `evolve_memory` call
        # below would raise anyway, but a KeyError originating here would look like
        # a scorer fault.
        importance = await self.scorer.score(
            memory["content"],
            memory.get("embedding"),
            tags=memory.get("tags"),
            message_type=memory.get("message_type"),
        )

        summary = await self._summarize(memory["content"])

        # Memory evolution check. `exclude_id` is not optional in practice: this
        # runs after the document is stored, so without it the search's top hit is
        # this very memory at similarity ~1.0, and the reinforce branch fires
        # against its own `_id` — inflating its own importance and never reaching
        # the real duplicates ranked below it.
        await self.memory_service.evolve_memory(
            memory["user_id"],
            memory["content"],
            memory["embedding"],
            exclude_id=memory_id,
        )

        update = {
            "enrichment_status": "complete",
            "importance": importance,
            "updated_at": datetime.now(UTC),
        }
        # Only set `summary` when there is one worth setting. Absent is the safe
        # state: readers fall back to `content`, which is the memory itself.
        if summary is not None:
            update["summary"] = summary

        await self.memories.update_one({"_id": memory_id}, {"$set": update})

    async def _summarize(self, content: str) -> str | None:
        """Summarize `content`, or return None when no usable summary exists.

        Short memories are skipped rather than summarized: a one-line
        conversational turn is already shorter than any summary of it, and asking
        the model to compress it produces a refusal — "I don't see the original
        text that needs to be summarized" — which reads as a successful call
        returning a string. See `is_usable_summary` for why storing that is worse
        than storing nothing.
        """
        if len(content) < MIN_SUMMARIZABLE_CHARS:
            return None

        summary_prompt = await self._get_prompt("summary_generation")
        if summary_prompt:
            summary = await self.providers.llm.generate_summary(
                content, prompt=summary_prompt,
            )
        else:
            summary = await self.providers.llm.generate_summary(content)

        if not is_usable_summary(summary, content):
            logger.debug("Discarded non-summary reply, keeping content as-is")
            return None
        return summary

    async def _process_merge(self, memory: dict) -> None:
        """Merge memory with its target via LLM, then soft-delete the target."""
        memory_id = memory["_id"]
        merge_target_id = memory.get("merge_target_id")

        # The target fetch is scoped to the *same user*, and to a live document.
        # `{"_id": merge_target_id}` alone trusted a stored id as proof of
        # ownership: a `merge_target_id` pointing at another tenant's memory would
        # have that memory's content read into this user's document and the victim's
        # record soft-deleted. The id is written by `evolve_memory` from a
        # user-filtered search today, so this is defence in depth — but it is one
        # field's worth of corruption away from a cross-tenant read, and the
        # correct filter costs nothing.
        #
        # `deleted_at: None` is part of the same fetch rather than a check after
        # it: merging an already-deleted target resurrects its content into a live
        # document, undoing a deletion the user asked for.
        target = await self.memories.find_one({
            "_id": merge_target_id,
            "user_id": memory.get("user_id"),
            "deleted_at": None,
        })
        if target is None:
            # Target was already deleted — just mark as complete
            await self.memories.update_one(
                {"_id": memory_id},
                {
                    "$set": {
                        "enrichment_status": "complete",
                        "updated_at": datetime.now(UTC),
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
        # `complete` builds the message in the provider's own shape. Passing an
        # OpenAI-shaped `[{"role": "user", "content": str}]` here — as this line
        # used to — makes every merge fail on Bedrock, the default provider, and
        # the failure is swallowed by the caller's `except` into a log line.
        merged_content = await self.providers.llm.complete(merge_text)

        # Re-embed, because the content just changed. Writing `content` and leaving
        # the old `embedding` in place produced a document that *reads* as the merged
        # memory and *searches* as its pre-merge half: the vector still encoded only
        # what this document said before the target's content was folded in, so the
        # information the merge existed to preserve became unretrievable. Worse, it
        # fails silently — the document looks right in Compass and simply never comes
        # back for the queries it should answer.
        #
        # Ordered before the write and allowed to raise: `_enrich_memory` catches,
        # counts a retry, and leaves the status as `merge_pending`, so the merge is
        # attempted again rather than committed half-done. A content/embedding pair
        # that disagrees is worse than a merge that has not happened yet.
        merged_embedding = await self.providers.embedding.generate_embedding(
            merged_content
        )

        now = datetime.now(UTC)

        # Update the new memory with merged content
        await self.memories.update_one(
            {"_id": memory_id},
            {
                "$set": {
                    "enrichment_status": "complete",
                    "content": merged_content,
                    "embedding": merged_embedding,
                    "importance": max(
                        target.get("importance", 0.5),
                        memory.get("importance", 0.5),
                    ),
                    "updated_at": now,
                }
            },
        )

        # Soft-delete the merge target — scoped to the same user and to a target that
        # is still live, the same filter the fetch used. Between the fetch and here
        # the user may have deleted it themselves; re-deleting is harmless, but
        # matching on `deleted_at: None` keeps the original deletion timestamp rather
        # than overwriting it with this one.
        #
        # These two writes are deliberately *not* in a transaction. Atlas
        # multi-document transactions require a session threaded through from the
        # client, which this worker does not hold, and the failure mode here is
        # benign in a way that a rollback would not improve: if the second write is
        # lost, the target remains live alongside a merged copy — duplicate content,
        # which the next evolution pass is built to detect and merge again. The
        # reverse order would be the dangerous one (target deleted, merge lost), and
        # that is why the merged write goes first.
        await self.memories.update_one(
            {
                "_id": merge_target_id,
                "user_id": memory.get("user_id"),
                "deleted_at": None,
            },
            {
                "$set": {
                    "deleted_at": now,
                    "is_deleted": True,
                    "updated_at": now,
                }
            },
        )
