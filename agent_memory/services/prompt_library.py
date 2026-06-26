"""Prompt library — versioned prompt templates with DB persistence.

Stores templates in MongoDB ``prompts`` collection with in-memory cache.
Falls back to hardcoded defaults when ``prompt_experiment_enabled=False``.
"""

import logging
import time
from datetime import datetime, timezone

from agent_memory.core.config import MCPConfig

logger = logging.getLogger(__name__)

# Hardcoded defaults used when prompt_experiment_enabled is False
_HARDCODED_PROMPTS = {
    "importance_assessment": (
        "Assess the importance of this memory on a scale of 0.0 to 1.0. "
        "Consider uniqueness, actionability, and emotional significance.\n\n"
        "Memory: {content}\n\nImportance (0.0-1.0):"
    ),
    "summary_generation": (
        "Summarize this memory in a concise sentence that captures the key information.\n\n"
        "Memory: {content}\n\nSummary:"
    ),
    "merge_prompt": (
        "Merge these two related memories into a single coherent memory:\n\n"
        "Memory 1: {memory_1}\n\n"
        "Memory 2: {memory_2}\n\nMerged:"
    ),
}


class PromptLibrary:
    """Versioned prompt template management with DB persistence and cache."""

    def __init__(self, prompts_collection, config: MCPConfig) -> None:
        self.collection = prompts_collection
        self.config = config
        self._cache: dict[str, dict] = {}
        self._cache_time: dict[str, float] = {}

    async def get_prompt(self, name: str, version: int | None = None) -> str:
        """Get a prompt template by name.

        If ``prompt_experiment_enabled`` is False, returns hardcoded default.
        Otherwise, fetches from DB with in-memory caching.
        """
        if not self.config.prompt_experiment_enabled:
            return _HARDCODED_PROMPTS.get(name, "")

        cache_key = f"{name}:{version or 'latest'}"
        now = time.time()
        cached_at = self._cache_time.get(cache_key, 0)

        if cache_key in self._cache and (now - cached_at) < self.config.prompt_cache_ttl_seconds:
            return self._cache[cache_key]

        query = {"name": name}
        if version is not None:
            query["version"] = version

        doc = await self.collection.find_one(
            query, sort=[("version", -1)]
        )

        if doc:
            template = doc["template"]
            self._cache[cache_key] = template
            self._cache_time[cache_key] = now
            return template

        # Fallback to hardcoded
        return _HARDCODED_PROMPTS.get(name, "")

    async def save_prompt(self, name: str, template: str, version: int | None = None) -> str:
        """Save or update a prompt template.

        If version is not specified, increments from the latest version.
        Returns the inserted document ID.
        """
        if version is None:
            latest = await self.collection.find_one(
                {"name": name}, sort=[("version", -1)]
            )
            version = (latest["version"] + 1) if latest else 1

        now = datetime.now(timezone.utc)
        doc = {
            "name": name,
            "template": template,
            "version": version,
            "created_at": now,
            "updated_at": now,
        }
        result = await self.collection.insert_one(doc)

        # Invalidate cache for this prompt
        for key in list(self._cache.keys()):
            if key.startswith(f"{name}:"):
                del self._cache[key]
                self._cache_time.pop(key, None)

        return str(result.inserted_id)

    async def seed_defaults(self) -> int:
        """Seed hardcoded prompts into DB as version 1. Returns count inserted.

        Idempotent: skips prompts that already have any version in the DB.
        """
        count = 0
        for name, template in _HARDCODED_PROMPTS.items():
            existing = await self.collection.find_one({"name": name})
            if not existing:
                now = datetime.now(timezone.utc)
                doc = {
                    "name": name,
                    "template": template,
                    "version": 1,
                    "created_at": now,
                    "updated_at": now,
                }
                await self.collection.insert_one(doc)
                count += 1
        return count
