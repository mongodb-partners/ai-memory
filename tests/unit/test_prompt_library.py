"""Tests for PromptLibrary."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory.core.config import MCPConfig
from agent_memory.services.prompt_library import PromptLibrary, _HARDCODED_PROMPTS


def _make_config(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


def _make_collection():
    col = MagicMock()
    col.find_one = AsyncMock(return_value=None)
    col.insert_one = AsyncMock(return_value=MagicMock(inserted_id="abc123"))
    return col


class TestGetPromptDisabled:
    """When prompt_experiment_enabled=False, returns hardcoded defaults."""

    async def test_disabled_returns_hardcoded(self):
        col = _make_collection()
        config = _make_config(prompt_experiment_enabled=False)
        lib = PromptLibrary(col, config)

        result = await lib.get_prompt("importance_assessment")
        assert "importance" in result.lower()
        col.find_one.assert_not_called()

    async def test_disabled_unknown_returns_empty(self):
        col = _make_collection()
        config = _make_config(prompt_experiment_enabled=False)
        lib = PromptLibrary(col, config)

        result = await lib.get_prompt("nonexistent_prompt")
        assert result == ""


class TestGetPromptEnabled:
    """When prompt_experiment_enabled=True, fetches from DB."""

    async def test_enabled_fetches_db(self):
        col = _make_collection()
        config = _make_config(prompt_experiment_enabled=True)
        lib = PromptLibrary(col, config)

        col.find_one = AsyncMock(return_value={
            "name": "importance_assessment",
            "template": "Custom template: {content}",
            "version": 2,
        })

        result = await lib.get_prompt("importance_assessment")
        assert result == "Custom template: {content}"
        col.find_one.assert_called_once()

    async def test_cache_hit(self):
        col = _make_collection()
        config = _make_config(
            prompt_experiment_enabled=True,
            prompt_cache_ttl_seconds=300,
        )
        lib = PromptLibrary(col, config)

        col.find_one = AsyncMock(return_value={
            "name": "test", "template": "cached", "version": 1,
        })

        await lib.get_prompt("test")
        await lib.get_prompt("test")

        assert col.find_one.call_count == 1

    async def test_fallback_when_not_in_db(self):
        col = _make_collection()
        config = _make_config(prompt_experiment_enabled=True)
        lib = PromptLibrary(col, config)

        col.find_one = AsyncMock(return_value=None)

        result = await lib.get_prompt("importance_assessment")
        assert result == _HARDCODED_PROMPTS["importance_assessment"]

    async def test_specific_version(self):
        col = _make_collection()
        config = _make_config(prompt_experiment_enabled=True)
        lib = PromptLibrary(col, config)

        col.find_one = AsyncMock(return_value={
            "name": "test", "template": "v1 template", "version": 1,
        })

        result = await lib.get_prompt("test", version=1)
        assert result == "v1 template"
        query = col.find_one.call_args[0][0]
        assert query["version"] == 1


class TestSavePrompt:
    """save_prompt persists and auto-increments version."""

    async def test_save_new(self):
        col = _make_collection()
        config = _make_config()
        lib = PromptLibrary(col, config)

        col.find_one = AsyncMock(return_value=None)

        doc_id = await lib.save_prompt("test", "new template")
        assert doc_id == "abc123"
        inserted = col.insert_one.call_args[0][0]
        assert inserted["name"] == "test"
        assert inserted["template"] == "new template"
        assert inserted["version"] == 1

    async def test_save_increments_version(self):
        col = _make_collection()
        config = _make_config()
        lib = PromptLibrary(col, config)

        col.find_one = AsyncMock(return_value={"version": 3})

        await lib.save_prompt("test", "updated template")
        inserted = col.insert_one.call_args[0][0]
        assert inserted["version"] == 4

    async def test_save_invalidates_cache(self):
        col = _make_collection()
        config = _make_config(prompt_experiment_enabled=True, prompt_cache_ttl_seconds=300)
        lib = PromptLibrary(col, config)

        # Populate cache
        col.find_one = AsyncMock(return_value={
            "name": "test", "template": "old", "version": 1,
        })
        await lib.get_prompt("test")
        assert "test:latest" in lib._cache

        # Save invalidates cache
        col.find_one = AsyncMock(return_value={"version": 1})
        await lib.save_prompt("test", "new template")
        assert "test:latest" not in lib._cache


class TestSeedDefaults:
    """seed_defaults inserts hardcoded prompts as version 1."""

    async def test_seed_inserts_all(self):
        """TC-E-004: All hardcoded prompts inserted on empty collection."""
        col = _make_collection()
        config = _make_config()
        lib = PromptLibrary(col, config)

        col.find_one = AsyncMock(return_value=None)

        count = await lib.seed_defaults()

        assert count == len(_HARDCODED_PROMPTS)
        assert col.insert_one.call_count == len(_HARDCODED_PROMPTS)

        # Verify inserted docs have version 1
        for call in col.insert_one.call_args_list:
            doc = call[0][0]
            assert doc["version"] == 1
            assert "name" in doc
            assert "template" in doc
            assert "created_at" in doc
            assert "updated_at" in doc

    async def test_seed_skips_existing(self):
        """TC-E-005: Existing prompts are not overwritten."""
        col = _make_collection()
        config = _make_config()
        lib = PromptLibrary(col, config)

        col.find_one = AsyncMock(return_value={"name": "existing", "version": 1})

        count = await lib.seed_defaults()

        assert count == 0
        col.insert_one.assert_not_called()

    async def test_seed_returns_count(self):
        """TC-E-006: Returns count of inserted prompts."""
        col = _make_collection()
        config = _make_config()
        lib = PromptLibrary(col, config)

        # First prompt exists, others don't
        call_count = 0
        async def find_one_side_effect(query, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"name": "exists", "version": 1}
            return None

        col.find_one = AsyncMock(side_effect=find_one_side_effect)

        count = await lib.seed_defaults()

        assert count == len(_HARDCODED_PROMPTS) - 1

    async def test_seed_skips_higher_version(self):
        """TC-E-034: Seed does not overwrite if higher version exists."""
        col = _make_collection()
        config = _make_config()
        lib = PromptLibrary(col, config)

        # A version 5 already exists
        col.find_one = AsyncMock(return_value={
            "name": "importance_assessment", "version": 5, "template": "custom",
        })

        count = await lib.seed_defaults()

        assert count == 0
        col.insert_one.assert_not_called()
