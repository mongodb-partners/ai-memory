"""Database migrations — idempotent index creation on startup.

Two-stage initialization following the conquer-code pattern:

Stage 1 (``ensure_indexes``): Standard B-tree indexes, runs during lifespan
    startup before the server accepts connections.  Fast, non-blocking.

Stage 2 (``ensure_search_indexes``): Atlas Search / Vector Search indexes,
    launched as a background task after startup.  Can take minutes on first
    run.  Non-fatal on failure (degrades to no vector/FTS search).
"""

import asyncio
import logging

from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from agent_memory.core.collections import STANDARD_INDEXES, get_search_indexes

logger = logging.getLogger(__name__)

_SEARCH_INDEX_POLL_INTERVAL = 5   # seconds between readiness checks
_SEARCH_INDEX_POLL_TIMEOUT = 120  # max seconds to wait per index


# ─── Stage 1: Standard Indexes ──────────────────────────────────


async def ensure_indexes(db) -> None:
    """Create all standard B-tree indexes.  Idempotent — safe to call on
    every startup.  PyMongo silently succeeds if the index already exists
    with the same spec.
    """
    for idx_def in STANDARD_INDEXES:
        collection_name: str = idx_def["collection"]
        keys: list[tuple[str, int]] = idx_def["keys"]
        name: str = idx_def["name"]
        extra_kwargs: dict = idx_def.get("kwargs", {})

        collection = db[collection_name]
        try:
            await collection.create_index(
                keys,
                name=name,
                background=True,
                **extra_kwargs,
            )
            logger.debug("Index '%s' on '%s' ensured.", name, collection_name)
        except OperationFailure as exc:
            if exc.code == 86 and name:
                # Index spec conflict — drop and recreate
                logger.info(
                    "Index '%s' on '%s' has conflicting options — "
                    "dropping and recreating.",
                    name,
                    collection_name,
                )
                try:
                    await collection.drop_index(name)
                    await collection.create_index(
                        keys,
                        name=name,
                        background=True,
                        **extra_kwargs,
                    )
                except Exception:
                    logger.exception(
                        "Failed to recreate index '%s' on '%s'.",
                        name,
                        collection_name,
                    )
            else:
                logger.exception(
                    "Failed to create index '%s' on '%s'.",
                    name,
                    collection_name,
                )

    logger.info("Standard indexes ensured for all Phase 0 collections.")


# ─── Stage 2: Atlas Search / Vector Search Indexes ───────────────


async def ensure_search_indexes(db, embedding_dimension: int = 1536) -> None:
    """Create Atlas Search and Vector Search indexes.

    Designed to run as a background task — non-fatal on failure.
    Gracefully detects non-Atlas deployments and skips.

    If the existing vector index has a different ``numDimensions`` than
    ``embedding_dimension``, the index is dropped and recreated.
    """
    search_indexes = get_search_indexes(embedding_dimension)

    for idx_def in search_indexes:
        collection_name: str = idx_def["collection"]
        index_name: str = idx_def["name"]
        index_type: str = idx_def["type"]
        definition: dict = idx_def["definition"]

        collection = db[collection_name]

        # Check if index already exists
        try:
            existing = await _list_search_indexes(collection, index_name)
            if existing:
                # Check for dimension mismatch on vector indexes
                if index_type == "vectorSearch":
                    existing_dims = _get_existing_dims(existing[0])
                    if existing_dims and existing_dims != embedding_dimension:
                        logger.info(
                            "Search index '%s' on '%s' has %d dimensions "
                            "but config requires %d — dropping and recreating.",
                            index_name,
                            collection_name,
                            existing_dims,
                            embedding_dimension,
                        )
                        await collection.drop_search_index(index_name)
                        # Wait for Atlas to fully remove the index
                        await _wait_for_search_index_dropped(
                            collection, index_name, _SEARCH_INDEX_POLL_TIMEOUT
                        )
                        # Fall through to creation below
                    else:
                        logger.debug(
                            "Search index '%s' on '%s' already exists — skipping.",
                            index_name,
                            collection_name,
                        )
                        continue
                else:
                    logger.debug(
                        "Search index '%s' on '%s' already exists — skipping.",
                        index_name,
                        collection_name,
                    )
                    continue
        except OperationFailure:
            logger.warning(
                "Atlas Search is not available on this deployment. "
                "Skipping all search/vector index creation. "
                "Vector and full-text search will not function."
            )
            return

        # Create the index
        try:
            model = SearchIndexModel(
                definition=definition,
                name=index_name,
                type=index_type,
            )
            await collection.create_search_index(model=model)
            logger.info(
                "Created search index '%s' on '%s'. Waiting for queryable state...",
                index_name,
                collection_name,
            )

            queryable = await _wait_for_search_index(
                collection, index_name, _SEARCH_INDEX_POLL_TIMEOUT
            )
            if queryable:
                logger.info("Search index '%s' is queryable.", index_name)
            else:
                logger.warning(
                    "Search index '%s' did not become queryable within %ds. "
                    "It may still be building.",
                    index_name,
                    _SEARCH_INDEX_POLL_TIMEOUT,
                )

        except OperationFailure as exc:
            logger.warning(
                "Failed to create search index '%s' on '%s': %s",
                index_name,
                collection_name,
                exc,
            )
        except Exception:
            logger.exception(
                "Unexpected error creating search index '%s' on '%s'.",
                index_name,
                collection_name,
            )

    logger.info("Atlas Search index setup complete.")


# ─── Helpers ─────────────────────────────────────────────────────


async def _list_search_indexes(collection, index_name: str) -> list[dict]:
    """List search indexes matching a name on a collection."""
    indexes = []
    async for idx in await collection.list_search_indexes(index_name):
        indexes.append(idx)
    return indexes


def _get_existing_dims(index_info: dict) -> int | None:
    """Extract numDimensions from an existing vector search index definition."""
    defn = index_info.get("latestDefinition") or index_info.get("definition", {})
    for field in defn.get("fields", []):
        if field.get("type") == "vector":
            return field.get("numDimensions")
    return None


async def _wait_for_search_index_dropped(
    collection, index_name: str, timeout: int
) -> None:
    """Poll until a search index no longer exists (fully deleted by Atlas)."""
    elapsed = 0
    while elapsed < timeout:
        try:
            indexes = await _list_search_indexes(collection, index_name)
            if not indexes:
                logger.debug("Search index '%s' fully removed.", index_name)
                return
        except Exception:
            return  # If listing fails, assume gone
        await asyncio.sleep(_SEARCH_INDEX_POLL_INTERVAL)
        elapsed += _SEARCH_INDEX_POLL_INTERVAL
    logger.warning("Timed out waiting for index '%s' to be removed.", index_name)


async def _wait_for_search_index(
    collection, index_name: str, timeout: int
) -> bool:
    """Poll until a search index becomes queryable or timeout is reached."""
    elapsed = 0
    while elapsed < timeout:
        try:
            indexes = await _list_search_indexes(collection, index_name)
            if indexes and indexes[0].get("queryable"):
                return True
        except Exception:
            pass
        await asyncio.sleep(_SEARCH_INDEX_POLL_INTERVAL)
        elapsed += _SEARCH_INDEX_POLL_INTERVAL
    return False
