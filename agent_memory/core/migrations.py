"""Database migrations — idempotent index creation on startup.

Two-stage initialization following the conquer-code pattern:

Stage 1 (``ensure_indexes``): Standard B-tree indexes, runs during lifespan
    startup before the server accepts connections.  Fast, non-blocking.

Stage 2 (``ensure_search_indexes``): Atlas Search / Vector Search indexes,
    launched as a background task after startup.  Can take minutes on first
    run.  Non-fatal on failure (degrades to no vector/FTS search).

Stage 2 reconciles *existing* indexes against the shipped definitions, not just
absent ones. It used to compare only ``numDimensions`` on vector indexes and
``continue`` past every existing full-text index, so an index created by an
earlier version kept that version's definition forever. Only a fresh cluster ever
got the current one.

That makes every future change to a definition undeliverable, and undeliverable
in the quietest possible way: a filter path missing from an index is not an error,
the query just matches nothing. The upgrade that added ``memory_type`` and
``tags`` as filter fields would have worked perfectly in testing — where the
cluster is new — and changed nothing on the deployment that needed it.
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
    """Create *and reconcile* Atlas Search and Vector Search indexes.

    Designed to run as a background task — non-fatal on failure.
    Gracefully detects non-Atlas deployments and skips.

    An index that does not exist is created. An index that exists is compared
    against the shipped definition and brought into line:

    * A changed ``numDimensions`` needs a **drop and recreate** — Atlas will not
      edit that field, and the stored vectors are the wrong width anyway.
    * Anything else — a new ``filter`` field, a new ``token`` mapping — is an
      **in-place update**, which preserves the built index and its stored vectors.
      Dropping here would take vector search offline for the minutes Atlas needs
      to rebuild, on every deployment, for a change that does not require it.

    Reconciliation is what makes a definition change deliverable at all. See the
    module docstring.
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
                existing_dims = (
                    _get_existing_dims(existing[0])
                    if index_type == "vectorSearch"
                    else None
                )
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
                elif _definition_matches(existing[0], definition):
                    logger.debug(
                        "Search index '%s' on '%s' is up to date — skipping.",
                        index_name,
                        collection_name,
                    )
                    continue
                else:
                    # Update rather than drop: the existing index is correctly
                    # built, and only the definition has moved on.
                    if await _update_search_index(
                        collection, index_name, collection_name, definition
                    ):
                        continue
                    # An update that failed is not a reason to drop a working
                    # index — the old definition still serves queries, just
                    # without whatever the new one added. Logged by the helper.
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


def _live_definition(index_info: dict) -> dict:
    """The definition Atlas currently has for an index.

    ``latestDefinition`` is the one to read: after an update it reflects the new
    definition while the index finishes rebuilding, so a second startup during
    the rebuild does not issue the same update again.
    """
    return index_info.get("latestDefinition") or index_info.get("definition") or {}


def _definition_matches(index_info: dict, wanted: dict) -> bool:
    """True when the live definition already provides everything ``wanted`` asks.

    Deliberately a *subset* check, not equality. Atlas echoes a definition back
    enriched with its own defaults — an analyzer on a string mapping, a
    ``quantization`` on a vector field, ``storedSource``, keys added by whatever
    Atlas version the cluster runs. Comparing for equality would find a
    difference on a cluster that is perfectly up to date and reissue an update on
    every single startup: harmless-looking, and a rebuild of the whole search
    index each time.

    So the question asked here is "is anything we require missing or different",
    which is the question that actually decides whether an update is needed.

    Vector-index ``fields`` are compared as a set keyed by ``(type, path)`` rather
    than positionally, because order carries no meaning in the definition and
    Atlas does not preserve ours.
    """
    live = _live_definition(index_info)
    if not live:
        # Nothing to compare against. Treating this as "matches" would make an
        # index we cannot read permanently unreconcilable; an update is idempotent
        # and safe, so prefer issuing one.
        return False
    if "fields" in wanted:
        live_fields = {
            (f.get("type"), f.get("path")): f for f in live.get("fields", [])
        }
        for field in wanted["fields"]:
            key = (field.get("type"), field.get("path"))
            if key not in live_fields:
                return False
            if not _is_subset(field, live_fields[key]):
                return False
        return True
    return _is_subset(wanted, live)


def _is_subset(wanted, live) -> bool:
    """True when every key/value in ``wanted`` appears in ``live``, recursively.

    Extra keys in ``live`` are fine — those are Atlas's defaults. Extra keys in
    ``wanted`` are not; that is a definition change we still owe the cluster.
    """
    if isinstance(wanted, dict):
        if not isinstance(live, dict):
            return False
        return all(
            key in live and _is_subset(value, live[key])
            for key, value in wanted.items()
        )
    return wanted == live


async def _update_search_index(
    collection, index_name: str, collection_name: str, definition: dict
) -> bool:
    """Push a changed definition to an existing index. Never raises.

    Returns False when the update could not be applied, so the caller can leave
    the existing index in place rather than dropping a working one.

    No wait for queryable: an updated index keeps serving the old definition
    while Atlas rebuilds, so there is nothing to block startup for. The next run
    will see ``latestDefinition`` already matching and do nothing.
    """
    try:
        await collection.update_search_index(index_name, definition)
    except OperationFailure as exc:
        logger.warning(
            "Could not update search index '%s' on '%s': %s. The index keeps its "
            "previous definition; queries relying on newly added filter fields "
            "will return no matches.",
            index_name,
            collection_name,
            exc,
        )
        return False
    except Exception:
        logger.exception(
            "Unexpected error updating search index '%s' on '%s'.",
            index_name,
            collection_name,
        )
        return False
    logger.info(
        "Updated search index '%s' on '%s' to the current definition. "
        "Atlas rebuilds it in the background; the old definition serves "
        "queries until it is ready.",
        index_name,
        collection_name,
    )
    return True


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
