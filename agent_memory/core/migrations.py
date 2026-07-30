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

One reconciliation is *not* safe to perform silently: a changed
``numDimensions``. Atlas will not edit that field, so the index must be dropped
and recreated, and the stored vectors are not touched by that — they stay the old
width and the rebuilt index returns none of them. See
``find_stranding_dimension_changes``.
"""

import asyncio
import logging
from dataclasses import dataclass

from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from agent_memory.core.collections import STANDARD_INDEXES, get_search_indexes
from agent_memory.exceptions import ConfigError

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


# ─── Preflight: dimension changes that would strand existing vectors ─────


@dataclass(frozen=True)
class StrandedVectors:
    """One vector index whose rebuild would orphan the documents beneath it."""

    collection: str
    index_name: str
    existing_dimension: int
    wanted_dimension: int
    document_count: int

    def describe(self) -> str:
        return (
            f"{self.collection}.{self.index_name}: index is "
            f"{self.existing_dimension}-dim, config wants "
            f"{self.wanted_dimension}-dim, and {self.document_count} document(s) "
            f"already hold {self.existing_dimension}-dim vectors"
        )


async def find_stranding_dimension_changes(
    db, embedding_dimension: int
) -> list[StrandedVectors]:
    """Vector indexes whose recreation would leave stored vectors unsearchable.

    A changed ``numDimensions`` forces a drop and recreate, because Atlas will not
    edit that field. The documents survive the drop untouched, which sounds like
    the safe outcome and is the dangerous one: every vector already stored is the
    old width, and the rebuilt index will not return a single one of them from
    ``$vectorSearch``.

    Nothing about that failure is visible. No exception, no changed document
    count, and ``find`` still returns every memory. Recall goes empty for the
    entire history while continuing to work perfectly for anything written
    afterwards — so it reads as "the user has no memories about that", which is
    indistinguishable from the truth. Recovery requires re-embedding every
    document with the *old* provider, which by then is the config the operator has
    just replaced.

    Reported rather than raised: this returns findings and the caller decides. The
    same question is asked at startup (to refuse) and could be asked by a
    migration tool (to plan), and only one of those wants an exception.

    An empty collection is not a finding. There are no vectors to strand, so
    changing the dimension there is exactly the ordinary first-run case.
    """
    findings: list[StrandedVectors] = []
    for idx_def in get_search_indexes(embedding_dimension):
        if idx_def["type"] != "vectorSearch":
            continue
        collection_name: str = idx_def["collection"]
        index_name: str = idx_def["name"]
        collection = db[collection_name]
        try:
            existing = await _list_search_indexes(collection, index_name)
        except OperationFailure:
            # Not an Atlas deployment, or search is unavailable. There is no
            # vector index to drop, so there is nothing to strand.
            return []
        except Exception:
            logger.warning(
                "Could not inspect search index '%s' on '%s' for a dimension "
                "change. Proceeding; if the dimension did change, existing "
                "vectors will stop being returned by $vectorSearch.",
                index_name,
                collection_name,
                exc_info=True,
            )
            continue
        if not existing:
            continue
        existing_dims = _get_existing_dims(existing[0])
        if not existing_dims or existing_dims == embedding_dimension:
            continue
        try:
            # Count documents that actually carry a vector, not documents in the
            # collection. A collection holding only un-embedded rows has nothing
            # to strand, and counting those would refuse a startup for no reason.
            count = await collection.count_documents({"embedding": {"$ne": None}})
        except Exception:
            # The dimension change is real; only the blast radius is unknown.
            # Report it with an unknown count rather than dropping the finding,
            # because "we could not count" is not "there is nothing there".
            logger.warning(
                "Could not count embedded documents in '%s'.",
                collection_name,
                exc_info=True,
            )
            count = -1
        if count == 0:
            continue
        findings.append(
            StrandedVectors(
                collection=collection_name,
                index_name=index_name,
                existing_dimension=existing_dims,
                wanted_dimension=embedding_dimension,
                document_count=count,
            )
        )
    return findings


def stranding_error(findings: list[StrandedVectors]) -> ConfigError:
    """The refusal message for a dimension change over existing vectors.

    Separate from the detection so the wording lives next to the check rather
    than at the call site, and so a migration tool can reuse it.
    """
    detail = "\n  ".join(f.describe() for f in findings)
    wanted = findings[0].wanted_dimension
    return ConfigError(
        "Refusing to start: the embedding dimension changed and existing "
        "vectors would be stranded.\n  "
        f"{detail}\n"
        "Rebuilding a vector index at a new numDimensions does not re-embed the "
        "documents beneath it, so those vectors stay the old width and "
        "$vectorSearch stops returning them — silently, with no error and no "
        "change in document count.\n"
        "Choose one:\n"
        "  * Restore the previous embedding_provider/model so the dimension "
        "matches the stored vectors.\n"
        f"  * Re-embed every document at {wanted} dimensions, then start.\n"
        "  * Drop the affected collections, if the history is expendable.\n"
        "  * Set allow_embedding_dimension_change=true "
        "(ALLOW_EMBEDDING_DIMENSION_CHANGE=true) to proceed anyway and accept "
        "that the existing vectors become unsearchable."
    )


# ─── Stage 2: Atlas Search / Vector Search Indexes ───────────────


async def ensure_search_indexes(
    db, embedding_dimension: int = 1536, allow_dimension_change: bool = False
) -> None:
    """Create *and reconcile* Atlas Search and Vector Search indexes.

    Designed to run as a background task — non-fatal on failure.
    Gracefully detects non-Atlas deployments and skips.

    An index that does not exist is created. An index that exists is compared
    against the shipped definition and brought into line:

    * A changed ``numDimensions`` needs a **drop and recreate** — Atlas will not
      edit that field. This is the one reconciliation that can destroy something,
      because the stored vectors are *not* recreated with it: they keep the old
      width and the new index never returns them. It is performed only when
      ``allow_dimension_change`` says so, or when there is nothing there to
      strand.
    * Anything else — a new ``filter`` field, a new ``token`` mapping — is an
      **in-place update**, which preserves the built index and its stored vectors.
      Dropping here would take vector search offline for the minutes Atlas needs
      to rebuild, on every deployment, for a change that does not require it.

    Reconciliation is what makes a definition change deliverable at all. See the
    module docstring.

    ``allow_dimension_change`` defaults to False here rather than deferring to
    the startup preflight, because this function's failures are deliberately
    non-fatal: it usually runs as a background task whose exceptions are logged
    and dropped. A guard that only existed upstream would be a guard that the
    default code path routes around.
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
                    if not await _may_rebuild_at_new_dimension(
                        collection,
                        collection_name,
                        index_name,
                        existing_dims,
                        embedding_dimension,
                        allow_dimension_change,
                    ):
                        continue
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


async def _may_rebuild_at_new_dimension(
    collection,
    collection_name: str,
    index_name: str,
    existing_dims: int,
    wanted_dims: int,
    allowed: bool,
) -> bool:
    """Whether it is safe to drop and recreate this index at a new dimension.

    True when the operator has said so, or when the collection holds no vectors
    to strand — the ordinary first-run case, where the "change" is from an index
    built by a previous config over an empty collection.

    False leaves the old index in place. That is the conservative outcome: search
    keeps working for every vector already stored, and anything written at the new
    width simply is not indexed until someone resolves the mismatch. The
    alternative — rebuilding — makes the *existing* history unsearchable instead,
    which is both larger and unrecoverable without the old provider config.

    Refusing is logged at error level, and says what to do. This runs inside a
    background task whose exceptions are swallowed, so an exception here would be
    the one shape of complaint guaranteed not to reach anyone.
    """
    if allowed:
        return True
    try:
        count = await collection.count_documents({"embedding": {"$ne": None}})
    except Exception:
        # Unknown, so assume there is something to lose. Guessing "empty" here
        # would turn an unreadable count into a silent data loss.
        logger.warning(
            "Could not count embedded documents in '%s' before rebuilding "
            "index '%s'. Assuming vectors exist and leaving the index alone.",
            collection_name,
            index_name,
            exc_info=True,
        )
        count = -1
    if count == 0:
        return True
    logger.error(
        "Not rebuilding search index '%s' on '%s': it is %d-dim, config wants "
        "%d-dim, and the collection holds %s embedded document(s) whose vectors "
        "would stop being returned by $vectorSearch. The existing index is left "
        "in place, so recall keeps working for those documents while anything "
        "written at %d dims goes unindexed. Re-embed the collection, restore the "
        "previous embedding model, or set "
        "allow_embedding_dimension_change=true to accept the loss.",
        index_name,
        collection_name,
        existing_dims,
        wanted_dims,
        "an unknown number of" if count < 0 else count,
        wanted_dims,
    )
    return False


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
