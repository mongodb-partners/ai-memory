"""Admin service — health statistics and full user-data wipe.

Extracted from the former ``memory_health`` / ``wipe_user_data`` MCP tools so
the facade orchestration layer (access-check → service → audit) stays thin and
identical across shells.
"""

from __future__ import annotations

from agent_memory.exceptions import MemoryError as MemoryLibError


class PartialWipeError(MemoryLibError):
    """A wipe deleted some collections and failed on others.

    Raised rather than returned. A partial erasure reported through the return
    value was audited as ``"success"`` — ``_run`` only labels an operation failed
    if it raises — so the record said the user's data was gone while some of it
    was still there. That is the one wrong answer this operation must never give:
    a deletion request is answered once, and an operator who reads "success" has
    no reason to look again.

    ``counts`` carries what *was* deleted so the caller can see how far it got,
    and ``errors`` maps collection name to the failure. Both reach the audit
    record, which is how a retry knows what is left.
    """

    def __init__(self, counts: dict, errors: dict) -> None:
        self.counts = counts
        self.errors = errors
        failed = ", ".join(sorted(errors))
        super().__init__(
            f"wipe incomplete: {failed} could not be cleared; "
            f"other collections were deleted. Retry to finish."
        )


def erasure_targets(user_id: str) -> list[tuple[str, str, dict]]:
    """Every user-scoped collection, as ``(result key, collection, filter)``.

    Module-level and shared rather than inlined in ``wipe_user_data``, because the
    facade's post-wipe residue check has to look in *exactly* these places. Two
    copies of this list would drift, and the failure mode of a residue check that
    is missing a collection is the one this whole path exists to prevent: a
    confident ``complete: true`` over data that is still there.

    ``episodes_counters`` is keyed by a composite ``_id`` rather than a top-level
    ``user_id``, so it carries its own filter — the one collection where the
    obvious query silently matches nothing.
    """
    from agent_memory.core.collections import (
        AUDIT_LOG,
        DECISIONS,
        EPISODES,
        EPISODES_COUNTERS,
        MEMORIES,
        RATE_LIMITS,
        SEMANTIC_CACHE,
    )

    return [
        ("memories_deleted", MEMORIES, {"user_id": user_id}),
        ("cache_deleted", SEMANTIC_CACHE, {"user_id": user_id}),
        ("audit_deleted", AUDIT_LOG, {"user_id": user_id}),
        ("episodes_deleted", EPISODES, {"user_id": user_id}),
        ("decisions_deleted", DECISIONS, {"user_id": user_id}),
        ("rate_limits_deleted", RATE_LIMITS, {"user_id": user_id}),
        ("episode_counters_deleted", EPISODES_COUNTERS, {"_id.user_id": user_id}),
    ]


class AdminService:
    """Cross-collection administrative operations for a single user."""

    def __init__(self, db) -> None:
        self.db = db

    async def health(self, user_id: str) -> dict:
        """Return tier counts, enrichment-status counts, and total memories."""
        memories_col = self.db["memories"]
        pipeline = [
            {"$match": {"user_id": user_id, "deleted_at": None}},
            {
                "$group": {
                    "_id": {
                        "tier": "$tier",
                        "enrichment_status": "$enrichment_status",
                    },
                    "count": {"$sum": 1},
                }
            },
        ]
        cursor = await memories_col.aggregate(pipeline)
        results = await cursor.to_list(None)

        tier_stats: dict = {}
        enrichment_stats: dict = {}
        total = 0
        for r in results:
            tier = r["_id"]["tier"]
            status = r["_id"]["enrichment_status"]
            count = r["count"]
            total += count
            tier_stats[tier] = tier_stats.get(tier, 0) + count
            enrichment_stats[status] = enrichment_stats.get(status, 0) + count

        return {
            "user_id": user_id,
            "total_memories": total,
            "tier_stats": tier_stats,
            "enrichment_stats": enrichment_stats,
        }

    async def wipe_user_data(self, user_id: str) -> dict:
        """Permanently delete every document this user owns, in every collection.

        The caller (facade) is responsible for the ``confirm`` gate; this method
        performs the irreversible deletion unconditionally.

        It used to clear three collections — ``memories``, ``semantic_cache``,
        ``audit_log`` — while its docstring promised "all user data". Episodic
        turns, sticky decisions, step counters, and rate-limit records survived,
        so a user who asked to be forgotten still had their activity log, and the
        answer to a deletion request was wrong rather than merely incomplete.

        Every user-scoped collection is now enumerated in one place —
        :func:`erasure_targets` — and each appears in the result with its own
        count so the caller can see exactly what was removed. A new user-scoped
        collection has to be added there; the test asserts that list against the
        collection names module, which is what makes an omission a failure rather
        than silent. The facade's post-wipe residue check reads the same function,
        so it cannot check fewer places than the deletion touched.

        Raises :class:`PartialWipeError` if any collection fails, so an incomplete
        erasure is audited as an error instead of a success.
        """
        from agent_memory.services.audit import ERASURE_PRINCIPAL

        if user_id == ERASURE_PRINCIPAL:
            # The erasure trail is filed against this reserved id, so accepting it
            # as a target would let anyone able to name it delete the record of
            # every deletion — by asking to be forgotten. It is not a real
            # identity: identifiers come from a token claim and this one leads with
            # an underscore.
            raise ValueError(
                f"{ERASURE_PRINCIPAL!r} is the reserved erasure-audit principal, "
                "not a user, and cannot be wiped"
            )

        targets = erasure_targets(user_id)

        out: dict = {"user_id": user_id}
        errors: dict = {}
        for key, collection, query in targets:
            # One failing collection must not abandon the rest: a partial wipe
            # that stops early leaves the most data behind and reports success
            # for nothing. Failures are collected and returned.
            try:
                result = await self.db[collection].delete_many(query)
                out[key] = result.deleted_count
            except Exception as exc:
                out[key] = 0
                errors[collection] = str(exc)

        if errors:
            # Raised, not returned. `_run` derives the audit status from whether
            # the coroutine raised, so returning this dict recorded a half-done
            # erasure as `"success"` — and the caller who reads a success has no
            # reason to retry. The counts travel on the exception so the audit
            # record still shows how far it got.
            raise PartialWipeError(out, errors)
        out["complete"] = True
        return out
