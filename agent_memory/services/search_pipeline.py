"""Shared ``$rankFusion`` pipeline builder for hybrid retrieval.

Two tiers search the same way over different collections: semantic memories
search `content`/`summary` in `memories`, episodic records search `search_text`
in `episodes`. The pipeline shape is identical, so it lives here once.

Why hybrid at all: vector search alone misses exact terms — SKUs, error codes, a
person's name — and full-text alone misses meaning. ``$rankFusion`` performs
reciprocal rank fusion over both ranked lists natively, in one round trip,
instead of two queries merged in application code.

The tenant filter is passed into *both* branches. That is the whole isolation
story for retrieval: the engine enforces it, so a caller cannot forget it by
building only half the filter.
"""

from __future__ import annotations

from typing import Any

# Over-fetch per branch before fusion. RRF ranks by position, so each branch
# needs enough depth for a document to be findable even when only one branch
# ranks it well.
DEFAULT_BRANCH_LIMIT = 20
DEFAULT_NUM_CANDIDATES = 100

VECTOR_BRANCH = "vectorPipeline"
TEXT_BRANCH = "fullTextPipeline"


def rank_fusion_pipeline(
    *,
    query: str,
    query_embedding: list[float],
    vector_index: str,
    fts_index: str,
    fts_paths: list[str],
    vs_filter: dict[str, Any],
    fts_filter_clauses: list[dict[str, Any]],
    limit: int,
    vector_weight: float,
    text_weight: float,
    num_candidates: int = DEFAULT_NUM_CANDIDATES,
    branch_limit: int = DEFAULT_BRANCH_LIMIT,
    project: dict[str, Any] | None = None,
    post_fusion_stages: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a `$rankFusion` aggregation over one vector and one full-text branch.

    ``vs_filter`` is a `$vectorSearch` pre-filter (MQL-shaped) and
    ``fts_filter_clauses`` are Atlas Search operators for the `compound.filter`
    slot — the same restriction expressed in each branch's own dialect. Every
    field referenced by ``vs_filter`` must be declared as a ``filter`` field in
    the vector index definition, or the branch silently returns nothing.

    ``project`` defaults to excluding ``embedding``, which is otherwise the
    largest field in the response by an order of magnitude.

    ``post_fusion_stages`` are inserted **between fusion and ``$limit``**, and the
    position is the whole point. A narrowing stage appended after the pipeline
    instead — which is what episodic ``since`` filtering did — runs on an
    already-truncated result set: fusion ranks across all time, ``$limit`` keeps
    the top *N*, and only then does the date filter run. Ask for the 5 most
    relevant turns since yesterday and you get however many of the 5
    best-all-time happen to be recent, which is frequently zero while the
    collection holds plenty of matching recent turns. Empty, not wrong-looking,
    so nothing draws attention to it.

    Use this for restrictions that cannot go in a branch pre-filter — typically a
    field not declared as a vector-index ``filter`` field, where declaring a
    high-cardinality value like a timestamp would bloat the index for a rarely
    used narrowing.

    This is a real improvement but not a complete substitute for a pre-filter, and
    the difference is worth knowing. Each branch contributes at most
    ``branch_limit`` documents, so a post-fusion narrowing selects from that depth
    rather than from the collection: if every matching document ranks below the
    branch cutoff, the result is still short. Raise ``branch_limit`` when a
    narrowing is both selective and routine. A pre-filter has no such ceiling
    because the engine applies it during the search — which is why ``user_id``
    lives in both branch filters and never here.
    """
    return [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        VECTOR_BRANCH: [
                            {
                                "$vectorSearch": {
                                    "index": vector_index,
                                    "path": "embedding",
                                    "queryVector": query_embedding,
                                    "numCandidates": num_candidates,
                                    "limit": branch_limit,
                                    "filter": vs_filter,
                                }
                            },
                        ],
                        TEXT_BRANCH: [
                            {
                                "$search": {
                                    "index": fts_index,
                                    "compound": {
                                        "must": [
                                            {"text": {"query": query, "path": fts_paths}}
                                        ],
                                        "filter": fts_filter_clauses,
                                    },
                                }
                            },
                            # $search has no limit option; cap it as a stage so
                            # the branch depth matches the vector branch.
                            {"$limit": branch_limit},
                        ],
                    }
                },
                "combination": {
                    "weights": {
                        VECTOR_BRANCH: vector_weight,
                        TEXT_BRANCH: text_weight,
                    },
                },
            }
        },
        # Narrow before truncating. See `post_fusion_stages` above.
        *(post_fusion_stages or []),
        {"$limit": limit},
        # $rankFusion does not project its fused rank; without this the caller
        # gets ranked documents carrying no score at all, and any consumer that
        # wants to show *why* a document ranked has nothing to show.
        #
        # It has to be $addFields rather than a computed field inside the
        # $project below: mixing `{"score": {"$meta": "score"}}` into an
        # exclusion projection does not error, it just silently yields null.
        # Verified against Atlas 8.3 — with the field inside $project every
        # score came back None; as its own stage the RRF values appear.
        {"$addFields": {"score": {"$meta": "score"}}},
        {"$project": project if project is not None else {"embedding": 0}},
    ]


__all__ = [
    "DEFAULT_BRANCH_LIMIT",
    "DEFAULT_NUM_CANDIDATES",
    "TEXT_BRANCH",
    "VECTOR_BRANCH",
    "rank_fusion_pipeline",
]
