"""Collection names and index definitions.

Index definitions are separated from migration logic so they serve as
the canonical reference for the database schema. ``core/migrations.py`` is
data-driven over these two lists, so adding a collection here is enough.
"""

# ─── Collection Names ────────────────────────────────────────────

MEMORIES: str = "memories"
EPISODES: str = "episodes"
# Per-thread step counters, kept out of EPISODES so that collection stays
# homogeneous — one document shape, one TTL policy, one index set.
EPISODES_COUNTERS: str = "episodes_counters"
SEMANTIC_CACHE: str = "semantic_cache"
AUDIT_LOG: str = "audit_log"
RATE_LIMITS: str = "rate_limits"
GOVERNANCE_PROFILES: str = "governance_profiles"
PROMPTS: str = "prompts"
DECISIONS: str = "decisions"

# Default episodic retention. A turn log is high-volume and loses value fast;
# 30 days covers "what did we do last month" without unbounded growth. Tunable
# at runtime via ``set_activity_retention`` (collMod) — STANDARD_INDEXES is a
# static list, so it cannot be config-driven here.
EPISODES_DEFAULT_TTL_SECONDS: int = 30 * 86400

# ─── Standard (B-tree) Indexes ───────────────────────────────────
#
# Each entry: collection, keys (list of (field, direction) tuples),
# name, optional unique flag, optional kwargs dict.

STANDARD_INDEXES: list[dict] = [
    # -- memories --
    {
        "collection": MEMORIES,
        "keys": [("expires_at", 1)],
        "name": "ix_memories_expires_at",
        "kwargs": {"expireAfterSeconds": 0},
    },
    {
        "collection": MEMORIES,
        "keys": [("user_id", 1), ("tier", 1), ("created_at", -1)],
        "name": "ix_memories_user_tier_created",
        "kwargs": {"partialFilterExpression": {"deleted_at": None}},
    },
    {
        "collection": MEMORIES,
        "keys": [("user_id", 1), ("conversation_id", 1)],
        "name": "ix_memories_conversation",
        "kwargs": {"partialFilterExpression": {"deleted_at": None}},
    },
    {
        "collection": MEMORIES,
        "keys": [("enrichment_status", 1), ("created_at", 1)],
        "name": "ix_memories_enrichment_queue",
    },
    {
        "collection": MEMORIES,
        "keys": [("deleted_at", 1)],
        "name": "ix_memories_deleted_at_ttl",
        "kwargs": {
            "expireAfterSeconds": 30 * 86400,  # 30 days
            "partialFilterExpression": {"deleted_at": {"$type": "date"}},
        },
    },
    # -- episodes --
    # Thread replay; the primary read path for "show me this conversation's
    # turns". The key order mirrors `get_thread` exactly: equality on
    # (user_id, thread_id), then the sort on (ts, step).
    #
    # `user_id` leads because every episodic read is tenant-scoped — a thread id
    # is not a capability — so an index keyed on thread_id alone leaves the
    # isolation filter as a residual predicate the server applies after the scan.
    # `ts` precedes `step` because that is the sort `get_thread` issues, and the
    # sort order is itself deliberate: `step` can be null when the durable counter
    # failed, and null sorts below every number, so leading on it would relocate a
    # turn to the front of the replay rather than keep it in place. See
    # `EpisodicService.get_thread`.
    {
        "collection": EPISODES,
        "keys": [("user_id", 1), ("thread_id", 1), ("ts", 1), ("step", 1)],
        "name": "ix_episodes_thread_step",
    },
    {
        "collection": EPISODES,
        "keys": [("user_id", 1), ("ts", -1)],
        "name": "ix_episodes_user_ts",
    },
    {
        "collection": EPISODES,
        "keys": [("thread_id", 1), ("ts", -1)],
        "name": "ix_episodes_thread_ts",
    },
    # Join a logged turn back to a trace or support ticket. Same shape as the
    # thread index: equality prefix, then the (ts, step) sort the read issues.
    {
        "collection": EPISODES,
        "keys": [
            ("user_id", 1), ("correlation_id", 1), ("ts", 1), ("step", 1),
        ],
        "name": "ix_episodes_correlation",
    },
    {
        "collection": EPISODES,
        "keys": [("ts", 1)],
        "name": "ix_episodes_ttl",
        "kwargs": {"expireAfterSeconds": EPISODES_DEFAULT_TTL_SECONDS},
    },
    # -- semantic_cache --
    {
        "collection": SEMANTIC_CACHE,
        "keys": [("created_at", 1)],
        "name": "ix_cache_ttl",
        "kwargs": {"expireAfterSeconds": 3600},
    },
    # -- audit_log --
    {
        "collection": AUDIT_LOG,
        "keys": [("user_id", 1), ("timestamp", -1)],
        "name": "ix_audit_user_timestamp",
    },
    {
        "collection": AUDIT_LOG,
        "keys": [("timestamp", 1)],
        "name": "ix_audit_ttl",
        "kwargs": {"expireAfterSeconds": 365 * 86400},
    },
    # -- rate_limits --
    {
        "collection": RATE_LIMITS,
        "keys": [("timestamp", 1)],
        "name": "ix_rate_limits_ttl",
        "kwargs": {"expireAfterSeconds": 86400},  # 24 hours
    },
    {
        "collection": RATE_LIMITS,
        "keys": [("user_id", 1), ("operation", 1), ("timestamp", -1)],
        "name": "ix_rate_limits_user_op",
    },
    # -- governance_profiles --
    {
        "collection": GOVERNANCE_PROFILES,
        "keys": [("role", 1)],
        "name": "ix_governance_profiles_role",
        "kwargs": {"unique": True},
    },
    # -- prompts --
    {
        "collection": PROMPTS,
        "keys": [("name", 1), ("version", -1)],
        "name": "ix_prompts_name_version",
        "kwargs": {"unique": True},
    },
    # -- decisions --
    {
        "collection": DECISIONS,
        "keys": [("expires_at", 1)],
        "name": "ix_decisions_ttl",
        "kwargs": {"expireAfterSeconds": 0},
    },
    {
        "collection": DECISIONS,
        "keys": [("user_id", 1), ("key", 1)],
        "name": "ix_decisions_user_key",
        "kwargs": {"unique": True},
    },
]

# ─── Atlas Search / Vector Search Indexes ────────────────────────
#
# Created asynchronously in the background after startup.
# Each entry: collection, name, type ("vectorSearch" | "search"),
# definition (passed to SearchIndexModel).

_DEFAULT_EMBEDDING_DIMENSION = 1536


def get_search_indexes(embedding_dimension: int = _DEFAULT_EMBEDDING_DIMENSION) -> list[dict]:
    """Return Atlas Search / Vector Search index definitions.

    ``embedding_dimension`` must match the output size of the configured
    embedding provider (e.g. 1536 for Bedrock Titan, 1024 for Voyage).
    """
    return [
        # Vector search on memories.
        #
        # `memory_type` and `tags` are here because `recall` and `hybrid_search`
        # both pre-filter on them. An undeclared filter path is not an error — the
        # branch just matches nothing — so a filtered recall returned zero results
        # while the memories sat in the collection. The failure looks like "the user
        # has no memories of that type", which is indistinguishable from the truth.
        {
            "collection": MEMORIES,
            "name": "memories_vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": embedding_dimension,
                        "similarity": "cosine",
                    },
                    {"type": "filter", "path": "user_id"},
                    {"type": "filter", "path": "tier"},
                    {"type": "filter", "path": "deleted_at"},
                    {"type": "filter", "path": "memory_type"},
                    # An array of strings. `filter` fields accept arrays of the
                    # scalar types, matching when any element matches — which is
                    # what lets an all-of tag query be expressed as an `$and` of
                    # equalities. See `tag_filter` in services/memory.py.
                    {"type": "filter", "path": "tags"},
                ]
            },
        },
        # Full-text search on memories. The scoping fields are `token`, not
        # `string`: an analyzed field cannot back an exact `equals` filter.
        {
            "collection": MEMORIES,
            "name": "memories_fts_index",
            "type": "search",
            "definition": {
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "content": {"type": "string"},
                        "summary": {"type": "string"},
                        "user_id": {"type": "token"},
                        "tier": {"type": "token"},
                        "is_deleted": {"type": "token"},
                        # Declared in *both* indexes on purpose. A pre-filter that
                        # only one branch of a `$rankFusion` applies is not a
                        # filter: the unfiltered branch contributes matches that
                        # ignore it, and fusion mixes them into the same ranked
                        # list, so a `memory_type`-scoped search returned documents
                        # of every other type — visibly wrong results rather than
                        # missing ones.
                        "memory_type": {"type": "token"},
                        "tags": {"type": "token"},
                    },
                }
            },
        },
        # Vector search on episodes. Every field used as a $vectorSearch
        # pre-filter must be declared here or the branch silently returns
        # nothing — hence thread_id and agent_name alongside user_id.
        {
            "collection": EPISODES,
            "name": "episodes_vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": embedding_dimension,
                        "similarity": "cosine",
                    },
                    {"type": "filter", "path": "user_id"},
                    {"type": "filter", "path": "thread_id"},
                    {"type": "filter", "path": "agent_name"},
                ]
            },
        },
        # Full-text search on episodes. The scoping fields are `token`, not
        # `string`: an analyzed field cannot back an exact `equals` filter.
        {
            "collection": EPISODES,
            "name": "episodes_fts_index",
            "type": "search",
            "definition": {
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "search_text": {"type": "string"},
                        "user_id": {"type": "token"},
                        "thread_id": {"type": "token"},
                        "agent_name": {"type": "token"},
                    },
                }
            },
        },
        # Vector search on semantic_cache
        {
            "collection": SEMANTIC_CACHE,
            "name": "cache_vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": embedding_dimension,
                        "similarity": "cosine",
                    },
                    {"type": "filter", "path": "user_id"},
                ]
            },
        },
    ]


# Backward-compatible constant for tests that reference SEARCH_INDEXES directly
SEARCH_INDEXES: list[dict] = get_search_indexes()
