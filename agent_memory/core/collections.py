"""Collection names and index definitions for Memory-MCP Phase 0.

Index definitions are separated from migration logic so they serve as
the canonical reference for the database schema.
"""

# ─── Collection Names ────────────────────────────────────────────

MEMORIES: str = "memories"
SEMANTIC_CACHE: str = "semantic_cache"
AUDIT_LOG: str = "audit_log"
RATE_LIMITS: str = "rate_limits"
GOVERNANCE_PROFILES: str = "governance_profiles"
PROMPTS: str = "prompts"
DECISIONS: str = "decisions"

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
        # Vector search on memories
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
                ]
            },
        },
        # Full-text search on memories
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
