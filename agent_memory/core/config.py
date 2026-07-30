"""Centralized configuration via Pydantic BaseSettings."""

from typing import ClassVar

from pydantic import model_validator
from pydantic_settings import BaseSettings

from agent_memory.version import __version__


class MCPConfig(BaseSettings):
    """Configuration for the agent-memory shells and services.

    All values can be overridden via environment variables (case-insensitive).
    ``mongodb_connection_string`` is the only required field.
    """

    # Server
    app_name: str = "agent-memory"
    # Read from the installed package rather than hardcoded: this value is
    # served by /health and stamped on audit records, and a second copy of the
    # version is a copy that goes stale.
    app_version: str = __version__
    port: int = 8000
    # The address the deployed shells bind. Loopback by default: a process that
    # binds every interface is reachable by anything that can route to the host,
    # and that is a deployment decision rather than a default. `0.0.0.0` was
    # hardcoded at every `uvicorn.run` call, so there was no way to ask for less.
    host: str = "127.0.0.1"
    transport: str = "streamable-http"
    debug: bool = False
    # The operator's assertion that serving unauthenticated on a routable
    # address is intended. Read only by `shells.runner.run` — see
    # `_refuse_to_serve_open` there for why the check cannot live in this model.
    allow_unauthenticated_network_access: bool = False

    # MongoDB
    mongodb_connection_string: str
    mongodb_database_name: str = "agent_memory"
    mongodb_max_pool_size: int = 20
    mongodb_min_pool_size: int = 2

    # Embedding Provider
    embedding_provider: str = "bedrock"
    embedding_model: str = "amazon.titan-embed-text-v1"
    embedding_dimension: int = 1536
    # The operator's assertion that changing the embedding dimension out from
    # under existing vectors is intended.
    #
    # A vector index cannot have its `numDimensions` edited, so reconciliation
    # drops and recreates it. The documents are untouched — and that is the
    # problem: every vector already stored is the old width, and the rebuilt index
    # will not return any of them from `$vectorSearch`. Nothing raises, no count
    # changes, and `find` still shows every memory. Recall simply goes empty for
    # the entire history while continuing to work for anything written after.
    #
    # Recovery means re-embedding every document, which needs the *old* provider
    # config that the operator has by then already replaced. So the default is to
    # refuse and say what to do, and this flag is how someone who has read that
    # and means it proceeds anyway.
    allow_embedding_dimension_change: bool = False

    # LLM Provider
    llm_provider: str = "bedrock"
    # A cross-region inference profile, not a bare model id: the newest Claude
    # models on Bedrock are only invocable through one. Sonnet rather than Opus
    # because the LLM here does importance scoring and summarization — short,
    # high-volume calls where latency matters more than depth.
    llm_model: str = "global.anthropic.claude-sonnet-5"

    # AWS (Bedrock)
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # Voyage AI
    # voyage_base_url selects the endpoint: the public Voyage API (default) or
    # the MongoDB Atlas embeddings gateway (https://ai.mongodb.com/v1/embeddings),
    # both via the VOYAGE_BASE_URL env var. The payload format is identical.
    voyage_api_key: str | None = None
    voyage_base_url: str = "https://api.voyageai.com/v1/embeddings"
    voyage_model: str = "voyage-3"

    # Memory Lifecycle
    stm_ttl_hours: int = 24
    ltm_retention_critical_days: int = 365
    ltm_retention_reference_days: int = 180
    ltm_retention_standard_days: int = 90
    ltm_retention_temporary_days: int = 7
    # How long a logged turn is kept. A turn log is high-volume and loses value
    # fast, so 30 days covers "what did we do last month" without unbounded
    # growth.
    #
    # This is the *declared* retention: startup reconciles the TTL index on
    # `episodes` to it. `set_activity_retention` changes the same index at
    # runtime via collMod, which is the right shape for "shorten this for the
    # next hour" but does not survive a restart — the next startup reconciles
    # back to the value here. Set this field to make a change permanent.
    episodic_retention_days: int = 30

    # Memory Evolution Thresholds
    reinforce_threshold: float = 0.85
    merge_threshold: float = 0.70

    # Retrieval Ranking Weights
    ranking_alpha: float = 0.2
    ranking_beta: float = 0.3
    ranking_gamma: float = 0.5

    # RRF Parameters
    rrf_k: int = 60
    rrf_vector_weight: float = 1.0
    rrf_text_weight: float = 0.7

    # Query Limits
    max_results_per_query: int = 100
    max_response_bytes: int = 16_777_216

    # Cache
    cache_ttl_seconds: int = 3600
    cache_similarity_threshold: float = 0.95

    # Consolidation (Phase 1)
    consolidation_interval_hours: int = 24
    stm_compression_age_hours: int = 24
    forgetting_score_threshold: float = 0.1
    promotion_importance_threshold: float = 0.6
    promotion_access_threshold: int = 2
    promotion_age_minutes: int = 60

    # Enrichment
    enrichment_interval_seconds: int = 30
    enrichment_batch_size: int = 50
    enrichment_concurrency: int = 5
    enrichment_max_retries: int = 3
    # How long a claimed enrichment stays claimed. A worker that dies mid-LLM-call
    # leaves its claim behind, and without an expiry that document would never be
    # enriched again — so the claim is a lease, not a lock, and any worker may take
    # over one this old.
    #
    # The floor is the slowest legitimate enrichment: a merge makes an LLM call and
    # then an embedding call, so a lease shorter than that would let a second worker
    # start work the first is still doing. 300s is roughly an order of magnitude
    # above a normal completion, which buys recovery within five minutes of a crash
    # while leaving no realistic chance of stealing live work.
    enrichment_lease_seconds: int = 300

    # Importance Scoring
    # "llm" (default) makes one LLM call per long-term memory. "local" evaluates
    # a small logistic model over the embedding that already exists by then —
    # microseconds, no network, no tokens. Default stays "llm" so an upgrade
    # changes nothing.
    importance_scorer: str = "llm"
    # Path to a JSON coefficient artifact. None means auto-select the bundled
    # artifact matching the configured embedder, falling back to the lexical one.
    importance_model_path: str | None = None

    # Audit
    audit_buffer_size: int = 10
    audit_flush_interval_seconds: int = 60
    audit_flush_on_write: bool = False
    audit_retention_days: int = 365

    # Soft Delete
    soft_delete_purge_days: int = 30

    # Identity & Auth (Phase 2)
    auth_enabled: bool = False
    auth_token_header: str = "Authorization"
    auth_secret: str = ""
    auth_token_expiry_seconds: int = 86400
    # Which JWT claim carries the identity a request acts as. Read by
    # `auth.identity.resolve_caller`, which is the only thing that decides the
    # caller's `user_id` — request bodies do not.
    auth_user_id_claim: str = "sub"
    auth_role_claim: str = "role"
    auth_default_role: str = "end_user"
    # Refuse to serve with authentication off. Off is the right default for a
    # library used in-process by a single-tenant app, and the wrong one for a
    # shell listening on a port — but the process cannot tell which it is, so
    # this is the operator's assertion that unauthenticated is intended.
    require_auth_for_multi_tenant: bool = False

    # Governance (Phase 2)
    governance_enabled: bool = False
    governance_default_profile: str = "default"
    governance_cache_ttl_seconds: int = 300
    rate_limit_enabled: bool = False
    rate_limit_window_seconds: int = 60
    rate_limit_max_requests: int = 100
    # How long a spent window counter is kept before Atlas expires it. Only
    # needs to outlive its own window; the generous default leaves the recent
    # ones readable while debugging a limit that fired. The TTL index is
    # reconciled to `max(this, rate_limit_window_seconds)` so a long window
    # cannot have its own counters expired out from under it mid-window.
    rate_limit_retention_seconds: int = 86400

    # Prompt Library (Phase 2)
    prompt_experiment_enabled: bool = True
    prompt_cache_ttl_seconds: int = 300

    # Auto-Capture (Phase 2)
    auto_capture_enabled: bool = True
    auto_capture_tools: list[str] = [
        "recall_memory", "hybrid_search",
        "store_decision", "recall_decision",
    ]
    auto_capture_min_length: int = 30
    auto_capture_max_content_length: int = 2000

    # Decision Stickiness (Phase 2)
    decision_stickiness_enabled: bool = False
    decision_default_ttl_days: int = 90

    model_config = {
        "env_prefix": "",
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
    }

    @model_validator(mode="after")
    def _auth_must_not_fail_open(self):
        """Refuse the configurations that silently disable authentication.

        ``AUTH_ENABLED=true`` with an empty ``AUTH_SECRET`` used to log a warning
        and serve every route unauthenticated. That is the worst available
        outcome: the operator asked for auth, the deployment reports healthy, and
        the only evidence is one line in a log written at startup. An operator who
        sets the flag has stated their intent, and a missing secret is an
        incomplete deployment rather than a request to turn the feature off.

        ``REQUIRE_AUTH_FOR_MULTI_TENANT=true`` is the inverse assertion — refuse to
        start *without* auth — for deployments where unauthenticated access is
        never acceptable regardless of how the flags end up set.
        """
        if self.auth_enabled and not self.auth_secret:
            raise ValueError(
                "AUTH_ENABLED=true requires AUTH_SECRET. Refusing to start with "
                "authentication silently disabled — set the secret, or set "
                "AUTH_ENABLED=false to accept unauthenticated access explicitly."
            )
        if self.require_auth_for_multi_tenant and not self.auth_enabled:
            raise ValueError(
                "REQUIRE_AUTH_FOR_MULTI_TENANT=true requires AUTH_ENABLED=true. "
                "Without authentication every caller supplies its own user_id, so "
                "tenant isolation cannot be enforced."
            )
        return self

    # ClassVar keeps this out of the model's fields. An un-annotated class
    # attribute would mostly work, but Pydantic v2 treats underscore-prefixed
    # attributes specially, and ClassVar states the intent outright.
    _IMPORTANCE_SCORERS: ClassVar[tuple[str, ...]] = ("llm", "local")

    @model_validator(mode="after")
    def _importance_scorer_must_be_known(self):
        """Refuse an unrecognized scorer rather than falling back to the LLM.

        Same reasoning as ``_auth_must_not_fail_open``: the operator has stated an
        intent, and a typo that silently keeps the old path produces no symptom
        except the invoice. ``IMPORTANCE_SCORER=locl`` would run every enrichment
        through the LLM while the deployment reports healthy.

        Normalizes case and surrounding whitespace first — ``IMPORTANCE_SCORER=Local``
        in a hand-edited ``.env`` is a correct intent expressed slightly wrong, and
        there is nothing to protect by rejecting it.
        """
        normalized = (self.importance_scorer or "").strip().lower()
        if normalized not in self._IMPORTANCE_SCORERS:
            raise ValueError(
                f"IMPORTANCE_SCORER={self.importance_scorer!r} is not recognized. "
                f"Valid values: {', '.join(self._IMPORTANCE_SCORERS)}. Refusing to "
                "start on a scorer the operator did not ask for."
            )
        if normalized != self.importance_scorer:
            # `mode="after"` gets a constructed model, so plain assignment is the
            # way to normalize. `validate_assignment` is not set on this model, so
            # this cannot recurse back into the validator.
            self.importance_scorer = normalized
        return self
