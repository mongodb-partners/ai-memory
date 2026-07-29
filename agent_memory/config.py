"""Programmatic configuration for agent-memory.

``MemoryConfig`` is the public, code-constructible config object. It extends the
substrate's ``MCPConfig`` (a Pydantic ``BaseSettings``) so it stays
backward-compatible with memory-mcp's environment-variable names, while adding
the SP3 fields: OpenAI/Anthropic provider settings and the ``workers_in_process``
lifecycle seam.

- Library callers build it directly: ``MemoryConfig(mongo_uri=..., ...)``.
- Deployed shells build it from the environment: ``MemoryConfig.from_env()``.
"""

from __future__ import annotations

from agent_memory.core.config import MCPConfig


class MemoryConfig(MCPConfig):
    """Programmatic config object. Superset of memory-mcp's ``MCPConfig``."""

    # OpenAI provider (LLM + embeddings; base_url enables the Grove gateway)
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # Anthropic provider (LLM only; no embeddings API)
    anthropic_api_key: str | None = None
    anthropic_base_url: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"

    # Worker lifecycle seam (SP1). True → run enrichment/consolidation/audit-flush
    # in-process. False → an external runtime (Atlas Triggers / worker process)
    # owns reactive work; in SP3 this is purely a disable switch.
    workers_in_process: bool = True

    # When True, create() blocks until Atlas Search indexes are queryable before
    # returning. Right for short-lived library/script callers (otherwise the
    # process can exit before background index creation finishes, making
    # search/recall silently return nothing). Default False keeps the
    # non-blocking background behaviour suited to long-running servers.
    await_search_indexes: bool = False

    # ─── Episodic memory (the agent activity log) ──────────────────
    #
    # The write path is fire-and-forget onto a bounded queue, so these knobs
    # trade memory footprint and staleness against write amplification. The
    # defaults suit a chat-shaped workload: a few turns per second per process.

    # Master switch. False → log_activity is accepted and discarded, so callers
    # do not need conditionals around it.
    episodic_enabled: bool = True

    # Bounded queue depth. When full, the *oldest* pending turn is evicted so
    # the newest is always retained — a stale turn is worth less than a fresh one.
    episodic_queue_size: int = 1000

    # Turns per insert_many. Higher amortizes round trips; lower bounds how much
    # is lost if the process dies without a flush.
    episodic_batch_size: int = 20

    # How long the worker waits for a full batch before writing a partial one.
    episodic_flush_interval_seconds: float = 1.0

    # Per-message content cap, in characters. Truncation is marked in the stored
    # text, so a reader can tell something was cut.
    episodic_content_cap: int = 4000

    # Cap on the text that gets embedded (first question + last answer). Kept
    # well under the content cap because embedding cost is per token.
    episodic_search_text_cap: int = 2000

    # Embed only final steps — a mid-turn step ending in a tool request has no
    # answer text worth searching, so its vector would represent half a turn.
    # False embeds every logged turn instead: more recall surface, more cost.
    episodic_embed_final_steps_only: bool = True

    # How long close() waits for queued turns to reach Atlas. Bounded, because a
    # hung shutdown is worse than a few lost log entries.
    episodic_shutdown_timeout_seconds: float = 5.0

    @classmethod
    def from_env(cls, **overrides) -> "MemoryConfig":
        """Build a config from environment variables (deployed-shell path).

        Pydantic ``BaseSettings`` already reads the environment; this classmethod
        is the explicit, named entry point the shells call. ``llm_provider`` and
        ``embedding_provider`` default to ``bedrock`` (inherited from ``MCPConfig``).
        """
        return cls(**overrides)
