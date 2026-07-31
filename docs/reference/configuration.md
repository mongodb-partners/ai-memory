# Reference: configuration

Every setting, its default, and what it controls. Two classes back this surface:
`MCPConfig` (`agent_memory/core/config.py`) holds the substrate, and
`MemoryConfig` (`agent_memory/config.py`) extends it with the provider and
lifecycle fields. `MemoryConfig` is the public one — construct it directly, or
build it from the environment:

```python
from agent_memory import MemoryConfig

MemoryConfig(mongodb_connection_string="mongodb+srv://...")   # in code
MemoryConfig.from_env()                                        # from the environment
```

Environment names are the field names, upper-cased, and matching is
**case-insensitive**. Unknown environment variables are ignored rather than
rejected. `mongodb_connection_string` is the only required field.

The library itself never reads a `.env` file on your behalf — but these are
pydantic-settings models, so *they* do: `env_file` is `.env`, so constructing a
config in a directory that has one picks it up. That is why tests pass
`_env_file=None`; see [tests/README.md](../../tests/README.md).

## Server and transport

| Setting | Default | Notes |
|---|---|---|
| `app_name` | `agent-memory` | Declared, but nothing reads it. Both shells use the literal name |
| `app_version` | the installed version | Defaults from the package rather than a literal. The OpenAPI document reads the package directly, so overriding this does not change what `/openapi.json` reports |
| `port` | `8000` | |
| `host` | `127.0.0.1` | Loopback. Binding a routable address without auth is refused — see [Deployment](../how-to/deployment.md) |
| `transport` | `streamable-http` | `mcp` \| `rest` \| `both`. `streamable-http` and `stdio` are legacy aliases for `mcp` |
| `debug` | `False` | |
| `allow_unauthenticated_network_access` | `False` | The operator's assertion that serving unauthenticated on a routable address is intended. Warns on every start |

`transport` is the field behind `TRANSPORT`. Any value outside the set above
raises `ValueError` at startup rather than defaulting.

## MongoDB

| Setting | Default | Notes |
|---|---|---|
| `mongodb_connection_string` | — | **Required** |
| `mongodb_database_name` | `agent_memory` | |
| `mongodb_max_pool_size` | `20` | |
| `mongodb_min_pool_size` | `2` | |

The client is shared per process and reference-counted, so several facades in one
process share one pool.

## Embedding provider

| Setting | Default | Notes |
|---|---|---|
| `embedding_provider` | `bedrock` | `bedrock` \| `voyage` \| `openai` |
| `embedding_model` | `amazon.titan-embed-text-v1` | |
| `embedding_dimension` | `1536` | Auto-aligned to the model for Voyage, but **only** while still at this default. Pin it and you own it |
| `allow_embedding_dimension_change` | `False` | Startup refuses a dimension change that would orphan stored vectors |

A vector index cannot have its `numDimensions` edited, so a changed dimension
means dropping and rebuilding the index — which leaves every already-stored
vector at the old width and unreturnable by `$vectorSearch`, with no error. That
is why the guard exists and why it is on by default.

## LLM provider

| Setting | Default | Notes |
|---|---|---|
| `llm_provider` | `bedrock` | `bedrock` \| `openai` \| `anthropic` |
| `llm_model` | `global.anthropic.claude-sonnet-5` | A cross-region inference profile, not a bare model id |

The LLM does importance scoring and summarization — short, high-volume calls
where latency matters more than depth.

## AWS (Bedrock)

| Setting | Default | Notes |
|---|---|---|
| `aws_region` | `us-east-1` | |
| `aws_access_key_id` | `None` | Falls back to the standard boto3 credential chain |
| `aws_secret_access_key` | `None` | Same |

## Voyage AI

| Setting | Default | Notes |
|---|---|---|
| `voyage_api_key` | `None` | |
| `voyage_base_url` | `https://api.voyageai.com/v1/embeddings` | Set to `https://ai.mongodb.com/v1/embeddings` for the Atlas gateway |
| `voyage_model` | `voyage-3` | |

The key decides the endpoint: a Voyage key against the Atlas gateway (or the
reverse) fails with a 403. Gateway models are 1024 dimensions — `voyage-3-lite`
is 512 — against the `1536` default.

## OpenAI

| Setting | Default | Notes |
|---|---|---|
| `openai_api_key` | `None` | |
| `openai_base_url` | `None` | Set for a compatible gateway |
| `openai_model` | `gpt-4o` | |
| `openai_embedding_model` | `text-embedding-3-small` | 1536 dimensions; `-3-large` is 3072 |

Needs the `openai` extra.

## Anthropic

| Setting | Default | Notes |
|---|---|---|
| `anthropic_api_key` | `None` | |
| `anthropic_base_url` | `None` | Set for a compatible gateway |
| `anthropic_model` | `claude-sonnet-5` | |

LLM only — Anthropic has no embeddings API. Needs the `anthropic` extra.

## Memory lifetimes

| Setting | Default | Expires |
|---|---|---|
| `stm_ttl_hours` | `24` | Short-term memories |
| `ltm_retention_critical_days` | `365` | Long-term, `critical` retention tier |
| `ltm_retention_reference_days` | `180` | Long-term, `reference` |
| `ltm_retention_standard_days` | `90` | Long-term, `standard` — where promotion lands |
| `ltm_retention_temporary_days` | `7` | Long-term, `temporary` |
| `episodic_retention_days` | `30` | Logged turns |
| `cache_ttl_seconds` | `3600` | Semantic-cache entries |
| `audit_retention_days` | `365` | Audit records |
| `soft_delete_purge_days` | `30` | Soft-deleted memories, permanently |
| `rate_limit_retention_seconds` | `86400` | Spent rate-limit window counters |
| `decision_default_ttl_days` | `90` | Sticky decisions, when none is given |

**Shortening a retention deletes data.** Startup reconciles each TTL index to
the configuration, and the rebuilt index applies to documents already stored, so
anything past the new cutoff is expired by Atlas's TTL monitor within a minute or
two of the restart — no confirmation step. Lengthening is safe.

The long-term durations work differently from the rest: they are applied as a
per-document `expires_at` at write time, not as one collection-wide duration, so
changing them affects memories written afterwards only.

`rate_limit_retention_seconds` is raised to `rate_limit_window_seconds` when that
is longer — a counter *is* the enforcement state, so expiring it inside its own
window would reset a caller who had exhausted the limit.

## What becomes long-term

| Setting | Default | Notes |
|---|---|---|
| `ltm_candidate_min_chars` | `31` | The shortest **human** message that becomes a long-term candidate |

Comparison is `>=`, so the value reads as "the shortest length that qualifies".
A message below it is stored as short-term but never enriched, promoted, or
returned by `recall`. `0` keeps every human message, which means one LLM
enrichment per turn. Assistant messages are never candidates at any threshold.

`scripts/train_importance.py` reads this same value to filter its corpus.

## Memory evolution and consolidation

| Setting | Default | Notes |
|---|---|---|
| `reinforce_threshold` | `0.85` | Similarity above which a new memory reinforces an existing one |
| `merge_threshold` | `0.70` | Similarity above which two memories are queued for merge |
| `consolidation_interval_hours` | `24` | |
| `stm_compression_age_hours` | `24` | |
| `forgetting_score_threshold` | `0.1` | Below this importance, a memory is deleted |
| `promotion_importance_threshold` | `0.6` | At or above this importance, a memory is promoted |
| `promotion_access_threshold` | `2` | Access count required for the access-based promotion path |
| `promotion_age_minutes` | `60` | |

The two thresholds are **absolute**, which is what makes importance-scorer
calibration matter rather than just ranking quality — see below.

## Enrichment

| Setting | Default | Notes |
|---|---|---|
| `enrichment_interval_seconds` | `30` | |
| `enrichment_batch_size` | `50` | |
| `enrichment_concurrency` | `5` | |
| `enrichment_max_retries` | `3` | |
| `enrichment_lease_seconds` | `300` | How long a claimed enrichment stays claimed |

The claim is a lease, not a lock: a worker that dies mid-LLM-call would otherwise
leave a document nobody enriches again. The floor is the slowest legitimate
enrichment — a merge makes an LLM call and then an embedding call.

## Importance scoring

| Setting | Default | Notes |
|---|---|---|
| `importance_scorer` | `llm` | `local` evaluates a logistic model in-process instead |
| `importance_model_path` | `None` | Explicit coefficient artifact. Unset auto-selects a bundled one |

A path pointing at a missing file refuses to start rather than falling back to an
artifact the operator did not ask for. Check `forget_agreement` and
`promote_agreement` in the artifact's `training.metrics` against your thresholds
before switching a production deployment: a model that ranks well but sits
systematically low forgets more and promotes less, and the symptom is degraded
recall weeks later rather than an error.

## Retrieval and ranking

| Setting | Default | Notes |
|---|---|---|
| `ranking_alpha` | `0.2` | Re-ranking weight |
| `ranking_beta` | `0.3` | Re-ranking weight |
| `ranking_gamma` | `0.5` | Re-ranking weight |
| `rrf_k` | `60` | Reciprocal-rank-fusion constant. A first-place document scores ~1/61 |
| `rrf_vector_weight` | `1.0` | Weight of the `$vectorSearch` branch |
| `rrf_text_weight` | `0.7` | Weight of the `$search` branch |
| `max_results_per_query` | `100` | |
| `max_response_bytes` | `16777216` | 16 MiB ceiling on a response |

## Semantic cache

| Setting | Default | Notes |
|---|---|---|
| `cache_ttl_seconds` | `3600` | |
| `cache_similarity_threshold` | `0.95` | Below this, a lookup is a miss |

## Episodic memory

| Setting | Default | Notes |
|---|---|---|
| `episodic_enabled` | `True` | `False` accepts and discards, so callers need no conditionals |
| `episodic_queue_size` | `1000` | Bounded. When full, the **oldest** pending turn is dropped |
| `episodic_batch_size` | `20` | Turns per `insert_many` |
| `episodic_flush_interval_seconds` | `1.0` | Max wait before writing a partial batch |
| `episodic_content_cap` | `4000` | Per-message character cap. Truncation is marked in the stored text |
| `episodic_search_text_cap` | `2000` | Cap on the text that gets embedded |
| `episodic_embed_final_steps_only` | `True` | A mid-turn step has a question but no answer worth embedding |
| `episodic_shutdown_timeout_seconds` | `5.0` | How long `close()` waits for the queue to drain |
| `episodic_retention_days` | `30` | Listed under lifetimes above |

See [the episodic document shape](episodic-document-shape.md) for what these
produce and [Observability](../how-to/observability.md) for the counters.

## Audit

| Setting | Default | Notes |
|---|---|---|
| `audit_buffer_size` | `10` | Entries buffered before a flush |
| `audit_flush_interval_seconds` | `60` | |
| `audit_flush_on_write` | `False` | |
| `audit_retention_days` | `365` | |
| `audit_fallback_path` | `audit_fallback.jsonl` | Where entries go when MongoDB refuses them. Resolved to an absolute path once, at startup. `""` discards them |
| `audit_fallback_max_bytes` | `52428800` | 50 MiB, then rotation to a single `.1` sibling — so twice this on disk. `0` disables rotation |

Set `audit_fallback_path` explicitly. The default is relative, which means
"wherever the process was started" — a systemd unit's `WorkingDirectory`, a
container's `WORKDIR`, a developer's shell.

## Authentication

| Setting | Default | Notes |
|---|---|---|
| `auth_enabled` | `False` | |
| `auth_token_header` | `Authorization` | |
| `auth_secret` | `""` | HS256 secret. `auth_enabled=True` with an empty secret **refuses to construct** |
| `auth_token_expiry_seconds` | `86400` | |
| `auth_user_id_claim` | `sub` | Which JWT claim carries the identity |
| `auth_role_claim` | `role` | Which claim carries the governance role |
| `auth_default_role` | `end_user` | Used when no role claim is present |
| `require_auth_for_multi_tenant` | `False` | Refuse to serve with auth off |

API keys come from the `MEMORY_MCP_API_KEYS` environment variable — not a config
field — formatted `key1=user1,key2=user2`.

With auth off, the caller supplies its own `user_id` and that is all there is;
with auth on, the token decides and a request naming anyone else is refused. See
[Governance](governance.md).

## Governance and rate limiting

| Setting | Default | Notes |
|---|---|---|
| `governance_enabled` | `False` | |
| `governance_default_profile` | `default` | No profile is named `default`, so this falls through to `end_user` |
| `governance_cache_ttl_seconds` | `300` | Profile cache lifetime |
| `rate_limit_enabled` | `False` | |
| `rate_limit_window_seconds` | `60` | |
| `rate_limit_max_requests` | `100` | Overridden by a governance profile's per-day quota when one applies |
| `rate_limit_retention_seconds` | `86400` | Listed under lifetimes above |

The limiter is a **fixed** window, so it can admit up to `2 × max` across a
boundary. That is the standard fixed-window trade, taken deliberately: the
alternatives are a racy count over per-request documents or a much more
expensive structure.

## Auto-capture (MCP only)

| Setting | Default | Notes |
|---|---|---|
| `auto_capture_enabled` | `True` | |
| `auto_capture_tools` | `["recall_memory", "hybrid_search", "store_decision", "recall_decision"]` | |
| `auto_capture_min_length` | `30` | |
| `auto_capture_max_content_length` | `2000` | |

Auto-capture is MCP-only by design; REST is the explicit-control surface.

## Prompt library

| Setting | Default | Notes |
|---|---|---|
| `prompt_experiment_enabled` | `True` | |
| `prompt_cache_ttl_seconds` | `300` | |

## Sticky decisions

| Setting | Default | Notes |
|---|---|---|
| `decision_stickiness_enabled` | `False` | |
| `decision_default_ttl_days` | `90` | |

## Worker lifecycle

| Setting | Default | Notes |
|---|---|---|
| `workers_in_process` | `True` | `False` → an external runtime owns background work |
| `await_search_indexes` | `False` | `True` blocks `create()` until Atlas Search indexes are queryable |

`workers_in_process=False` disables all four workers — enrichment,
consolidation, audit flush, and the episodic writer. Without a consumer,
`log_activity` fills its bounded queue and then discards the oldest turns, so set
`episodic_enabled=False` alongside it when that is what you intend.

Set `await_search_indexes=True` in short-lived scripts, or the process can exit
before its indexes are queryable and search returns nothing.

## Validated combinations

Two configurations are refused at construction or startup rather than degraded:

- `auth_enabled=True` with an empty `auth_secret` — this used to log a warning
  and serve every route unauthenticated.
- An `embedding_dimension` that disagrees with what the provider returns, or a
  change that would orphan stored vectors without
  `allow_embedding_dimension_change=True`.

And one is refused by the runner rather than the model: binding a routable
address with `auth_enabled=False`. The check cannot live on the config class,
because the same class configures the library used in-process, where auth-off is
correct and no socket exists.

## See also

- [Deployment](../how-to/deployment.md) — which of these matter when serving
- [Governance](governance.md) — profiles, quotas, identity
- [Configure retention](../how-to/configure-ttl.md) — changing episodic TTL at runtime
- `.env.example` — a commented starting point for the deployed shells
