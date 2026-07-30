# Tests

```bash
uv run pytest tests/unit -q          # no network, no Atlas, no credentials
uv run pytest -q                     # adds integration, which self-skips
uv run ruff check                    # whole repo, pinned in the dev extra
```

CI runs exactly the first and third of those. The unit suite is the gate.

## The `REQ-*` labels

Most test docstrings open with an id — `REQ-E-108`, `REQ-DB-002`. They are stable
names for behaviours, so that a test asserting one, a second test asserting its
inverse, and the comment in the source explaining why it holds are all reachable
from each other:

```bash
grep -rn REQ-E-108 tests/ agent_memory/
```

Nothing reads the ids at runtime, and they do not index into a document. A test
with no label is not a lesser test; plenty of them assert something that needs no
name.

The prefix says which surface, and the numbers group by area:

| Range | Area |
|---|---|
| `REQ-E-001`..`009` | Exceptions, the ranking formula, memory evolution and merge |
| `REQ-E-010`..`019` | `MemoryConfig` and `from_env()` |
| `REQ-E-020`..`039` | `AsyncMemory` facade — `create`, `recall`, `remember` |
| `REQ-E-040`..`049` | The synchronous `Memory` wrapper |
| `REQ-E-050`..`059` | OpenAI and Anthropic providers, `ProviderManager` |
| `REQ-E-060`..`069` | MCP shell |
| `REQ-E-070`..`079` | REST shell |
| `REQ-E-080`..`083` | Dual transport, packaging, the export surface |
| `REQ-E-084`..`089` | Audit redaction, auth, tool errors, response size limits |
| `REQ-E-090`..`099` | Episodic projection, context, correlation ids |
| `REQ-E-100`..`129` | The episodic write path and LLM reply handling |
| `REQ-E-140`..`150` | Prompt contract, promotion retention, identity binding, search filters, rate limits, refusals |
| `REQ-E-160`..`172` | Importance scoring — features, artifact, training, selection |
| `REQ-DB-*` | Collections and index migrations |
| `REQ-EC-*` | Memory and cache service edge cases |
| `REQ-VP-*` | The Voyage embedding provider |

The ranges have gaps, and `REQ-E-130`..`139` is empty entirely. Leave the holes:
handing a retired id to an unrelated behaviour makes every older grep result and
commit message point at the wrong thing.

## Unit tests take no credentials

`tests/unit/` runs offline. It needs no Atlas, no provider key, and no `.env`.
Construct config explicitly when a test needs one:

```python
MCPConfig(**defaults, _env_file=None)      # or MemoryConfig, which subclasses it
```

The `_env_file=None` is not decoration. Both are pydantic-settings models and
read `.env` on their own, so on a developer machine that is configured for real,
omitting it lets a live connection string and a live API key into the test —
which passes locally, fails in CI, and for the interval in between is a test
asserting something about your deployment rather than about the code.

The same trap applies to the demo modules under `examples/memory-ui/`. Both
`server/app.py` and `demo/seed.py` call `load_dotenv()` at module scope, which is
correct for an application and means importing one from a test loads the
repository root's real `.env`. Import them only through a fixture that patches
`dotenv.load_dotenv` first — `tests/unit/test_demo_seed_reset.py` has one.

## Integration tests skip themselves

`tests/integration/` needs a reachable MCP server and a configured Atlas. When
either is missing the tests are skipped, not failed, so a plain `pytest` stays
green on a laptop with no infrastructure. Point them somewhere with `MCP_HOST`
and `MCP_PORT` (default `localhost:8000`).

Skipping is the right default here and it is also the failure mode to know
about: a green run tells you nothing about the integration suite unless you
check that it actually ran. `-q` reports the skip count; a run that should have
exercised Atlas and reports skips has a configuration problem, not a passing
suite.

## Mutation testing

Several behaviours in the episodic write path were verified by breaking them on
purpose and confirming a test failed. The method matters more than the tooling:
check out the **commit**, not the working tree, into a worktree outside the
repository; patch one line; run the suite; restore.

A mutation that survives is information about the test, not only about the code.
Gathering the step-counter calls (`REQ-E-103`) survived its first run, and the
guard was fine — the test's fake was delaying the provider's *reply*, and reply
latency reorders nothing, because `asyncio.gather` starts its coroutines in order.
The real wait is for a connection from the pool, before the `$inc` is claimed. Move
the delay there and the mutation dies.
