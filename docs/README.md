# Documentation

Start with the [project README](../README.md) for installation and a quickstart.
These pages go deeper, organized by what you are trying to do.

## Reference: the contracts

- [Configuration](reference/configuration.md): every setting, its default, and
  what it controls, including the two combinations that are refused rather than
  degraded.
- [MCP tools](reference/mcp-tools.md): all sixteen tools, the response envelope,
  why a refusal is a return value rather than an exception, and auto-capture.
- [REST API](reference/rest-api.md): all twelve routes, the status-code mapping,
  and where the REST surface is narrower than the library.
- [Memory document shape](reference/memory-document-shape.md): what a memory
  looks like in the `memories` collection, the STM/LTM pair, the enrichment state
  machine, and how `recall` ranks.
- [Episodic document shape](reference/episodic-document-shape.md): what a logged
  turn looks like in the `episodes` collection, field by field, plus the index
  set and the two index settings that fail silently if you get them wrong.
- [Identity, governance, and rate limiting](reference/governance.md): the three
  profiles and their quotas, the order the access check runs in, and the two
  guards that hold even with governance switched off.

## How-to: task-shaped

- [Deploy the server](how-to/deployment.md): transport, auth, Docker, probes,
  and why the runner refuses to bind a routable address with auth disabled.
- [Monitor the episodic writer](how-to/observability.md): the counters, what
  each one means when it moves, and how to alert on them. Episodic logging is
  fire-and-forget, so nothing raises when it degrades.
- [Configure retention](how-to/configure-ttl.md): change the TTL in place with
  `collMod`, what TTL does and does not guarantee, and how to pick a number.
- [Scope memory to a user](how-to/per-user-scoping.md): `scoped_user` for the
  gap between the code that knows the user and the code that logs the turn.
- [Run the sample UI](../examples/memory-ui/README.md): the memory ON vs OFF
  demo, what each panel shows, and the presenter's preflight.

## Explanation: the why

- [Why episodic memory](explanation/why-episodic-memory.md): the tier most
  memory libraries skip, what it answers that a vector store cannot, and what it
  costs.
- [Architecture](explanation/architecture.md): facade over stateless services,
  the one deliberate exception to the audit path, and the seven properties holding
  the episodic write path together.

## Reading the tests

The test suite is the other half of the documentation, and the parts of it worth
knowing before you read it (what the `REQ-*` ids in the docstrings mean, why unit
tests pass `_env_file=None`, and why the integration suite skips itself) are in
[tests/README.md](../tests/README.md).
