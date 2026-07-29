# Documentation

Start with the [project README](../README.md) for installation and a quickstart.
These pages go deeper, organized by what you are trying to do.

## Reference — the contracts

- [Episodic document shape](reference/episodic-document-shape.md) — what a logged
  turn looks like in the `episodes` collection, field by field, plus the index
  set and the two index settings that fail silently if you get them wrong.

## How-to — task-shaped

- [Monitor the episodic writer](how-to/observability.md) — the counters, what
  each one means when it moves, and how to alert on them. Episodic logging is
  fire-and-forget, so nothing raises when it degrades.
- [Configure retention](how-to/configure-ttl.md) — change the TTL in place with
  `collMod`, what TTL does and does not guarantee, and how to pick a number.
- [Scope memory to a user](how-to/per-user-scoping.md) — `scoped_user` for the
  gap between the code that knows the user and the code that logs the turn.

## Explanation — the why

- [Why episodic memory](explanation/why-episodic-memory.md) — the tier most
  memory libraries skip, what it answers that a vector store cannot, and what it
  costs.
- [Architecture](explanation/architecture.md) — facade over stateless services,
  the one deliberate exception to the audit path, and the six properties holding
  the episodic write path together.

## Specs

`superpowers/specs/` holds the design documents and EARS requirements the
implementation was built against. They record decisions rather than describing
current behaviour — for that, prefer the pages above.
