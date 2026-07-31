# Compass pipelines

The aggregations behind the memory panel. Opening one of these is the talk's "why
it works" screen: the answer the audience just watched arrive came out of *this*,
and it is one pipeline against one database.

| File | Collection | Shows |
|---|---|---|
| `01-hybrid-recall-memories.json` | `memories` | `$rankFusion` fusing a `$vectorSearch` branch and a `$search` branch: semantic recall |
| `02-hybrid-recall-episodes.json` | `episodes` | the same fusion over the activity log: "what did we actually do?" |
| `03-stm-to-ltm-promotion.json` | `memories` | the promotion criteria, as a query: which short-term memories have earned durability |
| `04-tier-inventory.json` | `memories` | one row per tier with counts and mean importance: the four-tier slide, live |

Seed a user first, or these run over an empty collection. The order matters:
seeding before the server is up leaves pipeline 03 with no rows at all, for a
reason worth knowing: see [Optional: seed a user, and mind the
order](../../README.md#optional-seed-a-user-and-mind-the-order).

## Running them

Each file is a plain aggregation array. In Compass: pick the collection →
**Aggregations** → **⋯** → *Import pipeline from plain text* → paste.

Pipelines 3 and 4 run as-is. Pipelines 1 and 2 contain a
`"queryVector": "<<EMBEDDING>>"` placeholder, because a real one is 1024 floats
and a screenful of decimals is not a thing to project. Fill it in:

```bash
# From examples/memory-ui/
uv run --extra demo python -m demo.compass_pipelines \
    --query "what can't I eat?" --user memory-demo --out /tmp/pipelines
```

That writes runnable copies with the vector substituted, then prints the results
so you know they are non-empty *before* you paste anything in front of an
audience. `--query` and `--user` should match what you are about to demo.

## Reading the scores on stage

`$rankFusion` returns a reciprocal-rank sum. With the default `k` of 60, a
document ranked **first** in one branch contributes `1/61 ≈ 0.016`. So a perfect
top hit displays as `0.0164`, and on a projector that reads as a failed match.

Say the number is a rank sum, not a similarity, and move on, or point at the
panel, which pre-renders `#1 · rrf 0.0164` for exactly this reason. Do not rescale
it into a "relevance percentage": that number would be invented, and someone in a
booth crowd at an AI conference will ask how it was computed.

## Two details worth pointing at

**Isolation is in the engine, not the application.** `user_id` appears in the
`$vectorSearch` `filter` *and* in the `$search` `compound.filter`. Neither branch
can return another user's document, so isolation does not depend on a caller
remembering to add a `WHERE` clause.

**The `$addFields` stage is not decoration.** `$rankFusion` does not project its
own fused rank. Worse, writing `{"score": {"$meta": "score"}}` inside an exclusion
`$project` does not error, it silently yields `null`. It has to be its own stage.
Verified against Atlas 8.3; this is the kind of thing that costs an afternoon.
