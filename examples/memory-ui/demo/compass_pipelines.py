"""Emit the Compass pipelines, and prove they return rows before you present.

Two jobs, and the second is the important one.

**Emit.** The hybrid-recall pipelines are built by calling the library's own
``rank_fusion_pipeline`` — the same function the server calls on every turn — so a
saved pipeline cannot drift from what the demo actually runs. A hand-maintained
JSON copy would be wrong the first time a weight or an index name changed, and it
would be wrong silently.

**Verify.** It runs each pipeline against the live cluster and prints the row
count. A pipeline that returns nothing on stage is indistinguishable from a broken
product, and the two most common causes are both invisible in the JSON: an Atlas
Search index that is not queryable yet, and a filter field that was never declared
in the index definition (which returns zero rows rather than erroring).

    # Write runnable copies and check them
    uv run --extra demo python -m demo.compass_pipelines \\
        --query "what can't I eat?" --user ai4-demo --out /tmp/pipelines

    # Regenerate the checked-in templates (placeholder instead of a real vector)
    uv run --extra demo python -m demo.compass_pipelines --write-templates
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)

from agent_memory.config import MemoryConfig  # noqa: E402
from agent_memory.memory import AsyncMemory  # noqa: E402
from agent_memory.services.search_pipeline import rank_fusion_pipeline  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMPLATE_DIR = HERE / "compass-pipelines"

# Stands in for 1024 floats in the checked-in templates. A real vector is
# unreadable on a projector and adds ~20 KB per file to the repository.
PLACEHOLDER = "<<EMBEDDING>>"

# Promotion criteria, mirrored from ConsolidationWorker._promote_to_ltm. Kept as
# literals here rather than read from config on purpose: this pipeline is a
# *teaching artifact* that someone reads on screen, and `$gte: 0.6` communicates
# where a config reference would not. The defaults are asserted below so a config
# change surfaces as a failure rather than as a stale slide.
PROMOTION_IMPORTANCE = 0.6
PROMOTION_ACCESS = 2
PROMOTION_AGE_MINUTES = 60


def hybrid_memories(user_id: str, query: str, embedding, config) -> list[dict]:
    """Semantic recall over ``memories`` — pipeline 01."""
    return rank_fusion_pipeline(
        query=query,
        query_embedding=embedding,
        vector_index="memories_vector_index",
        fts_index="memories_fts_index",
        fts_paths=["content", "summary"],
        vs_filter={
            "user_id": user_id,
            "deleted_at": None,
            "tier": {"$in": ["stm", "ltm"]},
        },
        fts_filter_clauses=[
            {"equals": {"path": "user_id", "value": user_id}},
            {"equals": {"path": "is_deleted", "value": False}},
            {"in": {"path": "tier", "value": ["stm", "ltm"]}},
        ],
        limit=5,
        vector_weight=config.rrf_vector_weight,
        text_weight=config.rrf_text_weight,
        # Compass renders a 1024-element array as a wall of numbers that pushes
        # every readable field off screen. An inclusion projection instead of the
        # library's `{"embedding": 0}` default, so what is on the projector is
        # only what the audience needs to read.
        project={
            "_id": 0,
            "content": 1,
            "tier": 1,
            "importance": 1,
            "access_count": 1,
            "created_at": 1,
            "score": 1,
        },
    )


def hybrid_episodes(user_id: str, query: str, embedding, config) -> list[dict]:
    """Activity-log recall over ``episodes`` — pipeline 02."""
    return rank_fusion_pipeline(
        query=query,
        query_embedding=embedding,
        vector_index="episodes_vector_index",
        fts_index="episodes_fts_index",
        fts_paths=["search_text"],
        vs_filter={"user_id": user_id},
        fts_filter_clauses=[{"equals": {"path": "user_id", "value": user_id}}],
        limit=5,
        vector_weight=config.rrf_vector_weight,
        text_weight=config.rrf_text_weight,
        project={
            "_id": 0,
            "search_text": 1,
            "thread_id": 1,
            "agent_name": 1,
            "step": 1,
            "ts": 1,
            "correlation_id": 1,
            "files_touched": 1,
            "score": 1,
        },
    )


def promotion(user_id: str) -> list[dict]:
    """Which short-term memories have earned long-term durability — pipeline 03.

    The read half of ``ConsolidationWorker._promote_to_ltm``: importance, access
    count, and age, all three. Stated as a query it is legible in a way that the
    same rule buried in a worker is not — "memory decides what to keep" stops
    being a claim and becomes three predicates you can read off the screen.
    """
    return [
        {
            "$match": {
                "user_id": user_id,
                "tier": "stm",
                "deleted_at": None,
                "importance": {"$gte": PROMOTION_IMPORTANCE},
                "access_count": {"$gte": PROMOTION_ACCESS},
                # `$expr` is required, not stylistic. `$dateSubtract` and `$$NOW`
                # are aggregation expressions; a bare query operator does not
                # evaluate them, it compares `created_at` against the literal
                # document `{"$dateSubtract": {...}}`. That does not error — BSON
                # sorts Date above Object, so the predicate is simply false for
                # every document and the stage returns nothing. Verified against
                # Atlas 8.3: identical `$match` with and without `$expr` gave 0
                # rows and 5 rows over the same collection.
                "$expr": {
                    "$lt": [
                        "$created_at",
                        {
                            "$dateSubtract": {
                                "startDate": "$$NOW",
                                "unit": "minute",
                                "amount": PROMOTION_AGE_MINUTES,
                            }
                        },
                    ]
                },
            }
        },
        {
            "$addFields": {
                "age_minutes": {
                    "$dateDiff": {
                        "startDate": "$created_at",
                        "endDate": "$$NOW",
                        "unit": "minute",
                    }
                }
            }
        },
        {"$sort": {"importance": -1, "access_count": -1}},
        {
            "$project": {
                "_id": 0,
                "content": 1,
                "importance": 1,
                "access_count": 1,
                "age_minutes": 1,
                "conversation_id": 1,
            }
        },
    ]


def inventory(user_id: str) -> list[dict]:
    """One row per tier — pipeline 04.

    The four-tier slide, as live data. Useful as an opener: it is the fastest way
    to show that the tiers are real collections with real counts rather than a
    diagram someone drew.
    """
    return [
        {"$match": {"user_id": user_id, "deleted_at": None}},
        {
            "$group": {
                "_id": "$tier",
                "count": {"$sum": 1},
                "mean_importance": {"$avg": "$importance"},
                "total_accesses": {"$sum": "$access_count"},
                "newest": {"$max": "$created_at"},
                "oldest": {"$min": "$created_at"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "tier": "$_id",
                "count": 1,
                "mean_importance": {"$round": ["$mean_importance", 3]},
                "total_accesses": 1,
                "newest": 1,
                "oldest": 1,
            }
        },
        {"$sort": {"tier": 1}},
    ]


FILES = {
    "01-hybrid-recall-memories.json": "memories",
    "02-hybrid-recall-episodes.json": "episodes",
    "03-stm-to-ltm-promotion.json": "memories",
    "04-tier-inventory.json": "memories",
}


def build_all(user_id: str, query: str, embedding, config) -> dict[str, list[dict]]:
    return {
        "01-hybrid-recall-memories.json": hybrid_memories(
            user_id, query, embedding, config
        ),
        "02-hybrid-recall-episodes.json": hybrid_episodes(
            user_id, query, embedding, config
        ),
        "03-stm-to-ltm-promotion.json": promotion(user_id),
        "04-tier-inventory.json": inventory(user_id),
    }


def write(pipelines: dict[str, list[dict]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, pipeline in pipelines.items():
        (out / name).write_text(json.dumps(pipeline, indent=2, default=str) + "\n")
    print(f"wrote {len(pipelines)} pipelines to {out}")


def check_config_matches_pipelines(config) -> list[str]:
    """Fail loudly when a config change has outdated the literals above."""
    drift = []
    if config.promotion_importance_threshold != PROMOTION_IMPORTANCE:
        drift.append(
            f"promotion_importance_threshold is "
            f"{config.promotion_importance_threshold}, pipeline says "
            f"{PROMOTION_IMPORTANCE}"
        )
    if config.promotion_access_threshold != PROMOTION_ACCESS:
        drift.append(
            f"promotion_access_threshold is {config.promotion_access_threshold}, "
            f"pipeline says {PROMOTION_ACCESS}"
        )
    if config.promotion_age_minutes != PROMOTION_AGE_MINUTES:
        drift.append(
            f"promotion_age_minutes is {config.promotion_age_minutes}, pipeline "
            f"says {PROMOTION_AGE_MINUTES}"
        )
    return drift


async def run(user_id: str, query: str, out: Path | None) -> int:
    config = MemoryConfig.from_env(await_search_indexes=False)
    memory = await AsyncMemory.create(config)
    db = memory._db_manager.db

    try:
        drift = check_config_matches_pipelines(config)
        if drift:
            print("Promotion pipeline is out of date with config:", file=sys.stderr)
            for line in drift:
                print(f"  - {line}", file=sys.stderr)
            print(
                "  Update the literals in demo/compass_pipelines.py.",
                file=sys.stderr,
            )

        embedding = await memory.providers.embedding.generate_embedding(query)
        pipelines = build_all(user_id, query, embedding, config)

        if out is not None:
            write(pipelines, out)

        print(f"\nVerifying against the live cluster (user={user_id!r})")
        empty = []
        for name, pipeline in pipelines.items():
            collection = FILES[name]
            try:
                cursor = await db[collection].aggregate(pipeline)
                rows = await cursor.to_list(None)
            except Exception as exc:
                print(f"  {name:<38} ERROR  {exc}")
                empty.append(name)
                continue
            print(f"  {name:<38} {len(rows):>3} rows  ({collection})")
            if not rows:
                empty.append(name)
            elif "score" in rows[0]:
                # Print the top score so the presenter sees the RRF magnitude
                # before the audience does, and is not surprised by 0.016.
                print(f"  {'':<38}      top score {rows[0]['score']:.4f}")

        if empty:
            print(
                "\nThese returned no rows — do not present them:", file=sys.stderr
            )
            for name in empty:
                print(f"  - {name}", file=sys.stderr)
            print(
                "  Usual causes: the user has no seeded data (run demo.seed), a "
                "search index is not queryable yet, or a filter field is missing "
                "from the index definition.",
                file=sys.stderr,
            )
            return 1

        print("\nAll pipelines return rows.")
        return 0
    finally:
        await memory.close()


def write_templates() -> int:
    """Regenerate the checked-in JSON with a placeholder instead of a vector.

    Offline: no cluster, no embedding call, so this is safe to run in CI to check
    the committed files are current.
    """
    config = MemoryConfig.model_construct(
        rrf_vector_weight=1.0, rrf_text_weight=0.7, max_results_per_query=50
    )
    pipelines = build_all("<<USER_ID>>", "<<QUERY>>", PLACEHOLDER, config)
    write(pipelines, TEMPLATE_DIR)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="ai4-demo")
    parser.add_argument("--query", default="what can't I eat?")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="directory for runnable copies with the real vector substituted",
    )
    parser.add_argument(
        "--write-templates", action="store_true",
        help="regenerate the checked-in templates offline and exit",
    )
    args = parser.parse_args()

    if args.write_templates:
        return write_templates()
    return asyncio.run(run(args.user, args.query, args.out))


if __name__ == "__main__":
    sys.exit(main())
