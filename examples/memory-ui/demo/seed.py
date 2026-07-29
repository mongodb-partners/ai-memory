"""Plant a deterministic memory set so both booth mornings look identical.

Run this once before each talk. It creates a user whose memory already spans all
four tiers, so the Compass screen has real documents to open and the episodic
panel has history that predates the live demo — "what have we worked on together?"
is a much better question when the answer is not just the last two minutes.

    uv run --extra demo python -m demo.seed --user ai4-demo

Three things this script deliberately does *not* do.

**It does not hand-write LTM documents.** Every long-term memory here was created
by ``MemoryService.store_stm``, which queues an LTM candidate for each significant
human message, and enriched by the library's real ``EnrichmentWorker`` — so the
importance scores on screen are the model's own judgement, not numbers this file
chose. A seeded document that merely *says* ``tier: "ltm"`` would prove nothing,
and would rot silently the first time enrichment changed.

**It does not run the consolidation worker.** It backdates the STM past
``promotion_age_minutes`` and sets the importance and access count that promotion
requires, then *stops* — leaving those documents as live promotion candidates.
Two reasons. Compass pipeline 03 is the promotion criteria stated as a query, and
it needs rows to be worth projecting. And running the worker here would write a
*second* LTM document for facts that already have one from ``store_stm``, so every
recalled fact would appear twice in the memory panel with no explanation the
audience could see. Promotion is available as a live beat instead:
``--promote`` runs the worker and reports what moved.

**It does not seed the response cache.** A pre-warmed cache would make the demo's
cache beat a HIT on the first ask, which is backwards — the audience needs to see
the miss before the hit for the hit to mean anything.

Idempotent: ``--user`` is wiped before seeding unless ``--keep`` is passed.

The demo's *second* user is reset the other way round. It proves per-user isolation
by recalling nothing, so it must be empty — but a rehearsal types the demo's
question at it, and with memory ON that question gets stored. After one dry run the
"empty" user holds the exact words the next run asks about. ``--wipe-only`` clears a
user and plants nothing:

    uv run --extra demo python -m demo.seed --user alex --wipe-only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

# The library never loads a .env of its own — correct for a library, and it means
# a script has to be explicit. Same repo-root file the demo server reads.
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)

from agent_memory.config import MemoryConfig  # noqa: E402
from agent_memory.memory import AsyncMemory  # noqa: E402
from agent_memory.services.consolidation import ConsolidationWorker  # noqa: E402

log = logging.getLogger("seed")

# ── The planted history ──────────────────────────────────────────────────────
# Cooking, because it is a domain every audience can evaluate without context: a
# menu that ignores a stated allergy is *visibly* wrong from ten feet away, which
# is not true of a bug report or a support ticket.
#
# The shellfish allergy is the load-bearing fact. It appears once, early, in a
# thread the live demo never opens — so when the agent honours it later, the only
# possible source is the memory layer.

CONVERSATIONS: list[dict] = [
    {
        "conversation_id": "seed-thread-pantry",
        "thread_id": "seed-thread-pantry",
        # Days back. Spread out so `created_at` ordering is visible in Compass and
        # the recency term in the ranking has something to actually rank.
        "days_ago": 9,
        "turns": [
            ("human", "I'm allergic to shellfish — no shrimp, crab, or lobster, ever."),
            ("ai", "Understood. I'll keep shellfish out of everything I suggest."),
            ("human", "I keep a well-stocked pantry: olive oil, canned tomatoes, pasta, rice, lentils."),
            ("ai", "Good staples. That covers most weeknight dinners without a shop."),
        ],
        "files": [{"path": "pantry.md", "op": "write"}],
        "tools": ["note_preference"],
    },
    {
        "conversation_id": "seed-thread-weeknight",
        "thread_id": "seed-thread-weeknight",
        "days_ago": 6,
        "turns": [
            ("human", "I usually have about 30 minutes to cook on weeknights."),
            ("ai", "Then sheet-pan roasts and one-pot pasta are your best options."),
            ("human", "We made the lemon-herb chicken with roasted potatoes and it worked well."),
            ("ai", "Noting that as a keeper — it scales easily for guests too."),
        ],
        "files": [{"path": "recipes/lemon-herb-chicken.md", "op": "write"}],
        "tools": ["scale_recipe", "note_preference"],
    },
    {
        "conversation_id": "seed-thread-hosting",
        "thread_id": "seed-thread-hosting",
        "days_ago": 3,
        "turns": [
            ("human", "One of my regular guests is vegetarian, so I need a meatless main sometimes."),
            ("ai", "I'll keep a vegetarian option in mind for anything you host."),
            ("human", "I don't drink, so skip wine pairings — sparkling water or juice is fine."),
            ("ai", "Got it. No alcohol in pairings or in the cooking."),
        ],
        "files": [{"path": "guests.md", "op": "write"}],
        "tools": ["note_preference"],
    },
]

# Facts that should read as promotion candidates. Matched against STM content by
# substring, then given the importance and access count promotion requires — so
# pipeline 03 returns them and ``--promote`` has something to move.
#
# These are *substrings*, and short ones on purpose: matching on the full sentence
# would make the seed break silently if a turn above were reworded, and a seed
# that quietly plants nothing is worse than one that fails loudly.
PROMOTE = [
    "allergic to shellfish",
    "30 minutes to cook",
    "vegetarian",
    "don't drink",
]

# Above `promotion_importance_threshold` (0.6). Not 1.0 — an importance of exactly
# 1.0 on every seeded fact looks synthetic in Compass, and the ranking is more
# convincing when the scores differ.
SEED_IMPORTANCE = 0.78
# `promotion_access_threshold` is 2; a real recalled fact would have been touched
# more than the bare minimum.
SEED_ACCESS_COUNT = 3

# Width of the probe excerpt. One terminal line, so a presenter scanning the output
# sees one row per hit.
EXCERPT = 64


def _oneline(text: str | None) -> str:
    """Collapse whitespace and truncate, so one hit prints as one line."""
    collapsed = " ".join((text or "").split())
    return collapsed[:EXCERPT] + ("…" if len(collapsed) > EXCERPT else "")


async def _wipe(memory: AsyncMemory, db, user_id: str) -> None:
    """Remove every trace of `user_id` across all three collections.

    `wipe_user_data` covers `memories`, but `episodes` and the demo's response
    cache are outside the library's user-data contract and have to be named
    explicitly. Missing either one is not a visible error — it is a stale document
    that surfaces mid-demo.
    """
    result = await memory.wipe_user_data(user_id, confirm=True)
    episodes = await db["episodes"].delete_many({"user_id": user_id})
    cache = await db["demo_response_cache"].delete_many({"user_id": user_id})
    log.info(
        "wiped %s: memories=%s episodes=%s cache=%s",
        user_id, result.get("memories_deleted", "?"),
        episodes.deleted_count, cache.deleted_count,
    )


async def wipe_only(user_id: str) -> int:
    """Clear a user without seeding it — the reset for the isolation beat.

    The second user in the demo proves per-user isolation by recalling *nothing*,
    which means it must be empty and must stay empty. But a rehearsal types the
    demo's question at it, and with memory ON that question is itself stored: after
    one dry run the "empty" user holds the very words the next run will ask about,
    and the beat can recall its own residue instead of returning zero hits.

    `seed()` wipes only as a prelude to planting data, so it cannot express "empty
    this one and stop". This can.
    """
    config = MemoryConfig.from_env(await_search_indexes=True)
    memory = await AsyncMemory.create(config)
    try:
        await _wipe(memory, memory._db_manager.db, user_id)
        log.info("%s is now empty — that is the point; do not seed it", user_id)
        return 0
    finally:
        await memory.close()


async def seed(user_id: str, *, keep: bool, promote: bool) -> int:
    config = MemoryConfig.from_env(await_search_indexes=True)
    memory = await AsyncMemory.create(config)
    db = memory._db_manager.db

    try:
        if not keep:
            await _wipe(memory, db, user_id)

        now = datetime.now(timezone.utc)

        for convo in CONVERSATIONS:
            ts = now - timedelta(days=convo["days_ago"])

            # ── Semantic tiers ───────────────────────────────────────────
            messages = [
                {"content": text, "message_type": role}
                for role, text in convo["turns"]
            ]
            added = await memory.add(user_id, convo["conversation_id"], messages)

            # Backdate. `add` stamps `created_at` as now, and promotion requires
            # age; without this the whole seeded set sits below the age cutoff and
            # the consolidation pass below promotes nothing.
            await db["memories"].update_many(
                {"user_id": user_id, "conversation_id": convo["conversation_id"]},
                {"$set": {"created_at": ts, "updated_at": ts}},
            )

            # ── Episodic tier ────────────────────────────────────────────
            # One episode per conversation, carrying the tools and files that make
            # an `episodes` document look different from a `memories` document —
            # which is the point of showing both in Compass.
            # Tool calls go on the *first* assistant message, never the last.
            # `is_final_step` treats a trailing tool call as "the turn has not
            # answered yet" and correctly suppresses the embedding and search
            # text — so an episode whose last message requests a tool is written
            # but is not recallable. That is right for the library and fatal for a
            # seed, whose whole job is to be found later.
            episode_messages = []
            seen_ai = False
            for role, text in convo["turns"]:
                if role == "human":
                    episode_messages.append(
                        {"type": "human", "content": text, "tool_calls": []}
                    )
                    continue
                tool_calls = (
                    [{"name": name, "args": {}} for name in convo["tools"]]
                    if not seen_ai
                    else []
                )
                seen_ai = True
                episode_messages.append(
                    {"type": "ai", "content": text, "tool_calls": tool_calls}
                )

            await memory.log_activity(
                user_id,
                convo["thread_id"],
                messages=episode_messages,
                todos=[
                    {"id": "1", "content": "capture the preference", "status": "completed"}
                ],
                conversation_id=convo["conversation_id"],
                agent_name="cooking-assistant",
                ts=ts,
            )

            log.info(
                "seeded %s (%d days ago): %s memories",
                convo["conversation_id"], convo["days_ago"], added.get("count"),
            )

        # The episodic write path is a queue; flush it before anything reads.
        flushed = await memory.flush_activity(timeout=30.0)
        if not flushed:
            log.warning("episodic queue did not fully drain within 30s")

        # ── Make the promotion criteria true of real documents ───────────
        # These STM docs stay STM. Their LTM counterparts already exist — one per
        # significant human message, queued by `store_stm` and enriched by the real
        # worker — so what this loop produces is a *candidate*: a short-term memory
        # that satisfies all three promotion predicates and is waiting for the
        # consolidation worker. That is exactly what pipeline 03 selects for.
        marked = 0
        for fragment in PROMOTE:
            result = await db["memories"].update_many(
                {
                    "user_id": user_id,
                    "tier": "stm",
                    "content": {"$regex": fragment, "$options": "i"},
                },
                {
                    "$set": {
                        "importance": SEED_IMPORTANCE,
                        "access_count": SEED_ACCESS_COUNT,
                        "last_accessed": now - timedelta(days=1),
                    }
                },
            )
            if result.matched_count == 0:
                # Loudly, not silently: a fragment that matches nothing means the
                # conversation text drifted, and pipeline 03 will show a screen of
                # nothing to a standing audience.
                log.error("PROMOTE fragment matched no STM: %r", fragment)
            marked += result.modified_count
        log.info("marked %d STM as promotion candidates", marked)

        if promote:
            # Opt-in, because it is destructive to the candidate set: the worker
            # flips these documents to `tier: "ltm"` and pipeline 03 goes empty
            # afterwards. Useful as a live beat — show the candidates, run this,
            # show them again — but re-run the seed before the next talk.
            worker = ConsolidationWorker(db["memories"], config, memory.providers)
            stats = await worker.consolidate()
            log.info("consolidation ran: %s", stats)

        # ── Report what a presenter actually needs to know ───────────────
        counts = {
            "stm": await db["memories"].count_documents(
                {"user_id": user_id, "tier": "stm", "deleted_at": None}
            ),
            "ltm": await db["memories"].count_documents(
                {"user_id": user_id, "tier": "ltm", "deleted_at": None}
            ),
            "episodes": await db["episodes"].count_documents({"user_id": user_id}),
        }
        # Counted separately from `episodes`, because the two can differ and only
        # this one predicts whether the episodic panel will populate. An episode
        # without an embedding was written, is visible in Compass, and will never
        # be recalled — reporting only the total would overstate what works.
        searchable = await db["episodes"].count_documents(
            {"user_id": user_id, "embedding": {"$exists": True}}
        )
        # The same three predicates pipeline 03 uses. Counted here so the presenter
        # learns that screen is empty *now*, from a script they are already reading,
        # rather than in Compass with the audience watching.
        candidates = await db["memories"].count_documents(
            {
                "user_id": user_id,
                "tier": "stm",
                "deleted_at": None,
                "importance": {"$gte": config.promotion_importance_threshold},
                "access_count": {"$gte": config.promotion_access_threshold},
                "created_at": {
                    "$lt": now - timedelta(minutes=config.promotion_age_minutes)
                },
            }
        )

        print(f"\nSeeded {user_id}:")
        for tier, count in counts.items():
            print(f"  {tier:<10} {count}")
        print(f"  {'searchable':<10} {searchable} of {counts['episodes']} episodes")
        print(f"  {'candidates':<10} {candidates} STM eligible for promotion")

        # Prove the seed is queryable, not merely present. Atlas Search indexes are
        # eventually consistent, so "the document exists" and "recall finds it" are
        # different facts — and the second is the one the demo depends on.
        probe = "what can't I eat?"
        hits = (await memory.search(user_id, probe, limit=3)).get("results", [])
        episodic = (await memory.recall_activity(user_id, probe, limit=3)).get("results", [])
        print(f"\nRecall probe — {probe!r}")
        for hit in hits:
            print(f"  semantic  {hit.get('tier', '?'):<4} {_oneline(hit.get('content'))}")
        for hit in episodic:
            # `search_text` is the question joined to the answer with a newline, so
            # printing it raw splits one hit across two lines and the second looks
            # like a separate, unlabelled result.
            print(f"  episodic  {'':<4} {_oneline(hit.get('search_text'))}")

        problems = []
        if not hits:
            problems.append(
                "nothing recalled semantically — the documents are written but the "
                "search indexes may not be queryable yet; wait a minute and re-run"
            )
        if not episodic:
            problems.append(
                "nothing recalled episodically — check that `searchable` above "
                "equals the episode count; a zero there means the seeded turns end "
                "in a tool call rather than an answer"
            )
        if counts["ltm"] == 0:
            problems.append(
                "no long-term memories — `store_stm` queues an LTM candidate per "
                "significant human message and the enrichment worker completes it, "
                "so zero here means enrichment never ran"
            )
        if searchable < counts["episodes"]:
            problems.append(
                f"only {searchable} of {counts['episodes']} episodes are searchable"
            )
        if candidates == 0 and not promote:
            problems.append(
                "no promotion candidates — Compass pipeline 03 will return an "
                "empty screen; check that the PROMOTE fragments still match the "
                "conversation text above"
            )

        if problems:
            print("\nNOT READY TO PRESENT:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1

        print("\nAll four tiers seeded and queryable.")
        return 0
    finally:
        await memory.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user", default="ai4-demo",
        help="user id to seed; must match the UI's user field (default: ai4-demo)",
    )
    parser.add_argument(
        "--keep", action="store_true",
        help="add to existing data instead of wiping first",
    )
    parser.add_argument(
        "--promote", action="store_true",
        help="run the consolidation worker, promoting the candidates to LTM; "
             "leaves Compass pipeline 03 empty until the next seed",
    )
    parser.add_argument(
        "--wipe-only", action="store_true",
        help="clear the user and plant nothing — the reset for the second user, "
             "which proves isolation by recalling nothing and so must stay empty",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s"
    )
    if args.wipe_only:
        if args.keep:
            parser.error("--wipe-only and --keep contradict each other")
        return asyncio.run(wipe_only(args.user))
    return asyncio.run(seed(args.user, keep=args.keep, promote=args.promote))


if __name__ == "__main__":
    sys.exit(main())
