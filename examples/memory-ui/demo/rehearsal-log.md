# Rehearsal log

Measured, not estimated. Every number below came from running the beats in
`docs/talks/ai4-2026/demo-script.md` §1 against the live stack — Atlas 8.3,
Bedrock `global.anthropic.claude-sonnet-5`, voyage-4 embeddings at 1024 dims.

Re-run before each show and compare. A beat that has doubled in latency is a
signal about the network, not a reason to improvise on stage.

## 2026-07-30 — full dry run

Seeded state at start: `stm 12 · ltm 6 · episodes 3 · candidates 5`, user
`ai4-demo`. Second user `alex` deliberately **empty**.

| Beat | Latency | Result |
|---|---|---|
| A1 · state the constraint, memory OFF | 3.4s | 0 write frames. Acknowledges, retains nothing. |
| A3 · new thread, same question, OFF | 2.5s | 0 recall frames. Asks what you're in the mood for — the forgetting is visible. |
| B5 · same message, memory ON | 11.0s | 3 recall frames. Commits to a dish. |
| B7 · new thread, same question, ON | 12.7s | Recalls **ltm 3 · stm 1 · episodic 3**. Does not re-ask. Honours shellfish, the vegetarian guest, and the 30-minute window. |
| B8 · user `alex`, same question | 8.2s | **0 hits.** Different answer. Isolation proven live, not asserted. |
| C · repeat B7's question verbatim | **0.3s** | `cache_hit=true`, `match=exact`. |

**Total live agent time for the whole contrast: 38.1s.** The Screen-1 budget in
the demo script is 2:00, so the recording has ~80 seconds of headroom for the
UI interactions — clicking **New thread**, flipping the toggle, typing — that
this measurement excludes.

### What this confirms

- **The toggle is honest.** OFF and ON produce different answers to an identical
  prompt, and OFF emits no memory frames at all. No cache bleed-through, which is
  the failure mode that would kill the demo silently.
- **Cross-thread recall works, and it isn't chat history.** B7 runs in a thread
  with no prior turns. The 12.7s is the honest cost of recall plus inference.
- **Per-user isolation is enforced in the query.** `alex` sends the same words to
  the same model and gets nothing, because `user_id` sits inside both `$rankFusion`
  branches rather than in the prompt.
- **The cache beat is worth showing.** 12.7s → 0.3s is a 40× drop the audience can
  read off the clock without being told.

### Notes for the day

**B5 and B7 are 11–13 seconds each.** On a recording that is fine. Live, in front
of a standing crowd, it is long — which is the whole reason Screen 1 is recorded.
Do not decide to run it live because it worked in rehearsal on good wifi.

**The rehearsal drifts the state, in both directions.** Measured the morning after
this dry run, before touching anything:

| | Seeded | Next morning |
|---|---|---|
| `ai4-demo` | stm 12 · ltm 6 | **stm 7 · ltm 11** |
| `alex` (the "empty" user) | nothing | **2 stm · 1 episode · 1 cache entry** |

Both numbers matter and neither is an error. The demo user's short-term memories
were promoted by the running consolidation worker, which is the library working
correctly — but it leaves Compass tab 1 showing 7 where the script says 12.

The second row is the one that would have broken a beat. `alex` proves per-user
isolation by recalling *nothing*, and the rehearsal typed the isolation beat's own
question at it with memory ON — so the question got stored. One of those two
short-term documents was `What should I make Friday?` verbatim. Run the beat again
and it can recall its own residue and return hits, which is the opposite of the
point, with no error to warn you.

Hence `--wipe-only`, added after this run. Reset **both** users, server left up:

```bash
uv run --extra demo python -m demo.seed --user alex --wipe-only   # empty it
uv run --extra demo python -m demo.seed --user ai4-demo          # re-plant it
```

Verified after doing exactly that: `stm 12 · ltm 6 · episodes 3 · candidates 5`,
still holding 45 seconds later with the server running, `alex` at zero across all
three collections, and all four Compass pipelines returning their original row
counts and top scores. Re-run this between Tuesday and Wednesday.

**Ordering, again.** The server was already running when this seed was planted. If
you seed first and start the server second, its startup consolidation pass eats
the 5 promotion candidates and Compass pipeline 03 returns nothing. See
`compass-tabs.md`.

### Compass pipelines, same session

| Pipeline | Rows | Top score |
|---|---|---|
| `01-hybrid-recall-memories.json` | 5 | 0.0164 |
| `02-hybrid-recall-episodes.json` | 3 | 0.0274 |
| `03-stm-to-ltm-promotion.json` | 5 | — |
| `04-tier-inventory.json` | 2 | — |

Query: `what can't I eat?`, user `ai4-demo`. Re-verified after the reset above and
unchanged — row counts and both top scores identical, which is what makes them
usable as a pre-flight check rather than just a record.

## Session — Thu Jul 30, ~04:15 (after the Medium/Low fix batch)

**Why this session happened:** the :8100 server had been running since 01:14 on
pre-fix code. It held the *old* promotion logic, so every LTM document it had
written carried the 24-hour STM TTL. Restarting before re-seeding is not hygiene
here, it is the fix taking effect.

**What was actually wrong on disk.** Before the restart, `ai4-demo` held 12 LTM
documents and **5 of them expired in 21.7 hours** — around 01:45 on **Fri Jul 31**,
four days *before* the Tuesday talk. The TTL index would have deleted them quietly:
no error, no log line, just a demo where "long-term memory" recalls less each day.
The other 7 were already at 90 days, so a spot check of one document would have
looked fine. After restart + re-seed: **11 LTM, all at 2160.0h (90 days, into late
October), 0 expiring before Aug 5.**

To be clear about what is correct: LTM is *supposed* to carry a 90-day TTL
(`ltm_retention_standard_days`). The bug was LTM carrying the 24-hour *STM* TTL,
not LTM having a TTL at all.

**STM is a per-morning concern, by design.** The 7 STM documents expire ~24h after
seeding. That is correct behaviour and it is also why **the seed must be re-run on
each demo morning** — seed Monday and the Wednesday demo has no short-term tier to
show. Tuesday morning and Wednesday morning, both.

### Verification #10 — the toggle is honest (the demo-breaking failure mode)

Run OFF **first** on a query never asked before, then ON on the same query. If ON
gets a cache hit, the only place it came from is the OFF run.

| Check | Result |
|---|---|
| OFF performed recall | no — zero `recall` frames |
| OFF wrote to the response cache | no — cache docs 2 → 2 |
| ON saw a cached answer | no — `cache_hit: false` |
| ON recalled across tiers | yes — `ltm` + `stm` + `episodic` |
| ON wrote back | yes — `ltm` (queued for scoring) + `episodic` (turn logged) |
| Answers differ materially | yes |

The contrast, verbatim from the run — this is the talk's core moment:

- **OFF:** "Happy to help! To plan this well, could you tell me: any dietary
  restrictions or allergies among your guests…"
- **ON:** "I'd scale the lemon-herb chicken… plus a chickpea and spinach stew or
  stuffed portobellos **for the vegetarian option**…"

ON never had to ask. It already knew about the shellfish allergy and the vegetarian
guest, from two different tiers.

**Probe runs pollute the state.** Each verification turn writes STM + an episode +
a cache entry, so the seed was re-run afterwards to restore determinism. Final
state: `stm 7 · ltm 11 · episodes 3 · cache 0`. Do the same after any rehearsal
turns — check `cache 0` in particular, because a leftover cache entry is what makes
a live OFF/ON demo silently prove nothing.

## 2026-07-30, later — re-run on the post-security-fix build

**Why this session happened:** every measurement above was taken against code that
predates the seven security fixes in 4.2.0. Two of those fixes touch paths the demo
actually walks, so the numbers needed re-taking rather than re-reading:

- `wipe_user_data` now clears **every** user-scoped collection, `episodes`
  included, and now *raises* on a partial failure instead of returning it. Both
  `demo/seed.py` and the server's `/reset` predated that and were fixed in this
  session — `/reset` had been deleting `episodes` itself and then overwriting the
  library's real count with its own zero, so a reset that cleared nine episodes
  reported `episodes_deleted: 0` on stage.
- The search-index reconciliation now updates existing indexes in place. That
  raised the question of whether the first startup on the new build would rebuild
  the demo cluster's indexes. **It does not** — see below.

**No index rebuild on startup.** The config's default `embedding_dimension` is
1536, but `ProviderManager` aligns it to the configured embedding model before
migrations run, so all five indexes reconcile as up to date at voyage-4's 1024:

```
memories_vector_index   dims=1024 READY -> skip (up to date)
memories_fts_index                READY -> skip (up to date)
episodes_vector_index   dims=1024 READY -> skip (up to date)
episodes_fts_index                READY -> skip (up to date)
cache_vector_index      dims=1024 READY -> skip (up to date)
```

Worth keeping in mind for a *different* cluster: an index whose definition has
drifted gets an in-place `update_search_index`, which keeps serving queries while
it rebuilds. Only a `numDimensions` change drops and recreates, and that is the
one case that would take vector search offline mid-demo. Changing the embedding
model before a talk is therefore a rebuild, not a config edit.

### The beats, re-measured

| Beat | Then | Now | Result |
|---|---|---|---|
| A1 · state the constraint, OFF | 3.4s | 4.0s | 0 recall frames. |
| A3 · new thread, same question, OFF | 2.5s | 2.9s | 0 recall frames. Asks about dietary restrictions — two beats after being told. |
| B5 · same message, ON | 11.0s | 12.2s | Recalls **ltm 3 · stm 1 · episodic 3**. |
| B7 · new thread, same question, ON | 12.7s | 7.7s | Recalls **ltm 1 · stm 3 · episodic 3**. Does not re-ask. |
| C · repeat B7 verbatim | 0.3s | 0.9s | `cache_hit=true`, replaying B7's answer. |
| B8 · user `alex`, same question | 8.2s | 8.6s | **0 hits.** Isolation holds. |
| D · "what have we worked on together?" | — | 11.8s | **episodic 3.** Nothing from this thread. |

**Total 41.1s** across seven beats (the earlier 38.1s covered six). Still inside
the 2:00 Screen-1 budget. Every beat is within noise of its previous figure except
B7, which came in 5s *faster* — that is model latency, not a change worth
attributing to anything. **Zero errors, warnings, or tracebacks** in the server log
across the whole run.

The tier split shifts between B5 and B7 (`ltm 3 · stm 1` → `ltm 1 · stm 3`) because
the same fact exists in both tiers and `$rankFusion` orders them per query. Nothing
to fix; just don't script a specific split onto a slide.

**State drift reproduced exactly as documented above.** After the run, `alex` held
2 memories · 1 episode · 1 cache entry — including the isolation beat's own question.
`--wipe-only` cleared it and `ai4-demo` was re-seeded.

**And one more, worth its own line: Compass pipeline 03 empties itself.** A while
after the re-seed, with the server still up, `ai4-demo` had drifted to `stm 7 ·
ltm 12` and **promotion candidates were 0** — the consolidation worker had promoted
all five. Pipeline 03 *is* the promotion criteria stated as a query, so it returns
no rows, and nothing about the UI or the logs says so. This is the library working
correctly; it is also a screen that goes blank on its own between the pre-flight
check and the talk.

So the rule is tighter than "re-seed each morning": **re-seed last, immediately
before presenting**, and treat `candidates 5` in the seed output as the number to
read. The LTM TTLs were re-checked at the same time and are correct — 12 documents,
all at ~2160h (90 days), none expiring before the talk.

Final state as of this entry: `ai4-demo` at `stm 12 · ltm 6 · episodes 3 ·
candidates 5`, `alex` at zero across every collection.

## Still outstanding

The **recorded 60–90s OFF-vs-ON screen capture** is not done, and it is the item
to protect first — it is the agreed fallback for the one screen with no equally
good live alternative. Everything it needs is now verified working and the state
is seeded clean; what remains is the capture itself, which needs a person at the
keyboard driving the UI.
