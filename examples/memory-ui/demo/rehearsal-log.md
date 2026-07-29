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

**The rehearsal drifts the state.** Running the beats promotes and evolves
memories: LTM in `seed-thread-hosting` went 2 → 5 through the evolution path, and
three new threads appeared. Re-seed after any rehearsal, and re-seed between
Tuesday and Wednesday. That is what the two-show reset in the demo script is for.

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

Query: `what can't I eat?`, user `ai4-demo`.

## Still outstanding

The **recorded 60–90s OFF-vs-ON screen capture** is not done, and it is the item
to protect first — it is the agreed fallback for the one screen with no equally
good live alternative. Everything it needs is now verified working and the state
is seeded clean; what remains is the capture itself, which needs a person at the
keyboard driving the UI.
