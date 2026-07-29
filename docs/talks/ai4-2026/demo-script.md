# Demo script — "Give Your Agents a Memory"

Ai4 2026 · MongoDB Booth #1149 · Aug 4 and Aug 5, 11:00 AM PST

Companion to `slides.md`. This is the operational document: exact keystrokes, exact click
order, and a stated fallback for every screen. Read it once the morning of.

**Governing rule: nothing is live that doesn't have to be.** Booth wifi is shared with
8,000 people and a standing crowd will not wait on a spinner. The chat contrast is a
recording. Compass is a recording or screenshots. The only thing that must work is a
video player.

---

## Pre-flight (do this the night before, then again 20 min before)

**Night before**
- [ ] `python demo/seed.py --user daoud --user alex` — plants the deterministic memory set
- [ ] Record `screen1-off-on.mov` (60–90 sec, see §1). Re-record until it's clean; there is no live fallback that is as good.
- [ ] Capture `screen2-memory-panel.png` (all four groups populated, scores legible at 1080p)
- [ ] Capture Compass stills: `compass-ltm-doc.png`, `compass-episode-doc.png`, `compass-indexes.png`
- [ ] Export the deck to PDF as well as PPTX. PDF is the fallback that always opens.
- [ ] Copy all of it to a **USB stick** and to the laptop's local disk. Nothing loads from a network share or a cloud drive.

**20 minutes before, at the booth**
- [ ] Laptop plugged in. Sleep, screensaver, and auto-updates off.
- [ ] Notifications off — macOS Do Not Disturb, Slack quit, mail quit.
- [ ] Display mirroring confirmed at the booth resolution. Check the *bottom* of the memory panel is not cut off.
- [ ] Font size sanity check: stand 8 feet back and read the importance score. If you can't, zoom the recording.
- [ ] Video player already open with `screen1-off-on.mov` loaded, paused at frame 0.
- [ ] Compass open, connected, three tabs pre-opened and pre-scrolled (§3) — even if you plan to use stills.
- [ ] Deck open at slide 1, presenter notes on the laptop only.
- [ ] Water within reach. You do this twice; the second one is harder.

**If running the app live anyway** (only if the booth has wired ethernet):
- [ ] `TRANSPORT=both uv run agent-memory` up, `GET /health` returns ok
- [ ] `cd examples/memory-ui && npm run dev` up
- [ ] One full warm-up turn sent so the first crowd-facing request isn't a cold start
- [ ] Verify the memory-OFF path returns a *different* answer than memory-ON for the same prompt — if they match, the cache is bleeding through and the demo is dead. Fix or fall back to the recording.

---

## Screen 1 — Memory OFF vs ON (2:00)

**Slide 5. Full-bleed recording, narrated live over the top.**

### What the recording contains

**Pass A — toggle OFF** (header toggle reads `MEMORY: OFF`, visibly red/grey)

| # | Action | Expected |
|---|---|---|
| 1 | Type: `I'm allergic to shellfish, and I'm cooking for six people on Friday.` | Agent acknowledges. Nothing is written to memory. |
| 2 | Click **New thread** (visible in the recording — this is the proof) | Empty conversation |
| 3 | Type: `What should I make Friday?` | It has no idea. Asks who's coming, or suggests shrimp. |

**Pass B — toggle ON** (`MEMORY: ON`, spring green)

| # | Action | Expected |
|---|---|---|
| 4 | Toggle to ON | Memory panel appears, empty |
| 5 | Same message as step 1, verbatim | Agent acknowledges. Memory panel shows a **write**: `no shellfish`, `cooking for 6` |
| 6 | Click **New thread** | Empty conversation. Memory panel **still populated**. |
| 7 | Same question as step 3, verbatim | "For six, and no shellfish — here's a menu." Never re-asks. Panel shows the **recall** with scores. |
| 8 | Change the user ID in the header to `alex`, same question | Nothing recalled. Different answer. |

### Narration beats

- Over step 1: *"Watch the toggle. Memory off."*
- Over step 2 — **say this, it is the whole demo**: *"New conversation. Nothing in the context window."*
- Over step 3: *"And it has no idea. Everyone here has shipped this."*
- Over step 5: *"Same model, same prompt, same code. One toggle."*
- Over step 6: *"New thread again. Watch the panel — the memory survived the conversation."*
- Over step 7: *"It never re-asked. And that's not chat history — that thread is empty."*
- Over step 8: *"Different user. Nothing. The user filter is inside the query, not in the prompt."*

### Fallbacks

1. Recording won't play → describe steps 1–3 and 5–7 in 20 seconds, jump to slide 6 (a still that carries the same point).
2. Playing live and it hangs → don't wait past **five seconds**. Say *"and this is why I recorded it"*, cut to the recording or to slide 6. Never debug on stage.
3. Total AV failure → slides 6 and 7 are stills; slide 8 is the technical payload and needs no demo at all. The talk survives on 3 → 8 → 9 → 10 alone.

---

## Screen 2 — The memory panel (1:30)

**Slide 6. Still image, or the live panel if the app is up.**

Point at these four things, in this order:

1. **Long-term:** `no shellfish`, with its importance score and recall count — *"that score wasn't hand-written; it was assessed at write time and climbs on every use."*
2. **Episodic:** the logged turns, with step numbers and tool names — *"this is what it did, not what it knows."* (Plant slide 10 here.)
3. **Short-term:** this thread's state — *"expires on its own, TTL index."*
4. **Cache:** a miss, then a **hit** — *"second identical question, zero inference cost."*

**Read every number off the screen, never from this script.** Whatever the seeded values
are on the morning of the talk is what you say. Slide 6 prints no numbers for exactly this
reason — a memorized `0.9` next to a screen showing `0.82` is the one error the crowd is
guaranteed to catch.

The score numbers must be legible from 8 feet. If they aren't, that's a re-capture, not a
live zoom.

**Fallback:** the four bullets on slide 6 already say all of this. Read them and move on.

---

## Screen 3 — Compass (1:30)

**Slide 7. Pre-opened tabs, or stills.**

Click order, no exploring:

| Tab | Show | Say |
|---|---|---|
| 1 | `memories` — one LTM doc expanded, `importance` / `access_count` / `embedding` visible | *"The document the agent just read. Not an abstraction — a document."* |
| 2 | `episodes` — one doc with `messages[]`, `files_touched[]`, `step`, `parent_step` | *"And the turn log. What it did, in order, under a trace ID."* |
| 3 | Index list — vector, full-text, and TTL indexes in one list | *"Short-term state expires on its own. That's an index, not a cron job."* |

Do not quote index counts. Compass is showing the real list next to you; point at it and
say "vector, full-text, and TTL, same collection list." (For reference only, not for
speaking: 3 vector, 2 full-text, 7 TTL as the schema stands today.)

Do not scroll hunting for a field. Everything is pre-scrolled. Do not connect to a
cluster in front of the crowd.

**Fallback:** the three stills, shown in the same order, same lines. Indistinguishable to
the audience.

---

## Screen 4 — The pipeline (1:30)

**Slide 8. Static code on the slide. Nothing runs.**

Read it in this order — the order matters, it's an argument:

1. `vectorPipeline` — *"meaning"*
2. `fullTextPipeline` — *"exact terms: SKUs, error codes, names"*
3. `combination.weights` — *"native reciprocal rank fusion, one round trip"*
4. The `filter` inside **both** branches — *"per-user isolation enforced by the engine"*

Deliver slowly. This is the slide people photograph, so pause at the top of it for three
seconds before you start talking.

**If asked about versions:** `$rankFusion` is MongoDB 8.1+; before that, `$unionWith`
plus the RRF math in the pipeline, which the library also supports.

---

## Timing checkpoints

Glance at the clock at exactly two points. Don't watch it continuously.

| At | You should be | If you're behind |
|---|---|---|
| **4:00** | Starting the recording (slide 5) | Cut slide 3's second half — the table reads fine on its own |
| **9:00** | Leaving Compass, arriving at slide 8 | Skip slide 9 entirely; slide 10 is the more distinctive one |
| **13:30** | On slide 12 | — |

Never sacrifice slide 8 or slide 10 for time. Those two are the reason the talk is worth
standing for. Slide 9 is the one that cuts cleanly.

---

## Two-show notes

**Aug 4 → Aug 5.** Same script both mornings. After Tuesday, write down:
- Which slide the crowd's attention broke on
- Which question came up more than once (promote its answer into the talk)
- Whether the 4:00 checkpoint held

Reset for Wednesday: re-run `seed.py` (the demo user's `access_count` drifted on
Tuesday), re-check display mirroring, re-load the video at frame 0.

**Between shows:** do not re-record anything. Tuesday's recording worked; Wednesday is
not the day to improve it.

---

## Cross-promotion

Say it twice, deliberately: once at the top of slide 9 (*"this collapse is the whole
subject of my main session"*) and once on slide 12 as the closing line. Don't mention it
a third time — at that point it stops being a funnel and starts being an apology for the
booth talk.

**Closing line, verbatim:**
> "The full stack-collapse story is my main session, Thursday, August 6th. Come to that
> one. And this is all `pip install agent-memory` — go build it."
