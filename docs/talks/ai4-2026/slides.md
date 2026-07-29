# Give Your Agents a Memory

**Ai4 2026 — MongoDB Booth #1149**
Tue Aug 4, 2026 · 11:00–11:15 AM PST
Wed Aug 5, 2026 · 11:00–11:15 AM PST
Speaker: Mohammad Daoud Farooqi — Partner Solutions Architect for AI, MongoDB

> **Status: content final; screenshots outstanding.** The sample UI has landed at
> `examples/memory-ui/` and every beat behind slides 5–7 is verified running against
> live Atlas and Bedrock — see `examples/memory-ui/demo/rehearsal-log.md` for the
> measured timings and `demo/compass-tabs.md` for the four Compass tabs. What slides
> 5–7 still need is the capture itself: the 60–90s OFF-vs-ON recording and the
> stills. Everything here remains deliverable without them — the fallback for each
> demo slide is stated in its notes.

**Abstract as submitted:**
> Every team building agents hits the same wall: the model forgets, and a bigger context
> window won't fix it. I'll show how to give agents real memory on MongoDB Atlas:
> short-term state, long-term semantic memory, and smart retrieval, all in one database
> instead of four bolted together.

**Time budget — 15:00 total. The slides sum to 12:45, leaving 2:15 for churn and one question.**

| # | Slide | Time | Cumulative |
|---|---|---|---|
| 1 | Title | 0:20 | 0:20 |
| 2 | The wall | 1:10 | 1:30 |
| 3 | Four kinds of memory | 1:15 | 2:45 |
| 4 | Same agent, one difference | 0:30 | 3:15 |
| 5 | DEMO — memory OFF vs ON | 2:00 | 5:15 |
| 6 | DEMO — what it remembered | 1:30 | 6:45 |
| 7 | DEMO — Compass: the documents | 1:30 | 8:15 |
| 8 | One pipeline | 1:30 | 9:45 |
| 9 | Four boxes → one cluster · **CUT SLIDE** | 1:00 | 10:45 |
| 10 | The one people skip: episodic | 1:15 | 12:00 |
| 11 | Takeaway | 0:30 | 12:30 |
| 12 | Build it today | 0:15 | 12:45 |

**Two clocks to watch, not twelve.** At **3:15** you should be starting the demo; at
**9:45** you should be leaving slide 8. If you are late at the second checkpoint, drop
slide 9 (it is built to be dropped) and you land on time with slide 10 intact.

The cumulative column is *leave-by*, not arrive-by — 5:15 is when slide 5 should be behind
you, not when it starts. Both checkpoints above are read the same way as the table.

**Hard content rules for this deck**
- No framework logos, names, or imports on any slide. The library is framework-neutral;
  the demo backend calls it directly. Speaker co-presents with Mastra on Thu Aug 6 —
  naming a competing framework here is off-limits.
- Every number on a slide must be one I can defend if asked. **Live values live in the
  screenshots, not in slide text** — a printed number that disagrees with the screen behind
  it is worse than no number.
- Nothing claims a component persists to Atlas unless it demonstrably does.

**Naming — closed.** The package is **not published to PyPI**, so slide 12's CTA is a
`git+https` install from `github.com/mongodb-partners/ai-memory` — the repo's real name, no
rename required and no broken link. Say "install it from the repo," never "pip install
agent-memory." The distribution is still named `agent-memory` in `pyproject.toml`, which is
what `import agent_memory` follows from; that is internal and does not appear on a slide.
(Earlier drafts also mentioned `memory-mcp`; that name is retired and appears nowhere in
this deck.)

---

## Slide 1 — Title

**Give Your Agents a Memory**

Mohammad Daoud Farooqi
Partner Solutions Architect for AI · MongoDB

*(bottom-right, small)* Full stack-collapse story → main session, **Thursday Aug 6**

**Notes (0:20).** "I'm Daoud, I'm a Partner Solutions Architect for AI at MongoDB. Fifteen
minutes, one idea, and a demo. Let's go."

Don't introduce yourself twice. Booth crowds are already deciding whether to stay.

---

## Slide 2 — The wall

> ### Your agent forgot.
> ### A bigger context window will not fix it.

Three failures everyone in this room has already shipped:

- It re-asks a question the user answered yesterday
- It loses a decision made 30 turns ago
- It has no idea what it *did* last Tuesday

**Notes (1:10).** Get the nods first — this is the cheapest audience buy-in in the talk.

"Prompt engineering became context engineering for a reason. The bottleneck moved from
*how you word it* to *what data you put in front of the model*. And the moment your
problem is 'what data, selected how, for which user' — that is not a prompting problem.
That's a database problem."

Bigger windows lose on three axes at once: cost is linear in tokens, attention degrades
in the middle, and the window dies when the session does. Say it in one breath, don't
belabor it. Do **not** get drawn into a needle-in-a-haystack benchmark debate at a booth.

---

## Slide 3 — Four kinds of memory

Most teams build one. You need four.

| | Memory | Answers | Lifetime |
|---|---|---|---|
| **1** | **Short-term state** | "What are we talking about right now?" | Hours — TTL |
| **2** | **Long-term semantic** | "What do I know about this user?" | Months — importance-scored |
| **3** | **Episodic** | "What did I actually *do*, and when?" | Days–weeks — append-only + TTL |
| **4** | **Semantic cache** | "Have I answered this already?" | Minutes — similarity |

Two more things that are memory even though nobody calls them that:
**sticky decisions** (once you've decided, stop re-deciding) and the
**audit trail** (who recalled what, when).

**Notes (1:15).** This table is the spine of the talk — but *say* far less than is printed
on it. Name the four, land the one line, move. Slides 5–7 re-teach all of it concretely,
so anything you explain here you will explain twice.

**Spoken version — four names, one line:**
"Short-term. Long-term. Episodic. Cache. **Most teams build one. You need four.**"

That's it. Point at the table and let them read the rest — booth audiences read faster
than you talk.

**Only if the crowd is leaning in and you have the time**, add one sentence, never more:
- On 2: "You don't store transcripts, you store distilled facts with importance scores."
- On 3: "That's the one nobody builds — slide 10."
- On 4: "Same question twice, second one costs no inference."

Do not explain reinforcement, merging, decay, or promotion here. Every one of those is a
question you'd rather answer at the booth afterward than spend 20 seconds on now.

---

## Slide 4 — Same agent, one difference

> ### Same model.
> ### Same prompt.
> ### Same code path.
> ### One difference: where its memory lives.

**Notes (0:30).** Set the frame *before* the demo, so the crowd knows exactly what
variable is changing. Otherwise half of them will assume you swapped models and the
whole demo proves nothing.

"There is one toggle in this UI. It does not change the model, the temperature, or the
system prompt. It decides whether the agent is allowed to read and write memory."

---

## Slide 5 — DEMO: memory OFF vs ON

*Full-bleed screen recording, 60–90 sec, narrated live.*

**Left / first pass — toggle OFF**
1. "I'm allergic to shellfish, and I'm cooking for six on Friday."
2. → *fresh thread* → "What should I make Friday?"
3. It suggests shrimp. Or asks who's coming. It has no idea.

**Right / second pass — toggle ON**
1. Same first message.
2. → *fresh thread* → same question.
3. "For six, and no shellfish — here's a menu." It never re-asked.

**Notes (2:00).** The `new thread` step is not decoration — it is the whole proof. Same
thread would be indistinguishable from ordinary chat history. Say it out loud: **"Notice
I started a brand-new conversation. Nothing was in the context window."**

Then switch the user ID in the header and re-run: the other user gets nothing. Per-user
isolation, enforced in the query, not in the prompt. That is the second-most convincing
five seconds of this talk — and it only works if that second user id was **never seeded**.
Seed one user, type a different one.

Measured in rehearsal: the memory-ON turns take 11–13 seconds each, the memory-OFF turns
2–3. That asymmetry is honest — recall plus inference costs more than inference alone — but
13 seconds of silence in front of a standing crowd is why this screen is recorded rather
than run live. Do not talk yourself into going live because rehearsal went well on good wifi.

**Fallback if there is no recording:** describe it in 20 seconds and go straight to
slide 6, which is a static screenshot and carries the same point. Never debug live at a
booth. If the recording won't play, say "this runs live in my main session Thursday" and
keep moving.

---

## Slide 6 — DEMO: what it remembered

*Screenshot of the memory panel, four groups populated, scores visible.*

- **Short-term** — this thread's state
- **Long-term** — `no shellfish` · **importance score** · **recall count**
- **Episodic** — the turns it logged: step, tools called, files touched
- **Cache** — miss, then **hit**

> **No hard numbers on this slide.** The screenshot supplies them, and the screenshot is
> the source of truth. Printing `0.9` / `4×` / `0.96` as slide text means a reseed, a
> different embedding, or one extra rehearsal turn puts the slide and the screen in visible
> disagreement — in front of people looking for exactly that. Read the live values off the
> screenshot instead.

**Notes (1:30).** "Every one of those lines is a document the agent read, with the score
that got it there. This is not an abstraction over memory — it *is* the memory."

Point at the importance score on the screenshot and **read whatever value is on screen**.
"That score wasn't hand-written. An LLM assessed it at write time, and it climbs every time
the fact gets used. That's how the agent decides what's worth keeping when there's more
history than context."

Then point at the cache: "Second identical question — cache hit, zero inference cost."
Every exec in the crowd hears that as a line item. Quote the similarity number only if
it's legible on screen behind you; the *hit* is the point, not the decimal.

If the recording shows the clock, use it: rehearsal measured the same question at 12.7
seconds cold and **0.3 seconds** cached. That drop is legible from the back of the booth
without anyone having to read a score.

---

## Slide 7 — DEMO: Compass, the documents

*Four Compass tabs, clicked through in order. Stills are equally good.*

- A **short-term** document — `content`, `expires_at`, importance still at its placeholder
- A **long-term** document — `content`, `importance`, `access_count`, `source_stm_id`
- An **episodic** document — `messages[]`, `tool_calls`, `step`, `parent_step`, `correlation_id`
- The **index list** — vector, full-text, and TTL indexes, side by side in one collection list

Tabs 1 and 2 are the same collection. That is the point: the tier is a field, not a
second system.

**Notes (1:30).** "Same cluster. Same database. No sync job, no ETL, no second system.
The documents the agent retrieves are the documents it writes."

The exact filter for each tab, verified against the seeded data, is in
`examples/memory-ui/demo/compass-tabs.md`. Set them up from that file the night before —
not from this slide, and not by typing a filter in front of the crowd.

Short on time? **Cut tab 1, not tab 3.** Tab 1's `expires_at` point is also carried by the
TTL entry on tab 4's index list. Tab 3 is the episodic differentiator that slide 10 pays off.

Land the TTL point here because it's the one an ops person in the crowd is already
worried about: "Short-term state expires on its own — that's a TTL index, not a cron job
you have to remember to write. Retention for the activity log is one `collMod` away."

**Do not quote index counts from memory.** The schema currently declares 3 vector indexes
(`memories`, `episodes`, `cache`), 2 full-text indexes (`memories`, `episodes`), and 7 TTL
indexes. Those move as the library changes, and Compass is showing the truth right next to
you. Say "vector, full-text, and TTL, in the same index list" and point.

Two fields not to over-claim, both checked against the current seed. Short-term documents
carry **`conversation_id`, not `thread_id`** — `thread_id` lives on `episodes`. And
`files_touched` is **empty**, because the demo agent has no filesystem tools; it is a good
answer to a question and a bad thing to point at. Slide 10 shows the populated shape, which
is where that field belongs.

Have every Compass tab pre-opened, pre-scrolled, connected before the session starts.
Static screenshots are a perfectly good substitute and I will not apologize for them.

---

## Slide 8 — One pipeline

The retrieval layer, in one aggregation:

```js
{ $rankFusion: {
    input: { pipelines: {
      vectorPipeline:   [ { $vectorSearch: { index: "memories_vector_index",   // ← 1  MEANING
                                             path: "embedding",
                                             queryVector: [...],
                                             filter: { user_id, deleted_at: null } } } ],  // ← 4  ISOLATION
      fullTextPipeline: [ { $search: { index: "memories_fts_index",            // ← 2  EXACT TERMS
                                       compound: { must:   [ { text: { query, path: ["content","summary"] } } ],
                                                   filter: [ { equals: { path: "user_id", value: userId } } ] } } } ]  // ← 4
    } },
    combination: { weights: { vectorPipeline: 1.0, fullTextPipeline: 0.7 } }   // ← 3  MERGE BOTH
} }
```

**Four things to see, and nothing else:**

| | | |
|---|---|---|
| **1** | `$vectorSearch` | meaning |
| **2** | `$search` | exact terms |
| **3** | `$rankFusion` | merges both rankings, natively |
| **4** | `filter` in **both** branches | per-user isolation |

Meaning **and** exact terms. Native reciprocal rank fusion. **One round trip.**

**Notes (1:30).** The technical high point, and the slide people photograph — so the code
is there for the *photo*, not for reading at booth distance.

**Never read the code.** Point at the four numbered callouts in order, one clause each:
"Meaning. Exact terms. Merged natively. Filtered per user in both branches." Four
sentences, then talk to the callout table, not the JSON.

Design note for the built deck: dim the code block to ~60% and render the four callouts at
full contrast. Anyone who wants the pipeline will photograph it; anyone standing at the
back needs the four words.

"Vector search alone misses exact terms — product SKUs, error codes, a person's name.
Full-text alone misses meaning. You need both, and then you need to *merge two ranked
lists*, which is the part everyone gets wrong. `$rankFusion` is reciprocal rank fusion
running natively in the database. One query. One round trip. And notice the filter is
inside both branches — per-user isolation is enforced by the engine, not by hoping the
model behaves."

If someone asks: yes, `$rankFusion` needs MongoDB 8.1+; before that you compose it with
`$unionWith` and do the RRF math yourself, which the library also supports.

Then, in one line, the honest caveat if pressed: ranking after fusion is calibrated in
the application — recency, importance, relevance, weighted. That's a deliberate choice,
because "most similar" is not the same as "most worth showing."

---

## Slide 9 — Four boxes → one cluster

> **THIS IS THE CUT SLIDE.** If you are past 9:45 when you leave slide 8, or the crowd is
> thinning, skip it entirely and go to slide 10. Slide 10 is the differentiator; this one
> is the argument. Losing the argument costs you nothing — slide 11 makes it in two lines,
> and Thursday's main session *is* this slide for 45 minutes.
>
> **Compressed version (0:20), if you want it but not at full price:** show the two
> diagrams, say "**Four systems, four consistency models, four on-call rotations — or one
> cluster.**" Then advance. No trade-off discussion, no sync-job riff.

**Before**

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐
│ Vector   │  │  Cache   │  │  Search  │  │  System of   │
│   DB     │  │ (Redis)  │  │  engine  │  │   record     │
└──────────┘  └──────────┘  └──────────┘  └──────────────┘
      └─────────── sync jobs, drift, 4× ops ──────────┘
```

**After**

```
┌───────────────────────────────────────────────┐
│              MongoDB Atlas                    │
│  memories · episodes · semantic_cache ·       │
│  decisions · audit_log                        │
│  $vectorSearch · $search · $rankFusion · TTL  │
└───────────────────────────────────────────────┘
```

**Notes (1:00).** "Every arrow you delete is a failure mode you delete. Four systems
means four consistency models, four security reviews, four on-call rotations — and a sync
job that is always a little bit behind."

The word to use is *collapse*, because it hands off to Thursday.

Don't oversell it as free. If asked about the trade-off, the answer is real: you're
trading best-of-breed specialization for one operational surface, and for agent memory
specifically that trade is easy — the workload is small vectors, high write rate, and
mandatory per-user filtering, which is a document-database shape.

---

## Slide 10 — The one people skip: episodic

Semantic memory knows **facts**. Episodic memory knows **what happened**.

```jsonc
{ "user_id": "...", "thread_id": "...", "step": 7, "parent_step": 6,
  "agent_name": "researcher",
  "messages": [ /* what was said, with tool calls and token usage */ ],
  "todos":    [ /* what it planned */ ],
  "files_touched": [ { "path": "report.md", "op": "write" } ],
  "correlation_id": "trace-id-from-your-tracing-stack" }
```

Append-only · TTL'd · hybrid-searchable per user · never on the hot path

**Notes (1:15).** This is the slide that separates this talk from every other memory talk
at the show.

"Ask your agent 'what did we decide about the Q3 forecast, and what did you actually
change?' Semantic memory can't answer that — it stores facts, not events. You need the
turn log: what was said, which tools ran, which files got touched, in what order, under
which trace ID.

And it has to be free at the call site. Writing this is fire-and-forget — it goes on a
bounded queue and a background worker batches it to Atlas. If the queue fills, it drops
the *oldest* turn, never the newest. If the write fails, you get a counter, not an
exception. Your agent never waits on its own diary."

If someone asks why not use their observability vendor: those are built for humans
debugging after the fact. This is built for the *agent* to query mid-loop, per user, with
the same hybrid search as everything else — and it lives next to the rest of your data.

---

## Slide 11 — Takeaway

> ### Context engineering is a data problem.
> ### Solve it in the data layer.

Four memories, one cluster:
**state · facts · events · cache**

**Notes (0:30).** Say the two lines. Stop. Let it sit for a beat — this is the sentence
you want them repeating at the next booth.

---

## Slide 12 — Build it today

### 1 · Install it

```
uv add git+https://github.com/mongodb-partners/ai-memory.git
```

### 2 · Read it

`github.com/mongodb-partners/ai-memory`
Framework-neutral Python library. Use it directly, over MCP, or over REST.
Sample UI — the memory panel from slide 6 — included.

### 3 · Go deeper

Blog: *Build AI Memory Systems with MongoDB Atlas, AWS & Claude*

### 4 · See the whole story

**Main session — Thursday, Aug 6.**
*"For Agents: Deployment at Scale."*

**Notes (0:15).** Four things, in this order, in one breath — the order is the ask, from
cheapest commitment to biggest:

"**Install it straight from the repo** — one command, it's open source. **Repo's on
screen**, sample UI is in it. **Blog** if you want the long version. And **the whole
stack-collapse story is my main session Thursday** — come to that one."

Do not say "pip install agent-memory" — it is not on PyPI, and somebody in the front row
will try it while you are still talking.

QR to the repo on the booth backdrop, not on the slide — the slide is up for 15 seconds and
nobody scans a moving target.

Then stop talking and take one question. Do not start a second talk.

---

## Appendix — answers to hold ready

Not slides. Booth Q&A, one sentence each.

**"How is this different from a vector database + a cache?"**
It's the same query engine over the same documents — no sync, no drift, and the memory
the agent writes is the memory it reads.

**"Does it work with my agent framework?"**
The library has zero framework dependencies. You call `recall()` before your model call
and `add()` after. Adapters are a thin layer on top, not a requirement.

**"What model does the demo use?"**
Bedrock for chat and embeddings by default; OpenAI, Anthropic, and Voyage are one config
line. The library is provider-neutral too.

**"How do you keep one user's memory out of another's?"**
The user filter is inside the vector and full-text branches of the query, plus governance
profiles and per-user rate limits in front of it. It's not prompt-level, and the search
tool never accepts a user ID from the model.

**"What does the importance score do?"**
It's assessed at write time and reinforced on access; final ranking is a weighted blend
of recency, importance, and relevance, so "most similar" doesn't automatically win.

**"Cost?"**
The cache is the answer — repeated semantically-identical questions don't hit the model.
Everything else is small vectors and a high write rate, which is cheap in Atlas.

**"Is this production-ready?"**
It's an open-source library from MongoDB's partner solutions team, tested end to end. Use
it as a reference implementation or as a dependency — read the design doc before you bet
a roadmap on it.

**Do not answer:** anything about a competing framework's roadmap, or any benchmark
number I can't produce on the spot.
