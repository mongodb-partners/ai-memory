# Why episodic memory

Most memory libraries give an agent one kind of memory: a vector store of facts.
That covers "what do I know about this user." It does not cover "what did we
actually *do*."

Those are different questions, and only the second one can answer:

- Why did the agent book the wrong restaurant on Tuesday?
- Which tool call wrote to `plan.md`, and what was in the turn before it?
- Show me every turn under trace id `4bf92f…`, the one in the support ticket.
- Have I already tried this approach in this thread, and did it work?

## The two things agents forget

Semantic memory forgets nothing about *content* and everything about *process*.
Store "the user is vegetarian" and you can recall it forever, but you cannot
reconstruct the conversation that revealed it, the tool the agent called to
confirm it, or whether the agent acted on it three turns later.

That gap matters more for agents than for chatbots, because an agent takes
actions. A chatbot that forgets a turn produces a slightly worse answer. An agent
that forgets a turn re-runs a side effect, re-litigates a decision it already
made, or reports success on work it never did. The record of actions is not
observability trivia. It is state the agent needs to reason correctly.

## Why not just keep the transcript

Two reasons.

**A transcript is not queryable.** "What did we do last Tuesday" over a raw
message array means loading everything and hoping the model finds it. Episodic
memory is one document per turn with `step`, `parent_step`, `files_touched`,
`tool_calls`, and a `correlation_id`, so the same question is an index seek, and
"turns where the agent touched this file" is a query rather than a reading
exercise.

**A transcript has no retention policy.** Turn logs grow without bound and most
of their value decays in days. Episodic memory has a TTL you can change in place
(`set_activity_retention`), so you keep 30 days by default, 2 hours in a load
test, or forever for an audited workflow, without a migration.

## Why not just use your tracing stack

Traces answer the operator's question: what happened, how long did it take, where
did it fail. They are sampled, they expire on the vendor's schedule, and they are
not reachable from inside the agent's own reasoning loop.

Episodic memory answers the *agent's* question, and it is a MongoDB collection,
so `recall_activity` can put "here is what you did about this last week" straight
into the next prompt. That is not something a trace backend is built to do.

The two are complementary, and that is why `correlation_id` accepts a W3C
`traceparent`. An episodic record carries the same trace id the rest of your
stack uses, so you can pivot from a trace to the turns that produced it and back.
No competing id scheme.

## Why it lives in the same cluster

The alternative is a fourth system: a vector store for facts, a key-value store
for session state, a cache in front of the model, and a log for actions. Four
consistency stories, four failure modes, four things to operate, and no way to
ask a question that spans them.

In Atlas all four are queries over collections in one database. Episodic recall
uses the same `$rankFusion` hybrid search as semantic recall, and the pipeline
builder is literally shared code (`services/search_pipeline.py`), because a fix
to the fusion logic should not be able to land in only one tier.

And because it is one database, joining tiers is a query rather than an
integration. `conversation_id` is on both the `memories` and `episodes`
documents, so "the facts we learned during the turns of this conversation" is
answerable.

## What it costs

Three trade-offs, and they are the reason this tier usually gets skipped:

**Write volume.** One document per turn is a lot of documents. That is why
`log_activity` never awaits Atlas: it builds the document and enqueues, and a
single consumer task batches inserts. It is also why `log_activity` is the one
operation that does not write a per-call audit record: one audit write per turn
would mean logging the agent costs more writes than the agent.

**Embedding cost.** Only final steps get embedded by default. A step that ends in
a tool request has a question but no answer yet, so its vector would represent
half a turn, and the cost buys nothing. The steps that do get embedded are sent as
one call per batch rather than one per document, so a twenty-turn batch is one
request against the provider's rate limit.

**Retention pressure.** Hence the TTL index, and hence making it tunable at
runtime instead of at deploy time.

## Further reading

- [The document shape](../reference/episodic-document-shape.md): the contract
- [Observability](../how-to/observability.md): the counters and what they mean
- [Configuring TTL](../how-to/configure-ttl.md): changing retention in place
- [Architecture](architecture.md): how the write path is built
