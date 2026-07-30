#!/usr/bin/env python3
"""Train the coefficient artifacts that `IMPORTANCE_SCORER=local` loads.

Offline, optional, and never imported by the library: `pip install
'agent-memory[training]'` then run this. The dependency direction is one-way on
purpose — this script imports `agent_memory`, and nothing under `agent_memory/`
may ever import sklearn or numpy. `tests/unit/test_packaging.py` greps for that.

Four label sources, combined as sequential stages of one lifecycle rather than as
four alternatives to pick between:

1. **Benchmark anchor** (`--source benchmark`). `xiaowu0162/longmemeval-cleaned`
   and `adymaharana/locomo` are multi-session conversations with questions asked
   in later sessions. A turn cited as evidence for a later question was
   *demonstrably* useful later, which is the target quantity itself rather than a
   proxy for it. Stated limits, because they bound what this stage can do:

   - Labels are **sparse**. Most turns are unlabeled, not negative. Treating
     unlabeled as negative is an assumption, and it is opt-in via
     `--negative-ratio` rather than folded in silently.
   - Labels are near-binary, so this stage alone cannot populate a continuous
     0.1–1.0 scale.
   - The domain is casual conversation, not any particular deployment's domain.
   - **Neither dataset ships importance labels. This derivation is ours.**

2. **LLM distillation** (`--source llm`). Run the shipped `assess_importance`
   over the same corpora plus a synthetic set spanning the scale, and regress on
   what it returns. This is what makes the local path *agree with* the LLM rather
   than merely correlate with it, which is what matters when
   `forgetting_score_threshold` and `promotion_importance_threshold` are absolute
   cutoffs: two scorers only mean the same thing at 0.1 if their scales coincide.

3. **Combined bake-off** (`--source combined`, the default). Benchmark labels get
   `--benchmark-weight`, LLM labels supply density across the range.
   `LogisticRegression` and `Ridge` are fitted side by side, both scored, both
   printed, and only the winner is written.

   `--benchmark-weight` now defaults to **0.0** — the benchmark stage is collected
   (it supplies the corpus the LLM stage samples) but excluded from the fit. That
   default is a measurement, not a preference. Stage 1's labels mark a turn positive
   when a later question cited it as evidence, and the benchmark questions are
   time-anchored ("when did you..."), so `yesterday` carries a 21.7x lift toward
   positives and `today` 3.2x. Any nonzero weight trains `temporal` strongly
   positive, and the resulting model scores "Let's pick this up after lunch, I'm
   busy today and tomorrow" at 0.775 — above the 0.6 promotion threshold — while
   scoring "Our policy is that customer data never leaves the EU region" at 0.435.
   It promotes expiring chatter and drops standing policy. Every nonzero weight
   from 0.1 to 2.0 was tried; all fail `DISCRIMINATION_CASES`.

   The stage is kept rather than deleted because the confound is specific to the
   seven lexical features, which can only see the *word* `yesterday`. An embedding
   head can distinguish "we deployed yesterday" from "call me tomorrow", so the
   grounded-utility signal is still worth having there.

4. **Deployment retraining** (`--source mongodb`). Pull the operator's own
   memories and label them from signals the documents already carry —
   `access_count`, age, and whether consolidation soft-deleted them. The shipped
   artifact is the cold-start bootstrap; this is how a deployment moves past it.

Everything is scored through the *runtime's* arithmetic — `logistic` and the
`[0.1, 1.0]` clamp imported from `agent_memory.services.importance`, not
sklearn's `predict` — so a metric printed here is a metric the served model will
reproduce. That is the whole reason the reported numbers can be trusted enough to
commit.

Only sklearn and numpy are used. The `datasets` library is deliberately not:
locomo's HuggingFace repo holds no data files (the JSON lives in the
`snap-research/locomo` GitHub repo), and longmemeval ships three single-file
splits where the smallest useful one is 15MB next to a 2.7GB sibling. Two plain
downloads with a cache directory are less machinery than a loader script that
would still need per-dataset shape code.

Usage:
    python scripts/train_importance.py --source combined --space lexical --dry-run
    python scripts/train_importance.py --source combined --space lexical \\
        --out agent_memory/data/importance/lexical.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

# The script may import the library. The reverse must never happen.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The *class*, not an instance. Every other use of `MCPConfig` in this script is
# a function-local import because constructing one reads the environment and a
# live `.env`; reading a field's declared default touches neither.
from agent_memory.core.config import MCPConfig
from agent_memory.services.importance import (
    LEXICAL_FEATURE_NAMES,
    MAX_IMPORTANCE,
    MIN_IMPORTANCE,
    SCHEMA_VERSION,
    lexical_features,
    logistic,
)

logger = logging.getLogger("train_importance")

RANDOM_SEED = 20260730

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LONGMEMEVAL_URLS = {
    # The oracle split holds only sessions some question draws on, so it supplies
    # positives and almost no negatives. The `s` split carries ~23k uncited
    # sessions, which is where negatives come from — at 277MB, hence the choice.
    "oracle": (
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
        "resolve/main/longmemeval_oracle.json"
    ),
    "s": (
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
        "resolve/main/longmemeval_s_cleaned.json"
    ),
}

DEFAULT_CACHE_DIR = Path(
    os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
) / "agent-memory" / "importance"

# Below this, text is too short to carry a feature signal and mostly adds noise
# ("ok", "thanks"). Read from `ltm_candidate_min_chars` — the same threshold
# `MemoryService.store_stm` uses to decide what becomes an LTM candidate at all,
# because training on text the runtime never scores would be training on the
# wrong distribution.
#
# Imported rather than copied: this was a literal 31 with a comment pointing at
# the runtime's literal 30, which is two numbers that had already drifted by one
# and would drift further the moment either moved. A deployment that tunes the
# threshold and then retrains gets a matching training set for free.
MIN_CONTENT_CHARS = MCPConfig.model_fields["ltm_candidate_min_chars"].default

# Logit-space fitting needs labels strictly inside (0, 1): logit(1.0) is
# infinite. 0.02 keeps the transformed target inside ±3.9, which is well within
# the range a linear model can reach without saturating.
_LOGIT_EPS = 0.02

_SESSION_KEY_RE = re.compile(r"^session_\d+$")


# --------------------------------------------------------------------------
# Stage 1 — benchmark labels
# --------------------------------------------------------------------------


def derive_benchmark_labels(
    sessions: list[dict], questions: list[dict]
) -> list[tuple[str, float]]:
    """Positive if cited by a later question, negative if in an uncited session.

    Anything else is dropped. That third case is the whole point: a turn sitting in
    a session some question drew on, which that question did not cite, is
    *unlabeled* — we cannot tell whether it was useless or simply not what anybody
    happened to ask about. Folding it into the negatives is the easiest way to
    train a model that scores almost everything as forgettable, and every aggregate
    metric still looks reasonable while it happens.

    Returns ``[(content, label)]``. Empty when ``questions`` is empty: with nothing
    cited, "uncited session" describes the entire corpus, and a corpus labeled
    entirely negative trains a scorer that deletes everything.

    Both benchmark loaders normalize into this shape, so this function is dataset
    agnostic:

    - ``sessions``: ``[{"session_id": str, "turns": [{"turn_id": str,
      "content": str}]}]``
    - ``questions``: ``[{"evidence_turn_ids": [...], "evidence_session_ids":
      [...]}]``
    """
    if not questions:
        return []

    cited_turns: set[str] = set()
    cited_sessions: set[str] = set()
    for q in questions:
        cited_turns.update(q.get("evidence_turn_ids") or ())
        cited_sessions.update(q.get("evidence_session_ids") or ())

    labeled: list[tuple[str, float]] = []
    for session in sessions:
        session_is_cited = session.get("session_id") in cited_sessions
        for turn in session.get("turns") or ():
            content = turn.get("content")
            if not content:
                continue
            if turn.get("turn_id") in cited_turns:
                labeled.append((content, 1.0))
            elif not session_is_cited:
                # No question drew on this session at all, so nothing in it was
                # needed later. The closest thing to a real negative these
                # datasets offer.
                labeled.append((content, 0.0))
            # else: unlabeled. Dropped, not defaulted. See the docstring.
    return labeled


def unlabeled_benchmark_turns(
    sessions: list[dict], questions: list[dict]
) -> list[str]:
    """The turns `derive_benchmark_labels` drops — the middle case, on its own.

    Exposed so `--negative-ratio` can sample from them explicitly. An operator who
    wants "treat uncited turns in cited sessions as weak negatives" can have it;
    they just have to type the ratio, which keeps the assumption on the command
    line instead of buried in a label function.
    """
    if not questions:
        return []

    cited_turns: set[str] = set()
    cited_sessions: set[str] = set()
    for q in questions:
        cited_turns.update(q.get("evidence_turn_ids") or ())
        cited_sessions.update(q.get("evidence_session_ids") or ())

    dropped: list[str] = []
    for session in sessions:
        if session.get("session_id") not in cited_sessions:
            continue
        for turn in session.get("turns") or ():
            content = turn.get("content")
            if content and turn.get("turn_id") not in cited_turns:
                dropped.append(content)
    return dropped


# --------------------------------------------------------------------------
# Corpus loading
#
# Two datasets, two entirely different shapes, one normalized form. The
# normalization lives here rather than inside `derive_benchmark_labels` so the
# label rule stays dataset-agnostic and unit-testable without either download.
# --------------------------------------------------------------------------


def _download(url: str, dest: Path) -> Path:
    """Fetch to a cache path, reusing an existing file.

    Cached rather than re-fetched because longmemeval's `s` split is 277MB and a
    trainer that re-downloads it on every invocation is a trainer nobody iterates
    with. Written to a `.part` file and renamed, so an interrupted download does
    not leave a truncated file that looks cached.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("using cached %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return dest

    logger.info("downloading %s -> %s", url, dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as fh:
        while chunk := response.read(1 << 20):
            fh.write(chunk)
    tmp.rename(dest)
    logger.info("downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def load_longmemeval(cache_dir: Path, split: str = "oracle") -> tuple[list[dict], list[dict]]:
    """Normalize a longmemeval split.

    Record shape: ``haystack_sessions`` is a list of turn lists, parallel to
    ``haystack_session_ids``; ``answer_session_ids`` names the sessions holding the
    answer; and individual turns carry ``has_answer: true``. Turn ids are not in
    the data, so they are synthesized positionally — unique per record, which is
    all `derive_benchmark_labels` needs.

    The same session id can appear across records with different citation status,
    so ids are namespaced by ``question_id``. Without that, one question's cited
    session would suppress another question's negatives.
    """
    path = _download(LONGMEMEVAL_URLS[split], cache_dir / f"longmemeval_{split}.json")
    records = json.loads(path.read_text())

    sessions: list[dict] = []
    questions: list[dict] = []
    for record in records:
        qid = record["question_id"]
        answer_ids = set(record.get("answer_session_ids") or ())
        evidence_turns: list[str] = []
        evidence_sessions: list[str] = []

        haystack_ids = record.get("haystack_session_ids") or []
        for s_index, turns in enumerate(record.get("haystack_sessions") or []):
            raw_id = haystack_ids[s_index] if s_index < len(haystack_ids) else str(s_index)
            session_id = f"{qid}:{raw_id}"
            norm_turns = []
            for t_index, turn in enumerate(turns):
                turn_id = f"{session_id}:{t_index}"
                norm_turns.append({
                    "turn_id": turn_id,
                    "content": turn.get("content"),
                    "role": turn.get("role"),
                })
                if turn.get("has_answer"):
                    evidence_turns.append(turn_id)
            sessions.append({"session_id": session_id, "turns": norm_turns})
            if raw_id in answer_ids:
                evidence_sessions.append(session_id)

        questions.append(
            {
                "question": record.get("question"),
                "evidence_turn_ids": evidence_turns,
                "evidence_session_ids": evidence_sessions,
            }
        )

    return sessions, questions


def load_locomo(cache_dir: Path) -> tuple[list[dict], list[dict]]:
    """Normalize locomo.

    Record shape: ``conversation`` holds ``session_1``, ``session_2``, … each a
    list of turns with a ``dia_id`` like ``"D1:3"``, and ``qa[].evidence`` cites
    those ids directly. Every session in locomo is cited by something, so this
    corpus supplies positives and unlabeled turns but essentially no negatives —
    longmemeval's `s` split is where negatives come from.

    Ids are namespaced by ``sample_id``: ``dia_id`` restarts at ``D1:1`` in every
    conversation, so unnamespaced ids would cross-contaminate citations between
    unrelated conversations.
    """
    path = _download(LOCOMO_URL, cache_dir / "locomo10.json")
    records = json.loads(path.read_text())

    sessions: list[dict] = []
    questions: list[dict] = []
    for record in records:
        sample_id = record.get("sample_id", "?")
        conversation = record.get("conversation") or {}

        turn_to_session: dict[str, str] = {}
        for key in sorted(k for k in conversation if _SESSION_KEY_RE.match(k)):
            session_id = f"{sample_id}:{key}"
            norm_turns = []
            for t_index, turn in enumerate(conversation[key] or ()):
                raw = turn.get("dia_id") or f"{key}:{t_index}"
                turn_id = f"{sample_id}:{raw}"
                turn_to_session[turn_id] = session_id
                norm_turns.append({
                    "turn_id": turn_id,
                    "content": turn.get("text"),
                    # locomo is two humans talking to each other — there is no
                    # assistant role to exclude, so every turn is human-authored.
                    "role": "user",
                })
            sessions.append({"session_id": session_id, "turns": norm_turns})

        for qa in record.get("qa") or ():
            evidence_turns = [
                f"{sample_id}:{e}"
                for e in (qa.get("evidence") or ())
                if isinstance(e, str)
            ]
            questions.append(
                {
                    "question": qa.get("question"),
                    "evidence_turn_ids": evidence_turns,
                    # Derived rather than given: locomo cites turns, not sessions,
                    # and a session holding a cited turn is by definition drawn on.
                    "evidence_session_ids": sorted(
                        {turn_to_session[t] for t in evidence_turns if t in turn_to_session}
                    ),
                }
            )

    return sessions, questions


# --------------------------------------------------------------------------
# Stage 4 — labels from a live deployment
# --------------------------------------------------------------------------

# Accesses above which a memory is unambiguously useful. Chosen against
# `promotion_access_threshold` (default 2): a memory needs a couple of accesses
# just to be promoted, so "frequently accessed" has to mean well past that or the
# label is just measuring promotion.
_FREQUENT_ACCESS = 10

# Age past which "never accessed" starts to mean something. Below it, a memory has
# not had the chance to be useful yet.
_MATURITY_DAYS = 30


def label_from_mongo_document(doc: dict, *, now: datetime) -> float | None:
    """Label one memory document from signals it already carries, or ``None``.

    ``None`` is returned liberally and that is the point. A memory created an hour
    ago with ``access_count: 0`` is *unlabeled*, not unimportant — it has not had
    the chance to be useful. Scoring it 0 teaches the model that everything new is
    worthless, and new is exactly when scoring happens, so the error compounds on
    every future write.

    Signals, in priority order:

    1. Soft-deleted by consolidation for low importance — consolidation already
       made the judgement, so reuse it.
    2. Accessed at least ``_FREQUENT_ACCESS`` times — demonstrably useful, scaled
       by how often.
    3. Older than ``_MATURITY_DAYS`` and never accessed — had its chance.
    4. Anything else — ``None``.

    Note on signal 1: the shipped `_forget_low_importance` sets ``is_deleted`` but
    writes no ``deleted_reason``, so this branch only fires for deployments that
    add one, and a bare ``is_deleted`` is deliberately *not* treated as low —
    ``delete_memories`` soft-deletes on explicit user request too, and a memory the
    user deleted by hand says nothing about its importance.
    """
    if doc.get("is_deleted") and doc.get("deleted_reason") == "low_importance":
        return 0.05

    access_count = doc.get("access_count") or 0
    if access_count >= _FREQUENT_ACCESS:
        # log-scaled: the 10th access is far more informative than the 100th, and
        # a linear scale would make one hot memory an outlier that dominates the
        # fit. Reaches 1.0 at ~1000 accesses.
        return min(
            MAX_IMPORTANCE,
            0.7 + 0.3 * math.log10(access_count / _FREQUENT_ACCESS + 1) / 2.0,
        )

    created_at = doc.get("created_at")
    if isinstance(created_at, datetime):
        if created_at.tzinfo is None:
            # pymongo returns naive UTC by default. Comparing naive to aware
            # raises, and defaulting to local time would shift the age by hours.
            created_at = created_at.replace(tzinfo=timezone.utc)
        age = now - created_at
        if age >= timedelta(days=_MATURITY_DAYS) and access_count == 0:
            return 0.15

    return None


# --------------------------------------------------------------------------
# Synthetic scale coverage
#
# The benchmarks are near-binary and casual. These span the middle of the scale
# and the memory *kinds* a deployment actually stores — standing preferences,
# constraints, throwaway task chatter. They carry no hand-assigned labels: the LLM
# scores them, exactly as it would in production. Their only job is to give the
# distillation stage inputs across the range rather than two clusters at the ends.
# --------------------------------------------------------------------------

SYNTHETIC_CONTENT = [
    # Standing constraints and preferences — the archetypal long-term memory.
    "I'm allergic to penicillin, so it should never be prescribed to me.",
    "I am vegetarian and I really dislike cilantro in anything.",
    "My team always deploys behind a feature flag, never straight to production.",
    "Our convention is that every pull request needs two approvals before merge.",
    "I prefer written summaries over meeting invitations whenever there's a choice.",
    "Never schedule anything for me on Friday afternoons; that time is blocked.",
    "My daughter's birthday is on the 14th of March and I always take that day off.",
    "I require all invoices to be sent to accounts-payable, not to me directly.",
    "We standardised on Postgres for anything transactional and Mongo for documents.",
    "My mother lives in Pune and I visit her every December.",
    "I'm colourblind for red and green, so status colours alone don't work for me.",
    "Our policy is that customer data never leaves the EU region.",
    # Durable facts about the user — useful later, less imperative.
    "I have been working as a data engineer at a logistics company for six years.",
    "I'm learning Portuguese because we're relocating to Lisbon next year.",
    "My apartment building has no lift, which matters for any furniture delivery.",
    "I did my undergraduate degree in mechanical engineering before moving to software.",
    "I run about thirty kilometres a week and I'm training for a half marathon.",
    # Mid-scale: durable but narrow, or a decision that will expire.
    "We decided to use the blue variant of the logo for the conference booth.",
    "The staging database was migrated to the new cluster last quarter.",
    "I usually take the 8:15 train, though it varies with the school term.",
    "The vendor quoted around forty thousand for the annual licence renewal.",
    "I set the retry limit to five because three was too aggressive for our network.",
    # Scoped, expiring task chatter — the archetypal short-term memory.
    "Can you deploy branch fix-3 today?",
    "What time is the standup tomorrow?",
    "Please rerun the failing test one more time.",
    "I'm in a meeting right now, I'll look at it later.",
    "Could you resend that link? I lost the tab.",
    "Let's pick this up after lunch.",
    "The build is currently red, someone is looking at it.",
    "Yes, that works for me, go ahead.",
    "Thanks, that's exactly what I needed.",
    "Sorry, ignore my last message, I misread the dashboard.",
    "Are you around for a quick call in ten minutes?",
    "I'll be a few minutes late to the sync today.",
]


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, hand-rolled to avoid a scipy dependency.

    Average ranks for ties, then Pearson on the ranks. Ties matter here: labels
    from the benchmark stage are mostly two values, so a naive `argsort` ranking
    would invent an ordering inside each tied block and report a correlation that
    is partly an artifact of input order.
    """
    if len(a) < 2:
        return 0.0

    def ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="stable")
        out = np.empty(len(x), dtype=float)
        sorted_x = x[order]
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
                j += 1
            out[order[i : j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return out

    ra, rb = ranks(np.asarray(a, dtype=float)), ranks(np.asarray(b, dtype=float))
    if ra.std() == 0 or rb.std() == 0:
        # A constant prediction has no ranking at all. Reporting 0 rather than nan
        # keeps `composite_score` arithmetic finite, and the calibration columns
        # will already be showing how bad such a model is.
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(
    y_true,
    y_pred,
    *,
    forget_threshold: float = 0.1,
    promote_threshold: float = 0.6,
) -> dict:
    """Metrics for a scorer, weighted toward the decisions the score feeds.

    The thresholds default to `forgetting_score_threshold` and
    `promotion_importance_threshold` from `MCPConfig`, because those are what
    consume this number in production. They are parameters rather than constants so
    an operator who has tuned their own thresholds can measure agreement at the
    values they actually run.

    ``*_agreement`` is the fraction of samples where prediction and label fall on
    the same side of the threshold — i.e. where both scorers would make the same
    keep/delete or promote/hold decision. It is the operationally meaningful
    column: *of the memories the LLM would forget, how many does this model also
    forget?*

    ``mean_offset`` is signed (``mean(pred) - mean(true)``) because the direction
    tells you which failure you have: negative forgets too much, positive promotes
    too much. `composite_score` takes the magnitude; a human reading the table
    wants the sign.
    """
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)

    return {
        "spearman": _spearman(true, pred),
        "mae": float(np.mean(np.abs(pred - true))),
        "mean_offset": float(np.mean(pred) - np.mean(true)),
        "mean_pred": float(np.mean(pred)),
        "mean_label": float(np.mean(true)),
        # `<=`, not `<`, and the difference is load-bearing. Consolidation uses
        # `$lt`, but *no scorer can emit a value below the threshold*: both
        # `parse_importance` and `LocalScorer` floor at `MIN_IMPORTANCE` (0.1),
        # which equals the default `forgetting_score_threshold`. A strict `<`
        # therefore evaluates False for every prediction and every mapped label,
        # making the column identical across all candidates including a constant —
        # a metric that reports 0.11 for everything and discriminates nothing.
        # `<=` asks the question that is actually decidable: do both scorers put
        # this memory *at the forgetting boundary*? See the note in `_map_to_servable`.
        "forget_agreement": float(
            np.mean((pred <= forget_threshold) == (true <= forget_threshold))
        ),
        "promote_agreement": float(
            np.mean((pred >= promote_threshold) == (true >= promote_threshold))
        ),
        "n": int(len(true)),
    }


# Calibration outranks correlation because consolidation compares against absolute
# thresholds rather than sorting. A model with Spearman 0.85 and a +0.15 mean
# offset ranks memories beautifully and promotes nearly all of them.
# `generate_models.py:100-115` in the reference recommendation app is where this
# composite-metric shape comes from; the weights are inverted relative to it,
# because that app ranks and this one thresholds.
_WEIGHTS = {
    "forget_agreement": 0.30,
    "promote_agreement": 0.30,
    "mae": 0.20,          # inverted below
    "mean_offset": 0.15,  # magnitude, inverted below
    "spearman": 0.05,
}


def composite_score(metrics: dict) -> float:
    """One number for the bake-off, in ``[0, 1]``, higher is better.

    Each component is normalized so that 1.0 is perfect: agreements are already
    fractions, ``mae`` and ``|mean_offset|`` are inverted as ``1 - min(1, v)``, and
    ``spearman`` is mapped from ``[-1, 1]`` to ``[0, 1]``.

    Missing keys are treated as 0 (i.e. worst) rather than skipped. A metrics dict
    that lost a column should lose the bake-off, not quietly win it by being
    scored on fewer terms.
    """
    total = 0.0
    total += _WEIGHTS["forget_agreement"] * float(metrics.get("forget_agreement", 0.0))
    total += _WEIGHTS["promote_agreement"] * float(metrics.get("promote_agreement", 0.0))
    total += _WEIGHTS["mae"] * (1.0 - min(1.0, abs(float(metrics.get("mae", 1.0)))))
    total += _WEIGHTS["mean_offset"] * (
        1.0 - min(1.0, abs(float(metrics.get("mean_offset", 1.0))))
    )
    total += _WEIGHTS["spearman"] * ((float(metrics.get("spearman", -1.0)) + 1.0) / 2.0)
    return total


# --------------------------------------------------------------------------
# Discrimination gate
#
# `composite_score` is a calibration metric, and calibration is satisfiable by a
# model that discriminates nothing: predict the training mean for every input and
# `mae`, `mean_offset` and both agreement columns all look good. Measured, not
# hypothetical — on the LLM stage a Ridge fit scored composite 0.879 while
# separating archetypal-keep from archetypal-forget text by 0.002, and it won the
# bake-off against a LogisticRegression that separated them by 0.037.
#
# So the bake-off needs a second, independent question: does the model actually
# rank a standing preference above expiring task chatter? These pairs are the
# cheapest possible way to ask. They are *held out by construction* — none is in
# SYNTHETIC_CONTENT, so no candidate is ever fitted on them.
#
# This is a floor, not a benchmark. Passing it means the model is not a constant;
# it does not mean the model is good.
# --------------------------------------------------------------------------

DISCRIMINATION_CASES = (
    # Durable: standing constraints, policies, stable personal facts.
    ("I must never be prescribed anything containing codeine.", True),
    ("Our standard requires two reviewers on any schema migration.", True),
    ("My wheelchair means I always need step-free access to a venue.", True),
    ("I require every contract to be countersigned by our legal team.", True),
    # Expiring: scoped task chatter. The last two carry temporal markers *and* no
    # question mark, which is the combination a temporal-positive model scores
    # above the durable set. That defect is invisible to every aggregate metric
    # computed on a QA-evidence corpus, so it has to be named case-by-case here.
    ("Could you restart the staging job for me?", False),
    ("sounds good", False),
    ("I'm heads-down until tomorrow, so let's regroup today or tonight.", False),
    ("The pipeline is currently stuck, someone is looking at it now.", False),
)


def discrimination_margin(coefficients, intercept: float) -> float:
    """``min(durable) - max(expiring)`` over `DISCRIMINATION_CASES`.

    Positive means every durable case outranks every expiring one. The *worst*
    durable case against the *best* expiring one, deliberately: a mean-vs-mean
    comparison passes while individual pairs are inverted, and an inverted pair is
    a memory consolidation deletes for the wrong reason.

    Lexical only — the cases are raw text, and an embedding artifact would need the
    configured embedder to score them. `_run` handles that.
    """
    features = [lexical_features(text) for text, _ in DISCRIMINATION_CASES]
    scores = predict(coefficients, intercept, features)
    durable = [s for s, (_, keep) in zip(scores, DISCRIMINATION_CASES) if keep]
    expiring = [s for s, (_, keep) in zip(scores, DISCRIMINATION_CASES) if not keep]
    return float(min(durable) - max(expiring))


# --------------------------------------------------------------------------
# Artifact emission
# --------------------------------------------------------------------------


def build_artifact(
    kind: str,
    coefficients,
    intercept: float,
    *,
    embedding: dict | None = None,
    training: dict,
) -> dict:
    """Build the artifact document `load_artifact` consumes.

    Round-tripped through the real loader by the tests, so the trainer cannot emit
    a file the runtime rejects — which is the failure that would otherwise surface
    as a startup crash on an operator's machine rather than here.

    For ``kind == "lexical"`` the feature names are written into the training block.
    The coefficients are positional, and a file of seven bare numbers with no record
    of what they multiply is how a feature reordering becomes undiagnosable.
    """
    doc: dict = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "squash": "logistic",
        "coefficients": [float(c) for c in coefficients],
        "intercept": float(intercept),
        "training": dict(training),
    }
    if embedding is not None:
        doc["embedding"] = embedding
    if kind == "lexical":
        doc["training"]["feature_names"] = list(LEXICAL_FEATURE_NAMES)
    return doc


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def _logit(p: float) -> float:
    """Inverse of `logistic`, with the endpoints pulled inside.

    Needed because `Ridge` fits in the output space while the runtime applies
    `logistic` to the linear sum. Regressing on raw labels would produce weights
    that are correct for `dot(w, x)` and wrong for `logistic(dot(w, x))` — a
    mistake that yields a model whose printed metrics look fine and whose served
    scores all crowd toward 0.5.
    """
    p = min(1.0 - _LOGIT_EPS, max(_LOGIT_EPS, p))
    return math.log(p / (1.0 - p))


def _clamp(value: float) -> float:
    """The runtime's clamp, duplicated so predictions here match served scores.

    Imported constants, local function: `agent_memory.services.importance._clamp`
    is private, and reaching into it would couple the trainer to an internal name.
    The constants are the part that must not drift.
    """
    return max(MIN_IMPORTANCE, min(MAX_IMPORTANCE, value))


def _map_to_servable(labels: np.ndarray) -> np.ndarray:
    """Rescale labels into ``[MIN_IMPORTANCE, MAX_IMPORTANCE]``.

    Benchmark labels are 0.0 and 1.0, but no scorer can *emit* 0.0 — both
    `parse_importance` and `LocalScorer` floor at 0.1. Fitting against a target no
    model can reach guarantees a permanent error floor of 0.1 on every negative,
    and it reports as MAE rather than as the mismatch it is. Worse, the metrics
    become uninterpretable: `mean_offset` shows a systematic positive bias that is
    an artifact of the label space, not of the model.

    So labels are mapped onto the range the runtime can actually serve, and every
    metric is computed in that one shared space.

    Apply this to benchmark and Mongo-derived labels ONLY, never to LLM-distilled
    ones. It is an affine rescale, not a clamp: 0.2 becomes 0.28 and 0.3 becomes
    0.37. LLM labels come out of `parse_importance` already inside [0.1, 1.0], so
    rescaling them a second time shifts the whole distilled distribution upward and
    biases the model toward promotion — a quiet miscalibration against exactly the
    threshold that matters. `_servable_labels` handles the dispatch by stage.
    """
    span = MAX_IMPORTANCE - MIN_IMPORTANCE
    return MIN_IMPORTANCE + np.clip(np.asarray(labels, dtype=float), 0.0, 1.0) * span


def _servable_labels(labels: np.ndarray, sources: list[str]) -> np.ndarray:
    """Put every stage's labels in the servable range, rescaling only those that need it.

    `llm` labels are already there. `benchmark` labels are 0.0/1.0 indicators and
    `mongodb` labels are derived on a 0-1 scale, so both are rescaled.
    """
    labels = np.asarray(labels, dtype=float)
    needs_scaling = np.asarray([s != "llm" for s in sources])
    out = np.where(needs_scaling, _map_to_servable(labels), np.clip(labels, MIN_IMPORTANCE, MAX_IMPORTANCE))
    return out.astype(float)


def predict(coefficients, intercept: float, features) -> np.ndarray:
    """Score through the runtime's arithmetic, not sklearn's ``predict``.

    This is what makes the reported metrics trustworthy: `logistic` and the clamp
    are the library's own, so every number in the printed table is a number the
    served scorer will reproduce. Evaluating with `model.predict` instead would
    measure a model nobody runs — `Ridge.predict` skips the squash entirely, and
    `LogisticRegression.predict_proba` skips the clamp.
    """
    X = np.asarray(features, dtype=float)
    raw = X @ np.asarray(coefficients, dtype=float) + float(intercept)
    return np.array([_clamp(logistic(float(v))) for v in raw])


def fit_candidates(X, y, sample_weight=None) -> dict[str, tuple[np.ndarray, float]]:
    """Fit the candidate models and return ``{name: (coefficients, intercept)}``.

    Two linear models, deliberately. No RandomForest: a forest has no coefficient
    export, and the artifact format is coefficients plus an intercept because that
    is what a pure-Python scorer can evaluate with no runtime dependency. Trees
    would very likely beat linear on seven lexical features — that is a real
    capability given up, in exchange for `IMPORTANCE_SCORER=local` costing zero
    added install weight. Naming the trade here rather than leaving the absence to
    look like an oversight.

    `LogisticRegression` needs discrete classes, so continuous labels are
    thresholded at the promotion boundary and its `predict_proba` becomes the score
    — which is exactly the shape the artifact wants, since a logistic model's
    decision function *is* ``dot(w, x) + b`` under a `logistic` squash. `Ridge`
    regresses in logit space for the same reason.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    out: dict[str, tuple[np.ndarray, float]] = {}

    # Ridge: fit the logit of the label, so the served `logistic(dot(w, x) + b)`
    # inverts back to the label rather than to something 0.5-ish.
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, np.array([_logit(v) for v in y]), sample_weight=sample_weight)
    out["ridge"] = (np.asarray(ridge.coef_, dtype=float).ravel(), float(ridge.intercept_))

    # LogisticRegression: binarized at the promotion threshold, because that is the
    # decision boundary that matters most operationally. Skipped when the labels
    # land on one side of it — sklearn raises on a single class, and a model fitted
    # on one class would predict a constant.
    classes = np.unique(y >= 0.6)
    if len(classes) > 1:
        logistic_model = LogisticRegression(max_iter=2000, C=1.0)
        logistic_model.fit(X, (y >= 0.6).astype(int), sample_weight=sample_weight)
        out["logistic"] = (
            np.asarray(logistic_model.coef_, dtype=float).ravel(),
            float(np.ravel(logistic_model.intercept_)[0]),
        )
    else:
        logger.warning(
            "skipping LogisticRegression: every label is on one side of 0.6 "
            "(labels present: %s)",
            classes,
        )

    return out


# --------------------------------------------------------------------------
# Stage 2 — LLM distillation
# --------------------------------------------------------------------------


async def _score_with_llm(texts: list[str], concurrency: int = 8) -> list[float | None]:
    """Score each text with the shipped `assess_importance`.

    Uses the library's own provider and prompt path, so what is distilled is the
    behaviour actually being replaced — not a re-implementation of it that might
    differ in the prompt, the parsing, or the 1-10 vs 0.0-1.0 scale handling
    (`parse_importance` handles both; a hand-rolled call here could easily not).

    Failures return ``None`` for that text rather than a default. A timed-out call
    scored as 0.5 would look like a real label and pull the model toward the middle.
    """
    from agent_memory.core.config import MCPConfig
    from agent_memory.providers.manager import ProviderManager

    config = MCPConfig()
    manager = ProviderManager(config)
    semaphore = asyncio.Semaphore(concurrency)
    results: list[float | None] = [None] * len(texts)
    done = 0

    async def one(index: int, text: str) -> None:
        nonlocal done
        async with semaphore:
            try:
                results[index] = await manager.llm.assess_importance(text)
            except Exception as exc:
                logger.warning("assess_importance failed for sample %d: %s", index, exc)
            done += 1
            if done % 100 == 0:
                logger.info("scored %d/%d", done, len(texts))

    await asyncio.gather(*(one(i, t) for i, t in enumerate(texts)))
    return results


async def _embed(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed via the configured provider, batched.

    Same reasoning as `_score_with_llm`: the embeddings a model is trained on must
    come from the same provider and model the runtime will feed it, because the
    coefficients are positional in that vector space. Training on Titan and serving
    on Voyage produces a model that is confidently wrong.
    """
    from agent_memory.core.config import MCPConfig
    from agent_memory.providers.manager import ProviderManager

    manager = ProviderManager(MCPConfig())
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        out.extend(await manager.embedding.generate_embeddings_batch(batch))
        logger.info("embedded %d/%d", len(out), len(texts))
    return out


def embedding_metadata() -> dict:
    """The ``(provider, model, dimension)`` triple the artifact must declare.

    Read off a constructed `ProviderManager` rather than off the raw config,
    because the Voyage arm of `_create_embedding_provider` rewrites
    `embedding_model` and `embedding_dimension` during construction. Reading the
    config first yields Titan's defaults on a Voyage deployment, and the artifact
    would then claim to be trained on a model it was not — which
    `select_artifact_name` would go on to match against the wrong deployments.
    """
    from agent_memory.core.config import MCPConfig
    from agent_memory.providers.manager import ProviderManager

    config = MCPConfig()
    ProviderManager(config)
    return {
        "provider": config.embedding_provider,
        "model": config.embedding_model,
        "dimension": config.embedding_dimension,
    }


# --------------------------------------------------------------------------
# Stage 4 — pulling a live deployment's memories
# --------------------------------------------------------------------------


async def _load_mongo_samples(limit: int) -> list[tuple[str, float, list[float] | None]]:
    """Pull memories and label them from carried signals.

    Returns ``[(content, label, embedding)]``, skipping every document
    `label_from_mongo_document` declines to label. Includes soft-deleted documents
    on purpose: those are consolidation's own low-importance judgements, and
    filtering them out would drop the only strong negatives a live store holds.
    """
    from agent_memory.core.collections import MEMORIES
    from agent_memory.core.config import MCPConfig
    from agent_memory.core.database import DatabaseManager

    config = MCPConfig()
    db_manager = await DatabaseManager.initialize(config)
    try:
        collection = db_manager.db[MEMORIES]
        now = datetime.now(timezone.utc)
        samples: list[tuple[str, float, list[float] | None]] = []
        skipped = 0

        cursor = collection.find({"tier": "ltm"}, limit=limit)
        async for doc in cursor:
            content = doc.get("content") or ""
            if len(content) < MIN_CONTENT_CHARS:
                continue
            label = label_from_mongo_document(doc, now=now)
            if label is None:
                skipped += 1
                continue
            samples.append((content, label, doc.get("embedding")))

        logger.info(
            "mongodb: %d labeled, %d skipped as unlabeled "
            "(no signal yet — deliberately not scored 0)",
            len(samples),
            skipped,
        )
        return samples
    finally:
        await db_manager.close()


# --------------------------------------------------------------------------
# Assembling the training set
# --------------------------------------------------------------------------


def _dedupe(pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Collapse duplicate content, averaging the labels.

    longmemeval repeats the same haystack sessions across records, so the same turn
    can arrive many times — and with different labels, since a session cited by one
    question is uncited by another. Averaging is the honest resolution: a turn cited
    once out of five questions is genuinely more useful than one never cited and less
    than one always cited. Keeping the duplicates instead would let a heavily
    repeated turn dominate the fit by sheer count.
    """
    from collections import defaultdict

    grouped: dict[str, list[float]] = defaultdict(list)
    for content, label in pairs:
        grouped[content].append(label)
    return [(content, sum(v) / len(v)) for content, v in grouped.items()]


def _human_turns_only(sessions: list[dict]) -> list[dict]:
    """Drop assistant turns, because the runtime never scores them.

    `MemoryService.add_memories` creates LTM candidates only for
    ``message_type == "human"``, so an assistant reply is never passed to
    `assess_importance` and never will be to `LocalScorer`. Training on them is
    training on a distribution that does not exist at serving time.

    This is not a tidiness fix — it removes a confound that was actively inverting
    coefficients. In longmemeval the cited evidence turns are overwhelmingly user
    turns (842 of 896), while the uncited sessions supplying negatives are half
    assistant prose, which is far longer and more formal. Fitted on that, the model
    learns "short and first-person" as the signal, and the sign of every
    length-correlated feature reports whatever separates the two *speakers* rather
    than the two *importances*. Measured: positives averaged 0.23 on the length
    feature against 0.57 for negatives, and `temporal` came out +1.69 — the
    opposite of the design's prediction — because temporal language is
    conversational and conversational meant user meant cited.
    """
    filtered = []
    for session in sessions:
        turns = [
            t for t in (session.get("turns") or ())
            # Unknown role is kept: a dataset that stops labeling roles should not
            # silently produce an empty corpus.
            if t.get("role") in (None, "user")
        ]
        filtered.append({**session, "turns": turns})
    return filtered


def collect_benchmark_pairs(
    cache_dir: Path,
    *,
    negative_ratio: float,
    max_samples: int,
    rng: random.Random,
    max_negative_ratio: float = 3.0,
    include_s_split: bool = True,
) -> list[tuple[str, float]]:
    """Benchmark-derived ``(content, label)`` pairs from both corpora.

    locomo supplies positives and no negatives (every session in it is cited by
    something). longmemeval's oracle split is nearly the same. The `s` split is
    where negatives come from — ~23k sessions no question draws on — which is why it
    is downloaded despite being 277MB.
    """
    pairs: list[tuple[str, float]] = []
    dropped: list[str] = []

    loaders = [load_locomo, lambda d: load_longmemeval(d, "oracle")]
    if include_s_split:
        loaders.append(lambda d: load_longmemeval(d, "s"))

    for loader in loaders:
        sessions, questions = loader(cache_dir)
        sessions = _human_turns_only(sessions)
        pairs.extend(derive_benchmark_labels(sessions, questions))
        dropped.extend(unlabeled_benchmark_turns(sessions, questions))

    pairs = [(c, v) for c, v in pairs if c and len(c) >= MIN_CONTENT_CHARS]
    pairs = _dedupe(pairs)

    positives = sum(1 for _, v in pairs if v >= 0.5)
    logger.info(
        "benchmark: %d labeled turns (%d positive, %d negative), %d dropped as unlabeled",
        len(pairs),
        positives,
        len(pairs) - positives,
        len(dropped),
    )

    if negative_ratio > 0 and dropped:
        # Opt-in, and loud. This is the "treat unlabeled as negative" assumption
        # the docstring warns about; it is available because it can help, and it is
        # off by default because it is an assumption, not a label.
        n = min(len(dropped), int(positives * negative_ratio))
        sampled = rng.sample(dropped, n)
        logger.warning(
            "--negative-ratio %.2f: sampling %d of %d *unlabeled* turns as "
            "negatives. These were not observed to be useless; they were never "
            "asked about.",
            negative_ratio,
            n,
            len(dropped),
        )
        pairs.extend(
            (c, 0.0) for c in sampled if c and len(c) >= MIN_CONTENT_CHARS
        )
        pairs = _dedupe(pairs)

    pos = [p for p in pairs if p[1] >= 0.5]
    neg = [p for p in pairs if p[1] < 0.5]

    # Downsample the majority class to `max_negative_ratio` × positives. The raw
    # corpus runs ~80:1 negative (2.3k positives against 180k negatives, because
    # longmemeval's `s` split is 23k uncited sessions), and at that ratio a
    # least-squares fit minimizes error by predicting "forgettable" for everything:
    # it would be right 99% of the time on this corpus and useless in production,
    # where the input distribution is nothing like 80:1. Capping the ratio is what
    # makes the fitted intercept mean something.
    #
    # Not `class_weight="balanced"` instead: that reweights the loss but leaves
    # `mean_label` skewed, so the calibration columns would still be measured
    # against a distribution the deployment never sees.
    if neg and len(neg) > len(pos) * max_negative_ratio:
        keep = int(len(pos) * max_negative_ratio)
        logger.info(
            "downsampling negatives %d -> %d (%.1f:1) so the fit is not dominated "
            "by the majority class",
            len(neg), keep, max_negative_ratio,
        )
        neg = rng.sample(neg, keep)

    if len(pos) + len(neg) > max_samples:
        half = max_samples // 2
        pos = rng.sample(pos, min(len(pos), half))
        neg = rng.sample(neg, min(len(neg), max_samples - len(pos)))

    pairs = pos + neg
    rng.shuffle(pairs)
    logger.info(
        "benchmark final: %d samples (%d positive, %d negative)",
        len(pairs), len(pos), len(neg),
    )
    return pairs


async def build_dataset(
    source: str,
    space: str,
    cache_dir: Path,
    *,
    negative_ratio: float,
    benchmark_weight: float,
    max_samples: int,
    llm_samples: int,
    mongo_limit: int,
    rng: random.Random,
    max_negative_ratio: float = 3.0,
    synthetic_weight: float = 10.0,
) -> tuple[list[str], np.ndarray, np.ndarray, list[str], list, dict | None]:
    """Assemble ``(texts, labels, weights, label_sources, features, embedding_meta)``.

    Weighting is where the stages combine: benchmark labels carry
    ``benchmark_weight`` because they are grounded in observed future usefulness,
    while LLM labels carry 1.0 and exist to populate the scale continuously. Neither
    alone is sufficient — the benchmark is near-binary, and the LLM is the thing
    being replaced, so a model fitted on it alone can only ever approximate it.
    """
    texts: list[str] = []
    labels: list[float] = []
    weights: list[float] = []
    sources: list[str] = []
    embeddings: list[list[float]] | None = None

    if source == "mongodb":
        samples = await _load_mongo_samples(mongo_limit)
        if not samples:
            raise SystemExit(
                "No labelable memories found. A young store has no access signal "
                "yet — train from --source combined and revisit once memories have "
                "been recalled a few times."
            )
        texts = [s[0] for s in samples]
        labels = [s[1] for s in samples]
        weights = [1.0] * len(samples)
        sources = ["mongodb"] * len(samples)
        if space == "embedding":
            stored = [s[2] for s in samples]
            if all(e for e in stored):
                # Reuse what Atlas already holds: same provider, same model, and no
                # re-embedding cost for a corpus that was embedded on write.
                embeddings = stored
            else:
                logger.info("some documents lack embeddings; re-embedding all")
                embeddings = await _embed(texts)
    else:
        if source in ("benchmark", "combined"):
            pairs = collect_benchmark_pairs(
                cache_dir,
                negative_ratio=negative_ratio,
                max_samples=max_samples,
                max_negative_ratio=max_negative_ratio,
                rng=rng,
            )
            for content, label in pairs:
                texts.append(content)
                labels.append(label)
                weights.append(benchmark_weight)
                sources.append("benchmark")

        if source in ("llm", "combined"):
            # Score a subset of the benchmark corpus plus the synthetic set. The
            # synthetic set is small and always included: it is the only part of the
            # input that deliberately spans the middle of the scale.
            pool = [t for t in texts if len(t) >= MIN_CONTENT_CHARS]
            picked = rng.sample(pool, min(len(pool), llm_samples)) if pool else []
            if source == "llm" and not picked:
                sessions, _ = load_locomo(cache_dir)
                corpus = [
                    t["content"]
                    for s in sessions
                    for t in s["turns"]
                    if t.get("content") and len(t["content"]) >= MIN_CONTENT_CHARS
                ]
                picked = rng.sample(corpus, min(len(corpus), llm_samples))

            to_score = list(dict.fromkeys(picked + SYNTHETIC_CONTENT))
            logger.info("distilling %d LLM labels", len(to_score))
            scored = await _score_with_llm(to_score)
            synthetic = set(SYNTHETIC_CONTENT)
            for content, value in zip(to_score, scored):
                if value is None:
                    continue
                texts.append(content)
                labels.append(float(value))
                # The synthetic rows carry `--synthetic-weight` because they are the
                # only ones in the deployment's register; the corpus rows are casual
                # conversation. Both are LLM-labeled, so both are tagged `llm` — the
                # weight is what differs, not the label's provenance.
                weights.append(
                    synthetic_weight if content in synthetic else 1.0
                )
                sources.append("llm")

        if space == "embedding":
            embeddings = await _embed(texts)

    if space == "embedding":
        features = embeddings
        meta = embedding_metadata()
        if features and len(features[0]) != meta["dimension"]:
            raise SystemExit(
                f"Embedder returned {len(features[0])} dimensions but config "
                f"declares {meta['dimension']}. The artifact would claim a "
                "dimension it was not trained at, and LocalScorer would then refuse "
                "every real embedding. Fix embedding_dimension first."
            )
    else:
        features = [lexical_features(t) for t in texts]
        meta = None

    return (
        texts,
        np.asarray(labels, dtype=float),
        np.asarray(weights, dtype=float),
        sources,
        features,
        meta,
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

_COLUMNS = (
    "composite",
    # Second, immediately after the metric it exists to check. A reader comparing
    # candidates needs these two side by side: a high composite with a margin at or
    # below zero is a constant wearing a model's coefficients.
    "discrimination_margin",
    "forget_agreement",
    "promote_agreement",
    "mae",
    "mean_offset",
    "spearman",
    "mean_pred",
    "mean_label",
)


def print_metrics_table(rows: dict[str, dict]) -> None:
    """Both candidates, side by side, with the composite first.

    Every candidate is printed even though only the winner is written. A bake-off
    that reports one number gives no way to tell a good winner from the least bad of
    two poor options, and that distinction is the one an operator needs before
    trusting the artifact with deletion decisions.
    """
    name_width = max(len("model"), *(len(n) for n in rows))
    header = "  ".join([f"{'model':<{name_width}}"] + [f"{c:>17}" for c in _COLUMNS])
    print("\n" + header)
    print("-" * len(header))
    for name, metrics in rows.items():
        cells = [f"{name:<{name_width}}"]
        for column in _COLUMNS:
            cells.append(f"{metrics.get(column, float('nan')):>17.4f}")
        print("  ".join(cells))
    print()


def print_lexical_coefficients(coefficients, intercept: float) -> None:
    """Named coefficients, so the signs can be read against the design's premise.

    `temporal` and `interrogative` are predicted negative — a memory about *today*
    or a question asked in passing is the archetypal short-term memory. If either
    trains positive, the model is disagreeing with the reasoning the feature set was
    built on, and the right response is to investigate the labels rather than to
    ship it.
    """
    print("lexical coefficients:")
    for name, coefficient in zip(LEXICAL_FEATURE_NAMES, coefficients):
        print(f"  {name:>14}  {coefficient:+.4f}")
    print(f"  {'intercept':>14}  {intercept:+.4f}\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=("benchmark", "llm", "combined", "mongodb"),
        default="combined",
        help="Label source. 'combined' (default) is stage 3: benchmark labels "
             "weighted above LLM-distilled ones.",
    )
    parser.add_argument(
        "--space",
        choices=("lexical", "embedding"),
        default="lexical",
        help="Feature space. 'embedding' trains against the configured embedder and "
             "needs credentials; the resulting artifact is valid only for that "
             "provider/model/dimension triple.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Where to write the artifact. Omitted, nothing is written — so an "
             "exploratory run cannot overwrite a committed artifact by accident.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the metrics table and write nothing, even with --out.",
    )
    parser.add_argument(
        "--benchmark-weight",
        type=float,
        default=0.0,
        help="Sample weight for benchmark labels relative to LLM labels (default "
             "0.0, i.e. excluded from the fit). Any value above 0 has been measured "
             "to fail the discrimination gate on the lexical space: the benchmarks "
             "label a turn positive when a later question cited it, which rewards "
             "'yesterday' (21.7x lift toward positives) because the questions ask "
             "when things happened. That trains `temporal` strongly positive and "
             "promotes expiring chatter over standing policy. Raise it only for the "
             "embedding space, or with a corpus whose questions are not time-anchored.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.0,
        help="Sample this many unlabeled turns as negatives, as a multiple of the "
             "positive count. Default 0.0: 'unlabeled' is not 'negative', and "
             "assuming otherwise is the fastest way to train a model that forgets "
             "everything. On the command line so the assumption is visible.",
    )
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument(
        "--max-negative-ratio",
        type=float,
        default=3.0,
        help="Cap negatives at this multiple of positives (default 3.0). The raw "
             "benchmark corpus is ~80:1 negative, where a least-squares fit scores "
             "everything forgettable and is right 99%% of the time on that corpus "
             "and useless on a real one.",
    )
    parser.add_argument(
        "--llm-samples",
        type=int,
        default=1200,
        help="Benchmark turns to send to the LLM for distillation. Each is one API "
             "call, so this is the cost knob.",
    )
    parser.add_argument(
        "--synthetic-weight",
        type=float,
        default=10.0,
        help="Sample weight for SYNTHETIC_CONTENT rows relative to corpus rows "
             "(default 10.0). These are the only rows written in the deployment's "
             "own register — standing preferences, policies, durable personal facts "
             "— because the benchmark corpora are casual conversation and contain "
             "almost none. There are only ~34 of them against ~1200 corpus rows, so "
             "at weight 1.0 they cannot move the fit: `preference` trains to +0.08. "
             "This is a stand-in for a larger in-domain corpus, and it is honest to "
             "call it that; lower it as real in-domain labeled data arrives.",
    )
    parser.add_argument("--mongo-limit", type=int, default=20000)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--note",
        default="",
        help="Free text recorded in the artifact's training.note.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)

    texts, y_raw, weights, sources, features, embedding_meta = await build_dataset(
        args.source,
        args.space,
        args.cache_dir,
        negative_ratio=args.negative_ratio,
        benchmark_weight=args.benchmark_weight,
        max_samples=args.max_samples,
        llm_samples=args.llm_samples,
        mongo_limit=args.mongo_limit,
        max_negative_ratio=args.max_negative_ratio,
        synthetic_weight=args.synthetic_weight,
        rng=rng,
    )

    # Everything downstream — fitting, evaluation, thresholds — happens in the
    # range the runtime can actually serve. Per-stage, because the rescale is
    # correct for benchmark indicators and wrong for LLM scores that are already in
    # range. See `_servable_labels`.
    y = _servable_labels(y_raw, sources)

    # A stage weighted to zero must leave the dataset entirely, not just the fit.
    # `train_test_split` does not know about sample weights, so zero-weight rows
    # would still land in the test set and every reported metric would be measured
    # against labels the model was explicitly told to ignore.
    if np.any(weights <= 0):
        kept = weights > 0
        dropped = int((~kept).sum())
        from collections import Counter as _Counter

        logger.info(
            "dropping %d zero-weight samples before the split (%s)",
            dropped,
            dict(_Counter(s for s, k in zip(sources, kept) if not k)),
        )
        features = [f for f, k in zip(features, kept) if k]
        texts = [t for t, k in zip(texts, kept) if k]
        sources = [s for s, k in zip(sources, kept) if k]
        y = y[kept]
        y_raw = y_raw[kept]
        weights = weights[kept]

    X = np.asarray(features, dtype=float)
    if len(X) < 20:
        raise SystemExit(
            f"Only {len(X)} labeled samples — too few to fit or to trust the "
            "metrics of. Widen the source or raise --max-samples."
        )

    logger.info(
        "training on %d samples, %d features, label mean %.3f (range %.2f–%.2f)",
        len(X), X.shape[1], float(y.mean()), float(y.min()), float(y.max()),
    )

    # Stratified on the promotion boundary so both splits contain both classes:
    # a test set that is all negatives reports promote_agreement 1.0 while having
    # measured nothing.
    stratify = (y >= 0.6).astype(int)
    if len(np.unique(stratify)) < 2 or min(np.bincount(stratify)) < 2:
        stratify = None
    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, weights, test_size=args.test_size,
        random_state=args.seed, stratify=stratify,
    )

    candidates = fit_candidates(X_train, y_train, sample_weight=w_train)
    if not candidates:
        raise SystemExit("No model could be fitted — check the label distribution.")

    rows: dict[str, dict] = {}
    for name, (coefficients, intercept) in candidates.items():
        metrics = evaluate(y_test, predict(coefficients, intercept, X_test))
        metrics["composite"] = composite_score(metrics)
        rows[name] = metrics

    # Two baselines, because they answer different questions and a single one is
    # easy to beat by accident.
    #
    # `constant-0.5` is the shipped placeholder — what an operator gets today. It is
    # the bar for "is this artifact worth committing at all".
    #
    # `constant-mean` predicts the *training* label mean, which is the hardest
    # constant available and the one that exposes a model that has learned nothing
    # but the prior. A trained model beating 0.5 while losing to the mean has
    # learned the average and not the discrimination — and on a skewed corpus that
    # is the most likely way to be fooled.
    baselines = {
        "constant-0.5": np.full(len(y_test), 0.5),
        "constant-mean": np.full(len(y_test), _clamp(float(y_train.mean()))),
    }
    for name, prediction in baselines.items():
        metrics = evaluate(y_test, prediction)
        metrics["composite"] = composite_score(metrics)
        rows[name] = metrics

    trained = {k: v for k, v in rows.items() if k in candidates}

    # Selection is gated on discrimination *before* composite, not ranked by a
    # blend of the two. A weighted blend lets a strongly-calibrated non-model
    # outscore a weakly-calibrated real one, which is the failure this exists to
    # prevent; and the margin is not on the composite's scale, so any blend weight
    # would be arbitrary. Lexical only — see `discrimination_margin`.
    if args.space == "lexical":
        for name, (coefficients_, intercept_) in candidates.items():
            margin = discrimination_margin(coefficients_, intercept_)
            trained[name]["discrimination_margin"] = round(margin, 4)
            rows[name]["discrimination_margin"] = round(margin, 4)
        for name in baselines:
            # A constant separates nothing, by definition. Recorded so the column
            # reads as a comparison rather than as a property only models have.
            rows[name]["discrimination_margin"] = 0.0

        passing = {k: v for k, v in trained.items() if v["discrimination_margin"] > 0}
        if not passing:
            print_metrics_table(rows)
            print(
                "\nFAILED: no candidate ranks the durable cases above the expiring "
                "ones.\n\n"
                "Per-case scores for each candidate:\n"
            )
            for name, (coefficients_, intercept_) in candidates.items():
                print(f"  {name}:")
                scores = predict(
                    coefficients_,
                    intercept_,
                    [lexical_features(t) for t, _ in DISCRIMINATION_CASES],
                )
                for (text, keep), score in zip(DISCRIMINATION_CASES, scores):
                    label = "durable " if keep else "expiring"
                    print(f"    {score:.3f}  {label}  {text}")
            print(
                "\nThis is not a tuning problem to be worked around by loosening the "
                "gate. It means the labels disagree with the premise that standing "
                "preferences outlive task chatter — check which stage dominates the "
                "sample weights, and read `--help` on --benchmark-weight.\n"
            )
            return 1
        trained = passing

    print_metrics_table(rows)

    winner = max(trained, key=lambda k: trained[k]["composite"])
    coefficients, intercept = candidates[winner]
    print(f"winner: {winner} (composite {trained[winner]['composite']:.4f})")
    if args.space == "lexical":
        print(
            f"discrimination margin: {trained[winner]['discrimination_margin']:+.4f} "
            f"(min durable - max expiring, over {len(DISCRIMINATION_CASES)} held-out cases)"
        )
        if len(candidates) > len(trained):
            rejected = sorted(set(candidates) - set(trained))
            print(
                f"rejected on discrimination despite composite: {', '.join(rejected)}"
            )

    best_baseline = max(baselines, key=lambda k: rows[k]["composite"])
    if trained[winner]["composite"] <= rows[best_baseline]["composite"]:
        # Not fatal, and on the lexical space not even necessarily bad. `composite`
        # is pure calibration, and a constant predicting the label mean is
        # *optimally* calibrated by construction — it cannot be beaten on
        # `mean_offset` and is very hard to beat on `mae`. Measured: the shipped
        # artifact lands within ±0.001 of `constant-mean` on composite across
        # repeated runs, on either side depending on LLM sampling, while separating
        # the held-out cases by a stable +0.02.
        #
        # So this warning says "the composite cannot distinguish these two", not
        # "the model is worthless". Read the discrimination margin, which is the
        # column that still carries signal here: a constant scores exactly 0.0 on
        # it, and a constant is what the calibration metrics are rewarding.
        print(
            f"\nNOTE: the winning model does not beat {best_baseline} on the "
            "composite metric.\n"
            "  `composite` is a calibration metric and a mean-predicting constant "
            "is optimally calibrated,\n"
            "  so a tie here is expected rather than disqualifying. What separates "
            "them is discrimination:\n"
            f"  this model {trained[winner].get('discrimination_margin', 0.0):+.4f} "
            "vs 0.0000 for any constant.\n"
            "  Ship it only if that margin is what you need; do not read the tie as "
            "'no better than a constant'.\n"
        )

    if args.space == "lexical":
        print_lexical_coefficients(coefficients, intercept)

    from collections import Counter

    training = {
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "source": args.source,
        "space": args.space,
        "model": winner,
        "labels": sorted(set(sources)),
        "label_counts": dict(Counter(sources)),
        "n_samples": int(len(X)),
        "n_test": int(len(y_test)),
        "benchmark_weight": args.benchmark_weight,
        "synthetic_weight": args.synthetic_weight,
        "negative_ratio": args.negative_ratio,
        "seed": args.seed,
        "metrics": {k: round(float(v), 4) for k, v in trained[winner].items()},
        # Recorded so the artifact's own numbers can be read against the floor they
        # had to clear. "forget_agreement 0.94" means nothing on its own.
        "baselines": {
            name: {k: round(float(v), 4) for k, v in rows[name].items()}
            for name in baselines
        },
        "label_range": [MIN_IMPORTANCE, MAX_IMPORTANCE],
        "datasets": (
            ["xiaowu0162/longmemeval-cleaned", "adymaharana/locomo"]
            if args.source in ("benchmark", "combined")
            else []
        ),
    }
    if args.note:
        training["note"] = args.note

    doc = build_artifact(
        "lexical" if args.space == "lexical" else "embedding_linear",
        coefficients,
        intercept,
        embedding=embedding_meta,
        training=training,
    )

    if args.dry_run or not args.out:
        reason = "--dry-run" if args.dry_run else "no --out given"
        print(f"not writing ({reason}). Artifact would be:\n")
        print(json.dumps({**doc, "coefficients": "<omitted>"}, indent=2))
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {args.out}")

    # Load it back through the runtime's own loader. If the artifact this script
    # just wrote cannot be loaded by the library, the operator should find out here
    # and not at their next deployment's startup.
    from agent_memory.services.importance import load_artifact

    artifact = load_artifact(args.out)
    print(
        f"verified: kind={artifact.kind}, "
        f"{len(artifact.coefficients)} coefficients, "
        f"intercept {artifact.intercept:+.4f}"
    )
    return 0


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    return asyncio.run(_run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
