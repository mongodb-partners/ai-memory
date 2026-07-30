"""What becomes a long-term memory is a configured length, not a literal.

``MemoryService.store_stm`` writes every message as STM and *additionally* queues
an LTM candidate for human messages it judges significant. That judgement was a
bare ``len(msg["content"]) > 30`` in the middle of the method.

It is the cheapest filter in the system and the most consequential. A message
below the threshold is never enriched, never promoted, and never recalled — so
the number decides what the agent is *able* to remember, and it decided it
somewhere an operator could neither read it nor change it without forking.

30 is a length, not a measure of meaning, which is the whole reason it belongs in
config rather than in a cleverer inline rule. It drops the acknowledgements that
dominate a real transcript, and it also drops "I'm allergic to penicillin" (26
characters). A deployment whose users write telegraphically needs it lower; one
where every turn is a paragraph and enrichment cost matters needs it higher.

Pinned in both directions: a threshold that is read but ignored, and one that
silently keeps everything, would each satisfy half of this.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

from agent_memory.core.config import MCPConfig
from agent_memory.providers.manager import ResolvedEmbedding
from agent_memory.services.memory import MemoryService


def _config(**overrides) -> MCPConfig:
    # `_env_file=None`: a live .env would otherwise set the threshold under test.
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


def _providers():
    providers = MagicMock()
    providers.embedding = AsyncMock()
    providers.embedding.generate_embeddings_batch = AsyncMock(
        side_effect=lambda texts: [[0.1] * 1536 for _ in texts]
    )
    providers.embedding_spec = ResolvedEmbedding(
        model="amazon.titan-embed-text-v2:0", dimension=1536
    )
    return providers


def _service(**config_overrides):
    collection = AsyncMock()
    collection.insert_many = AsyncMock(
        side_effect=lambda docs: MagicMock(
            inserted_ids=[ObjectId() for _ in docs]
        )
    )
    return MemoryService(collection, _config(**config_overrides), _providers()), collection


def _ltm_contents(collection) -> list[str]:
    """The contents queued as LTM candidates, across every insert after the first.

    The first ``insert_many`` is always the STM batch; a second one appears only
    when at least one candidate qualified.
    """
    calls = collection.insert_many.call_args_list
    return [
        doc["content"]
        for call in calls[1:]
        for doc in call[0][0]
    ]


def _human(content: str) -> dict:
    return {"content": content, "message_type": "human"}


class TestTheThresholdIsRead:
    """A configured value that the code ignores is worse than a literal: it reads
    as a working control."""

    @pytest.mark.parametrize("min_chars", [5, 20, 31, 100])
    async def test_the_boundary_follows_the_configured_value(self, min_chars):
        service, collection = _service(ltm_candidate_min_chars=min_chars)
        at = "a" * min_chars
        below = "b" * (min_chars - 1)
        await service.store_stm("u1", "c1", [_human(at), _human(below)])

        assert _ltm_contents(collection) == [at]

    async def test_a_lower_threshold_admits_a_short_fact(self):
        """The motivating case. "I'm allergic to penicillin" is 26 characters —
        below the default, and exactly the kind of thing a deployment would want
        remembered."""
        fact = "I'm allergic to penicillin"
        assert len(fact) < _config().ltm_candidate_min_chars

        service, collection = _service(ltm_candidate_min_chars=20)
        await service.store_stm("u1", "c1", [_human(fact)])
        assert _ltm_contents(collection) == [fact]

    async def test_a_higher_threshold_excludes_what_the_default_would_keep(self):
        service, collection = _service(ltm_candidate_min_chars=500)
        await service.store_stm("u1", "c1", [_human("a" * 200)])

        # STM only. The message is still stored; it just is not a candidate.
        assert collection.insert_many.call_count == 1
        assert _ltm_contents(collection) == []

    async def test_zero_keeps_every_human_message(self):
        """The honest way to say "let importance scoring decide" — and it means
        one enrichment per turn, which is why it is not the default."""
        service, collection = _service(ltm_candidate_min_chars=0)
        await service.store_stm("u1", "c1", [_human("ok"), _human("")])

        assert _ltm_contents(collection) == ["ok", ""]

    async def test_the_threshold_reads_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("LTM_CANDIDATE_MIN_CHARS", "7")
        config = MCPConfig(
            mongodb_connection_string="mongodb://localhost:27017", _env_file=None
        )
        assert config.ltm_candidate_min_chars == 7


class TestTheDefaultPreservesTheOldBoundary:
    """The literal was ``> 30`` and the comparison is now ``>=``, so the default
    has to be 31. Anything already stored was judged by the old boundary, and a
    silent one-character shift would change which messages a *running* deployment
    remembers on upgrade."""

    def test_the_default_is_thirty_one(self):
        assert _config().ltm_candidate_min_chars == 31

    @pytest.mark.parametrize(
        ("length", "qualifies"),
        [(30, False), (31, True)],
    )
    async def test_the_default_boundary_is_unchanged(self, length, qualifies):
        service, collection = _service()
        content = "a" * length
        await service.store_stm("u1", "c1", [_human(content)])

        assert _ltm_contents(collection) == ([content] if qualifies else [])


class TestTheOtherConditionsAreUntouched:
    """The threshold is one of two conditions, and only one of them moved."""

    async def test_a_long_ai_message_is_still_not_a_candidate(self):
        """Length is not the only gate. An assistant turn is the agent's own
        output; promoting it would let the agent's summaries become the facts it
        later recalls."""
        service, collection = _service()
        await service.store_stm(
            "u1", "c1", [{"content": "a" * 500, "message_type": "ai"}]
        )
        assert _ltm_contents(collection) == []

    async def test_a_long_ai_message_is_still_stored_as_stm(self):
        service, collection = _service()
        await service.store_stm(
            "u1", "c1", [{"content": "a" * 500, "message_type": "ai"}]
        )
        stm_docs = collection.insert_many.call_args_list[0][0][0]
        assert [d["tier"] for d in stm_docs] == ["stm"]

    async def test_a_message_below_the_threshold_is_still_stored_as_stm(self):
        """The threshold decides what is *additionally* queued, never what is
        kept. A short message is still recallable for the STM window."""
        service, collection = _service()
        await service.store_stm("u1", "c1", [_human("ok")])

        stm_docs = collection.insert_many.call_args_list[0][0][0]
        assert len(stm_docs) == 1
        assert stm_docs[0]["content"] == "ok"
        assert stm_docs[0]["tier"] == "stm"

    async def test_candidates_keep_their_own_message_embedding(self):
        """The LTM loop indexes `embeddings[i]` by *message* position while
        building a list that skips non-candidates. Mixing those two indexes would
        pair a message with another message's vector, and the threshold moving is
        what makes the offset vary."""
        service, collection = _service(ltm_candidate_min_chars=10)
        providers = service.providers
        providers.embedding.generate_embeddings_batch = AsyncMock(
            side_effect=lambda texts: [[float(i)] * 1536 for i, _ in enumerate(texts)]
        )
        # Only messages 1 and 3 qualify.
        await service.store_stm("u1", "c1", [
            _human("short"),
            _human("long enough to qualify"),
            _human("tiny"),
            _human("also long enough to qualify"),
        ])

        ltm_docs = collection.insert_many.call_args_list[1][0][0]
        assert [d["embedding"][0] for d in ltm_docs] == [1.0, 3.0]

    async def test_candidates_point_at_their_own_stm_document(self):
        """Same indexing argument for `source_stm_id`: a candidate whose
        `source_stm_id` names a different message's STM row makes the provenance
        link a lie."""
        service, collection = _service(ltm_candidate_min_chars=10)
        stm_ids = [ObjectId() for _ in range(3)]
        collection.insert_many = AsyncMock(
            side_effect=[MagicMock(inserted_ids=stm_ids), MagicMock(inserted_ids=[])]
        )
        await service.store_stm("u1", "c1", [
            _human("tiny"),
            _human("long enough to qualify"),
            _human("no"),
        ])

        ltm_docs = collection.insert_many.call_args_list[1][0][0]
        assert [d["source_stm_id"] for d in ltm_docs] == [stm_ids[1]]


class TestTheTrainerUsesTheSameThreshold:
    """`scripts/train_importance.py` filters its training corpus by the same
    length. Training on text the runtime never scores would fit the wrong
    distribution — and the two used to be separate literals, 30 and 31, already
    off by one."""

    def test_the_trainer_constant_tracks_the_config_default(self):
        pytest.importorskip("numpy", reason="the trainer needs the training extra")
        pytest.importorskip("sklearn", reason="the trainer needs the training extra")

        import importlib.util
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "_train_importance_probe", root / "scripts" / "train_importance.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["_train_importance_probe"] = module
        try:
            spec.loader.exec_module(module)
            assert module.MIN_CONTENT_CHARS == _config().ltm_candidate_min_chars
        finally:
            sys.modules.pop("_train_importance_probe", None)
