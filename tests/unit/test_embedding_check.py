"""A provider reply that does not describe its input must not reach the database.

Two silent losses are under test, and they are silent in different ways:

- **A short batch.** ``zip(messages, embeddings)`` stops at the shorter sequence,
  so nine vectors for ten messages wrote nine documents and reported success. No
  exception, no log line, and the returned id list is the only evidence — a caller
  comparing lengths could notice, and none did.
- **A wrong width.** Atlas accepts a 1024-wide vector into a 1536-wide index,
  stores it, returns it from ``find``, and never returns it from
  ``$vectorSearch``. The memory exists and is not recallable.

Every guard is tested in both directions. A refusal test alone is satisfied by a
function that refuses everything, which would break storing entirely; the paired
"and a correct reply still stores" case is what makes each refusal a *narrow* one.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

from agent_memory.core.config import MCPConfig
from agent_memory.core.embedding_check import (
    check_batch,
    check_one,
    expected_dimension,
)
from agent_memory.exceptions import EmbeddingError, MemoryError
from agent_memory.providers.manager import ResolvedEmbedding
from agent_memory.services.memory import MemoryService

DIM = 8


def _cfg(**overrides) -> MCPConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017"}
    defaults.update(overrides)
    return MCPConfig(**defaults, _env_file=None)


def _vec(width: int = DIM) -> list[float]:
    return [0.1] * width


def _providers(*, dimension: int | None = DIM, batch=None, single=None):
    """A provider stack shaped like ``ProviderManager``.

    ``dimension=None`` omits ``embedding_spec`` entirely — the stub/hand-assembled
    case — rather than setting it to None, because the two are different states and
    the code distinguishes them.
    """
    providers = MagicMock()
    providers.embedding = AsyncMock()
    providers.embedding.generate_embeddings_batch = AsyncMock(
        side_effect=batch or (lambda texts: [_vec() for _ in texts])
    )
    providers.embedding.generate_embedding = AsyncMock(
        return_value=single if single is not None else _vec()
    )
    if dimension is None:
        del providers.embedding_spec
    else:
        providers.embedding_spec = ResolvedEmbedding(model="m", dimension=dimension)
    return providers


class TestTheCountMustMatchTheInput:
    """A batch shorter or longer than its input is refused, not zipped."""

    async def test_a_short_batch_is_refused(self):
        with pytest.raises(EmbeddingError) as exc:
            check_batch([_vec(), _vec()], ["a", "b", "c"], expected=DIM)
        # The numbers are in the message because "an embedding failed" sends an
        # operator to the provider's status page; "2 vectors for 3 inputs" sends
        # them to the response body, which is where the fault is.
        assert "2" in str(exc.value) and "3" in str(exc.value)

    async def test_a_long_batch_is_refused(self):
        # Longer is not harmless: `zip` would silently discard the extra, so the
        # reply is still not the one that was asked for and something is wrong
        # upstream. Refusing surfaces it.
        with pytest.raises(EmbeddingError):
            check_batch([_vec(), _vec(), _vec()], ["a", "b"], expected=DIM)

    async def test_a_matching_batch_passes_through_unchanged(self):
        vectors = [_vec(), _vec(), _vec()]
        assert check_batch(vectors, ["a", "b", "c"], expected=DIM) is vectors

    async def test_an_empty_batch_for_no_input_is_fine(self):
        assert check_batch([], [], expected=DIM) == []

    async def test_nothing_at_all_is_refused(self):
        # A provider that returns None rather than a list: `len(None)` raises
        # TypeError, which the caller's `except Exception` would log as an
        # unexplained fault.
        with pytest.raises(EmbeddingError):
            check_batch(None, ["a"], expected=DIM)

    async def test_the_count_is_checked_without_a_declared_dimension(self):
        # The count check needs no dimension, and this is the case that proves it
        # is not accidentally gated on one — an unrecognised provider stack still
        # cannot silently drop inputs.
        with pytest.raises(EmbeddingError):
            check_batch([_vec()], ["a", "b"], expected=None)


class TestTheWidthMustMatchTheIndex:
    """A vector of the wrong width never reaches a collection."""

    @pytest.mark.parametrize("width", [DIM - 1, DIM + 1, 0, 1536])
    async def test_a_wrong_width_is_refused(self, width):
        with pytest.raises(EmbeddingError):
            check_batch([_vec(width)], ["a"], expected=DIM)

    async def test_the_right_width_passes(self):
        assert check_batch([_vec(DIM)], ["a"], expected=DIM) == [_vec(DIM)]

    async def test_a_wrong_width_anywhere_in_the_batch_is_refused(self):
        # Not just the first — a single bad vector in position three is exactly the
        # case a "check the first one" implementation would miss.
        with pytest.raises(EmbeddingError) as exc:
            check_batch(
                [_vec(), _vec(), _vec(DIM - 2), _vec()], ["a", "b", "c", "d"],
                expected=DIM,
            )
        assert "position 2" in str(exc.value)

    async def test_the_message_names_the_silent_failure(self):
        # The text matters more than usual here. An operator who reads "wrong
        # dimension" assumes the database rejected the write and nothing is lost;
        # the truth is that Atlas would have accepted it and the memory would have
        # become unrecallable. The message has to say so.
        with pytest.raises(EmbeddingError) as exc:
            check_batch([_vec(DIM - 1)], ["a"], expected=DIM)
        message = str(exc.value)
        assert "$vectorSearch" in message
        assert "Nothing was written" in message

    async def test_no_declared_width_skips_the_comparison(self):
        # Deliberate: a provider stack that does not publish a spec must not have
        # every write refused. The count check still applies.
        odd = [_vec(3), _vec(99)]
        assert check_batch(odd, ["a", "b"], expected=None) is odd

    async def test_a_non_vector_is_refused_rather_than_stored(self):
        with pytest.raises(EmbeddingError):
            check_batch([0.5], ["a"], expected=DIM)

    async def test_a_missing_vector_inside_a_full_length_batch_is_refused(self):
        # The count matches, so only the per-vector check can catch this one.
        with pytest.raises(EmbeddingError):
            check_batch([_vec(), None], ["a", "b"], expected=DIM)


class TestTheSingleVectorCheck:
    """``check_one`` for the paths that embed one string."""

    async def test_a_wrong_width_is_refused(self):
        with pytest.raises(EmbeddingError):
            check_one(_vec(DIM + 4), expected=DIM)

    async def test_the_right_width_passes_through(self):
        vector = _vec()
        assert check_one(vector, expected=DIM) is vector

    async def test_no_vector_is_refused(self):
        with pytest.raises(EmbeddingError):
            check_one(None, expected=DIM)

    async def test_no_declared_width_accepts_any_width(self):
        assert check_one(_vec(77), expected=None) == _vec(77)

    async def test_the_message_omits_a_position(self):
        # One vector has no position; "at position 0" would be noise pointing at an
        # index the caller never supplied.
        with pytest.raises(EmbeddingError) as exc:
            check_one(_vec(DIM - 1), expected=DIM)
        assert "position" not in str(exc.value)


class TestTheExpectedWidthComesFromTheResolvedSpec:
    """``embedding_dimension`` is the declared value; the spec is the real one."""

    async def test_the_spec_dimension_is_used(self):
        assert expected_dimension(_providers(dimension=1024)) == 1024

    async def test_a_stack_without_a_spec_declares_nothing(self):
        assert expected_dimension(_providers(dimension=None)) is None

    async def test_a_mock_attribute_is_not_mistaken_for_a_dimension(self):
        # A bare MagicMock answers *any* attribute with another MagicMock, so
        # `spec.dimension` is truthy but not an int. Treating it as a width would
        # refuse every vector in any test using a plain mock provider stack.
        assert expected_dimension(MagicMock()) is None

    @pytest.mark.parametrize("bad", [0, -1])
    async def test_a_degenerate_declared_width_is_ignored(self, bad):
        # Comparing against 0 would refuse every real vector; there is no width a
        # provider could return that satisfies it.
        assert expected_dimension(_providers(dimension=bad)) is None

    async def test_the_service_reads_it_at_construction(self):
        service = MemoryService(AsyncMock(), _cfg(), _providers(dimension=1024))
        assert service._expected_dimension == 1024

    async def test_the_service_reads_the_spec_not_the_config(self):
        # The Voyage case: the config still says 1536 because it inherited Titan's
        # default, while the resolved spec says 1024. Reading the config here would
        # refuse every correct Voyage vector — the check would fail closed on a
        # working deployment.
        config = _cfg()
        assert config.embedding_dimension == 1536
        service = MemoryService(AsyncMock(), config, _providers(dimension=1024))
        assert service._expected_dimension == 1024


class TestStoreRefusesRatherThanWritingPartially:
    """``store_stm`` end to end — the defect this exists to fix."""

    def _service(self, providers):
        collection = AsyncMock()
        # One id per document actually inserted, as the driver does. A fixed-length
        # return would make "stored every message" pass regardless of how many
        # documents the code built.
        collection.insert_many = AsyncMock(
            side_effect=lambda docs, **kw: MagicMock(
                inserted_ids=[ObjectId() for _ in docs]
            )
        )
        return MemoryService(collection, _cfg(), providers), collection

    async def test_a_short_reply_writes_nothing(self):
        # Three messages, two vectors: the pre-fix code stored two STM documents
        # and returned two ids to a caller that handed over three.
        providers = _providers(batch=lambda texts: [_vec() for _ in texts[:-1]])
        service, collection = self._service(providers)
        messages = [
            {"content": f"message {i}", "message_type": "human"} for i in range(3)
        ]

        with pytest.raises(EmbeddingError):
            await service.store_stm("u", "c", messages)

        # The point of raising *before* the loop: not one document, not a partial
        # batch, nothing.
        collection.insert_many.assert_not_called()

    async def test_a_complete_reply_stores_every_message(self):
        # The other half. A guard that refused everything would satisfy the test
        # above and break storing; this is what makes the refusal narrow.
        providers = _providers()
        service, collection = self._service(providers)
        messages = [
            {"content": f"message {i}", "message_type": "ai"} for i in range(3)
        ]

        result = await service.store_stm("u", "c", messages)

        assert len(result) == 3
        stored = collection.insert_many.await_args_list[0].args[0]
        assert len(stored) == 3
        assert [d["content"] for d in stored] == ["message 0", "message 1", "message 2"]

    async def test_a_wrong_width_reply_writes_nothing(self):
        providers = _providers(batch=lambda texts: [_vec(DIM - 1) for _ in texts])
        service, collection = self._service(providers)

        with pytest.raises(EmbeddingError):
            await service.store_stm(
                "u", "c", [{"content": "hello", "message_type": "human"}]
            )
        collection.insert_many.assert_not_called()

    async def test_the_ltm_candidates_are_not_lost_silently(self):
        # The second half of the original defect. The LTM loop indexes
        # `embeddings[i]` by *message* position, so a short reply raised IndexError
        # inside the `try` at the end of `store_stm` — which logs "Failed to insert
        # LTM candidates" and returns success. The candidates vanished and the
        # caller was told everything worked.
        long_human = "x" * 40
        providers = _providers(batch=lambda texts: [_vec() for _ in texts[:1]])
        service, collection = self._service(providers)
        messages = [
            {"content": long_human, "message_type": "human"},
            {"content": long_human, "message_type": "human"},
        ]

        with pytest.raises(EmbeddingError):
            await service.store_stm("u", "c", messages)
        collection.insert_many.assert_not_called()

    async def test_an_empty_message_list_still_short_circuits(self):
        # No provider call at all, so no reply to validate. Guarding this before
        # the check keeps a no-op cheap.
        providers = _providers()
        service, _ = self._service(providers)
        assert await service.store_stm("u", "c", []) == []
        providers.embedding.generate_embeddings_batch.assert_not_called()

    async def test_a_stack_without_a_spec_still_stores(self):
        # A provider stack that publishes no dimension must keep working — the
        # width is unknown, not wrong.
        providers = _providers(dimension=None)
        service, _ = self._service(providers)
        result = await service.store_stm(
            "u", "c", [{"content": "hello", "message_type": "human"}]
        )
        assert len(result) == 1

    async def test_the_refusal_is_catchable_as_a_memory_error(self):
        # `EmbeddingError` subclasses `MemoryError`, so an existing caller that
        # catches the base keeps catching this. That is the whole reason for the
        # hierarchy — see exceptions.py.
        providers = _providers(batch=lambda texts: [])
        service, _ = self._service(providers)
        with pytest.raises(MemoryError):
            await service.store_stm(
                "u", "c", [{"content": "hello", "message_type": "human"}]
            )
