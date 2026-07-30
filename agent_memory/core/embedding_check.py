"""Validate an embedding provider's reply before anything is written from it.

A batch embedding call returns a list, and the code that consumed it assumed the
list was parallel to its input::

    embeddings = await provider.generate_embeddings_batch(texts)
    for msg, emb in zip(messages, embeddings):

``zip`` stops at the shorter sequence. A provider that returns nine vectors for
ten texts therefore produces nine documents, ``insert_many`` succeeds, and
``store_stm`` returns nine ids to a caller that handed over ten messages. Nothing
raises and nothing logs. The tenth message is gone, and because the response is a
list of ids rather than a per-message result, the caller cannot tell which one.

The width has the same shape of failure one level down. Atlas accepts a 1024-wide
vector into a 1536-wide index without complaint — the document is stored, the count
goes up, ``find`` returns it — and ``$vectorSearch`` never returns it again. The
memory exists and is not recallable, which reads as "the user never told us that".

Both are silent, both destroy data the caller still had a moment ago, and neither
is recoverable afterwards because nothing anywhere records what was lost. So the
call fails instead, with ``EmbeddingError``, while the caller still holds its own
input and can retry.

Why not repair rather than refuse: there is no honest repair. Padding a short
vector, truncating a long one, or re-embedding the missing tail all invent data
and store it as if the provider had produced it. A refusal is the only outcome
that does not lie about what happened.
"""

from __future__ import annotations

from agent_memory.exceptions import EmbeddingError


def expected_dimension(providers) -> int | None:
    """The width the configured embedder is expected to emit, if it is knowable.

    Reads ``providers.embedding_spec``, which ``ProviderManager`` publishes. Any
    other provider stack — a hand-assembled one in an embedding test, a stub — may
    not have it, and ``None`` means "no declared width to compare against" rather
    than zero.

    Deliberately tolerant, because the alternative is worse than the gap it
    leaves. A width check that raised on an unrecognised provider stack would
    make ``store`` fail for a caller whose vectors are perfectly fine, and the
    count check below — the one that catches silent data loss — needs no dimension
    at all and stays unconditional either way.
    """
    spec = getattr(providers, "embedding_spec", None)
    dimension = getattr(spec, "dimension", None)
    return dimension if isinstance(dimension, int) and dimension > 0 else None


def check_batch(
    embeddings, texts, *, expected: int | None = None, operation: str = "embedding"
) -> list[list[float]]:
    """Return ``embeddings`` once it is known to describe ``texts``.

    Raises :class:`EmbeddingError` when the provider returned a different number
    of vectors than there were inputs, or a vector of a width other than
    ``expected``. ``expected=None`` skips the width comparison — see
    :func:`expected_dimension`.

    The count is checked before the widths, because a length mismatch is the more
    destructive of the two (documents silently dropped rather than documents
    silently unsearchable) and because reporting "8 vectors for 10 texts" is more
    useful than reporting whichever of the eight happened to be measured first.
    """
    if embeddings is None:
        raise EmbeddingError(
            f"{operation}: the embedding provider returned nothing for "
            f"{len(texts)} input(s). Nothing was written."
        )

    if len(embeddings) != len(texts):
        raise EmbeddingError(
            f"{operation}: the embedding provider returned {len(embeddings)} "
            f"vector(s) for {len(texts)} input(s). Refusing to write a partial "
            "batch — zipping these would silently drop "
            f"{abs(len(texts) - len(embeddings))} input(s). Nothing was written."
        )

    if expected is None:
        return embeddings

    for index, vector in enumerate(embeddings):
        _check_one(vector, expected, operation=operation, index=index)
    return embeddings


def check_one(
    embedding, *, expected: int | None = None, operation: str = "embedding"
) -> list[float]:
    """Return a single ``embedding`` once its width is known to be ``expected``.

    The single-vector counterpart of :func:`check_batch`, for the paths that embed
    one string: recall queries, an episodic turn, a merged memory.
    """
    if embedding is None:
        raise EmbeddingError(
            f"{operation}: the embedding provider returned no vector. "
            "Nothing was written."
        )
    if expected is not None:
        _check_one(embedding, expected, operation=operation, index=None)
    return embedding


def _check_one(vector, expected: int, *, operation: str, index: int | None) -> None:
    where = "" if index is None else f" at position {index}"
    if vector is None:
        raise EmbeddingError(
            f"{operation}: the embedding provider returned no vector{where}. "
            "Nothing was written."
        )
    try:
        width = len(vector)
    except TypeError as exc:
        raise EmbeddingError(
            f"{operation}: the embedding provider returned "
            f"{type(vector).__name__}{where} where a vector was expected. "
            "Nothing was written."
        ) from exc
    if width != expected:
        # Named as the *silent* failure it prevents, because "wrong dimension" on
        # its own reads as something the database would have rejected.
        raise EmbeddingError(
            f"{operation}: the embedding provider returned a {width}-dimensional "
            f"vector{where} but the configured embedder emits {expected}. Atlas "
            "would accept this vector into the index and then never return it "
            "from $vectorSearch — no error, no visible symptom, the memory simply "
            "stops being recallable. Nothing was written."
        )


__all__ = ["check_batch", "check_one", "expected_dimension"]
