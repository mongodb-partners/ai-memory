"""Abstract base classes for embedding and LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class EmbeddingProvider(ABC):
    @abstractmethod
    async def generate_embedding(self, text: str) -> list[float]:
        ...

    @abstractmethod
    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        ...


class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs) -> str:
        ...

    async def chat_stream(
        self, messages: list[dict], **kwargs
    ) -> AsyncIterator[str]:
        """Yield text deltas as the model produces them.

        Concrete, not abstract: the default implementation awaits :meth:`chat`
        and yields the whole answer as one chunk. That keeps every existing
        provider working and makes streaming an optimization rather than a
        breaking change — a caller written against this interface behaves
        correctly on a provider that cannot stream, it just sees one large
        delta instead of many small ones.

        Yields text only. Tool-call and usage events are deliberately out of
        scope: they would make the return type provider-shaped, and the whole
        point of this seam is that the caller does not know which provider it
        has.
        """
        yield await self.chat(messages, **kwargs)

    @abstractmethod
    async def assess_importance(self, content: str) -> float:
        ...

    @abstractmethod
    async def generate_summary(self, content: str, max_length: int = 100) -> str:
        ...
