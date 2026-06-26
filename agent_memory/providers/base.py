"""Abstract base classes for embedding and LLM providers."""

from abc import ABC, abstractmethod


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

    @abstractmethod
    async def assess_importance(self, content: str) -> float:
        ...

    @abstractmethod
    async def generate_summary(self, content: str, max_length: int = 100) -> str:
        ...
