from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, model: str, prompt: str) -> str:
        """Generate a complete response for the given prompt."""

    @abstractmethod
    async def stream(self, model: str, prompt: str) -> AsyncIterator[str]:
        """Stream response tokens for the given prompt."""

    @abstractmethod
    async def health(self) -> bool:
        """Return True if the provider is reachable."""
