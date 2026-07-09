"""
NiveshAI — LLM Provider Base Class
Abstract interface for all LLM providers.
"""
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base for all LLM providers (Gemini, OpenAI, Groq, etc.)."""

    @abstractmethod
    def generate(self, prompt: str, system: str = None) -> str:
        """Generate a text response."""
        ...

    @abstractmethod
    def generate_with_tools(self, prompt: str, tools: list, system: str = None) -> dict:
        """Generate a response with tool-calling support."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model name."""
        ...

    @property
    @abstractmethod
    def is_free(self) -> bool:
        """Whether this provider offers a free tier."""
        ...

    @property
    @abstractmethod
    def daily_limit(self) -> int:
        """Max requests per day."""
        ...
