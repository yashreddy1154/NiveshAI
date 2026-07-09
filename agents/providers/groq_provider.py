"""
NiveshAI — Groq Provider
Free tier with 14,400 RPD. Requires user API key.
"""
# TODO: Implement in Phase 2

from agents.providers.base import LLMProvider


class GroqProvider(LLMProvider):
    """Groq API provider — free tier, fast inference."""

    def __init__(self, api_key: str = None, model_id: str = "llama-3.1-70b-versatile"):
        self._model_id = model_id
        self._api_key = api_key

    def generate(self, prompt: str, system: str = None) -> str:
        raise NotImplementedError("Will be implemented in Phase 2")

    def generate_with_tools(self, prompt: str, tools: list, system: str = None) -> dict:
        raise NotImplementedError("Will be implemented in Phase 2")

    @property
    def model_name(self) -> str:
        return "Groq LLaMA 3 70B"

    @property
    def is_free(self) -> bool:
        return True

    @property
    def daily_limit(self) -> int:
        return 14400
