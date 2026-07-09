"""
NiveshAI — Google Gemini Provider
Pre-configured with built-in free API key.
"""
# TODO: Implement in Phase 2

from agents.providers.base import LLMProvider


class GeminiProvider(LLMProvider):
    """Google Gemini API provider — free tier with 1,500 RPD."""

    def __init__(self, api_key: str = None, model_id: str = "gemini-2.0-flash"):
        self._model_id = model_id
        self._api_key = api_key
        # TODO: Initialize Gemini client

    def generate(self, prompt: str, system: str = None) -> str:
        raise NotImplementedError("Will be implemented in Phase 2")

    def generate_with_tools(self, prompt: str, tools: list, system: str = None) -> dict:
        raise NotImplementedError("Will be implemented in Phase 2")

    @property
    def model_name(self) -> str:
        return "Google Gemini 2.0 Flash"

    @property
    def is_free(self) -> bool:
        return True

    @property
    def daily_limit(self) -> int:
        return 1500
