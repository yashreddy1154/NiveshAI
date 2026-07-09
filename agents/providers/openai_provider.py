"""
NiveshAI — OpenAI Provider
Requires user-provided API key.
"""
# TODO: Implement in Phase 2

from agents.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI API provider — requires user API key."""

    def __init__(self, api_key: str = None, model_id: str = "gpt-4o"):
        self._model_id = model_id
        self._api_key = api_key

    def generate(self, prompt: str, system: str = None) -> str:
        raise NotImplementedError("Will be implemented in Phase 2")

    def generate_with_tools(self, prompt: str, tools: list, system: str = None) -> dict:
        raise NotImplementedError("Will be implemented in Phase 2")

    @property
    def model_name(self) -> str:
        return "OpenAI GPT-4o"

    @property
    def is_free(self) -> bool:
        return False

    @property
    def daily_limit(self) -> int:
        return float("inf")
