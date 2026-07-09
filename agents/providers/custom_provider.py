"""
NiveshAI — Custom OpenAI-Compatible Provider
For self-hosted or third-party OpenAI-compatible endpoints.
"""
# TODO: Implement in Phase 2

from agents.providers.base import LLMProvider


class CustomProvider(LLMProvider):
    """Custom OpenAI-compatible endpoint provider."""

    def __init__(self, api_key: str = None, base_url: str = None, model_id: str = "custom"):
        self._model_id = model_id
        self._api_key = api_key
        self._base_url = base_url

    def generate(self, prompt: str, system: str = None) -> str:
        raise NotImplementedError("Will be implemented in Phase 2")

    def generate_with_tools(self, prompt: str, tools: list, system: str = None) -> dict:
        raise NotImplementedError("Will be implemented in Phase 2")

    @property
    def model_name(self) -> str:
        return f"Custom ({self._model_id})"

    @property
    def is_free(self) -> bool:
        return False

    @property
    def daily_limit(self) -> int:
        return float("inf")
