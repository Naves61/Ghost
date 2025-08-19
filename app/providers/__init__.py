from .llm_base import LLMProvider
from .embeddings_base import EmbeddingsProvider
from .clock import Clock
from .stubs import StubLLM, StubEmbeddings, SystemClock
from .ollama import OllamaProvider
from ..settings import settings


def create_llm() -> LLMProvider:
    if settings.PROVIDER_LLM == "ollama":
        return OllamaProvider()
    return StubLLM()

__all__ = [
    "LLMProvider",
    "EmbeddingsProvider",
    "Clock",
    "StubLLM",
    "StubEmbeddings",
    "SystemClock",
    "OllamaProvider",
    "create_llm",
]
