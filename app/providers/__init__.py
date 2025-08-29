from .llm_base import LLMProvider
from .embeddings_base import EmbeddingsProvider
from .clock import Clock
from .stubs import StubLLM, StubEmbeddings, SystemClock
from .ollama import OllamaProvider
from .ollama_embed import OllamaEmbeddings
from ..settings import settings


def create_llm() -> LLMProvider:
    if settings.PROVIDER_LLM == "ollama":
        return OllamaProvider()
    return StubLLM()


def create_embeddings() -> EmbeddingsProvider:
    if settings.PROVIDER_EMBED == "ollama":
        return OllamaEmbeddings()
    return StubEmbeddings()

__all__ = [
    "LLMProvider",
    "EmbeddingsProvider",
    "Clock",
    "StubLLM",
    "StubEmbeddings",
    "SystemClock",
    "OllamaProvider",
    "OllamaEmbeddings",
    "create_llm",
    "create_embeddings",
]
