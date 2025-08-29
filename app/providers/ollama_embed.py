from .embeddings_base import EmbeddingsProvider

class OllamaEmbeddings(EmbeddingsProvider):
    """Placeholder embeddings provider for tests."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("OllamaEmbeddings not implemented in test environment")
