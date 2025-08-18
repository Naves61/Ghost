from __future__ import annotations

from typing import List, Protocol


class EmbeddingsProvider(Protocol):
    def embed(self, texts: List[str]) -> List[List[float]]: ...
