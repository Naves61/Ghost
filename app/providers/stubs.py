from __future__ import annotations

import hashlib
import math
from typing import Any, List

from .llm_base import LLMProvider
from .embeddings_base import EmbeddingsProvider
from .clock import SystemClock
from ..settings import settings


class StubLLM(LLMProvider):
    """
    Deterministic, offline LLM:
    - Emits a micro-thought <= 60 words with one of tags #plan/#question/#insight chosen by hashing the prompt.
    - StoreDecider JSON when prompted with 'STORE_DECIDER:' prefix in system string.
    """

    def generate(self, system: str, prompt: str, max_tokens: int = 256, **kwargs: Any) -> str:
        seed = int(hashlib.sha256((system + "|" + prompt).encode("utf-8")).hexdigest(), 16)
        tags = ["#plan", "#question", "#insight"]
        tag = tags[seed % 3]
        if system.strip().startswith("STORE_DECIDER:"):
            # Return simple JSON with deterministic choices
            ttypes = ["episodic", "semantic", "procedural"]
            ttype = ttypes[(seed // 3) % 3]
            importance = ((seed % 100) / 100.0)
            should = "true" if importance > 0.55 or tag in ("#plan", "#insight") else "false"
            return (
                f'{{"should_store": {should}, "importance": {min(1.0, importance):.2f}, '
                f'"type": "{ttype}", "tags": ["auto","{tag[1:]}"]}}'
            )
        words = prompt.strip().split()
        body = " ".join(words[: min(len(words), 60)])
        return f"{tag} {body}"


class StubEmbeddings(EmbeddingsProvider):
    """
    Deterministic pseudo-embeddings from SHA256.
    Dimension from settings.VECTOR_DIM.
    """
    def embed(self, texts: List[str]) -> List[List[float]]:
        dim = settings.VECTOR_DIM
        out: List[List[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # repeat digest to fill dim
            buf = (h * ((dim // len(h)) + 1))[:dim]
            vec = [(b - 128) / 128.0 for b in buf]
            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec)) + 1e-8
            out.append([v / norm for v in vec])
        return out


SystemClock = SystemClock
