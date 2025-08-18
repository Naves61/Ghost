from __future__ import annotations

import hashlib
import math
import re
from typing import Any, List

from .llm_base import LLMProvider
from .embeddings_base import EmbeddingsProvider
from .clock import SystemClock
from ..settings import settings


def _pick(seed: int, items: List[str]) -> str:
    return items[seed % len(items)]


class StubLLM(LLMProvider):
    """
    Deterministic, offline LLM that produces concise micro-thoughts (<= 60 words).
    It avoids echoing the whole prompt and instead synthesizes one sentence.
    Also returns deterministic JSON for the store-decider.
    """

    def generate(self, system: str, prompt: str, max_tokens: int = 256, **kwargs: Any) -> str:
        seed = int(hashlib.sha256((system + "|" + prompt).encode("utf-8")).hexdigest(), 16)

        if system.strip().startswith("STORE_DECIDER:"):
            ttypes = ["episodic", "semantic", "procedural"]
            ttype = _pick(seed // 3, ttypes)
            importance = ((seed % 100) / 100.0)
            should = importance > 0.55
            tag = _pick(seed, ["plan", "question", "insight"])
            return (
                f'{{"should_store": {str(should).lower()}, '
                f'"importance": {min(1.0, importance):.2f}, '
                f'"type": "{ttype}", "tags": ["auto","{tag}"]}}'
            )

        # Extract minimal context to craft a sentence
        # grab last "Stimuli:" block content
        m = re.search(r"Stimuli:\s*(.+)", prompt, flags=re.DOTALL)
        stimuli = (m.group(1) if m else "").strip().splitlines()
        snippet = ""
        for ln in stimuli[::-1]:
            ln = ln.strip()
            if ln:
                snippet = re.sub(r"^\w+:\s*", "", ln)  # drop "cli:" etc.
                break

        tag = _pick(seed, ["#plan", "#question", "#insight"])
        templates = {
            "#plan": [
                "Plan: break '{x}' into two actions and schedule the first.",
                "Next step on '{x}': draft outline and set a 30-min timer.",
                "Prioritize '{x}' then review outcomes briefly."
            ],
            "#question": [
                "What key facts are missing to advance '{x}'?",
                "Which constraints block '{x}' and who can clarify?",
                "What single source can verify assumptions about '{x}'?"
            ],
            "#insight": [
                "Small, timed chunks reduce drift on '{x}'.",
                "Link '{x}' to a concrete, testable result to cut uncertainty.",
                "If '{x}' lacks data, capture a single metric before proceeding."
            ]
        }[tag]
        sentence = _pick(seed // 5, templates).replace("{x}", (snippet or "the goal"))

        # ensure <= 60 words
        words = sentence.split()
        sentence = " ".join(words[:60])
        return f"{tag} {sentence}"


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
            buf = (h * ((dim // len(h)) + 1))[:dim]
            vec = [(b - 128) / 128.0 for b in buf]
            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec)) + 1e-8
            out.append([v / norm for v in vec])
        return out


SystemClock = SystemClock
