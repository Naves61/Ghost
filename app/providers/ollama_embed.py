from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Any, List

import httpx

from .embeddings_base import EmbeddingsProvider
from ..settings import settings

logger = logging.getLogger(__name__)

DEFAULT_MODELS = [
    "nomic-embed-text",
    "jina-embeddings-v2",
    "all-minilm",
]


class OllamaEmbeddings(EmbeddingsProvider):
    def __init__(self) -> None:
        self.base_url = settings.OLLAMA_URL.rstrip("/")
        self.client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=None)
        )
        self.model = settings.EMB_MODEL or self._select_model()

        if not self._ollama_available():
            raise RuntimeError(
                "No local Ollama embeddings backend. Start Ollama or set PROVIDER_EMBED=stub."
            )

        if not self._model_available(self.model):
            self._pull_model(self.model)
            if not self._model_available(self.model):
                raise RuntimeError(
                    f"Embedding model '{self.model}' unavailable. Run `ollama pull {self.model}`."
                )

        logger.info("Embeddings backend=ollama model=%s", self.model)

    # ------------------------------------------------------------------
    def _select_model(self) -> str:
        if settings.EMB_MODEL:
            return settings.EMB_MODEL
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            tags = {m.get("name") for m in resp.json().get("models", [])}
            for cand in DEFAULT_MODELS:
                if cand in tags:
                    return cand
        except Exception:
            pass
        return DEFAULT_MODELS[0]

    def _ollama_available(self) -> bool:
        try:
            self.client.get(f"{self.base_url}/api/tags")
            return True
        except Exception:
            return False

    def _model_available(self, model: str) -> bool:
        try:
            resp = self.client.get(f"{self.base_url}/api/tags")
            tags = {m.get("name") for m in resp.json().get("models", [])}
            return model in tags
        except Exception:
            return False

    def _pull_model(self, model: str) -> None:
        lock = Path("/tmp/ghost_ollama_pull.lock")
        now = time.time()
        if lock.exists() and now - lock.stat().st_mtime < 3600:
            return
        try:
            lock.touch()
            subprocess.run(["ollama", "pull", model], check=True)
        except Exception as e:
            logger.error("ollama pull failed: %s", e)
        finally:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass

    def _request_with_retries(self, method: str, url: str, **kwargs: Any):
        for attempt in range(3):
            try:
                r = self.client.request(method, url, **kwargs)
            except httpx.HTTPError:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
                continue
            if r.status_code >= 500:
                if attempt == 2:
                    r.raise_for_status()
                time.sleep(2**attempt)
                continue
            return r
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------
    def embed(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/api/embeddings"
        out: List[List[float]] = []
        for t in texts:
            payload = {"model": self.model, "input": t}
            r = self._request_with_retries("POST", url, json=payload)
            data = r.json()
            vec = data.get("embedding")
            if not isinstance(vec, list):
                vec = []
            out.append([float(x) for x in vec])
        return out

