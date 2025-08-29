import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Generator, List, Optional, Union

import httpx

from .llm_base import LLMProvider
from ..settings import settings

logger = logging.getLogger(__name__)

DEFAULT_MODELS = [
    "llama3.1:8b-instruct-q5_K_M",
    "llama3.1:8b-instruct",
    "mixtral:8x7b-instruct-q5_K_M",
    "qwen2.5:7b-instruct-q5_K_M",
]


class OllamaProvider(LLMProvider):
    def __init__(self) -> None:
        self.base_url = settings.OLLAMA_URL.rstrip("/")
        self.lcpp_url = settings.LLAMACPP_SERVER_URL
        self.client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=None)
        )
        self.ctx = settings.LLM_CTX
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.temp = settings.LLM_TEMP
        self.top_p = settings.LLM_TOP_P
        self.top_k = settings.LLM_TOP_K
        self.seed = settings.LLM_SEED
        self.backend = "ollama"
        self.model = self._select_model()

        if not self._ollama_available():
            if self.lcpp_url:
                self.backend = "llama.cpp"
            else:
                raise RuntimeError(
                    "No local LLM backend found. Start Ollama or set LLAMACPP_SERVER_URL."
                )

        if self.backend == "ollama" and not self._model_available(self.model):
            self._pull_model(self.model)
            if not self._model_available(self.model):
                raise RuntimeError(
                    f"Model '{self.model}' unavailable. Run scripts/dev_setup_local_llm.sh or `ollama pull {self.model}`."
                )

        logger.info(
            "LLM backend=%s model=%s ctx=%s max_tokens=%s temp=%s seed=%s streaming=yes",
            self.backend,
            self.model,
            self.ctx,
            self.max_tokens,
            self.temp,
            self.seed,
        )

    # ------------------------------------------------------------------
    def _select_model(self) -> str:
        if settings.LLM_MODEL:
            return settings.LLM_MODEL
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

    def _request_with_retries(self, method: str, url: str, *, stream: bool = False, **kwargs: Any):
        for attempt in range(3):
            try:
                if stream:
                    r = self.client.stream(method, url, **kwargs)
                    if r.status_code >= 500:
                        raise httpx.HTTPStatusError("server error", request=r.request, response=r)
                    return r
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
    def generate(
        self,
        system: str,
        prompt: str,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> Union[str, Generator[str, None, None]]:
        stream: bool = kwargs.get("stream", False)
        stop: Optional[List[str]] = kwargs.get("stop")

        print(f"[providers/ollama.py] generate: backend={self.backend} model={self.model} max_tokens={max_tokens} stream={stream}")

        if len(system) + len(prompt) > self.ctx - 512:
            logger.warning("Prompt length approaching context window")

        if self.backend == "ollama":
            return self._gen_ollama(system, prompt, max_tokens, stop, stream)
        return self._gen_llamacpp(system, prompt, max_tokens, stop, stream)

    # ------------------------------------------------------------------
    def _gen_ollama(
        self,
        system: str,
        prompt: str,
        max_tokens: int,
        stop: Optional[List[str]],
        stream: bool,
    ) -> Union[str, Generator[str, None, None]]:
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": stream,
            "options": {
                "temperature": self.temp,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "num_ctx": self.ctx,
                "num_predict": max_tokens,
            },
        }
        if stop:
            payload["stop"] = stop
        if self.seed is not None:
            payload["options"]["seed"] = self.seed

        if not stream:
            resp = self._request_with_retries("POST", url, json=payload)
            data = resp.json()
            return data.get("message", {}).get("content", "")

        def gen() -> Generator[str, None, None]:
            with self._request_with_retries("POST", url, json=payload, stream=True) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("done"):
                        break
                    chunk = data.get("message", {}).get("content")
                    if chunk:
                        yield chunk

        return gen()

    # ------------------------------------------------------------------
    def _gen_llamacpp(
        self,
        system: str,
        prompt: str,
        max_tokens: int,
        stop: Optional[List[str]],
        stream: bool,
    ) -> Union[str, Generator[str, None, None]]:
        url = self.lcpp_url.rstrip("/") + "/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temp,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "n_ctx": self.ctx,
            "n_predict": max_tokens,
            "stream": stream,
        }
        if stop:
            payload["stop"] = stop
        if self.seed is not None:
            payload["seed"] = self.seed

        if not stream:
            resp = self._request_with_retries("POST", url, json=payload)
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        def gen() -> Generator[str, None, None]:
            with self._request_with_retries("POST", url, json=payload, stream=True) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[len("data:"):].strip()
                    if line.strip() == "[DONE]":
                        break
                    data = json.loads(line)
                    delta = data["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta

        return gen()
