import os
import httpx
import pytest

from app.providers.ollama import OllamaProvider
from app.settings import settings


def _backend_available() -> bool:
    try:
        httpx.get(f"{settings.OLLAMA_URL.rstrip('/')}/api/tags", timeout=1.0)
        return True
    except Exception:
        url = settings.LLAMACPP_SERVER_URL or os.environ.get("LLAMACPP_SERVER_URL")
        if not url:
            return False
        try:
            httpx.get(url.rstrip("/") + "/v1/models", timeout=1.0)
            return True
        except Exception:
            return False


@pytest.mark.skipif(not _backend_available(), reason="No local LLM backend")
def test_non_stream_generation():
    p = OllamaProvider()
    out = p.generate(system="You are a test bot", prompt="Say hello", max_tokens=32)
    assert isinstance(out, str) and out.strip()
    assert len(out.split()) <= settings.LLM_MAX_TOKENS + 50


@pytest.mark.skipif(not _backend_available(), reason="No local LLM backend")
def test_stream_generation():
    p = OllamaProvider()
    chunks = list(
        p.generate(
            system="You are a test bot",
            prompt="Say hello",
            max_tokens=32,
            stream=True,
        )
    )
    text = "".join(chunks)
    assert text.strip()
    assert len(text.split()) <= settings.LLM_MAX_TOKENS + 50


@pytest.mark.skipif(not _backend_available(), reason="No local LLM backend")
def test_seed_determinism(monkeypatch):
    monkeypatch.setattr(settings, "LLM_SEED", 42)
    p1 = OllamaProvider()
    out1 = p1.generate(system="You are a test bot", prompt="Tell a joke", max_tokens=32)
    p2 = OllamaProvider()
    out2 = p2.generate(system="You are a test bot", prompt="Tell a joke", max_tokens=32)
    assert out1 == out2


@pytest.mark.skipif(not _backend_available(), reason="No local LLM backend")
def test_stop_tokens():
    p = OllamaProvider()
    stop_token = "qqqSTOPzzz"
    out = p.generate(
        system="You are a test bot",
        prompt="Say this should end",
        max_tokens=32,
        stop=[stop_token],
    )
    assert stop_token.lower() not in out.lower()
