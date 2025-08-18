from __future__ import annotations

from typing import Protocol, Optional, Any


class LLMProvider(Protocol):
    def generate(
        self,
        system: str,
        prompt: str,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> str: ...
