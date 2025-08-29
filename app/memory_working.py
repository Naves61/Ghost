from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, List, Tuple
from collections import deque
import time

from .schema import Thought
from .settings import settings


@dataclass
class WorkingMemory:
    char_budget: int = settings.WM_CHAR_BUDGET
    half_life_seconds: int = settings.WM_HALF_LIFE_SECONDS
    _buf: Deque[Thought] = field(default_factory=deque)
    _total_chars: int = 0

    def add(self, thought: Thought) -> None:
        """O(1) amortized append; trims to budget."""
        self._buf.append(thought)
        self._total_chars += len(thought.content)
        print(f"[memory_working.py] add: thought.id={thought.id} tags={thought.tags} chars={len(thought.content)}")
        self.truncate_to_budget()

    def view(self, n: int = 10) -> List[Thought]:
        return list(self._buf)[-n:]

    def score_decay(self, now: float | None = None) -> List[Tuple[Thought, float]]:
        """Return decayed scores per thought using exponential half-life on recency."""
        now = now or time.time()
        scores: List[Tuple[Thought, float]] = []
        for th in self._buf:
            dt = max(0.0, now - th.ts)
            # decay = 0.5 ** (dt / half_life)
            if self.half_life_seconds <= 0:
                decay = 1.0
            else:
                decay = 0.5 ** (dt / self.half_life_seconds)
            score = th.importance * decay
            scores.append((th, score))
        return scores

    def truncate_to_budget(self) -> None:
        """Ensure char budget by popping left."""
        while self._buf and self._total_chars > self.char_budget:
            left = self._buf.popleft()
            self._total_chars -= len(left.content)
