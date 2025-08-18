from __future__ import annotations

import time

from app.memory_working import WorkingMemory
from app.schema import Thought


def test_wm_budget_and_decay():
    wm = WorkingMemory(char_budget=50, half_life_seconds=10)
    now = time.time()
    for i in range(10):
        th = Thought(
            id=f"t{i}",
            content="x" * 15,
            tags=["insight"],
            importance=0.5,
            uncertainty=0.5,
            ts=now - i,
        )
        wm.add(th)

    # Ensure budget
    total = sum(len(t.content) for t in wm.view(100))
    assert total <= 50

    scores = wm.score_decay(now=now + 20)
    assert len(scores) == len(wm.view(100))
    # older items should have lower score
    assert scores[0][1] >= scores[-1][1]
