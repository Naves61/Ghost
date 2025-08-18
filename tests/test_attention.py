from __future__ import annotations

import time
from app.attention import AttentionScheduler
from app.schema import Task


def make_task(id_: str, imp: float, unc: float, deadline: float | None) -> Task:
    now = time.time()
    return Task(
        id=id_,
        title=f"T {id_}",
        goal="test",
        context="",
        deadline_ts=deadline,
        importance=imp,
        uncertainty=unc,
        status="open",
        created_ts=now,
        last_update_ts=now,
        metadata={}
    )


def test_attention_scoring_and_heap():
    attn = AttentionScheduler()
    now = time.time()
    t1 = make_task("1", 0.9, 0.3, now + 3600)  # urgent
    t2 = make_task("2", 0.4, 0.7, None)
    attn.add_or_update(t1, goal_fit=0.8)
    attn.add_or_update(t2, goal_fit=0.2)

    peek = attn.peek()
    assert peek is not None
    assert peek[0] in ("1", "2")
    nxt = attn.next_task()
    assert nxt is not None
