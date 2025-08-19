from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .schema import Task
from .settings import settings


def _urgency(deadline_ts: Optional[float], now: float) -> float:
    if not deadline_ts:
        return 0.0
    dt = max(0.0, deadline_ts - now)
    # map dt to [0,1], where near 0 -> urgency ~1
    return max(0.0, min(1.0, 1.0 - (dt / (60 * 60 * 24))))  # within 24h => high urgency


def _recency(last_update: float, now: float) -> float:
    dt = max(0.0, now - last_update)
    # older -> lower recency
    return max(0.0, min(1.0, 1.0 / (1.0 + dt / 3600.0)))


@dataclass(order=True)
class _Item:
    priority: float
    task_id: str = field(compare=False)


class AttentionScheduler:
    def __init__(self) -> None:
        self.heap: List[_Item] = []
        self.cache: Dict[str, float] = {}
        self.tasks: Dict[str, Task] = {}

    def _score(self, task: Task, goal_fit: float, now: Optional[float] = None) -> float:
        now = now or time.time()
        urgency = _urgency(task.deadline_ts, now)
        recency = _recency(task.last_update_ts, now)
        score = (
            settings.W_URGENCY * urgency
            + settings.W_IMPORTANCE * task.importance
            + settings.W_UNCERTAINTY * task.uncertainty
            + settings.W_RECENCY * recency
            + settings.W_GOALFIT * max(0.0, min(1.0, goal_fit))
        )
        return score

    def add_or_update(self, task: Task, goal_fit: float) -> float:
        s = self._score(task, goal_fit)
        self.tasks[task.id] = task
        self.cache[task.id] = s
        heapq.heappush(self.heap, _Item(priority=-s, task_id=task.id))
        return s

    def _refresh(self) -> None:
        # lazy invalidation
        while self.heap:
            top = self.heap[0]
            sid = top.task_id
            current = self.cache.get(sid)
            if current is None or -top.priority != current:
                heapq.heappop(self.heap)
            else:
                break

    def peek(self) -> Optional[Tuple[str, float]]:
        self._refresh()
        if not self.heap:
            return None
        top = self.heap[0]
        return (top.task_id, -top.priority)

    def next_task(self) -> Optional[Task]:
        self._refresh()
        if not self.heap:
            return None
        item = heapq.heappop(self.heap)
        tid = item.task_id
        return self.tasks.get(tid)

    def top_k(self, k: int = 5) -> List[Tuple[Task, float]]:
        self._refresh()
        items: List[Tuple[Task, float]] = []
        for tid, score in sorted(self.cache.items(), key=lambda x: x[1], reverse=True)[:k]:
            task = self.tasks.get(tid)
            if task:
                items.append((task, score))
        return items
