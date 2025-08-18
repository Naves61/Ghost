from __future__ import annotations

import time
from typing import Dict, List, Optional

from .schema import Query


class InterruptManager:
    def __init__(self) -> None:
        self._store: Dict[str, Query] = {}
        self.user_available: bool = True  # toggleable externally

    def create(self, question: str, rationale: str, required_fields: List[str]) -> str:
        q = Query(
            id=f"q:{int(time.time()*1000)}",
            question=question,
            rationale=rationale,
            required_fields=required_fields,
            created_ts=time.time(),
        )
        self._store[q.id] = q
        return q.id

    def list_open(self) -> List[Query]:
        return [q for q in self._store.values() if not q.answered]

    def answer(self, qid: str, answer_text: str) -> bool:
        q = self._store.get(qid)
        if not q:
            return False
        q.answered = True
        q.answer_text = answer_text
        q.answer_ts = time.time()
        return True

    def get(self, qid: str) -> Optional[Query]:
        return self._store.get(qid)
