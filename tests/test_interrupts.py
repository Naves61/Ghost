from __future__ import annotations

from app.interrupts import InterruptManager


def test_interrupt_flow():
    im = InterruptManager()
    qid = im.create("Need info on X", "rationale", [])
    lst = im.list_open()
    assert any(q.id == qid for q in lst)
    ok = im.answer(qid, "answer")
    assert ok
    assert not any(not q.answered for q in im.list_open())
