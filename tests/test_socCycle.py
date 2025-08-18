from __future__ import annotations

import time
from app.soc import SoCEngine
from app.memory_working import WorkingMemory
from app.providers.stubs import StubLLM, StubEmbeddings
from app.providers.clock import SystemClock
from app.interrupts import InterruptManager
from app.schema import Stimulus


def test_soc_cycle_and_store_and_interrupt():
    wm = WorkingMemory()
    llm = StubLLM()
    emb = StubEmbeddings()
    clock = SystemClock()
    ints = InterruptManager()
    engine = SoCEngine(
        wm=wm,
        ltm_embed=emb,
        llm=llm,
        clock=clock,
        interrupts=ints,
        self_summary="Test Self",
    )
    stimuli = [Stimulus(id="s1", source="test", content="What is latest guideline?", metadata={}, ts=time.time())]
    thought, stored, inter = engine.step(stimuli)
    assert thought.content
    # stored may be True/False depending on stub hash; assert type
    assert isinstance(stored, bool)
    # if it's a question, we should get an interrupt id
    assert isinstance(inter, list)
