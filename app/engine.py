from __future__ import annotations

import signal
import time
from typing import Dict, List, Optional, Tuple

from .schema import Stimulus, Thought, Task
from .attention import AttentionScheduler
from .soc import SoCEngine
from .providers import LLMProvider, EmbeddingsProvider, Clock
from .interrupts import InterruptManager


class Engine:
    def __init__(
        self,
        attn: AttentionScheduler,
        soc: SoCEngine,
        interrupts: InterruptManager,
    ) -> None:
        self.attn = attn
        self.soc = soc
        self.interrupts = interrupts
        self._stop = False
        signal.signal(signal.SIGTERM, self._handle_stop)
        signal.signal(signal.SIGINT, self._handle_stop)

    def _handle_stop(self, *_: object) -> None:
        self._stop = True

    def cycle(self, stimuli: List[Stimulus]) -> Dict[str, object]:
        thought, stored, interrupts = self.soc.step(stimuli)
        nxt = self.attn.peek()
        return {
            "thought": thought.model_dump(),
            "stored": stored,
            "interrupts": interrupts,
            "next_task": nxt[0] if nxt else None,
        }
