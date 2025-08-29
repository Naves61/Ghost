from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Dict, List, Optional, Tuple

from .schema import Memory, Stimulus, Thought, now_ts
from .settings import settings
from .providers import (
    LLMProvider,
    EmbeddingsProvider,
    Clock,
    SystemClock,
    create_llm,
    create_embeddings,
)
from .memory_working import WorkingMemory
from .memory_longterm import upsert_memory, vector_search
from .interrupts import InterruptManager


class SoCEngine:
    def __init__(
        self,
        wm: WorkingMemory,
        ltm_embed: EmbeddingsProvider,
        llm: LLMProvider,
        clock: Clock,
        interrupts: InterruptManager,
        self_summary: str,
    ) -> None:
        self.wm = wm
        self.embed = ltm_embed
        self.llm = llm
        self.clock = clock
        self.interrupts = interrupts
        self.self_summary = self_summary

    def _retrieve_ltm(self, query_text: str, top_k: int = 5) -> List[Memory]:
        qv = self.embed.embed([query_text])[0]
        hits = vector_search(qv, top_k=top_k)
        print(f"[soc.py] _retrieve_ltm: query='{query_text[:60]}' top_k={top_k} -> {len(hits)} hits")
        return [m for m, _ in hits]

    def _wm_summary(self, k: int = 5) -> str:
        last = self.wm.view(k)
        return "\n".join([f"- {t.tags} {t.content[:80]}" for t in last])

    def _prompt(self, stimuli: List[Stimulus]) -> Tuple[str, str]:
        stim_text = "\n".join([f"{s.source}: {s.content}" for s in stimuli[-5:]])
        ltm_hits = self._retrieve_ltm(stim_text or "general", top_k=3)
        ltm_text = "\n".join([f"{m.type}:{m.content[:120]}" for m in ltm_hits])
        wm_text = self._wm_summary(5)
        system = (
            "Internal SoC micro-thought generator. Output <= 60 words and begin with one tag "
            "#plan|#question|#insight."
        )
        prompt = (
            f"Self: {self.self_summary}\n"
            f"WM:\n{wm_text}\n"
            f"LTM:\n{ltm_text}\n"
            f"Stimuli:\n{stim_text}\n"
            "Produce one helpful micro-thought."
        )
        print("[soc.py] _prompt: built prompt with sections sizes:",
              f"Stimuli={len(stimuli)} lines, WM_chars={len(wm_text)}, LTM_chars={len(ltm_text)}")
        return system, prompt

    def _store_decider(self, content: str) -> Dict[str, object]:
        system = "STORE_DECIDER:"
        prompt = (
            "Given the micro-thought and context, decide if it should be stored in LTM. "
            'Return JSON: {"should_store": bool, "importance": float, "type": "episodic|semantic|procedural", "tags": [..]}.\n'
            f"thought: {content}"
        )
        raw = self.llm.generate(system=system, prompt=prompt, max_tokens=96)
        print(f"[soc.py] _store_decider: raw='{str(raw)[:120]}'")
        try:
            obj = json.loads(raw)
            print(f"[soc.py] _store_decider: parsed -> {obj}")
            return obj  # type: ignore[return-value]
        except json.JSONDecodeError:
            print("[soc.py] _store_decider: JSON parse failed; defaulting to no-store")
            return {"should_store": False, "importance": 0.3, "type": "episodic", "tags": ["auto"]}

    def _maybe_interrupt(self, text: str) -> Optional[str]:
        if text.startswith("#question"):
            # Extract question after tag
            q = text[len("#question") :].strip()
            if q:
                return q
        return None

    def step(self, stimuli: List[Stimulus]) -> Tuple[Thought, bool, List[str]]:
        print(f"[soc.py] step: received {len(stimuli)} stimuli -> {[s.content[:60] for s in stimuli]}")
        system, prompt = self._prompt(stimuli)
        gen = self.llm.generate(system=system, prompt=prompt, max_tokens=settings.SOC_MAX_TOKENS)
        print(f"[soc.py] step: llm.generate -> '{str(gen)[:200]}'")
        tag = "#insight"
        for t in ("#plan", "#question", "#insight"):
            if gen.startswith(t):
                tag = t
                break
        th = Thought(
            id=f"thought:{int(self.clock.now()*1000)}",
            content=gen,
            tags=[tag[1:]],
            importance=0.6 if tag != "#insight" else 0.5,
            uncertainty=0.4 if tag == "#plan" else 0.5,
            ts=self.clock.now(),
        )
        self.wm.add(th)
        print(f"[soc.py] step: WM.add -> thought.id={th.id} tags={th.tags} len={len(th.content)}")

        stored = False
        decide = self._store_decider(gen)
        if bool(decide.get("should_store", False)):
            mem = Memory(
                id=f"mem:{int(self.clock.now()*1000)}",
                type=str(decide.get("type", "episodic")),  # type: ignore[arg-type]
                content=gen,
                embedding=self.embed.embed([gen])[0],
                importance=float(decide.get("importance", 0.5)),
                created_at=self.clock.now(),
                last_access=self.clock.now(),
                metadata={"tags": decide.get("tags", [])},
            )
            upsert_memory(mem)
            print(f"[soc.py] step: stored LTM id={mem.id} type={mem.type} imp={mem.importance}")
            stored = True

        interrupts: List[str] = []
        q = self._maybe_interrupt(gen)
        if q:
            iq = self.interrupts.create(question=q, rationale="Missing external info", required_fields=[])
            interrupts.append(iq)
            print(f"[soc.py] step: interrupt created id={iq} question='{q[:100]}'")

            # Auto-escalate to scraper if user unavailable AND the question contains URLs on the allowlist.
            if not self.interrupts.user_available:
                import re
                from .scraper import ScrapePlan, Scraper
                from .settings import settings as _settings
                urls = re.findall(r"https?://[^\s)]+", q)
                seeds = [u for u in urls if any(
                    u.startswith(f"https://{d}") or u.startswith(f"http://{d}") for d in _settings.allowlist())]
                if seeds:
                    Scraper().execute(ScrapePlan(question=q, seeds=seeds, max_pages=_settings.SCRAPER_MAX_PAGES,
                                                 allow_domains=_settings.allowlist()))

        return th, stored, interrupts


async def run_soc_loop(engine: SoCEngine, get_stimuli_cb, stop_evt: asyncio.Event) -> None:
    jitter = settings.SOC_JITTER_SECONDS
    while not stop_evt.is_set():
        stimuli = await get_stimuli_cb()
        print(f"[soc.py] run_soc_loop: stepping with {len(stimuli)} stimuli")
        thought, stored, interrupts = engine.step(stimuli)
        print(
            json.dumps(
                {
                    "event": "soc_thought",
                    "thought": thought.model_dump(),
                    "stored": stored,
                    "interrupts": interrupts,
                }
            )
        )
        cadence = settings.SOC_CADENCE_SECONDS
        sleep_s = cadence + random.uniform(-jitter, jitter)
        await asyncio.wait_for(stop_evt.wait(), timeout=max(0.01, sleep_s))


async def run_soc_main() -> None:
    wm = WorkingMemory()
    llm: LLMProvider = create_llm()
    emb: EmbeddingsProvider = create_embeddings()
    clock: Clock = SystemClock()
    from .interrupts import InterruptManager

    interrupts = InterruptManager()
    engine = SoCEngine(
        wm=wm,
        ltm_embed=emb,
        llm=llm,
        clock=clock,
        interrupts=interrupts,
        self_summary="Ghost v0.1, pragmatic assistant following principles and goals.",
    )

    async def _get_stimuli() -> List[Stimulus]:
        return []

    stop_evt = asyncio.Event()
    try:
        await run_soc_loop(engine, _get_stimuli, stop_evt)
    except asyncio.TimeoutError:
        pass


if __name__ == "__main__":
    asyncio.run(run_soc_main())
