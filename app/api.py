from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

import orjson
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles

from .schema import Stimulus, StimulusIn
from .settings import settings
from .memory_working import WorkingMemory
from .memory_longterm import keyword_search, vector_search, list_recent
from .providers import (
    LLMProvider,
    EmbeddingsProvider,
    Clock,
    SystemClock,
    create_llm,
    create_embeddings,
)
from .soc import SoCEngine, run_soc_loop
from .attention import AttentionScheduler
from .interrupts import InterruptManager
from .scraper import ScrapePlan, Scraper

# Observability
from prometheus_client import CollectorRegistry, Counter, generate_latest, CONTENT_TYPE_LATEST


def orjson_response(data: Any) -> ORJSONResponse:
    return ORJSONResponse(content=data)


app = FastAPI(title="Ghost API", version="0.1.0")
app.mount("/ui", StaticFiles(directory="app/ui", html=True), name="ui")

REG = CollectorRegistry()
INGEST_COUNTER = Counter("ghost_ingest_total", "Stimuli ingested", registry=REG)
SOC_CYCLE_COUNTER = Counter("ghost_soc_cycles_total", "SoC cycles", registry=REG)

# Simple in-proc singletons
WM = WorkingMemory()
EMB: EmbeddingsProvider = create_embeddings()
LLM: LLMProvider = create_llm()
CLOCK: Clock = SystemClock()
INT = InterruptManager()
ATTN = AttentionScheduler()
SOC = SoCEngine(WM, EMB, LLM, CLOCK, INT, "Ghost v0.1: pragmatic assistant.")

STOP_EVT = asyncio.Event()

# Pending stimuli buffer for the background SoC loop
PENDING_STIMULI: asyncio.Queue[Stimulus] = asyncio.Queue()


async def _get_stimuli() -> List[Stimulus]:
    """Drain pending stimuli (non-blocking) for consumption by the SoC loop."""
    items: List[Stimulus] = []
    try:
        while True:
            items.append(PENDING_STIMULI.get_nowait())
    except asyncio.QueueEmpty:
        pass
    if items:
        print(f"[api.py] _get_stimuli: drained {len(items)} stimuli for SoC loop")
    return items


@app.on_event("startup")
async def _startup() -> None:
    if settings.SOC_ENABLED:
        # create stop event & background task
        app.state.soc_stop = asyncio.Event()
        app.state.soc_task = asyncio.create_task(run_soc_loop(SOC, _get_stimuli, app.state.soc_stop))
        print("[api.py] startup: SoC background loop started (SOC_ENABLED=true)")

@app.on_event("shutdown")
async def _shutdown() -> None:
    # signal loop to stop and await task
    stop_evt = getattr(app.state, "soc_stop", None)
    task = getattr(app.state, "soc_task", None)
    if stop_evt is not None:
        stop_evt.set()
    if task is not None:
        try:
            await task
        except Exception:
            pass


def _auth_dep(request: Request) -> None:
    if settings.DEV_DISABLE_AUTH:
        return
    token = request.headers.get("Authorization", "")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/", response_class=HTMLResponse)
def index() -> Any:
    with open("app/ui/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/stimuli")
async def ingest(stimuli: list[StimulusIn], _: None = Depends(_auth_dep)) -> Any:
    payload = [s.model_dump() for s in stimuli]
    print("[api.py] POST /stimuli payload:", json.dumps(payload, indent=2))
    INGEST_COUNTER.inc(len(stimuli))
    # For demo: convert into episodic LTM immediately (low importance)
    from .memory_longterm import upsert_memory
    from .schema import Memory, now_ts
    try:
        for s in payload:
            mem = Memory(
                id=f"stim:{int(time.time()*1000)}",
                type="episodic",
                content=str(s.get("content", "")),
                embedding=EMB.embed([str(s.get("content",""))])[0],
                importance=0.3,
                created_at=now_ts(),
                last_access=now_ts(),
                metadata={"source": s.get("source", "api")},
            )
            upsert_memory(mem)
            print(f"[api.py] /stimuli: upsert LTM episodic id={mem.id} chars={len(mem.content)} source={mem.metadata.get('source')}")
    except Exception as e:
        print("ERROR in upsert_memory:", repr(e))
        raise HTTPException(status_code=500, detail=f"upsert_memory failed: {type(e).__name__}: {e}")

    # Build Stimulus objects with required fields
    now = time.time()
    soc_stimuli = [
        Stimulus(
            id=f"stim:{uuid4()}",
            source=s["source"],
            content=s["content"],
            metadata={"source": s["source"]},
            ts=now,
        )
        for s in payload
    ]

    # sanity check to avoid the validation error
    assert all(isinstance(x, Stimulus) for x in soc_stimuli), "SOC input must be Stimulus objects"

    # If background SoC is running, enqueue and return to avoid double-processing.
    if settings.SOC_ENABLED and getattr(app.state, "soc_task", None) is not None:
        print("[api.py] /stimuli: enqueueing stimuli for background SoC loop")
        for st in soc_stimuli:
            try:
                PENDING_STIMULI.put_nowait(st)
                print(f"[api.py] /stimuli: enqueued id={st.id} content='{st.content[:80]}'")
                print(f"[api.py] /stimuli: number of PENDING STIMULI = {PENDING_STIMULI.qsize()}")
            except asyncio.QueueFull:
                print("[api.py] /stimuli: WARNING queue full; dropping stimulus")
                pass
        wm_view = [t.model_dump() for t in WM.view(5)]
        return {"ok": True, "queued": len(soc_stimuli), "wm": wm_view}

    # Otherwise, run a synchronous SoC step and include the result
    SOC_CYCLE_COUNTER.inc()
    try:
        print("[api.py] /stimuli: SoC disabled -> running SOC.step() synchronously")
        thought, stored, interrupts = SOC.step(soc_stimuli)
        wm_view = [t.model_dump() for t in WM.view(5)]
        print(f"[api.py] /stimuli: SOC.step -> thought.id={thought.id} tags={thought.tags} stored={stored} interrupts={len(interrupts)}")
    except Exception as e:
        print("ERROR in SOC.step/WM.view:", repr(e))
        raise HTTPException(status_code=500, detail=f"SOC/WM failed: {type(e).__name__}: {e}")

    try:
        _get_stimuli()
    except Exception as e:
        print("ERROR in _get_stimuli:", repr(e))

    return {"ok": True, "thought": thought.model_dump(), "stored": stored, "interrupts": interrupts, "wm": wm_view}


@app.get("/wm")
def get_wm(_: None = Depends(_auth_dep)) -> Any:
    return orjson_response([t.model_dump() for t in WM.view(20)])


@app.get("/ltm/search")
def ltm_search(q: str, top_k: int = 5, _: None = Depends(_auth_dep)) -> Any:
    if not q.strip():
        return []
    kw = keyword_search(q, top_k=top_k)
    vq = EMB.embed([q])[0]
    vs = vector_search(vq, top_k=top_k)
    return {
        "keyword": [m.model_dump() for m in kw],
        "vector": [{"mem": m.model_dump(), "sim": sim} for m, sim in vs],
    }


@app.get("/ltm/recent")
def ltm_recent(top_k: int = 5, _: None = Depends(_auth_dep)) -> Any:
    return [m.model_dump() for m in list_recent(top_k)]


@app.get("/ltm/recent")
def ltm_recent(top_k: int = 5, _: None = Depends(_auth_dep)) -> Any:
    return [m.model_dump() for m in list_recent(top_k)]


@app.get("/tasks/next")
def tasks_next(_: None = Depends(_auth_dep)) -> Any:
    nxt = ATTN.peek()
    return {"next": nxt[0] if nxt else None, "score": nxt[1] if nxt else None}


@app.get("/tasks/top")
def tasks_top(k: int = 5, _: None = Depends(_auth_dep)) -> Any:
    return [
        {"task": t.model_dump(), "score": s} for t, s in ATTN.top_k(k)
    ]

@app.get("/config")
def get_config(_: None = Depends(_auth_dep)) -> Any:
    return {
        "user_available": INT.user_available,
        "allowlist": settings.allowlist(),
        "soc_enabled": settings.SOC_ENABLED,
        "soc_cadence": settings.SOC_CADENCE_SECONDS,
    }

@app.post("/config/user_available")
def set_user_available(available: bool, _: None = Depends(_auth_dep)) -> Any:
    INT.user_available = available
    return {"ok": True, "user_available": INT.user_available}


@app.post("/config/soc_cadence")
def set_soc_cadence(seconds: float, _: None = Depends(_auth_dep)) -> Any:
    settings.SOC_CADENCE_SECONDS = max(0.01, float(seconds))
    return {"ok": True, "soc_cadence": settings.SOC_CADENCE_SECONDS}

@app.get("/interrupts")
def list_interrupts(_: None = Depends(_auth_dep)) -> Any:
    return [q.model_dump() for q in INT.list_open()]


@app.post("/interrupts/answer")
def answer_interrupt(qid: str, text: str, _: None = Depends(_auth_dep)) -> Any:
    ok = INT.answer(qid, text)
    if not ok:
        raise HTTPException(status_code=404, detail="Unknown interrupt")

    from .memory_longterm import upsert_memory
    from .schema import Memory, now_ts

    # treat answer as stimulus and store in LTM
    stim = Stimulus(id=f"ans:{int(time.time() * 1000)}", source="user", content=text, metadata={"qid": qid},
                    ts=now_ts())
    mem = Memory(
        id=f"mem:{int(time.time() * 1000)}",
        type="episodic",
        content=text,
        embedding=EMB.embed([text])[0],
        importance=0.4,
        created_at=now_ts(),
        last_access=now_ts(),
        metadata={"source": "interrupt", "qid": qid},
    )
    upsert_memory(mem)

    SOC_CYCLE_COUNTER.inc()
    thought, stored, interrupts = SOC.step([stim])
    wm_view = [t.model_dump() for t in WM.view(5)]
    return {
        "ok": True,
        "thought": thought.model_dump(),
        "stored": stored,
        "interrupts": interrupts,
        "wm": wm_view,
    }


@app.post("/scrape")
def scrape(question: str, seeds: List[str] | None = None, seeds_csv: str | None = None, _: None = Depends(_auth_dep)) -> Any:
    if seeds is None:
        seeds = []
    if seeds_csv:
        seeds.extend([s.strip() for s in seeds_csv.split(",") if s.strip()])
    plan = ScrapePlan(
        question=question,
        seeds=seeds,
        max_pages=settings.SCRAPER_MAX_PAGES,
        allow_domains=settings.allowlist(),
    )
    s = Scraper()
    saved = s.execute(plan)
    return {"saved": saved}


@app.get("/metrics")
def metrics() -> Any:
    data = generate_latest(REG)
    return HTMLResponse(data, media_type=CONTENT_TYPE_LATEST)
