from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

import orjson
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles

from .schema import Stimulus
from .settings import settings
from .memory_working import WorkingMemory
from .memory_longterm import keyword_search, vector_search
from .providers import StubLLM, StubEmbeddings, SystemClock, LLMProvider, EmbeddingsProvider, Clock
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
EMB: EmbeddingsProvider = StubEmbeddings()
LLM: LLMProvider = StubLLM()
CLOCK: Clock = SystemClock()
INT = InterruptManager()
ATTN = AttentionScheduler()
SOC = SoCEngine(WM, EMB, LLM, CLOCK, INT, "Ghost v0.1: pragmatic assistant.")

STOP_EVT = asyncio.Event()


async def _get_stimuli() -> List[Stimulus]:
    return []


@app.on_event("startup")
async def _startup() -> None:
    if settings.SOC_ENABLED:
        # create stop event & background task
        app.state.soc_stop = asyncio.Event()
        app.state.soc_task = asyncio.create_task(run_soc_loop(SOC, _get_stimuli, app.state.soc_stop))

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
async def ingest(stimuli: List[Dict[str, Any]], _: None = Depends(_auth_dep)) -> Any:
    INGEST_COUNTER.inc(len(stimuli))
    # For demo: convert into episodic LTM immediately (low importance)
    from .memory_longterm import upsert_memory
    from .schema import Memory, now_ts

    for s in stimuli:
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
    # Trigger one SoC step
    SOC_CYCLE_COUNTER.inc()
    SOC.step([Stimulus(**s) for s in stimuli])
    return {"ok": True}


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


@app.get("/tasks/next")
def tasks_next(_: None = Depends(_auth_dep)) -> Any:
    nxt = ATTN.peek()
    return {"next": nxt[0] if nxt else None, "score": nxt[1] if nxt else None}

@app.get("/config")
def get_config(_: None = Depends(_auth_dep)) -> Any:
    return {
        "user_available": INT.user_available,
        "allowlist": settings.allowlist(),
        "soc_enabled": settings.SOC_ENABLED,
    }

@app.post("/config/user_available")
def set_user_available(available: bool, _: None = Depends(_auth_dep)) -> Any:
    INT.user_available = available
    return {"ok": True, "user_available": INT.user_available}

@app.get("/interrupts")
def list_interrupts(_: None = Depends(_auth_dep)) -> Any:
    return [q.model_dump() for q in INT.list_open()]


@app.post("/interrupts/answer")
def answer_interrupt(qid: str, text: str, _: None = Depends(_auth_dep)) -> Any:
    ok = INT.answer(qid, text)
    if not ok:
        raise HTTPException(status_code=404, detail="Unknown interrupt")
    return {"ok": True}


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
