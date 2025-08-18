from __future__ import annotations

import os
import time
from app.memory_longterm import upsert_memory, get_memory, keyword_search, vector_search, export_jsonl, import_jsonl
from app.schema import Memory
from app.providers.stubs import StubEmbeddings
from app.settings import settings


def test_ltm_crud_and_search(tmp_path):
    settings.DB_PATH = str(tmp_path / "ghost.sqlite3")
    emb = StubEmbeddings()

    m1 = Memory(
        id="m1",
        type="episodic",
        content="hello world about ophthalmology",
        embedding=emb.embed(["hello world"])[0],
        importance=0.5,
        created_at=time.time(),
        last_access=time.time(),
        metadata={}
    )
    upsert_memory(m1)

    got = get_memory("m1")
    assert got and got.content.startswith("hello")

    kw = keyword_search("ophthalmology", top_k=5)
    assert any("ophthalmology" in m.content for m in kw)

    qv = emb.embed(["hello"])[0]
    vs = vector_search(qv, top_k=5)
    assert vs and vs[0][0].id == "m1"

    outp = tmp_path / "export.jsonl"
    n = export_jsonl(str(outp))
    assert n >= 1

    settings.DB_PATH = str(tmp_path / "ghost2.sqlite3")
    n2 = import_jsonl(str(outp))
    assert n2 == n
