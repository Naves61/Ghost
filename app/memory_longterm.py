from __future__ import annotations

import sqlite3
import json
import time
import hashlib
from typing import Iterable, List, Optional, Tuple

import numpy as np
from nacl import secret, utils

from .schema import Memory, MemoryType
from .settings import settings


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(settings.DB_PATH)
    con.execute(
        """CREATE TABLE IF NOT EXISTS memories (
           id TEXT PRIMARY KEY,
           type TEXT NOT NULL,
           content BLOB NOT NULL,
           embedding BLOB,
           importance REAL,
           created_at REAL,
           last_access REAL,
           metadata BLOB
        )"""
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(type)"
    )
    return con


def _sb() -> Optional[secret.SecretBox]:
    if not settings.ENCRYPT_LTM:
        return None
    key = settings.SECRET_KEY.encode("utf-8")
    if len(key) < 32:
        # Derive a 32-byte key deterministically from provided secret
        key = hashlib.sha256(key).digest()
    else:
        key = key[:32]
    return secret.SecretBox(key)


def _enc(raw: bytes) -> bytes:
    sb = _sb()
    if not sb:
        return raw
    nonce = utils.random(secret.SecretBox.NONCE_SIZE)
    return nonce + sb.encrypt(raw, nonce).ciphertext


def _dec(enc: bytes) -> bytes:
    sb = _sb()
    if not sb:
        return enc
    nonce = enc[: secret.SecretBox.NONCE_SIZE]
    ct = enc[secret.SecretBox.NONCE_SIZE :]
    return sb.decrypt(nonce + ct)


def upsert_memory(mem: Memory) -> None:
    con = _db()
    try:
        emb_bytes = None
        if mem.embedding is not None:
            emb = np.asarray(mem.embedding, dtype=np.float32)
            emb_bytes = emb.tobytes()
        meta_bytes = json.dumps(mem.metadata, separators=(",", ":")).encode("utf-8")
        con.execute(
            """INSERT INTO memories(id,type,content,embedding,importance,created_at,last_access,metadata)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 type=excluded.type,
                 content=excluded.content,
                 embedding=excluded.embedding,
                 importance=excluded.importance,
                 last_access=excluded.last_access,
                 metadata=excluded.metadata
            """,
            (
                mem.id,
                mem.type,
                _enc(mem.content.encode("utf-8")),
                emb_bytes,
                mem.importance,
                mem.created_at,
                mem.last_access,
                _enc(meta_bytes),
            ),
        )
        con.commit()
    finally:
        con.close()


def get_memory(mem_id: str) -> Optional[Memory]:
    con = _db()
    try:
        cur = con.execute(
            "SELECT id,type,content,embedding,importance,created_at,last_access,metadata FROM memories WHERE id=?",
            (mem_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        emb = None
        if row[3] is not None:
            emb = list(np.frombuffer(row[3], dtype=np.float32))
        return Memory(
            id=row[0],
            type=row[1],  # type: ignore
            content=_dec(row[2]).decode("utf-8"),
            embedding=emb,
            importance=row[4],
            created_at=row[5],
            last_access=row[6],
            metadata=json.loads(_dec(row[7]).decode("utf-8")),
        )
    finally:
        con.close()


def keyword_search(query: str, top_k: int = settings.TOPK_DEFAULT) -> List[Memory]:
    con = _db()
    try:
        cur = con.execute(
            "SELECT id,type,content,embedding,importance,created_at,last_access,metadata FROM memories"
        )
        rows = cur.fetchall()
        scored: List[Tuple[float, Memory]] = []
        q = query.lower()
        for row in rows:
            content = _dec(row[2]).decode("utf-8")
            score = content.lower().count(q)
            if score > 0:
                emb = None
                if row[3] is not None:
                    emb = list(np.frombuffer(row[3], dtype=np.float32))
                scored.append(
                    (
                        float(score),
                        Memory(
                            id=row[0],
                            type=row[1],  # type: ignore
                            content=content,
                            embedding=emb,
                            importance=row[4],
                            created_at=row[5],
                            last_access=row[6],
                            metadata=json.loads(_dec(row[7]).decode("utf-8")),
                        ),
                    )
                )
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]
    finally:
        con.close()


def vector_search(
    query_vec: List[float], top_k: int = settings.TOPK_DEFAULT
) -> List[Tuple[Memory, float]]:
    con = _db()
    try:
        cur = con.execute("SELECT id,type,content,embedding,importance,created_at,last_access,metadata FROM memories WHERE embedding IS NOT NULL")
        rows = cur.fetchall()
        q = np.asarray(query_vec, dtype=np.float32)
        qn = np.linalg.norm(q) + 1e-8
        results: List[Tuple[Memory, float]] = []
        for row in rows:
            emb = np.frombuffer(row[3], dtype=np.float32)
            sim = float(np.dot(q, emb) / (qn * (np.linalg.norm(emb) + 1e-8)))
            mem = Memory(
                id=row[0],
                type=row[1],  # type: ignore
                content=_dec(row[2]).decode("utf-8"),
                embedding=list(emb),
                importance=row[4],
                created_at=row[5],
                last_access=row[6],
                metadata=json.loads(_dec(row[7]).decode("utf-8")),
            )
            results.append((mem, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    finally:
        con.close()


def export_jsonl(path: str) -> int:
    con = _db()
    try:
        cur = con.execute("SELECT id,type,content,embedding,importance,created_at,last_access,metadata FROM memories")
        rows = cur.fetchall()
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                # embedding -> list[float] (native Python floats)
                emb = None
                if row[3] is not None:
                    emb_np = np.frombuffer(row[3], dtype=np.float32)
                    emb = [float(x) for x in emb_np.tolist()]

                obj = {
                    "id": row[0],
                    "type": row[1],
                    "content": _dec(row[2]).decode("utf-8"),
                    "embedding": emb,
                    "importance": float(row[4]) if row[4] is not None else 0.0,
                    "created_at": float(row[5]) if row[5] is not None else 0.0,
                    "last_access": float(row[6]) if row[6] is not None else 0.0,
                    "metadata": json.loads(_dec(row[7]).decode("utf-8")),
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n += 1
        return n
    finally:
        con.close()


def import_jsonl(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            mem = Memory(**obj)
            upsert_memory(mem)
            n += 1
    return n


def consolidate_recent(
    max_items: int = 50, dedup_threshold: float = 0.95
) -> List[str]:
    """
    Consolidation job:
    - pick recent episodic memories
    - deduplicate near-duplicates by cosine similarity
    - summarize clusters to semantic memories (stub summary)
    Returns list of created semantic memory IDs.
    """
    con = _db()
    try:
        cur = con.execute(
            "SELECT id,type,content,embedding,importance,created_at,last_access,metadata FROM memories WHERE type='episodic' ORDER BY created_at DESC LIMIT ?",
            (max_items,),
        )
        rows = cur.fetchall()
    finally:
        con.close()

    # Simple greedy clustering by cosine similarity
    clusters: list[list[Tuple[str, np.ndarray, str, float, float, dict]]] = []
    for row in rows:
        _id = row[0]
        content = _dec(row[2]).decode("utf-8")
        emb = np.frombuffer(row[3], dtype=np.float32) if row[3] else None
        if emb is None:
            # no embedding: put alone
            clusters.append([(_id, np.zeros(settings.VECTOR_DIM, dtype=np.float32), content, row[4], row[5], json.loads(_dec(row[7]).decode("utf-8")))])
            continue
        placed = False
        for cl in clusters:
            cvecs = [v for _, v, _, _, _, _ in cl]
            ccent = np.mean(cvecs, axis=0) if cvecs else emb
            sim = float(np.dot(emb, ccent) / ((np.linalg.norm(emb) + 1e-8) * (np.linalg.norm(ccent) + 1e-8)))
            if sim >= dedup_threshold:
                cl.append((_id, emb, content, row[4], row[5], json.loads(_dec(row[7]).decode("utf-8"))))
                placed = True
                break
        if not placed:
            clusters.append([(_id, emb, content, row[4], row[5], json.loads(_dec(row[7]).decode("utf-8")))])

    created_ids: List[str] = []
    now = time.time()
    for idx, cl in enumerate(clusters):
        if len(cl) < 2:
            continue
        merged_text = "\n".join([c for _, _, c, _, _, _ in cl])
        summary = f"Summary of {len(cl)} events:\n" + merged_text[:512]
        mem = Memory(
            id=f"semantic:{int(now)}:{idx}",
            type="semantic",
            content=summary,
            embedding=None,
            importance=min(1.0, 0.6 + 0.02 * len(cl)),
            created_at=now,
            last_access=now,
            metadata={"source": "consolidation", "members": [m for m, *_ in cl]},
        )
        upsert_memory(mem)
        created_ids.append(mem.id)
    return created_ids
