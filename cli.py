from __future__ import annotations

import json
import time
from typing import Optional

import typer
import requests

app = typer.Typer(help="Ghost CLI")

API = "http://localhost:8000"


@app.command()
def run() -> None:
    """Run API server (dev)."""
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)


@app.command()
def ingest(text: str, source: str = "cli") -> None:
    payload = [
        {
            "id": f"stim:{int(time.time()*1000)}",
            "source": source,
            "content": text,
            "metadata": {},
            "ts": time.time(),
        }
    ]
    r = requests.post(f"{API}/stimuli", json=payload, timeout=10)
    typer.echo(r.json())


@app.command()
def search(query: str, top_k: int = 5) -> None:
    r = requests.get(f"{API}/ltm/search", params={"q": query, "top_k": top_k}, timeout=10)
    typer.echo(json.dumps(r.json(), indent=2))


@app.command()
def interrupts() -> None:
    r = requests.get(f"{API}/interrupts", timeout=10)
    typer.echo(json.dumps(r.json(), indent=2))


@app.command()
def export(path: str = "data/export.jsonl") -> None:
    from app.memory_longterm import export_jsonl
    n = export_jsonl(path)
    typer.echo(f"Exported {n} records to {path}")


if __name__ == "__main__":
    app()
