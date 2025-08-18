# Ghost: a production-grade MVP “ghost in the machine” for LLMs

**Goals:** Persistent continuity across conversations with:
- Working Memory (WM)
- Long-Term Memory (LTM)
- Attention Scheduler (multi-conversation prioritization)
- Always-on Stream of Consciousness (SoC)
- Interrupts for user input
- Web scraper (robots-aware)
- Clear APIs, CLI, tests, dockerization, observability

**Status:** MIT-licensed, provider-agnostic. Deterministic defaults with stub LLM/embeddings. No secrets required.

---

## Architecture (ASCII)
```
+---------------------+ +------------------+ +------------------------+
| FastAPI / CLI | uses | | Attention Scheduler |
| - /stimuli -------->+-->+ Engine +--->+ (priority heap) |
| - /wm /ltm/search | | - cycle() | +------------------------+
| - /interrupts | | - graceful stop |
| - /metrics /ui | +------------------+
+---------------------+ |
|
+-------------+-------------+
| Working Memory (WM) |
| ring buffer + decay |
+-------------+-------------+
|
v
+---------------------------+
| Stream of Consciousness |
| - cadence loop |
| - store decider |
| - emits interrupts |
| - triggers scraper |
+-------------+-------------+
|
v
+---------------------------+
| Long-Term Memory |
| SQLite + Vec index |
| encryption optional |
+-------------+-------------+
^
|
+-------------+-------------+
| Scraper |
| robots-aware |
| allowlisted only |
+-----------------------------+
```
**Observability:** JSON logs, `/metrics` (Prometheus), minimal dashboard at `/ui/`.

---

## Quick Start

### 1) Requirements
- Docker + docker-compose **or** Python 3.11+
- No API keys needed for the stub LLM/embeddings.

### 2) One-liner (Docker)
```bash
    make up
```
This builds and launches the API (FastAPI + uvicorn) and the SoC background loop. Dashboard at http://localhost:8000/ui/
### 3) Without Docker (local dev)
```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e .
    make test
    uvicorn app.api:app --reload
```
### 4) Smoke test (CLI)
```bash
    ghost ingest "hello world"
    curl -s localhost:8000/wm | jq
    curl -s localhost:8000/tasks/next | jq
```
### Acceptance checks

make up runs API/UI/SoC.

ghost ingest "hello" creates a stimulus → SoC produces a micro-thought; /wm and /tasks/next respond.

Toggle USER_UNAVAILABLE=true then trigger a #question: SoC routes an interrupt to scraper (if SCRAPER_ALLOWLIST contains the domain), stores doc memories, resumes.

### Endpoints

POST /stimuli – ingest stimuli

GET /wm – view working memory

GET /ltm/search?q=...&k=5 – hybrid search (keyword+vector)

GET /tasks/next – attention-based next best task

GET /metrics – Prometheus metrics

POST /interrupts/answer – answer an open interrupt

GET /ui/ – dashboard

CLI
```bash
    ghost run                  # start SoC loop (when running purely via CLI)
    ghost ingest "text"        # ingest a stimulus
    ghost search "query"       # search LTM
    ghost interrupts           # list pending interrupts
    ghost export               # export LTM as JSONL
```

### Configuration (via .env or env vars)

| Key                       |         Default | Description                                      |
| ------------------------- | --------------: | ------------------------------------------------ |
| `DB_PATH`                 | `data/ghost.db` | SQLite path                                      |
| `WM_CHAR_BUDGET`          |          `8000` | WM capacity (chars)                              |
| `WM_HALF_LIFE_SEC`        |          `1800` | Exponential decay half-life                      |
| `SOC_ENABLED`             |          `true` | Run SoC loop at startup                          |
| `SOC_CADENCE_SEC`         |            `10` | Base interval between cycles                     |
| `SOC_JITTER_SEC`          |             `2` | Random jitter added/subtracted                   |
| `SOC_TOPK_LTM`            |             `5` | LTM retrieval per cycle                          |
| `EMBEDDINGS_PROVIDER`     |          `stub` | Provider name                                    |
| `LLM_PROVIDER`            |          `stub` | Provider name                                    |
| `ENCRYPT_AT_REST`         |         `false` | Toggle PyNaCl encryption for LTM                 |
| `ENCRYPTION_KEY_HEX`      |            \`\` | 32-byte hex secret for SecretBox                 |
| `PII_REDACT`              |         `false` | Redact emails/phones on ingestion                |
| `SCRAPER_ALLOWLIST`       |            \`\` | Comma-separated host allowlist                   |
| `SCRAPER_MAX_PAGES`       |             `5` | Max pages per plan                               |
| `SCRAPER_MAX_RUNTIME_SEC` |            `30` | Safety cap                                       |
| `USER_UNAVAILABLE`        |         `false` | Route interrupts to scraper if no user           |
| `ATTN_WEIGHTS`            |     `1,1,1,1,1` | urgency,importance,uncertainty,recency,goal\_fit |
| `OPEN_TELEMETRY`          |         `false` | Enable OTLP hooks (no exporter by default)       |

### Providers

app/providers/stubs.py contains deterministic stubs:

StubLLM: returns a ≤60-word micro-thought tagged #plan, #insight, or #question based on input hash.

StubEmbeddings: fixed-dimension numeric hashing → stable cosine behavior.

You can add adapters for OpenAI, local models, etc. Implement:

```python
LLMProvider.generate(system: str, prompt: str, max_tokens: int, **kw) -> str
EmbeddingsProvider.embed(texts: List[str]) -> List[List[float]]
```

### Security & Privacy

Robots-aware scraping (urllib.robotparser).

Per-domain rate limits, retries, allowlist, runtime/page caps.

Optional PII redaction on ingestion (emails/phones masked).

Optional encryption at rest (PyNaCl SecretBox). Toggle via env.

### Observability

Structured JSON logs.

/metrics via prometheus_client.

Minimal dashboard: WM contents, current thought, pending interrupts, tasks.

### Benchmarks (Macbook Air M2, Python 3.11)

SoC cycle (stub providers, k=5): ~8–15 ms end-to-end

LTM hybrid search (10k rows, pure-Python cosine): ~5–20 ms

Attention peek: ~1–2 ms

### Tests & Quality

pytest -q with coverage ≥80%.

ruff, mypy, pinned versions for reproducibility.

No network calls in tests (scraper mocked).

### Notes on FAISS/Chroma

Spec allowed FAISS/Chroma. For deterministic, portable CI with zero native deps, we implement a pure-Python cosine index that is provider-pluggable. Drop-in FAISS/Chroma adapters can be added if desired.