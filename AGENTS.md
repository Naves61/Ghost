# Ghost — Agents Overview (Architecture & Extension Guide)

This document is a practical reference for code assistants (e.g., Codex) and developers.
It enumerates Ghost’s agents, their responsibilities, entry points, data flows, and how to extend them.

Repo layout (partial)
- app/
  - api.py                     # FastAPI routes (HTTP API + UI serving)
  - engine.py                  # Orchestrator: one-cycle coordination
  - soc.py                     # Stream-of-Consciousness engine (async loop)
  - attention.py               # Attention scheduler / task priority heap
  - interrupts.py              # Interrupt manager (question/answer flow)
  - memory_working.py          # Working Memory ring buffer
  - memory_longterm.py         # Long-Term Memory (SQLite, search, export)
  - scraper/
    - plan.py                  # ScrapePlan schema
    - runner.py                # Scraper executor (requests/BS4)
    - robots.py                # RobotsGate (allowlist + robots.txt)
  - providers/
    - llm_base.py              # LLMProvider interface
    - embeddings_base.py       # EmbeddingsProvider interface
    - clock.py                 # Clock interface
    - stubs.py                 # Stub providers (offline deterministic)
  - schema.py                  # Pydantic models (Stimulus, Thought, Memory, Task, SelfModel)
  - settings.py                # Configuration (Pydantic BaseSettings)
  - ui/index.html              # Minimal dashboard (WM, interrupts, config, scrape)
- cli.py                       # CLI entrypoints
- tests/                       # Unit/integration tests

#######################################################################
# 1) AGENTS (runtime modules)
#######################################################################

[Working Memory Agent]
- File: app/memory_working.py
- Purpose: bounded ring buffer for recent thoughts; token/char budget + time decay
- Key API:
  - WM.add(thought: Thought) -> None
  - WM.view(n: int) -> List[Thought]
  - WM.score_decay(now: float) -> float
  - WM.truncate_to_budget() -> None
- Notes: O(1) amortized append; decay uses half-life from settings.

[Long-Term Memory Agent]
- File: app/memory_longterm.py
- Purpose: durable knowledge store with keyword + vector search; export/import
- Key API:
  - upsert_memory(mem: Memory) -> None
  - get_memory(id: str) -> Optional[Memory]
  - keyword_search(query: str, top_k: int) -> List[Memory]
  - vector_search(query_vec: List[float], top_k: int) -> List[Tuple[Memory, float]]
  - export_jsonl(path: str) -> int
  - import_jsonl(path: str) -> int
- Storage: SQLite; embeddings stored as float32 blobs; optional encryption toggle (libsodium).
- Consolidation (job): cluster episodics → semantic summaries; dedup near-duplicates; age importance.

[Attention Scheduler Agent]
- File: app/attention.py
- Purpose: select the next best task to attend to
- Key API:
  - add_or_update(task: Task, goal_fit: float) -> None
  - next_task() -> Optional[Task]
  - peek() -> Optional[Task]
- Scoring:
  score = w_u*urgency + w_i*importance + w_unc*uncertainty + w_r*recency + w_g*goal_fit
- Data: lazy invalidation heap; recency based on last_update_ts; urgency from deadline proximity.

[Stream-of-Consciousness Agent]
- File: app/soc.py
- Purpose: always-on background loop; produces concise micro-thoughts (<= 60 words)
- Cycle steps:
  1) retrieve top-k LTM for context
  2) generate micro-thought via LLMProvider (stub by default)
  3) push to WM
  4) decide storage via StoreDecider (LLM JSON)
  5) emit Interrupt when the thought is a #question requiring info
- Config: cadence with jitter, rate limits, graceful shutdown (startup/shutdown hooks in api.py)

[Interrupts Agent]
- File: app/interrupts.py
- Purpose: pause SoC on missing info; query user; resume when answered
- Key API:
  - create(question: str, rationale: str, required_fields: List[str]) -> str  # returns id
  - list_open() -> List[Query]
  - answer(id: str, text: str) -> bool
- Policy: if user unavailable and question has allowlisted URLs, escalate to Scraper.

[Scraper Agent]
- Files: app/scraper/{plan.py, runner.py, robots.py}
- Purpose: robots.txt-aware web retrieval; rate limits; per-domain allowlist
- Inputs: ScrapePlan(question, seeds, max_pages, allow_domains)
- Behavior:
  - Deduplicate normalized URLs; only text/html|text/plain
  - Extract clean text + metadata; create document memories (semantic, low importance)
  - Optional cross-domain crawling via SCRAPER_SAME_DOMAIN_ONLY=false
- Gate: RobotsGate.allowed(url) checks allowlist and robots.txt

#######################################################################
# 2) PROVIDER INTERFACES
#######################################################################

[LLMProvider]
- File: app/providers/llm_base.py
- Method: generate(system: str, prompt: str, max_tokens: int, **kwargs) -> str
- Default: StubLLM in stubs.py (deterministic, offline)
- Swap-in adapters: implement class and wire via env PROVIDER_LLM

[EmbeddingsProvider]
- File: app/providers/embeddings_base.py
- Method: embed(texts: List[str]) -> List[List[float]]
- Default: StubEmbeddings (SHA256-based, normalized, dim from settings.VECTOR_DIM)
- Swap-in adapters: implement and set PROVIDER_EMBED

[Clock]
- File: app/providers/clock.py
- Method: now() -> float
- Default: SystemClock()

#######################################################################
# 3) DATA MODELS (Pydantic)
#######################################################################

- Stimulus {id, source, content, metadata, ts}
- Thought {id, content, tags[], importance∈[0,1], uncertainty∈[0,1], ts}
- Memory {id, type∈{episodic, semantic, procedural}, content, embedding[], importance, created_at, last_access, metadata}
- Task {id, title, goal, context, deadline_ts?, importance, uncertainty, status∈{open,doing,done,blocked}, created_ts, last_update_ts, metadata}
- SelfModel {id, name, principles[], long_term_goals[], preferences, last_review_ts}

#######################################################################
# 4) ORCHESTRATOR & API
#######################################################################

[Engine]
- File: app/engine.py
- Function: cycle(stimuli: List[Stimulus]) -> {thought, stored?, next_task, interrupts[]}
- Role: integrates WM/LTM, attention, SoC outputs; persists updates atomically

[FastAPI]
- File: app/api.py
- Endpoints:
  - POST /stimuli                # ingest external stimuli
  - GET  /wm                     # list working memory
  - GET  /ltm/search             # keyword + vector search
  - GET  /tasks/next             # recommended next task
  - GET  /interrupts             # list open interrupts
  - POST /interrupts/answer      # answer and resume
  - GET  /metrics                # Prometheus metrics (basic)
  - GET  /config                 # user availability, allowlist, flags
  - POST /config/user_available  # toggle availability (bool)
  - POST /scrape                 # run scraper (question + seeds or seeds_csv)
  - GET  /ui                     # static dashboard
- Startup/shutdown: creates/cancels the SoC background task using app.state.soc_task

[CLI]
- File: cli.py
- Commands:
  - ghost run                    # run engine/loop (if implemented)
  - ghost ingest "text"          # send a stimulus via API
  - ghost search "query"         # LTM search
  - ghost interrupts             # list/answer interrupts
  - ghost export                 # export memories to JSONL

#######################################################################
# 5) CONFIGURATION (env/.env → app/settings.py)
#######################################################################

- SOC_ENABLED=true|false
- SOC_CADENCE_SECONDS=5
- VECTOR_DIM=128
- ALLOWLIST_DOMAINS=example.com,localhost   # "*" for unrestricted
- SCRAPER_SAME_DOMAIN_ONLY=true|false       # follow links only within domain
- SCRAPER_MAX_PAGES=8
- SCRAPER_MAX_RUNTIME_SECONDS=30
- SCRAPER_HTTP_TIMEOUT=15
- SCRAPER_DOC_MAX_CHARS=4000
- PROVIDER_LLM=stub
- PROVIDER_EMBED=stub
- PROVIDER_CLOCK=system
- DB_PATH=data/ghost.sqlite3
- LOG_DIR=logs

Helper:
- settings.allowlist() returns ["*"] when ALLOWLIST_DOMAINS="*", else parsed domains.

#######################################################################
# 6) FLOWS (pseudocode)
#######################################################################

[SoC loop]
repeat every SOC_CADENCE_SECONDS ± jitter:
  ctx = {
    self_summary, WM.tail(k), LTM.topk(embed(prompt)), new stimuli batch
  }
  gen = LLMProvider.generate(system_prompt, prompt_from(ctx))
  thought = parse_microthought(gen)  # "#plan|#question|#insight" + sentence
  WM.add(thought); WM.truncate_to_budget()
  dec = LLMProvider.generate("STORE_DECIDER", store_prompt(thought, ctx)) -> JSON
  if dec.should_store:
      LTM.upsert({... importance=dec.importance, type=dec.type, tags=dec.tags })
  if "#question" and missing_info(ctx):
      interrupts.create(question, rationale, required_fields)
      if not user_available and question contains allowlisted URLs:
          Scraper.execute(plan(question, seeds_from_urls, max_pages, allowlist))

[Scraper]
for url in plan.seeds:
  if RobotsGate.allowed(url): queue.push(url)
while queue and within_limits:
  url = queue.pop()
  if RobotsGate.allowed(url) and not visited[url]:
    html = GET(url, timeout)
    text, links = extract_text_and_links(html)
    if text: LTM.upsert(doc_memory(url, text))
    for link in links:
      n = normalize(link)
      if RobotsGate.allowed(n) and (not cross_domain unless allowed):
        queue.push(n)

#######################################################################
# 7) EXTENSION POINTS
#######################################################################

- LLM/Embeddings:
  - Create app/providers/my_llm.py with class MyLLM(LLMProvider) and implement generate().
  - Set PROVIDER_LLM=my_llm in .env (wire import in providers/__init__.py if needed).
- Vector DB:
  - Replace FAISS/Chroma by adapting memory_longterm.py (keep embed() contract).
- Encryption:
  - Enable libsodium for at-rest encryption; toggle via env and wrap blob I/O.
- UI:
  - app/ui/index.html is plain JS/HTML; add routes in api.py for richer interactions.
- Policies:
  - Edit interrupts/escalation policy (when to scrape, who to ask).

#######################################################################
# 8) TEST TARGETS (high level)
#######################################################################

- tests/test_working_memory.py          # ring buffer budget/decay
- tests/test_long_term_memory.py        # CRUD/search/export
- tests/test_attention.py               # scoring + heap behavior
- tests/test_socCycle.py                # step generation/store/interrupt
- tests/test_interrupts.py              # create/answer lifecycle
- tests/test_scraper.py                 # robots allowlist; link discovery

#######################################################################
# 9) QUICK API EXAMPLES (curl)
#######################################################################

# ingest stimulus
curl -s -X POST 'http://localhost:8000/stimuli' \
  -H 'Content-Type: application/json' \
  --data '{"source":"cli","content":"Plan: review clinic notes", "metadata":{}}'

# read WM
curl -s http://localhost:8000/wm | jq .

# LTM search
curl -s 'http://localhost:8000/ltm/search?q=ophthalmology' | jq .

# interrupts
curl -s http://localhost:8000/interrupts | jq .
curl -s -X POST "http://localhost:8000/interrupts/answer?qid=q:123&text=Here+you+go"

# scraper (manual)
curl -s -X POST "http://localhost:8000/scrape?question=collect+notes&seeds_csv=https://www.who.int/health-topics/influenza" | jq .

#######################################################################
# 10) ASCII ARCHITECTURE
#######################################################################

User ──▶ FastAPI (/stimuli, /interrupts/answer, /ui)
             │
             ▼
        +-----------+       +-----------------+
        |  Engine   |──────▶|  Attention      |
        +-----------+       +-----------------+
             │                      ▲
             ▼                      │
        +-----------+        +-------------+
        |   SoC     |──────▶ | Interrupts  |
        +-----------+        +-------------+
             │                      │
     ┌───────┴───────┐              │ auto-escalate when unavailable
     ▼               ▼              │
+-----------+   +-----------+       │
|   WM      |   |   LTM     |◀─────┘
+-----------+   +-----------+
                     ▲
                     │
                +----------+
                | Scraper  |
                +----------+  (RobotsGate → allowlist + robots.txt)

#######################################################################
# 11) CONVENTIONS
#######################################################################

- Thought content starts with tag: "#plan | #question | #insight".
- IDs: "thought:<ts_ms>", "mem:<ts_ms>", "doc:<ts_ms>", "q:<ts_ms>".
- JSON logging; avoid printing secrets.
- No network calls unless scraper is invoked and allowed.