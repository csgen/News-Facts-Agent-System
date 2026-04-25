# news_facts_system

Integrated codebase for the 3-member grad-school project on agentic AI for
real-time news fact-checking. Three modules live side-by-side as subfolders:

| Folder | Owner | What it does |
|---|---|---|
| `scraper_preprocessing_memory/` | Kelvin (Task 1) | Scrapes news, extracts claims + entities + image captions, ingests to Neo4j + ChromaDB. Also hosts the shared `MemoryAgent` facade used by the other two. |
| `FakeNewsAgent/` | Shantam (Task 2) | LangGraph fact-check pipeline: cache check → freshness → live search → RAG → verdict. Reads/writes the same DBs via `MemoryAgent`. |
| `PredictionAgent/` | Wasae (Task 3) | Streamlit UI + entity tracker + trend-based prediction agent. Invokes `FakeNewsAgent` on each user query and renders results. |

No logic is duplicated. The three parts communicate through the DB and
through thin adapter files in `PredictionAgent/agents/` (`fact_check_agent.py`,
`memory_agent.py`).

---

## Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux) with `docker compose` v2
- API keys for whichever cloud services you use (see `.env.example`)

Everything else (Python 3.12, deps, spaCy model) is installed inside the image.

---

## Setup (once)

```bash
cp .env.example .env
# open .env and fill the sections you'll use (see modes below)
```

Build the image once — subsequent runs reuse it:

```bash
docker compose -f docker/docker-compose.yml build
```

---

## Mode A — Fully cloud (the recommended default for Kelvin)

All databases and LLMs are cloud services. Zero local infra.

**In `.env`:**
```
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_PASSWORD=...
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=...
CHROMA_HOST=                        # leave blank
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
TAVILY_API_KEY=tvly-...
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=gemini
```

**First-time init (one-shot):**
```bash
docker compose -f docker/docker-compose.yml run --rm init-db
```

**Populate the DB with some articles (one-shot, can be re-run anytime):**
```bash
docker compose -f docker/docker-compose.yml run --rm scraper
```

**Start the UI (long-running):**
```bash
docker compose -f docker/docker-compose.yml up -d ui
# open http://localhost:8501
```

Stop / logs:
```bash
docker compose -f docker/docker-compose.yml logs -f ui
docker compose -f docker/docker-compose.yml down
```

---

## Mode B — Local DBs + cloud LLMs (teammates' existing setup)

Neo4j and ChromaDB run as local Docker containers, LLMs stay on the OpenAI/Gemini cloud APIs.

**In `.env`:**
```
NEO4J_URI=bolt://neo4j:7687
NEO4J_PASSWORD=fakenews123
CHROMA_API_KEY=                     # blank → HttpClient
CHROMA_HOST=chroma
CHROMA_PORT=8000
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
TAVILY_API_KEY=tvly-...
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=gemini
```

**Start the local DB containers + UI:**
```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.local.yml \
  --profile neo4j --profile chroma up -d

docker compose -f docker/docker-compose.yml run --rm init-db
docker compose -f docker/docker-compose.yml run --rm scraper
```

Open http://localhost:8501 (UI), http://localhost:7474 (Neo4j browser).

---

## Mode C — Fully local (DBs + local LLM via Ollama)

Offline-capable. Everything runs in containers.

**In `.env`:**
```
NEO4J_URI=bolt://neo4j:7687
NEO4J_PASSWORD=fakenews123
CHROMA_HOST=chroma
CHROMA_PORT=8000
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434/v1
OLLAMA_LLM_MODEL=gemma2:2b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

**Start everything:**
```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.local.yml \
  --profile neo4j --profile chroma --profile ollama up -d

# one-time model pull (runs inside the ollama container)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.local.yml \
  --profile ollama run --rm ollama-init
```

> ⚠️ **Embedding dimension change**: switching `EMBEDDING_PROVIDER` from `gemini`/`openai`
> (1536-dim) to `ollama nomic-embed-text` (768-dim) is **not** transparent. Before
> starting Mode C for the first time after running Mode A/B, wipe the local Chroma
> volume so collections can be re-created at the new dimension:
> ```bash
> docker compose -f docker/docker-compose.yml -f docker/docker-compose.local.yml down
> rm -rf docker/data/chroma
> ```

---

## Services reference

| Service | Role | Invocation |
|---|---|---|
| `ui` | Streamlit frontend, port 8501, long-running | `docker compose up -d ui` |
| `scraper` | One-shot scrape + preprocess + ingest | `docker compose run --rm scraper` |
| `init-db` | Create Neo4j constraints + seed sources (idempotent) | `docker compose run --rm init-db` |
| `bench` | Run the fact-check benchmark on LIAR/FEVER/Factify | `docker compose run --rm bench` |
| `shell` | Interactive bash inside the image | `docker compose run --rm shell` |
| `neo4j`, `chroma`, `ollama`, `langfuse` | Local infra (in `docker-compose.local.yml`, gated by `--profile`) | see modes above |

`ui` does **not** depend on `scraper` at the service level — they communicate only through the DB. The UI keeps working whether the scraper is running or not.

---

## Where code lives

```
news_facts_system/
├── scraper_preprocessing_memory/   # Kleivn's repo (near-verbatim)
│   └── src/
│       ├── pipeline.py             # `python -m src.pipeline` entry
│       ├── memory/agent.py         # Unified MemoryAgent — the single DB facade
│       ├── memory/vector_store.py  # Chroma (Cloud / HTTP / embedded) + source_credibility
│       └── memory/graph_store.py   # Neo4j + extended queries used by all 3 tasks
│
├── FakeNewsAgent/                  # Shantam's repo (near-verbatim)
│   └── fact_check_agent/src/
│       ├── _bootstrap.py           # Points updated at sibling ../../scraper_preprocessing_memory
│       ├── pipeline.py             # run_fact_check(PreprocessingOutput)
│       └── graph/                  # LangGraph nodes + routing
│
├── PredictionAgent/                # Wasae's repo (task3 branch) + 2 adapter files
│   ├── frontend/app.py             # Streamlit UI
│   ├── agents/
│   │   ├── fact_check_agent.py     # NEW adapter — wraps FakeNewsAgent's pipeline
│   │   ├── memory_agent.py         # NEW adapter — re-exports scapper's MemoryAgent
│   │   ├── entity_tracker.py       # Task 3 (unchanged)
│   │   └── prediction_agent.py     # Task 3 (unchanged)
│   └── evaluation/
│
├── docker/
│   ├── Dockerfile                  # Unified Python 3.12 image
│   ├── docker-compose.yml          # Cloud-default services (ui, scraper, …)
│   └── docker-compose.local.yml    # Overlay with neo4j/chroma/ollama/langfuse profiles
├── requirements.txt                # Merged deps installed in the image
├── .env.example                    # Copy to .env
└── README.md                       # This file
```

---

## How the three parts talk to each other

```
  [user query]                                                [periodic]
       │                                                           │
       ▼                                                           ▼
   ┌────────────────────────┐    fact_check_claim(str)   ┌──────────────────┐
   │ PredictionAgent UI     │ ──────────────────────────▶│ FakeNewsAgent    │
   │ (Streamlit, port 8501) │◀── FactCheckOutput ────────│ (LangGraph)      │
   └────────────┬───────────┘                            └────────┬─────────┘
                │                                                 │
                │ MemoryAgent (singleton, shared import)          │
                ▼                                                 ▼
        ┌──────────────────────────────────────────────────────────────┐
        │     ChromaDB  (claims, articles, verdicts, captions,         │
        │                source_credibility)                           │
        │                     +                                         │
        │     Neo4j     (Source/Article/Claim/Entity/Verdict/            │
        │                CredibilitySnapshot/Prediction)                │
        └──────────────────────────────────────────────────────────────┘
                ▲
                │ ingest_preprocessed()
                │
   ┌────────────────────────┐
   │ scapper pipeline       │   one-shot, periodic or on-demand
   │ (RSS / Tavily / …)     │
   └────────────────────────┘
```

---

## Public API for teammates: `decompose_input`

`scraper_preprocessing_memory` exposes a single function teammates can call to
push any user-supplied input (URL, article text, or short claim) into Neo4j +
ChromaDB and get back the resulting `claim_id`s:

```python
from src.preprocessing.decompose import decompose_input

claim_ids = decompose_input("Trump claims peace with Iran today.")
claim_ids = decompose_input("https://www.bbc.com/news/articles/c62lp853214o")
claim_ids = decompose_input(article_text_pasted_by_user)  # any length

# → list[str], e.g. ["clm_a3f8b2c1", ...]
```

**Behaviour:**
- **URL** → fetched as clean markdown via Jina Reader, then run through the same preprocessing pipeline as scraped articles.
- **Article text** (≥500 chars or ≥2 sentences) → `gpt-4o-mini` infers / generates a title + body split, then full preprocessing.
- **Short claim** → wrapped as a synthetic single-claim article so entity extraction still runs.
- **Idempotent** — re-calling with the same input returns the same `claim_id`s (matched by content hash).
- **Failure** — URLs that Jina cannot read raise `URLFetchError`; callers should catch and surface a clear message.

Optional env vars (in `.env`):
```
JINA_READER_BASE_URL=https://r.jina.ai/   # default
JINA_API_KEY=                              # leave blank for free tier; set for higher rate limits
```

---

## Troubleshooting

- **Streamlit can't connect to Neo4j** — verify `NEO4J_URI` in `.env` matches the mode.
  Cloud uses `neo4j+s://…`; local uses `bolt://neo4j:7687` (the container DNS name, not `localhost`).
- **ChromaDB "collection not found" after a provider switch** — you changed
  `EMBEDDING_PROVIDER`. Wipe `docker/data/chroma` and re-run `scraper`.
- **Streamlit "ModuleNotFoundError: agents"** — the UI must be started from `PredictionAgent/`
  (which the `ui` service's `working_dir` already handles). If running outside Docker, `cd PredictionAgent`
  then `streamlit run frontend/app.py`.
- **`scraper` writes nothing** — first run `init-db`, and confirm the `.env` API keys are non-empty.
- **Disabling Langfuse** — the two flags `LANGFUSE_ENABLED` and `LANGFUSE_TRACING_ENABLED` gate only `FakeNewsAgent`'s Langfuse code; `scraper_preprocessing_memory` ignores them. To fully silence Langfuse for both modules, leave `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` blank in `.env`. With empty keys the SDK silently no-ops on every `@observe` and every `langfuse.openai` call.
