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
| `prometheus`, `grafana`, `cadvisor`, `metrics-collector` | Monitoring stack (in `docker-compose.monitoring.yml`, gated by `--profile monitoring`) | see "Monitoring" section below |

`ui` does **not** depend on `scraper` at the service level — they communicate only through the DB. The UI keeps working whether the scraper is running or not.

---

## Where code lives

```
news_facts_system/
├── scraper_preprocessing_memory/   # Kelvin's repo (near-verbatim)
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
│   │   ├── memory_agent.py         # NEW adapter — re-exports scraper's MemoryAgent
│   │   ├── entity_tracker.py       # Task 3 (unchanged)
│   │   └── prediction_agent.py     # Task 3 (unchanged)
│   └── evaluation/
│
├── docker/
│   ├── Dockerfile                       # Unified Python 3.12 image
│   ├── docker-compose.yml               # Cloud-default services (ui, scraper, …)
│   ├── docker-compose.local.yml         # Overlay with neo4j/chroma/ollama/langfuse profiles
│   └── docker-compose.monitoring.yml    # Overlay with prometheus/grafana/cadvisor/metrics-collector
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
   │ scraper pipeline       │   one-shot, periodic or on-demand
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

## Monitoring (Prometheus + Grafana)

The repo ships with an opt-in monitoring stack so you can see what the system is doing in real time. It is **off by default** — start it with the `monitoring` profile only when you want it.

### What you get

| Layer | Tool | What it shows |
|---|---|---|
| Application | Prometheus + Grafana | Verdict rate by label, fact-check latency p50/p95/p99, LLM API errors, queries/hour |
| Database | metrics-collector sidecar | Neo4j node counts (Article, Claim, Verdict, …) and Chroma collection sizes |
| Scheduled scrapes | Implicit via `nfs_db_node_count` | Successful cron scrapes show up as growth in `count(:Article)`, `count(:Claim)`, etc. (See "How the scraper appears on the dashboard" below for the future explicit-panel path.) |
| Infrastructure | cAdvisor | Per-container CPU/RAM/network |

LLM-level traces (per-prompt token usage, latency, full prompt/response) live separately in **Langfuse** if you've configured it — see the Langfuse env block in `.env.example`.

### Starting the stack

The monitoring stack lives in its own overlay file (`docker-compose.monitoring.yml`) so cloud-default users don't need to pull in `docker-compose.local.yml` (which is for local DB / LLM alternatives). The two overlays are independent and combinable.

| Mode | Command |
|---|---|
| Cloud DBs only | `docker compose -f docker/docker-compose.yml up -d ui` |
| Cloud DBs + monitoring | `docker compose -f docker/docker-compose.yml -f docker/docker-compose.monitoring.yml --profile monitoring up -d` |
| Local DBs (Mode B) | `docker compose -f docker/docker-compose.yml -f docker/docker-compose.local.yml --profile neo4j --profile chroma up -d` |
| Local DBs + monitoring | `docker compose -f docker/docker-compose.yml -f docker/docker-compose.local.yml -f docker/docker-compose.monitoring.yml --profile neo4j --profile chroma --profile monitoring up -d` |

⚠️ When you also pass `--profile <X>`, run `up -d` with **no service name**. If you write `up -d ui`, Compose starts only the UI — naming a service narrows the start to that service plus its `depends_on:` chain, so anything brought in by `--profile X` is silently skipped. With no service name, Compose starts everything eligible: services without a profile *and* services whose profile is currently active.

Stop the stack symmetrically — repeat the same `-f` files with `down` in place of `up -d`.

### Endpoints

| Service | URL | Purpose |
|---|---|---|
| Grafana | http://localhost:3001 (admin/admin → change on first login) | Dashboards. The `news_facts_system → overview` dashboard auto-loads. |
| Prometheus | http://localhost:9090 | Direct PromQL exploration. `Status → Targets` shows scrape health. |
| cAdvisor | http://localhost:8080 | Per-container metrics UI (also serves `/metrics` for Prometheus) |
| UI metrics | http://localhost:8000/metrics | Live counter/histogram values exposed by the Streamlit process |
| Collector metrics | (only on Docker network) `metrics-collector:8001/metrics` | DB node-count gauges, polled every 5 min |

### How the scraper appears on the dashboard

The `scraper` and `init-db` services are one-shot containers — Prometheus can't scrape them because they exit before the next 15 s tick. Instead, `pipeline.py` writes a `(:ScrapeRun {run_id, started_at, finished_at, scraped, ingested, skipped, failed, source})` node to Neo4j at the end of every run.

**Indirect view (what's wired today):** the metrics-collector polls Neo4j every 5 min and exports `nfs_db_node_count` for every node label, including `Article` / `Claim` / `Verdict`. So when a scheduled scrape ingests new articles, the "Database growth" panel ticks up. That's enough signal for the dashboard to reflect cron activity without any Grafana plugin.

**Explicit per-run view (future, optional):** for a panel showing one row per `ScrapeRun` (with `source: "local" | "github_actions"`, durations, failure counts), the standard pattern is to add a tiny REST endpoint to `scripts/metrics_collector.py` that returns the most recent `ScrapeRun` records as JSON, install the `yesoreyeram-infinity-datasource` plugin in Grafana, and add a Table panel pointing at the new endpoint. Skipped today to keep the stack lightweight and avoid Grafana plugin-registry churn.

The same code path runs in the GitHub Actions cron, with `source="github_actions"` set automatically (Actions runners populate `GITHUB_ACTIONS=true`). Local dev runs land with `source="local"`. The `(:ScrapeRun)` data is in Neo4j Aura whether or not the dashboard surfaces it — you can always inspect via the Aura browser:
```cypher
MATCH (s:ScrapeRun) RETURN s ORDER BY s.started_at DESC LIMIT 20
```

### Notes / quirks

- **First-time `init-db`** — re-run it once after switching to monitoring; the new schema adds a uniqueness constraint and a time index on `:ScrapeRun`. Idempotent — it's only `CREATE … IF NOT EXISTS`.
- **Aura quota** — the metrics collector polls every 5 min; ~6 short Cypher counts per round, ~1 700/day. Bump `METRICS_COLLECT_INTERVAL_S` in the env if you ever throttle.
- **Streamlit reruns** — the UI starts a Prometheus HTTP server on port 8000 once; subsequent script reruns hit `OSError` on the bound port and silently no-op (intentional).
- **Volumes** — all data lives under `docker/data/{prometheus,grafana}` (gitignored). `docker compose down` keeps it; `docker compose down -v` would NOT delete bind-mount data either, but `rm -rf docker/data/prometheus` would.
- **No GH Actions plugin needed** — we read run outcomes via `(:ScrapeRun)` nodes (richer than what the GitHub API exposes anyway, and no PAT to manage).

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

---

## CI/CD (GitHub Actions)

Workflows live under `.github/workflows/`:

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | every PR + push to main | Three parallel jobs: lint, unit tests, mocked-integration tests + coverage |
| `release-image.yml` | push to main + tag `v*` | Build & publish Docker image, then smoke-test it |
| `release.yml` | tag `v*` only | Generate changelog, create GitHub Release |
| `scrape.yml` | twice daily cron + manual | Run scraper pipeline against cloud DBs |
| `health-check.yml` | Mon + Thu cron + manual | Probe live OpenAI / Gemini / Tavily / Neo4j / Chroma APIs |

The test pyramid running across these workflows:

| Tier | Where | What it tests | Cost |
|---|---|---|---|
| **Unit + lint** | `ci.yml`, every PR | Pure-logic functions, mocked deps | Free, ~3 min |
| **Mocked-integration** | `ci.yml`, every PR | Real Neo4j + Chroma sidecars; OpenAI/Gemini/Tavily mocked at HTTP layer (respx) | Free, ~3-5 min |
| **Smoke** | `release-image.yml` `smoke` job, post-merge | Just-built image actually boots: imports, schema, round-trip, Streamlit health endpoint | Free, ~3 min |
| **Health check** | `health-check.yml`, twice weekly | Tiny real-API pings to detect vendor drift / quota / auth issues | ~$0.05/run, $0.50/month |

(The unit + mocked-integration tiers run in the same CI job; a single `pytest -m "not integration"` invocation includes both because `mocked_integration` isn't excluded by that filter.)

### `ci.yml` — Three parallel jobs: lint, unit tests, mocked-integration

Runs on every push and PR to `main`, plus on manual trigger. Three independent jobs run in parallel on separate runners:

| Job | What it does | Runtime |
|---|---|---|
| `lint` | `ruff check .` (ruleset in `pyproject.toml`) | ~30 s |
| `test` | `pytest -m "not integration and not mocked_integration"` — pure unit tests with mocked deps; uploads `coverage.xml` | ~3 min |
| `mocked-integration` | `pytest -m "mocked_integration"` — testcontainers Neo4j + Chroma + respx-mocked LLM HTTP; uploads `coverage-mocked.xml` | ~5 min |

Total wall-clock = max of the three (~5 min) instead of sum (~9 min). Fast lint/unit feedback shows in the Actions tab within ~1 min, regardless of how long the mocked-integration job takes.

The `@pytest.mark.mocked_integration` tests use:
- `respx` to mock OpenAI / Gemini / Tavily HTTP calls (already in `requirements.txt`)
- `testcontainers` to spin up real Neo4j + Chroma sidecars in the runner

Tests that hit live external services should be tagged `@pytest.mark.integration` and run only via the `health-check.yml` workflow.

**Coverage upload** — both test jobs upload coverage XML to Codecov with distinct flags (`unit` and `mocked-integration`), so the Codecov UI shows per-tier coverage breakdown.

#### Test coverage

Coverage runs as part of every CI test job and prints a per-file line-coverage table at the end of the test output:

```
---------- coverage: platform linux, python 3.12.x ----------
Name                                                          Stmts   Miss  Cover
---------------------------------------------------------------------------------
scraper_preprocessing_memory/src/memory/agent.py                 142     38   73%
scraper_preprocessing_memory/src/memory/vector_store.py           58     12   79%
FakeNewsAgent/fact_check_agent/src/graph/nodes.py                206     91   56%
PredictionAgent/agents/entity_tracker.py                          81     22   73%
...
```

The XML report is uploaded as a workflow artifact (`coverage-xml`, 14-day retention) — download it from the Actions run page and open in any coverage viewer (or feed to codecov / coveralls if you ever set those up).

To run coverage locally inside your container:

```bash
docker compose -f docker/docker-compose.yml run --rm shell
pytest -m "not integration" \
  --cov=scraper_preprocessing_memory/src \
  --cov=FakeNewsAgent/fact_check_agent/src \
  --cov=PredictionAgent/agents \
  --cov-report=term-missing
```

`--cov-report=term-missing` prints uncovered line numbers next to each file — handy for spotting which branches need a test.

For an HTML report you can browse:

```bash
pytest -m "not integration" --cov=PredictionAgent/agents --cov-report=html
# then open htmlcov/index.html in a browser
```

### `scrape.yml` — Scheduled scraper

Runs the scraper pipeline against cloud DBs twice a day (cron `0 6,18 * * *`), and on manual trigger. Each run:
1. Spins up an Ubuntu runner
2. Installs deps + spaCy
3. Runs `python -m src.pipeline` from `scraper_preprocessing_memory/`
4. Articles + claims + captions land in your Aura + ChromaDB Cloud
5. Runner exits — no state kept between runs

The pipeline deduplicates by content hash, so missed runs / retries are safe.

### `health-check.yml` — Live API probes (Mon + Thu)

Twice-weekly probes verify the external services we depend on still answer correctly:
- OpenAI — 1-token chat completion
- Gemini — 1 short embedding (must still emit 1536-dim)
- Tavily — minimal search query
- Neo4j Aura — `RETURN 1` connectivity check
- ChromaDB Cloud — `list_collections()` auth check

Each probe is wrapped in `@pytest.mark.skipif(not os.getenv("..."))` so a missing secret skips just that probe rather than failing the whole workflow. When a probe fails (e.g. quota exceeded, key revoked, vendor changed response shape), GitHub auto-emails the workflow owner.

Cost: ~$0.05 per run × 2 runs/week ≈ $0.50/month.

### `release-image.yml` — Build, publish, smoke-test the image

Every push to `main` (and every tag push) does three things in two parallel jobs:
1. **`build-and-push`** — builds the Dockerfile, computes tags via `docker/metadata-action` (`:latest`, `:sha-<short>`, `:vX.Y.Z` on tags), pushes to GHCR.
2. **`smoke`** (depends on the above) — pulls the just-published image, spins up sidecar Neo4j + Chroma containers in the runner, then runs four checks:
   - **Smoke 1: imports** — `from src.memory.agent import MemoryAgent; from agents.fact_check_agent import fact_check_claim` etc., proving the container's PYTHONPATH is wired correctly
   - **Smoke 2: schema init** — `MemoryAgent.init_schema()` against sidecar Neo4j
   - **Smoke 3: round-trip write/read** — create a test entity, look it up by name
   - **Smoke 4: Streamlit boot** — start `streamlit run frontend/app.py`, poll `http://localhost:8501/_stcore/health` until it answers

If any smoke step fails, the image is published but the failure is loud (red workflow check + email). The bad image keeps its `:sha-…` tag for forensic reference.

### Required GitHub Secrets

In repo: **Settings → Secrets and variables → Actions → New repository secret**, add:

| Secret | Required for | Notes |
|---|---|---|
| `OPENAI_API_KEY` | scrape | Claim isolation, entity extraction, GPT-4o vision |
| `GOOGLE_API_KEY` | scrape | Gemini embeddings |
| `TAVILY_API_KEY` | scrape | Live news search |
| `NEO4J_URI` | scrape | e.g. `neo4j+s://xxxxxxxx.databases.neo4j.io` |
| `NEO4J_USER` | scrape | Usually `neo4j` |
| `NEO4J_PASSWORD` | scrape | Aura password |
| `CHROMA_API_KEY` | scrape | ChromaDB Cloud key |
| `CHROMA_TENANT` | scrape | ChromaDB Cloud tenant ID |
| `CHROMA_DATABASE` | scrape | ChromaDB Cloud database name |
| `TELEGRAM_SCRAPER_API_URL` | scrape (optional) | Only if using Telegram fetcher |
| `TELEGRAM_SCRAPER_API_KEY` | scrape (optional) | |
| `ENSEMBLEDATA_API_TOKEN` | scrape (optional) | Only if using Reddit/EnsembleData fetcher |
| `LANGFUSE_PUBLIC_KEY` | scrape (optional) | Leave unset to disable Langfuse |
| `LANGFUSE_SECRET_KEY` | scrape (optional) | |
| `LANGFUSE_HOST` | scrape (optional) | e.g. `https://cloud.langfuse.com` |

CI (`ci.yml`) needs **no secrets** — it runs against placeholder env vars.

### Manual / one-off triggers

From the GitHub UI: **Actions** tab → pick a workflow → **Run workflow**. Useful for:
- Re-running the scraper out-of-cycle after pushing fetcher changes
- Smoke-testing a workflow before relying on the cron

### Cost/Frequency guardrails

- To throttle: edit the cron in `.github/workflows/scrape.yml`, or lower `max_per_source` in `src/pipeline.py`'s `ScraperAgent.scrape(...)` call.
- The 30-minute `timeout-minutes` cap stops a hung run from racking up runner-minutes.

### `release-image.yml` — Docker image to GHCR

Every push to `main` (and every tag push) builds the Docker image and publishes it to GitHub Container Registry. Anyone with read access to the repo can then `docker pull` and run the system:

```bash
docker pull ghcr.io/<owner>/news_facts_system:latest
docker run --rm --env-file .env ghcr.io/<owner>/news_facts_system:latest \
  python -c "import src.config; print('ok')"
```

Tag scheme:
- `:latest` — most recent push to `main`
- `:sha-<short>` — every commit, regardless of branch
- `:vX.Y.Z` — published when you push a `v*.*.*` git tag

The workflow uses GitHub Actions cache (`type=gha`) for the `pip install` layer, so first build is ~5–10 min, subsequent builds with no `requirements.txt` change drop to ~1–2 min. No secrets needed — auth uses the auto-provided `GITHUB_TOKEN`.

### `release.yml` — Auto-generated GitHub Releases

When you push a tag matching `v*.*.*`, this workflow runs in parallel with `release-image.yml`:
- Pulls the full git history
- Generates a changelog from commits since the previous tag (grouped by `feat:` / `fix:` / `ci:` / `docs:` prefixes)
- Creates a GitHub Release with the changelog as the body, named after the tag

Cutting a release:
```bash
git tag v0.1.0
git push origin v0.1.0
```

After ~2 minutes you'll have:
- `ghcr.io/<owner>/news_facts_system:v0.1.0` (image)
- A new entry on the repo's **Releases** page with auto-generated notes

If you tag with a hyphen (e.g. `v0.2.0-rc1`), the Release is automatically marked as a pre-release.

### `dependabot.yml` — Automated dependency PRs

Three ecosystems are watched weekly:
- **`pip`** — Python packages in `requirements.txt`. Patch + minor bumps only; majors are ignored (too risky to auto-propose).
- **`github-actions`** — `uses:` version pins in workflow files. Near-zero risk; safe to merge after CI passes.
- **`docker`** — base image digest in `docker/Dockerfile`. Triggers when Docker Hub republishes `python:3.12-slim` (Python point releases, security patches).

Each PR is labeled `dependencies` + an ecosystem-specific tag (`python`, `github-actions`, `docker`). CI runs automatically; merge after green. No PR is auto-merged — review the changelog link before clicking Merge.

To check Dependabot's last scan or trigger an early scan: **Insights → Dependency Graph → Dependabot** in the repo UI.

### Reproducible builds (digest-pinned base image)

The `docker/Dockerfile` first line pins to a specific `sha256:…` digest:

```dockerfile
FROM python:3.12-slim@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3
```

This means every build (today, in 6 months, on a fresh machine) starts from byte-identical bits — no surprise OS patches, no surprise Python point upgrades. Dependabot's `docker` ecosystem opens PRs to bump the digest when the upstream tag moves.

To bump manually:
```bash
docker pull python:3.12-slim
docker inspect --format='{{index .RepoDigests 0}}' python:3.12-slim
# paste the new digest into Dockerfile line 1
```
