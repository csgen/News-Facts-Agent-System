# FactGuard — AI-Powered Fact-Checking System

FactGuard is a multi-agent system that automatically scrapes news, extracts claims, and produces evidence-backed verdicts using a LangGraph pipeline, Neo4j graph memory, and ChromaDB vector retrieval.

---

## What It Does

1. **Scrapes** news articles from RSS feeds, NewsAPI/Tavily, and Telegram
2. **Preprocesses** each article: extracts claims, named entities, and image captions
3. **Fact-checks** each claim through a multi-node LangGraph pipeline:
   - Cache check → freshness gate → live web search → RAG retrieval → verdict synthesis
   - Optional: multi-agent debate, cross-modal image consistency check
4. **Tracks** entity credibility over time (Neo4j snapshots)
5. **Predicts** future credibility changes using linear-regression trend detection
6. **Evaluates** pipeline quality on LIAR, FEVER, and Factify benchmarks
7. **Displays** everything in a Streamlit dashboard

---

## Project Structure

```
factguard-merged/
├── agents/
│   ├── memory_agent.py         # Unified facade over Neo4j + ChromaDB (Task 1)
│   ├── fact_check_agent.py     # Entry point for the LangGraph fact-check pipeline (Task 2)
│   ├── entity_tracker.py       # Computes + stores credibility snapshots (Task 3)
│   ├── prediction_agent.py     # Trend detection + rules-based predictions (Task 3)
│   ├── reflection_agent.py     # Per-source credibility tracking via ChromaDB (Task 2)
│   ├── scraper_agent.py        # Orchestrates news fetchers (Task 1)
│   └── preprocessing_agent.py  # Claim extraction + NER + captioning (Task 1)
│
├── memory/
│   ├── graph_db.py             # Neo4j GraphStore — nodes, relationships, Cypher queries
│   ├── vector_db.py            # ChromaDB VectorStore — 5 collections
│   └── embeddings.py           # OpenAI embedding helper with retry
│
├── models/
│   ├── article.py              # Source, Article Pydantic models
│   ├── claim.py                # Claim, Entity, MentionSentiment models
│   ├── credibility.py          # CredibilitySnapshot, Prediction models
│   ├── verdict.py              # Verdict model
│   ├── caption.py              # ImageCaption model
│   ├── pipeline.py             # PreprocessingOutput model
│   └── schemas.py              # FactCheckInput/Output, EntityRef (Task 2 schemas)
│
├── orchestration/
│   ├── graph.py                # LangGraph StateGraph builder
│   ├── nodes.py                # All 13 graph node functions
│   ├── router.py               # Conditional edge routing logic
│   └── state.py                # FactCheckState TypedDict
│
├── tools/
│   ├── rag_tool.py             # ChromaDB retrieval + RRF fusion
│   ├── live_search_tool.py     # Tavily live web search
│   ├── freshness_tool.py       # Claim freshness / time-sensitivity check
│   ├── cross_modal_tool.py     # Image ↔ claim consistency check (SigLIP/CLIP)
│   └── reranker.py             # Reciprocal Rank Fusion + cross-encoder reranking
│
├── scraper/                    # Low-level fetchers (from Task 1)
│   └── fetchers/
│       ├── base.py             # RawArticle dataclass + BaseFetcher ABC
│       ├── newsapi.py          # Tavily news fetcher
│       ├── rss.py              # RSS feed fetcher
│       └── telegram.py         # Telegram channel fetcher
│
├── preprocessing/              # LLM sub-processors (from Task 1)
│   ├── claim_isolator.py       # GPT-4 claim extraction
│   ├── entity_extractor.py     # Batched NER via GPT-4
│   ├── caption_generator.py    # VLM image caption generation
│   └── prompts.py              # Preprocessing prompt templates
│
├── evaluation/
│   └── evaluation.py           # Benchmark runner: LIAR, FEVER, Factify
│
├── frontend/
│   └── app.py                  # Streamlit dashboard (Task 3)
│
├── config.py                   # Unified pydantic-settings config
├── llm_factory.py              # LLM + embedding client factory
├── prompts.py                  # Fact-check LLM prompt templates
├── id_utils.py                 # UUID-based ID generator
├── requirements.txt
├── .env.example
└── docker-compose.yml          # Neo4j + ChromaDB containers
```

---

## How Agents Connect

```
ScraperAgent
    │  list[RawArticle]
    ▼
PreprocessingAgent
    │  PreprocessingOutput (Source, Article, list[Claim], ImageCaption)
    ├──► MemoryAgent.add_article / add_claims / add_image_caption   → Neo4j + ChromaDB
    ▼
FactCheckAgent (fact_check_claim / run_fact_check)
    │  calls LangGraph pipeline per claim
    ├──► MemoryAgent.query_similar_claims                           → ChromaDB RAG
    ├──► MemoryAgent.get_verdict_by_claim / write_verdict           → Neo4j
    ├──► ReflectionAgent (source credibility)                       → ChromaDB
    └──► live_search_tool (Tavily)                                  → web
    │  FactCheckOutput (verdict, confidence, evidence_links, bias)
    ▼
EntityTracker  (run after fact-check)
    ├──► MemoryAgent.get_entity_claims                              → Neo4j
    └──► MemoryAgent.add_credibility_snapshot / update_entity       → Neo4j

PredictionAgent  (run periodically)
    ├──► MemoryAgent.get_entity_snapshots                           → Neo4j
    ├──► MemoryAgent.get_claim_count_for_entity                     → Neo4j
    └──► MemoryAgent.add_prediction / resolve_prediction            → Neo4j

Streamlit frontend
    ├──► FactCheckAgent.fact_check_claim                            → live verdict
    ├──► MemoryAgent.get_entity_snapshots                           → credibility chart
    └──► MemoryAgent.get_predictions_for_entity                     → prediction table
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker (for Neo4j and ChromaDB)

### 2. Start the databases

```bash
docker-compose up -d
```

This starts:
- **Neo4j** on `bolt://localhost:7687` (browser at http://localhost:7474)
- **ChromaDB** on `http://localhost:8000`

### 3. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY and TAVILY_API_KEY
```

### 5. Run the Streamlit dashboard

```bash
streamlit run frontend/app.py
```

Open http://localhost:8501 in your browser.

---

## Running Individual Agents

```bash
# Fact-check a single claim
python -c "from agents.fact_check_agent import fact_check_claim; print(fact_check_claim('Tesla recalled 500k cars'))"

# Run entity tracker for an entity
python -m agents.entity_tracker    # defaults to "Tesla", last 24h

# Run prediction agent
python -m agents.prediction_agent  # defaults to "Tesla"

# Run evaluation benchmarks
python -m evaluation.evaluation --dataset liar --samples 50
python -m evaluation.evaluation --dataset all  --samples 100
```

---

## Feature Flags (`.env`)

| Variable | Default | Description |
|---|---|---|
| `USE_GRAPH_RAG` | `false` | Add Neo4j entity-graph context to retrieval |
| `USE_DEBATE` | `false` | Run multi-agent advocate/arbiter debate when confidence < threshold |
| `DEBATE_CONFIDENCE_THRESHOLD` | `70` | Trigger debate below this confidence (0–100) |
| `USE_CROSS_MODAL` | `false` | Run SigLIP/CLIP image ↔ claim consistency check |
| `USE_RERANKER` | `false` | Cross-encoder reranking after RRF fusion |
| `OFFLINE_MODE` | `false` | Skip Tavily live search |
| `DRY_RUN` | `false` | Skip all DB writes |

---

## Evaluation Benchmarks

| Dataset | Task | Metric |
|---|---|---|
| LIAR | Verdict classification | Macro-F1, Entity-F1, Precision@5 |
| FEVER | Wikipedia fact-checking | Macro-F1, Entity-F1, Precision@5 |
| Factify | Multi-modal claim + image | Macro-F1, Image-Mismatch F1 |

Real datasets are loaded via HuggingFace `datasets` (install with `pip install datasets`). If unavailable, the evaluation script falls back to synthetic mock data.

---

## Input Guardrail (Task 3)

All user inputs are screened through a two-layer defence before reaching the fact-checking pipeline.

### Layer A — Rule-Based (instant, zero cost)

Regex patterns catch known attack patterns without any API call:

| Check | Examples blocked |
|---|---|
| Prompt injection | `ignore previous instructions`, `[INST]`, `<\|im_start\|>` |
| Jailbreak / persona | `DAN`, `developer mode`, `god mode`, `roleplay as` |
| System prompt leak | `repeat your system prompt`, `what were your instructions` |
| PII detection | credit cards, SSNs, API keys (`sk-…`), passwords |
| Hate / harmful | bomb-making, violent calls to action |
| Gibberish / spam | low alpha-ratio inputs, keyboard spam |
| Length bounds | < 5 chars or > 5 000 chars |

### Layer B — LLM Classifier (GPT-4o-mini, ~300 ms)

Only fires when Layer A passes. Classifies inputs as `SAFE / UNSAFE` with a `NONE / LOW / MEDIUM / HIGH` risk score. Catches subtle attacks that regex cannot — educational framing, indirect persona substitution, base64-encoded injections.

**Return schema:**
```json
{"blocked": true, "reason": "…", "layer": "A", "risk": "HIGH"}
```

### Guardrail Benchmark

Run the evaluation suite:
```bash
python -m evaluation.guardrail_benchmark          # Layer A + B
python -m evaluation.guardrail_benchmark --no-llm  # Layer A only
```

44 test cases across 8 categories:

| Category | Cases | What is tested |
|---|---|---|
| Prompt Injection | 6 | Classic overrides, delimiter attacks, ChatML tokens |
| Jailbreak | 6 | DAN, persona hijack, developer/god mode, roleplay |
| Prompt Extraction | 4 | Direct prompt leak, instruction extraction |
| PII | 4 | SSN, credit card, API key, password |
| Harmful Content | 2 | Bomb-making, violent content |
| Gibberish/Spam | 2 | Random chars, too-short input |
| Garak-Style Probes | 6 | Base64 encoding, GCG suffix, continuation, simulation |
| Legitimate Inputs | 10 | Real news claims, URLs, controversial-but-genuine topics |

Metrics reported: **True Positive Rate**, **False Positive Rate**, accuracy, per-category breakdown, Layer A vs B catch breakdown, latency.

---

## Entity Credibility Formula (Task 3)

### Credibility Score

Recency-weighted with Bayesian volume shrinkage — few claims pull the score toward the neutral prior (0.5).

```
label_score  : supported=1.0, misleading=0.35, refuted=0.0
time_decay_i : exp(−λ × days_ago),   λ = ln(2)/14  (14-day half-life)
weight_i     : confidence_i × time_decay_i

evidence     = Σ(label_score_i × weight_i) / Σ(weight_i)

volume_factor = 1 − exp(−n/3)         # 0.63 at n=1, 0.86 at n=3, 0.99 at n=10
credibility   = 0.5×(1−volume_factor) + evidence×volume_factor
```

### Sentiment Score

Recency-weighted in [−1, +1]: `positive→+1, neutral→0, negative→−1`, same 14-day half-life.

---

## Prediction Agent (Task 3)

After 3 or more credibility snapshots are stored for an entity, the Prediction Agent runs automatically:

1. Loads the N most recent `CredibilitySnapshot` nodes from Neo4j
2. Fits a linear regression on `(snapshot_index → credibility_score)`
3. Extrapolates to a 7-day horizon
4. Applies threshold rules (drop < 0.3 → "credibility likely to fall critically")
5. Writes a `Prediction` node to Neo4j with `prediction_text`, `confidence`, and `deadline`

The Streamlit dashboard shows open predictions under the **Entity & Trend** tab.

---

## Team

| Name | Task | Role |
|---|---|---|
| Shantam Sharma | Task 2 | Core Agentic AI Engineer |
| Chen Sigen | Task 1 | Data & Memory Engineer |
| Ahmed Abdul Wasae | Task 3 | Full-Stack & Evaluation Engineer |

| Task | Branch |
|---|---|
| Task 1 | `data_memory_dev` |
| Task 2 | `main` (FakeNewsAgent) |
| Task 3 | `feature/input-guardrail` |
