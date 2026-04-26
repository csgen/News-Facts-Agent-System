# FakeNewsAgent — Fact-Check Agent Reference

> Created: 2026-04-25 | Not tracked in git | For internal engineering reference only

---

## 1. Overview

The Fact-Check Agent is a **LangGraph-based agentic pipeline** that takes a single extracted `Claim` from preprocessed news articles and produces a `FactCheckOutput` verdict (`supported` / `refuted` / `misleading`). It is one sub-system in a larger News-Facts-Agent-System that also includes a Scraper, Preprocessing, and a MemoryAgent (ChromaDB + Neo4j).

**Key technology:** LangGraph state machine, OpenAI gpt-4o (or Ollama local), Tavily live search, ChromaDB (vector memory), Neo4j (graph memory), SigLIP / Gemma4 vision (cross-modal).

---

## 2. Input Contract

Entry point: `graph.invoke({"input": fact_check_input})`

```
FactCheckInput (Pydantic, schemas.py)
├── claim_id:          str          e.g. "clm_abc123"
├── claim_text:        str          The claim sentence to verify
├── entities:          list[EntityRef]  Named entities with sentiment (can be [])
│   └── EntityRef { entity_id, name, entity_type, sentiment }
├── source_url:        str          Article origin URL
├── article_id:        str          For image caption correlation
├── image_caption:     Optional[str]  Pre-fetched VLM text caption (None if no image)
├── image_url:         Optional[str]  Raw image URL / base64 URI (for SigLIP/vision)
├── timestamp:         datetime     When the claim was extracted
└── prefetched_chunks: list[str]    Pre-fetched evidence (skips Tavily when non-empty)
```

**How `FactCheckInput` is populated:**

- **Real pipeline** (`pipeline.py`): Built from a `PreprocessingOutput` → `Claim` object. Image caption is fetched once per article via `memory.get_caption_by_article()` before graph invocation. Entities come from the Claim model.
- **Benchmark path**: Built by `BenchmarkRecord.to_fact_check_input()`. Entities are usually `[]`; captions are pre-generated offline.

---

## 3. Output Contract

Final output: `state["output"]`

```
FactCheckOutput (Pydantic, schemas.py)
├── verdict_id:              str          e.g. "vrd_xyz789"
├── claim_id:                str          Matches the input claim_id
├── verdict:                 str          "supported" | "refuted" | "misleading"
├── confidence_score:        int (0–100)  Stored as float/100 in memory
├── evidence_links:          list[str]    Source URLs used as evidence
├── reasoning:               str          2–3 sentence chain-of-thought
├── bias_score:              float (0–1)  0.0 = unbiased, 1.0 = highly biased
├── cross_modal_flag:        bool         True if image/text conflict detected
├── cross_modal_explanation: Optional[str]  One sentence; None if no conflict
├── last_verified_at:        Optional[datetime]  Populated on cache hits only
└── revalidation_needed:     bool         True if freshness check triggered live re-check
```

---

## 4. Graph State (`FactCheckState`)

The LangGraph state is a `TypedDict` that flows through all nodes. Each node returns a partial dict of keys it mutates.

| Field | Type | Set by | Purpose |
|---|---|---|---|
| `input` | `FactCheckInput` | Caller | Immutable claim data |
| `memory_results` | `MemoryQueryResponse` | `query_memory` | Similar claims from ChromaDB |
| `entity_context` | `list[dict]` | `query_memory` | Entity credibility from Neo4j |
| `route` | `str` | Router / nodes | `"cache"` or `"live_search"` |
| `revalidation_needed` | `bool` | `freshness_check` | Whether cache needs re-verification |
| `retrieval_gate_needed` | `bool` | `retrieval_gate` | Whether to call Tavily |
| `retrieved_chunks` | `list[str]` | `live_search`, `rag_retrieval`, `return_cached` | Evidence blocks for synthesis |
| `sub_claims` | `list[str]` | `decompose_claim` | Atomic sub-claims (SOTA only) |
| `debate_transcript` | `str` | `multi_agent_debate` | Full advocate/arbiter transcript |
| `source_credibility` | `dict` | `query_memory` | Weighted source stats from Reflection Agent |
| `cross_modal_flag` | `bool` | `cross_modal_check` | Image/text conflict |
| `cross_modal_explanation` | `str` | `cross_modal_check` | Human-readable conflict description |
| `clip_similarity_score` | `float` | `cross_modal_check` | SigLIP match probability |
| `last_verified_at` | `datetime` | `query_memory` | Timestamp of best cache hit |
| `output` | `FactCheckOutput` | `synthesize_verdict`, updated by debate + cross-modal, finalized by `emit_output` | Final verdict |

---

## 5. Graph Architecture & Data Flow

```
[CALLER] graph.invoke({"input": FactCheckInput})
          │
          ▼
    receive_claim          ← Initialises all state fields to defaults; loads prefetched_chunks
          │
          ▼
    decompose_claim        ← S3 (disabled by default): splits compound claim into atomic sub-claims
          │
          ▼
    query_memory           ← 1) ChromaDB vector similarity search (top-k similar claims)
          │                   2) GraphRAG via Neo4j entity traversal (if use_graph_rag=True)
          │                   3) RRF merge + optional cross-encoder rerank
          │                   4) Reflection Agent READ: source credibility from ChromaDB
          │
          ▼
    [router]  ──── max_confidence >= 0.80? ────────────────────────────────┐
          │                                                                  │
          └── < 0.80 (live_search path)                                    │ (cache path)
                │                                                           ▼
                ▼                                                    freshness_check
          retrieval_gate     ← S2 (disabled by default):             │
          │                    LLM decides if Tavily needed           ├── fresh → return_cached
          │                                                           │            │
          ├── needed (default)                                        └── stale → live_search (below)
          │       │
          │       ▼
          │   live_search    ← Tavily API call (skipped if prefetched_chunks exist)
          │       │             Enforces ≥3 distinct source domains; retries with broader query
          │       │
          └── skip ──────────┐
                             │
                             ▼
                       rag_retrieval   ← Formats similar claims from memory_results into context block;
                             │           appends to retrieved_chunks
                             │
                             ▼
                    return_cached ──────┐  (appends CACHE HIT chunk)
                             │          │
                             ▼          │
                   synthesize_verdict ◄─┘
                             │    ← gpt-4o call with:
                             │      - claim_text
                             │      - evidence_block (all retrieved_chunks joined)
                             │      - source_credibility_note (from Reflection Agent)
                             │      Normalises verdict to {supported, refuted, misleading}
                             │
                    [debate_check]
                             │
                    confidence < debate_confidence_threshold (default 70)?
                             │
                    ├── yes: multi_agent_debate  ← S4: two advocates + arbiter, updates output
                    │
                    └── no: (skip)
                             │
                             ▼
                    cross_modal_check   ← Priority order:
                             │            1. SigLIP (local embedding similarity) if use_siglip=True
                             │            2. Gemma4 vision via Ollama if llm_provider=="ollama"
                             │            3. LLM caption check (fallback)
                             │           Sets cross_modal_flag + explanation on output
                             │
                             ▼
                      write_memory    ← Writes Verdict to ChromaDB + Neo4j
                             │         Reflection Agent WRITE: appends (source, topic, credibility, bias)
                             │
                             ▼
                       emit_output    ← Stamps last_verified_at + revalidation_needed onto output
                             │
                             ▼
                           [END]
                             │
                    state["output"] = FactCheckOutput
```

---

## 6. Node Details

### `receive_claim`
- No-op initializer. Copies `prefetched_chunks` from input into `retrieved_chunks` so downstream nodes see them.

### `decompose_claim` (S3, disabled by default)
- LLM call with `DECOMPOSITION_PROMPT` → JSON list of `{text, verifiable}` sub-claims.
- Gated by `settings.use_claim_decomposition = False`.

### `query_memory`
- **Vector search**: `memory.search_similar_claims(claim_text, top_k=5)` → ChromaDB cosine similarity.
- **GraphRAG** (optional): `memory.get_entity_ids_for_claims()` + `memory.get_graph_claims_for_entities()` → Neo4j 1-hop traversal.
- **Reranking**: RRF merge of vector + graph results; optional cross-encoder rerank (`use_cross_encoder=False` by default).
- **Reflection Agent READ**: `memory.query_source_credibility()` → weighted kNN of past (source, topic) observations.
- Sets: `memory_results`, `entity_context`, `last_verified_at`, `source_credibility`.

### `freshness_check`
- Only runs on the cache path (confidence ≥ 0.80).
- LLM call with `FRESHNESS_CHECK_PROMPT` categorises claim type and decides if re-verification needed.
- **ReAct mode** (`use_freshness_react=True`): LLM can call Tavily before deciding.
- Thresholds by category: political/election → 7 days, ongoing events → 3 days, economic → 14 days, scientific → 180 days, historical → rarely.

### `retrieval_gate` (S2, disabled by default)
- LLM call with `IS_RETRIEVAL_NEEDED_PROMPT` + existing memory context.
- Returns `retrieval_gate_needed: bool`. When False, skips Tavily entirely.

### `live_search`
- Calls `search_live(claim_text, tavily_api_key)`.
- Skipped entirely when `retrieved_chunks` is non-empty (prefetched evidence case).
- Minimum 3 distinct source domains enforced; retries with `"fact check: {claim_text}"` if fewer.

### `rag_retrieval`
- Formats `memory_results` into a `[RETRIEVED EVIDENCE FROM MEMORY]` block.
- Appends to `retrieved_chunks` (so both live search + memory context go to synthesis).

### `return_cached`
- On high-confidence cache hit: creates a `[CACHE HIT]` chunk from the best matching claim.
- Synthesis still runs (using cached chunk as evidence).

### `synthesize_verdict`
- Single gpt-4o call with `VERDICT_SYNTHESIS_PROMPT`.
- Input: claim text + all `retrieved_chunks` joined + `source_credibility_note`.
- Source credibility note includes: domain-level credibility (weighted kNN mean), bias score ± std, entity credibility from Neo4j.
- Output: JSON → `FactCheckOutput` (partial; cross-modal fields filled later).
- Normalises non-standard verdict labels to valid 3-label set.

### `multi_agent_debate` (S4, disabled by default)
- Two advocate LLM calls (for/against) + one arbiter LLM call.
- Updates `output.verdict`, `confidence_score`, `reasoning`, `bias_score`.
- Stores full transcript in `state["debate_transcript"]`.
- Triggered when `confidence_score < debate_confidence_threshold` (default 70).

### `cross_modal_check`
- Skipped if no `image_url` and no `image_caption`.
- Mode 1 (SigLIP): local HuggingFace `google/siglip-base-patch16-224` — sigmoid probability < `siglip_threshold` (default 0.10) = conflict.
- Mode 2 (Gemma4 vision via Ollama): sends base64 image + claim to local VLM.
- Mode 3 (LLM caption fallback): `CROSS_MODAL_PROMPT` with text caption.
- Updates `output.cross_modal_flag` and `output.cross_modal_explanation`.

### `write_memory`
- Constructs `Verdict` object (memory_agent model) and calls `memory.add_verdict()`.
- Also calls `update_source_credibility()` — Reflection Agent WRITE — inserts one (source_id, topic_embedding, credibility, bias) observation into ChromaDB.
- Skipped in `dry_run` or `offline_mode`.

### `emit_output`
- Stamps `last_verified_at` and `revalidation_needed` onto the output (these are only known after freshness_check runs, so they're set at the very end).

---

## 7. Routing Logic

| Router | Condition | Routes to |
|---|---|---|
| `router` (after `query_memory`) | `max_confidence >= 0.80` | `freshness_check` (cache path) |
| `router` | `max_confidence < 0.80` | `retrieval_gate` (live path) |
| `freshness_router` | `revalidation_needed == False` | `return_cached` |
| `freshness_router` | `revalidation_needed == True` (or None) | `live_search` |
| `retrieval_gate_router` | `retrieval_gate_needed == True` | `live_search` |
| `retrieval_gate_router` | `retrieval_gate_needed == False` | `rag_retrieval` (skip Tavily) |
| `debate_check` | `use_debate=True` AND `confidence < threshold` | `multi_agent_debate` |
| `debate_check` | otherwise | `cross_modal_check` |

---

## 8. Real Pipeline Invocation (`pipeline.py`)

```python
# Entry: called after MemoryAgent.ingest_preprocessed() in the main pipeline
from fact_check_agent.src.pipeline import run_fact_check

outputs: list[FactCheckOutput] = run_fact_check(preprocessing_output)
```

Internally:
1. Gets singleton MemoryAgent via `get_memory()`.
2. Pre-fetches article image caption once: `memory.get_caption_by_article(article_id)`.
3. For each `Claim` in `preprocessing_output.claims`:
   - Builds `FactCheckInput` via `claim_to_fact_check_input()`.
   - Calls `graph.invoke({"input": fact_check_input})`.
   - Appends `state["output"]` to results list.

The graph is compiled once at module level and reused across calls.

---

## 9. Reflection Agent (Source Credibility Subsystem)

A stateless helper — no LLM, no loops — that accumulates source reputation across verdicts.

**READ** (called by `query_memory`):
- Queries `source_credibility` ChromaDB collection for k=20 nearest (source, topic) observations.
- Returns weighted-kNN statistics: `credibility_mean`, `bias_mean`, `bias_std`, `sample_count`.
- Injected into synthesis prompt so the LLM can factor in source history.

**WRITE** (called by `write_memory`):
- Maps verdict to credibility signal: `supported` → `confidence/100`, `refuted` → `1 - confidence/100`, `misleading` → `0.5`.
- Inserts one new point into ChromaDB collection (always insert, never upsert — full history preserved).

---

## 10. Configuration Flags (`config.py`)

| Flag | Default | Effect |
|---|---|---|
| `llm_provider` | `"openai"` | `"openai"` or `"ollama"` |
| `llm_model` | `"gpt-4o"` | Model for all LLM calls |
| `use_graph_rag` | `False` | Enable Neo4j entity traversal in memory query |
| `use_cross_encoder` | `False` | Enable cross-encoder reranking after RRF |
| `use_siglip` | `False` | Enable SigLIP image-text similarity |
| `siglip_threshold` | `0.10` | Below this sigmoid prob → conflict |
| `use_retrieval_gate` | `False` | S2: LLM decides if Tavily call needed |
| `use_claim_decomposition` | `False` | S3: Decompose compound claims |
| `use_debate` | `False` | S4: Advocate/arbiter debate for low confidence |
| `debate_confidence_threshold` | `70` | Trigger debate when confidence < this |
| `use_freshness_react` | `False` | S6: ReAct loop for freshness decision |
| `dry_run` | `False` | Skip all DB writes |
| `offline_mode` | `False` | Skip all DB reads + writes |
| `reranker_top_k` | `5` | Final chunks passed to synthesis |

---

## 11. External Dependencies

| Service | Used for | Config keys |
|---|---|---|
| OpenAI API | LLM calls (all nodes), embeddings | `openai_api_key`, `llm_model` |
| Tavily API | Live web search | `tavily_api_key` |
| ChromaDB | Vector store (claims, verdicts, source credibility) | `chroma_api_key` / `chroma_host:port` |
| Neo4j Aura | Graph store (entities, source-topic credibility) | `neo4j_uri`, `neo4j_user`, `neo4j_password` |
| Ollama (local) | Alternative LLM + vision (Gemma4) | `ollama_base_url`, `ollama_llm_model` |
| HuggingFace | SigLIP / cross-encoder models | local, no API key |
| LangSmith | Tracing | `langchain_api_key`, `langchain_tracing_v2` |

---

## 12. Verdict Label Normalisation

The LLM sometimes returns non-standard labels. The synthesizer normalises:
- Contains "support" → `"supported"`
- Contains "refut", "contradict", or "false" → `"refuted"`
- Anything else → `"misleading"` (safe default)

---

## 13. Key File Locations

| Purpose | File |
|---|---|
| Input/Output schemas | `fact_check_agent/src/models/schemas.py` |
| Graph state | `fact_check_agent/src/models/state.py` |
| Graph assembly | `fact_check_agent/src/graph/graph.py` |
| All node functions | `fact_check_agent/src/graph/nodes.py` |
| Routing logic | `fact_check_agent/src/graph/router.py` |
| All prompts | `fact_check_agent/src/prompts.py` |
| Settings | `fact_check_agent/src/config.py` |
| Real pipeline entry | `fact_check_agent/src/pipeline.py` |
| MemoryAgent singleton | `fact_check_agent/src/memory_client.py` |
| Reflection Agent | `fact_check_agent/src/agents/reflection_agent.py` |
| Live search tool | `fact_check_agent/src/tools/live_search_tool.py` |
| RAG tool | `fact_check_agent/src/tools/rag_tool.py` |
| Cross-modal tool | `fact_check_agent/src/tools/cross_modal_tool.py` |
| Freshness tool | `fact_check_agent/src/tools/freshness_tool.py` |
| Reranker | `fact_check_agent/src/tools/reranker.py` |
