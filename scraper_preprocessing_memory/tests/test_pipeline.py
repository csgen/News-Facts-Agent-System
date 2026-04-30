"""Tests for the scraper pipeline's ScrapeRun summary write.

The contract:
  - On a successful run, `memory.add_scrape_run(...)` is called once with
    accurate counts and `source="local"` (or `"github_actions"` inside Actions
    runners).
  - The write happens inside a `try/finally` so partial failures (e.g. a
    preprocessor exception, a Gemini quota hiccup during reconcile) still
    produce a ScrapeRun node — the dashboard panel must not lose runs to
    silent crashes.

All three agents (Scraper, Preprocessing, Memory) are mocked; no real
network or DB calls.
"""

import os
from unittest.mock import MagicMock, patch

# pydantic-settings reads these at module-import time when src.config loads
# (transitively imported by src.pipeline). Placeholders are fine for unit tests.
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test_unused")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-unused")

import pytest  # noqa: E402
from src import pipeline  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def patched_pipeline(monkeypatch):
    """Patch ScraperAgent, PreprocessingAgent, MemoryAgent, and time.sleep.

    Returns a tuple of the mock instance objects (not the classes), so tests
    can configure return values / side effects on them directly.
    """
    # Skip the 60s reconcile cooldown so tests run instantly.
    monkeypatch.setattr(pipeline.time, "sleep", lambda *_a, **_kw: None)

    scraper = MagicMock()
    preprocessor = MagicMock()
    memory = MagicMock()

    # Default: scraper produces 3 raw articles, preprocessor + memory succeed.
    raw_articles = [MagicMock(title=f"raw_{i}") for i in range(3)]
    scraper.scrape.return_value = raw_articles
    preprocessor.process.side_effect = lambda raw: MagicMock(
        article=MagicMock(article_id=f"art_{raw.title}")
    )
    memory.ingest_preprocessed.return_value = True
    memory.reconcile_entities.return_value = {"merged": 0}
    memory.promote_canonical_candidates.return_value = []

    monkeypatch.setattr(pipeline, "ScraperAgent", lambda *_a, **_kw: scraper)
    monkeypatch.setattr(pipeline, "PreprocessingAgent", lambda *_a, **_kw: preprocessor)
    monkeypatch.setattr(pipeline, "MemoryAgent", lambda *_a, **_kw: memory)

    return scraper, preprocessor, memory


# ──────────────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────────────


def test_pipeline_writes_scrape_run_on_success(patched_pipeline, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    scraper, preprocessor, memory = patched_pipeline

    summary = pipeline.run_pipeline()

    # Returned summary reflects the happy-path counts
    assert summary["scraped"] == 3
    assert summary["ingested"] == 3
    assert summary["skipped_duplicates"] == 0
    assert summary["failed"] == 0

    # ScrapeRun was written exactly once with matching counts
    memory.add_scrape_run.assert_called_once()
    kwargs = memory.add_scrape_run.call_args.kwargs
    assert kwargs["scraped"] == 3
    assert kwargs["ingested"] == 3
    assert kwargs["skipped"] == 0
    assert kwargs["failed"] == 0
    assert kwargs["source"] == "local"
    assert kwargs["run_id"].startswith("sr_")
    assert kwargs["duration_s"] >= 0


def test_pipeline_skipped_duplicate_reflected_in_scrape_run(patched_pipeline, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    scraper, preprocessor, memory = patched_pipeline
    # Two of three articles are duplicates (ingest returns False).
    memory.ingest_preprocessed.side_effect = [True, False, False]

    pipeline.run_pipeline()

    kwargs = memory.add_scrape_run.call_args.kwargs
    assert kwargs["scraped"] == 3
    assert kwargs["ingested"] == 1
    assert kwargs["skipped"] == 2
    assert kwargs["failed"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# Partial-failure paths — the `finally` must still write ScrapeRun
# ──────────────────────────────────────────────────────────────────────────────


def test_pipeline_writes_scrape_run_on_partial_failure(patched_pipeline, monkeypatch):
    """Preprocessor raises on the second article — pipeline should NOT crash,
    `failed` count should reflect it, and ScrapeRun should still land."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    scraper, preprocessor, memory = patched_pipeline

    call_state = {"n": 0}

    def maybe_fail(_raw):
        call_state["n"] += 1
        if call_state["n"] == 2:
            raise RuntimeError("simulated preprocessing failure")
        return MagicMock(article=MagicMock(article_id=f"art_{call_state['n']}"))

    preprocessor.process.side_effect = maybe_fail

    pipeline.run_pipeline()

    memory.add_scrape_run.assert_called_once()
    kwargs = memory.add_scrape_run.call_args.kwargs
    assert kwargs["scraped"] == 3
    assert kwargs["ingested"] == 2
    assert kwargs["failed"] == 1


def test_pipeline_writes_scrape_run_when_reconcile_fails(patched_pipeline, monkeypatch):
    """reconcile_entities throwing must not block the ScrapeRun write."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    scraper, preprocessor, memory = patched_pipeline
    memory.reconcile_entities.side_effect = RuntimeError("Gemini 429 RESOURCE_EXHAUSTED")

    pipeline.run_pipeline()

    memory.add_scrape_run.assert_called_once()


def test_pipeline_writes_scrape_run_when_promote_fails(patched_pipeline, monkeypatch):
    """promote_canonical_candidates throwing must not block the ScrapeRun write."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    scraper, preprocessor, memory = patched_pipeline
    memory.promote_canonical_candidates.side_effect = RuntimeError("simulated")

    pipeline.run_pipeline()

    memory.add_scrape_run.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────────
# Source label — controls "where this run came from" in the Grafana panel
# ──────────────────────────────────────────────────────────────────────────────


def test_pipeline_source_label_local_by_default(patched_pipeline, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    scraper, preprocessor, memory = patched_pipeline

    pipeline.run_pipeline()

    assert memory.add_scrape_run.call_args.kwargs["source"] == "local"


def test_pipeline_source_label_github_actions_when_env_set(patched_pipeline, monkeypatch):
    """Inside GH Actions runners GITHUB_ACTIONS=true is auto-set; honor it."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    scraper, preprocessor, memory = patched_pipeline

    pipeline.run_pipeline()

    assert memory.add_scrape_run.call_args.kwargs["source"] == "github_actions"


def test_pipeline_source_label_local_when_env_is_other_value(patched_pipeline, monkeypatch):
    """Treat anything besides exactly 'true' as local — Actions only ever sets 'true'."""
    monkeypatch.setenv("GITHUB_ACTIONS", "false")
    scraper, preprocessor, memory = patched_pipeline

    pipeline.run_pipeline()

    assert memory.add_scrape_run.call_args.kwargs["source"] == "local"


# ──────────────────────────────────────────────────────────────────────────────
# Cleanup invariants
# ──────────────────────────────────────────────────────────────────────────────


def test_pipeline_closes_memory_at_end(patched_pipeline, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    scraper, preprocessor, memory = patched_pipeline

    pipeline.run_pipeline()

    memory.close.assert_called_once()


def test_pipeline_closes_memory_even_when_scrape_run_write_fails(patched_pipeline, monkeypatch):
    """If the ScrapeRun write itself fails (Aura down), pipeline still closes Memory."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    scraper, preprocessor, memory = patched_pipeline
    memory.add_scrape_run.side_effect = RuntimeError("Aura unreachable")

    pipeline.run_pipeline()

    memory.close.assert_called_once()
