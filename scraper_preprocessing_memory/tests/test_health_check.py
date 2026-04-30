"""Twice-weekly probes against real APIs to detect vendor drift / quota / auth issues.

Costs ~$0.05 per run. Only invoked by .github/workflows/health-check.yml — never
runs in the PR-time CI pipeline (filtered out by the default `not integration`
marker selection).

Each test is decorated with skipif on the relevant secret, so a partial-config
run (e.g. dev forgot to set TAVILY_API_KEY) skips that probe rather than failing
the whole workflow.
"""
import os

import pytest

# Treated like the rest of the live-service tests so pytest -m "not integration"
# excludes them from regular CI runs.
pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_alive():
    """1-token completion proves OpenAI key + endpoint + model still work."""
    import openai

    r = openai.OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "say ok"}],
        max_tokens=2,
    )
    assert r.choices[0].message.content, "OpenAI returned empty content"


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
def test_gemini_embedding_alive():
    """1 short embed call proves the Gemini API still emits 1536-dim vectors."""
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    r = client.models.embed_content(
        model="gemini-embedding-001",
        contents="health probe",
        config=genai_types.EmbedContentConfig(output_dimensionality=1536),
    )
    assert len(r.embeddings) == 1
    assert len(r.embeddings[0].values) == 1536, (
        f"expected 1536-dim vector, got {len(r.embeddings[0].values)}"
    )


@pytest.mark.skipif(not os.getenv("TAVILY_API_KEY"), reason="TAVILY_API_KEY not set")
def test_tavily_alive():
    """Smallest possible Tavily call — proves the search endpoint + auth still work."""
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    r = client.search("test", max_results=1)
    assert "results" in r, f"unexpected Tavily response shape: {r}"


@pytest.mark.skipif(not os.getenv("NEO4J_URI"), reason="NEO4J_URI not set")
def test_neo4j_aura_alive():
    """Connectivity probe — `RETURN 1` on the live Aura instance."""
    from neo4j import GraphDatabase

    drv = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
    )
    try:
        with drv.session() as s:
            x = s.run("RETURN 1 AS x").single()["x"]
        assert x == 1
    finally:
        drv.close()


@pytest.mark.skipif(
    not os.getenv("CHROMA_API_KEY"),
    reason="CHROMA_API_KEY not set (this probe targets ChromaDB Cloud, not local)",
)
def test_chroma_cloud_alive():
    """List collections on ChromaDB Cloud — proves auth + tenant + database still resolve."""
    import chromadb

    c = chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
    )
    # Just calling this proves auth works; we don't care about the result shape.
    c.list_collections()
