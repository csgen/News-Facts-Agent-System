"""Top-level FakeNewsAgent conftest.

In the integrated monorepo (news_facts_system), the memory module lives at the
sibling `scraper_preprocessing_memory/` subfolder, not at `./memory_agent/`.
The top-level `pytest.ini` already adds the right paths to PYTHONPATH; this
file only sets default env vars so tests that touch `src.config.Settings`
don't fail at import time on missing required fields.
"""
import os

# Required by `src.config.Settings` (raises if missing). Use safe placeholders;
# any test that actually hits the network is decorated with @pytest.mark.skipif
# or @pytest.mark.integration.
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test_unused")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-unused")
