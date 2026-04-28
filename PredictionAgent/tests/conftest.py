"""PredictionAgent test fixtures + env defaults.

Importing `agents.entity_tracker` or `agents.prediction_agent` ultimately
loads `src.config.Settings` (via the memory_agent adapter), which raises if
required env vars are missing. We seed safe placeholders here so unit tests
never need real credentials.
"""

import os

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test_unused")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-unused")
os.environ.setdefault("GOOGLE_API_KEY", "")
