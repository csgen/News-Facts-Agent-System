"""Memory Agent adapter — thin re-export of scraper_preprocessing_memory's MemoryAgent.

The real implementation lives at `../../scraper_preprocessing_memory/src/memory/agent.py`.
This file only:
  1. Adjusts sys.path so scraper's `src.*` imports resolve.
  2. Re-exports `MemoryAgent` plus a process-level `get_memory()` singleton.

Usage:
    from agents.memory_agent import MemoryAgent, get_memory, close_memory
    memory = get_memory()
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Path setup so the three subfolders can import from each other:
#   <repo>/scraper_preprocessing_memory       → makes `src.*` imports work
#   <repo>/scraper_preprocessing_memory/src   → makes flat `id_utils`, `config`, `models.*`
#                                               imports work (used by task3 code
#                                               in entity_tracker.py / prediction_agent.py)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCAPPER = _REPO_ROOT / "scraper_preprocessing_memory"
for _p in (str(_SCAPPER), str(_SCAPPER / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.config import settings as _settings
from src.memory.agent import MemoryAgent

logger = logging.getLogger(__name__)

_memory: Optional[MemoryAgent] = None


def get_memory() -> MemoryAgent:
    """Return (or create) the process-level MemoryAgent singleton."""
    global _memory
    if _memory is None:
        logger.info("Initialising MemoryAgent singleton")
        _memory = MemoryAgent(_settings)
    return _memory


def close_memory() -> None:
    """Close the MemoryAgent and its Neo4j driver. Call at process shutdown."""
    global _memory
    if _memory is not None:
        _memory.close()
        _memory = None
        logger.info("MemoryAgent closed")


__all__ = ["MemoryAgent", "get_memory", "close_memory"]
