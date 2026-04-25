"""Bootstrap: add the memory module to sys.path so its internal `src.*` imports resolve.

In the integrated monorepo (news_facts_system), the memory module lives at the sibling
`scraper_preprocessing_memory/` folder next to `FakeNewsAgent/`, i.e. three levels up
from this file. We add that path so `from src.memory.agent import MemoryAgent` works.

Import this module first in any file that needs MemoryAgent. Idempotent.
"""
import sys
from pathlib import Path

_MEMORY_ROOT = Path(__file__).resolve().parents[3] / "scraper_preprocessing_memory"

if str(_MEMORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEMORY_ROOT))
