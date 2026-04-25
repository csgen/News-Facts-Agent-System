"""Automated promotion of significant entities to the canonical list.

After the entity reconciliation pass (entity_merger), variants have been
unified and their mention counts aggregated. This module finds entities
that have crossed a significance threshold and weren't already canonical,
then appends them to data/canonical_entities.json so they're treated as
canonical in future pipeline runs.

Rationale for no LLM check:
Because the merger clusters variants BEFORE counting, an entity hitting
10+ unified mentions is almost certainly a real, significant entity.
The threshold provides a safety margin against clustering errors.
"""

import json
import logging
from pathlib import Path

from src.memory.graph_store import GraphStore
from src.preprocessing.canonical_names import (
    _CANONICAL_FILE,  # re-use the same path
    get_all_canonical_names,
    reload_canonical_lookup,
)

logger = logging.getLogger(__name__)

SIGNIFICANCE_THRESHOLD = 10


class CanonicalPromoter:
    def __init__(self, graph_store: GraphStore):
        self._graph = graph_store

    def promote(self, threshold: int = SIGNIFICANCE_THRESHOLD) -> list[dict]:
        """Find non-canonical entities with >= threshold mentions and append to canonical list.

        Returns list of promoted entries (for logging).
        """
        entities = self._graph.get_all_entities()
        canonical_set = get_all_canonical_names()

        candidates = []
        for e in entities:
            mention_count = self._graph.count_entity_mentions(e["entity_id"])
            key = (e["entity_type"].lower(), e["name"])
            if key not in canonical_set and mention_count >= threshold:
                candidates.append({
                    "canonical_name": e["name"],
                    "entity_type": e["entity_type"],
                    "aliases": [],
                    "mention_count": mention_count,
                })

        if not candidates:
            logger.info("Canonical promotion: no candidates crossed threshold (%d)", threshold)
            return []

        self._append_to_canonical_file(candidates)

        # Refresh the in-memory canonical lookup so future entity extractions
        # within the same process see the new entries.
        reload_canonical_lookup()

        for c in candidates:
            logger.info(
                "Promoted to canonical list: '%s' (type=%s, mentions=%d)",
                c["canonical_name"], c["entity_type"], c["mention_count"],
            )

        return candidates

    def _append_to_canonical_file(self, new_entries: list[dict]) -> None:
        """Append new entries to data/canonical_entities.json, preserving existing content."""
        path: Path = _CANONICAL_FILE

        try:
            with path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception as e:
            logger.error("Failed to read canonical file for append: %s", e)
            return

        # Strip the transient `mention_count` field before writing
        for entry in new_entries:
            existing.append({
                "canonical_name": entry["canonical_name"],
                "entity_type": entry["entity_type"],
                "aliases": entry["aliases"],
            })

        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
                f.write("\n")
        except Exception as e:
            logger.error("Failed to write canonical file: %s", e)
