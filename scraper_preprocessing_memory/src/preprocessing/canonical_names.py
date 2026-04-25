"""Canonical entity name registry.

Loads data/canonical_entities.json at import time and provides a fast
alias-to-canonical-name lookup. Used by the entity extractor to unify
variant names (USA / US / United States) into a single canonical form
before generating entity_id.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Path relative to project root (works both locally and in Docker container)
_CANONICAL_FILE = Path(__file__).resolve().parents[2] / "data" / "canonical_entities.json"


def _load_canonical_lookup() -> dict[tuple[str, str], str]:
    """Build a {(entity_type_lower, name_lower): canonical_name} dict.

    Includes both the canonical_name (as its own alias) and each alias.
    """
    lookup: dict[tuple[str, str], str] = {}

    if not _CANONICAL_FILE.exists():
        logger.warning("Canonical entities file not found at %s", _CANONICAL_FILE)
        return lookup

    try:
        with _CANONICAL_FILE.open("r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception as e:
        logger.error("Failed to load canonical entities: %s", e)
        return lookup

    for entry in entries:
        canonical = entry.get("canonical_name", "").strip()
        entity_type = entry.get("entity_type", "").strip().lower()
        aliases = entry.get("aliases", [])

        if not canonical or not entity_type:
            continue

        # The canonical form itself maps to itself (case-insensitive lookup)
        lookup[(entity_type, canonical.lower())] = canonical

        # All aliases map to the canonical form
        for alias in aliases:
            if alias:
                lookup[(entity_type, alias.strip().lower())] = canonical

    logger.info("Loaded %d canonical aliases from %s", len(lookup), _CANONICAL_FILE)
    return lookup


# Module-level singleton loaded once at import time
_CANONICAL_LOOKUP: dict[tuple[str, str], str] = _load_canonical_lookup()


def _build_any_type_lookup(typed_lookup: dict[tuple[str, str], str]) -> dict[str, str]:
    """Collapse a {(type, name_lower): canonical} dict into {name_lower: canonical}.

    Used at query time when the caller doesn't know the entity_type
    (e.g. a frontend lookup of "Trump"). Cross-type collisions resolve to
    whichever entry was inserted last — acceptable since most aliases are
    type-unique in practice.
    """
    return {name_lower: canonical for (_et, name_lower), canonical in typed_lookup.items()}


_CANONICAL_LOOKUP_ANY_TYPE: dict[str, str] = _build_any_type_lookup(_CANONICAL_LOOKUP)


def canonicalize(name: str, entity_type: str) -> str:
    """Return the canonical form of an entity name if it's a known alias.

    If not found in the canonical registry, returns the original name stripped.
    Lookup is case-insensitive and scoped by entity_type.
    """
    if not name:
        return name

    stripped = name.strip()
    key = (entity_type.strip().lower(), stripped.lower())
    return _CANONICAL_LOOKUP.get(key, stripped)


def canonicalize_any_type(name: str) -> str | None:
    """Return canonical form for `name` ignoring entity_type, or None if unknown.

    Used by query-time lookups (e.g. `MemoryAgent.get_entity_by_name`) where
    the caller doesn't know which entity_type the name belongs to.

    Example:
        canonicalize_any_type("USA")    → "United States"
        canonicalize_any_type("trump")  → None  (not in registry)
    """
    if not name or not name.strip():
        return None
    return _CANONICAL_LOOKUP_ANY_TYPE.get(name.strip().lower())


def get_all_canonical_names() -> set[tuple[str, str]]:
    """Return set of (entity_type_lower, canonical_name) for all known canonicals.

    Used by entity_merger to identify which entities are already canonical
    (should not be merged as sources).
    """
    canonicals: set[tuple[str, str]] = set()
    if not _CANONICAL_FILE.exists():
        return canonicals

    try:
        with _CANONICAL_FILE.open("r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            canonical = entry.get("canonical_name", "").strip()
            entity_type = entry.get("entity_type", "").strip().lower()
            if canonical and entity_type:
                canonicals.add((entity_type, canonical))
    except Exception as e:
        logger.error("Failed to read canonical names: %s", e)

    return canonicals


def reload_canonical_lookup() -> None:
    """Reload the lookup table from disk. Call after canonical_promoter appends new entries."""
    global _CANONICAL_LOOKUP, _CANONICAL_LOOKUP_ANY_TYPE
    _CANONICAL_LOOKUP = _load_canonical_lookup()
    _CANONICAL_LOOKUP_ANY_TYPE = _build_any_type_lookup(_CANONICAL_LOOKUP)
