"""ID generation utilities with type prefixes."""

import hashlib
import uuid


def make_id(prefix: str) -> str:
    """Generate a random ID with a type prefix.

    Examples: art_a3f8b2c1d4e5, clm_7d2e9f01b3c4
    """
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def make_entity_id(name: str, entity_type: str) -> str:
    """Generate a deterministic entity ID from name + type.

    This ensures the same entity (e.g. "Tesla" / "organization")
    always maps to the same ID regardless of which article it appears in.
    """
    key = f"{name.strip().lower()}|{entity_type.strip().lower()}"
    hash_hex = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"ent_{hash_hex}"
