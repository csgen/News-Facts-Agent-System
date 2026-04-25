"""Content hash deduplication using SHA-256."""

import hashlib


def compute_content_hash(title: str, body_text: str) -> str:
    """Compute a SHA-256 hash of normalized title + body for exact deduplication."""
    normalized = f"{title.strip().lower()}|{body_text.strip().lower()}"
    hash_hex = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"sha256_{hash_hex}"
