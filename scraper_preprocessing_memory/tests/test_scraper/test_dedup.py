"""Tests for content hash deduplication."""

from src.scraper.dedup import compute_content_hash


class TestContentHash:
    def test_basic_hash(self):
        h = compute_content_hash("Test Title", "Test body text")
        assert h.startswith("sha256_")
        assert len(h) == 7 + 64  # "sha256_" + 64 hex chars

    def test_deterministic(self):
        h1 = compute_content_hash("Title", "Body")
        h2 = compute_content_hash("Title", "Body")
        assert h1 == h2

    def test_case_insensitive(self):
        h1 = compute_content_hash("TITLE", "BODY")
        h2 = compute_content_hash("title", "body")
        assert h1 == h2

    def test_whitespace_normalized(self):
        h1 = compute_content_hash("  Title  ", "  Body  ")
        h2 = compute_content_hash("Title", "Body")
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = compute_content_hash("Title A", "Body A")
        h2 = compute_content_hash("Title B", "Body B")
        assert h1 != h2
