"""Tests for ID generation utilities."""

from src.id_utils import make_entity_id, make_id


class TestMakeId:
    def test_prefix(self):
        assert make_id("art_").startswith("art_")
        assert make_id("clm_").startswith("clm_")

    def test_length(self):
        id_ = make_id("art_")
        assert len(id_) == 4 + 12  # prefix + 12 hex chars

    def test_unique(self):
        ids = {make_id("art_") for _ in range(100)}
        assert len(ids) == 100


class TestMakeEntityId:
    def test_deterministic(self):
        id1 = make_entity_id("Tesla", "organization")
        id2 = make_entity_id("Tesla", "organization")
        assert id1 == id2

    def test_case_insensitive(self):
        id1 = make_entity_id("Tesla", "Organization")
        id2 = make_entity_id("tesla", "organization")
        assert id1 == id2

    def test_prefix(self):
        assert make_entity_id("Tesla", "organization").startswith("ent_")

    def test_different_entities(self):
        id1 = make_entity_id("Tesla", "organization")
        id2 = make_entity_id("Apple", "organization")
        assert id1 != id2
