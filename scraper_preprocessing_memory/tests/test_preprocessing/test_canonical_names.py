"""Tests for canonical entity name lookup (Tier 1 of entity canonicalization).

Scenario 1: a new entity extracted from an article is an alias of a known
canonical entity (e.g. "USA" → "United States"). The canonicalize() function
should map the alias to the canonical form before entity_id is generated,
so Neo4j's MERGE treats variant mentions as a single node.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.id_utils import make_entity_id


@pytest.fixture
def canonical_fixture(tmp_path, monkeypatch):
    """Create a temp canonical_entities.json and patch the module to use it."""
    fixture_data = [
        {
            "canonical_name": "United States",
            "entity_type": "country",
            "aliases": ["USA", "US", "U.S.", "America"],
        },
        {
            "canonical_name": "Federal Bureau of Investigation",
            "entity_type": "organization",
            "aliases": ["FBI", "F.B.I."],
        },
    ]
    fixture_file = tmp_path / "canonical_entities.json"
    fixture_file.write_text(json.dumps(fixture_data), encoding="utf-8")

    # Reload the canonical module pointing at the fixture
    import src.preprocessing.canonical_names as cn

    monkeypatch.setattr(cn, "_CANONICAL_FILE", fixture_file)
    cn.reload_canonical_lookup()
    yield cn
    # Cleanup: restore the real lookup after the test
    cn._CANONICAL_FILE = Path(__file__).resolve().parents[2] / "data" / "canonical_entities.json"
    cn.reload_canonical_lookup()


class TestCanonicalize:
    def test_exact_alias_match_returns_canonical(self, canonical_fixture):
        """"USA" should be mapped to "United States" (Tier 1 hit)."""
        assert canonical_fixture.canonicalize("USA", "country") == "United States"

    def test_case_insensitive_lookup(self, canonical_fixture):
        """Aliases should match regardless of casing."""
        assert canonical_fixture.canonicalize("usa", "country") == "United States"
        assert canonical_fixture.canonicalize("Usa", "country") == "United States"

    def test_canonical_name_itself_matches(self, canonical_fixture):
        """The canonical name is its own alias (no-op lookup)."""
        assert (
            canonical_fixture.canonicalize("United States", "country")
            == "United States"
        )

    def test_alias_not_cross_contaminated_between_types(self, canonical_fixture):
        """'FBI' as 'country' should NOT match the organization canonical."""
        # "FBI" is an alias only for organization. Looking it up as country
        # should return the original string (no match).
        assert canonical_fixture.canonicalize("FBI", "country") == "FBI"
        assert (
            canonical_fixture.canonicalize("FBI", "organization")
            == "Federal Bureau of Investigation"
        )

    def test_unknown_entity_returns_stripped_original(self, canonical_fixture):
        """Entities not in the registry pass through unchanged (except strip)."""
        assert canonical_fixture.canonicalize("NovelOrg  ", "organization") == "NovelOrg"

    def test_empty_name_returns_empty(self, canonical_fixture):
        assert canonical_fixture.canonicalize("", "country") == ""


class TestCanonicalizeUnifiesEntityId:
    """The whole point of canonicalization: aliases should produce the same entity_id.

    Without canonicalization, "USA" and "US" generate different make_entity_id outputs,
    causing Neo4j to create separate nodes. With canonicalization, both map to
    "United States" first, then produce the SAME id.
    """

    def test_aliases_produce_same_entity_id(self, canonical_fixture):
        usa_canonical = canonical_fixture.canonicalize("USA", "country")
        us_canonical = canonical_fixture.canonicalize("US", "country")
        america_canonical = canonical_fixture.canonicalize("America", "country")

        usa_id = make_entity_id(usa_canonical, "country")
        us_id = make_entity_id(us_canonical, "country")
        america_id = make_entity_id(america_canonical, "country")

        assert usa_id == us_id == america_id, (
            "All aliases of 'United States' should produce the same entity_id "
            "so Neo4j MERGE unifies them."
        )

    def test_different_entities_produce_different_ids(self, canonical_fixture):
        """Sanity check: canonicalization doesn't collapse unrelated entities."""
        us_id = make_entity_id(
            canonical_fixture.canonicalize("USA", "country"), "country"
        )
        fbi_id = make_entity_id(
            canonical_fixture.canonicalize("FBI", "organization"), "organization"
        )
        assert us_id != fbi_id


class TestGetAllCanonicalNames:
    def test_returns_all_canonicals_as_tuples(self, canonical_fixture):
        canonicals = canonical_fixture.get_all_canonical_names()
        assert ("country", "United States") in canonicals
        assert ("organization", "Federal Bureau of Investigation") in canonicals
        # Aliases should NOT be in the canonical set
        assert ("country", "USA") not in canonicals
