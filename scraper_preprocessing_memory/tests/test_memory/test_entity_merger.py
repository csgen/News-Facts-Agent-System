"""Tests for EntityMerger's 2-tier clustering + merge logic.

Covers:
- Scenario 2: entities caught by fuzzy string match (Tier 2)
- Scenario 3: entities caught by embedding similarity (Tier 3) with cluster
  formation and representative selection

The GraphStore and EmbeddingHelper are mocked so tests don't require Neo4j
or OpenAI connections.
"""

import math
from unittest.mock import MagicMock

import numpy as np
from src.memory.entity_merger import EntityMerger, UnionFind

# ──────────────────────────────────────────────────────────────────────────────
# Test helpers
# ──────────────────────────────────────────────────────────────────────────────


class MockGraphStore:
    """In-memory fake of GraphStore for merger tests.

    Records merge_entity() calls so the test can assert which merges happened.
    """

    def __init__(self, entities: list[dict]):
        self._entities = entities
        self.merge_calls: list[tuple[str, str]] = []

    def get_all_entities(self) -> list[dict]:
        return list(self._entities)

    def count_entity_mentions(self, entity_id: str) -> int:
        for e in self._entities:
            if e["entity_id"] == entity_id:
                return e.get("total_claims", 0)
        return 0

    def merge_entity(self, source_id: str, target_id: str) -> None:
        self.merge_calls.append((source_id, target_id))


def _make_embedder_mock(name_to_vec: dict[str, list[float]]):
    """Build a fake EmbeddingHelper that returns preset vectors per name."""
    helper = MagicMock()

    def embed_batch(texts: list[str]) -> list[list[float]]:
        return [name_to_vec[t] for t in texts]

    helper.embed_batch.side_effect = embed_batch
    return helper


def _orthogonal_vec(seed: int, dim: int = 64) -> list[float]:
    """Produce a deterministic unit vector along a standard basis axis.

    Using one-hot basis vectors guarantees orthogonality between different
    seeds (since e_i · e_j = 0 for i != j), which is what `_blend` requires
    to produce exact target cosine similarities.
    """
    assert 0 <= seed < dim, f"seed {seed} must be in [0, {dim})"
    v = [0.0] * dim
    v[seed] = 1.0
    return v


def _blend(a: list[float], b: list[float], target_cos: float) -> list[float]:
    """Produce a unit vector x such that cosine(x, a) == target_cos.

    Requires a and b to be orthogonal unit vectors. Constructs:
        x = target_cos * a + sqrt(1 - target_cos^2) * b
    which has |x| = 1 and cos(x, a) = target_cos (since a · b = 0).
    """
    va, vb = np.array(a), np.array(b)
    blended = target_cos * va + math.sqrt(1 - target_cos**2) * vb
    return blended.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# SCENARIO 2 — Tier 2: fuzzy string match
# ──────────────────────────────────────────────────────────────────────────────


class TestFuzzyStringMatch:
    """Entities that are obviously the same via string similarity should cluster.

    Example: "Donald Trump" and "Donald J. Trump" have ~88% token_sort_ratio
    similarity — above our 85 threshold — so they merge without needing
    the embedding tier.
    """

    def test_similar_names_get_merged(self):
        entities = [
            {
                "entity_id": "ent_donald_trump",
                "name": "Donald Trump",
                "entity_type": "person",
                "total_claims": 10,
                "accurate_claims": 7,
                "current_credibility": 0.5,
            },
            {
                "entity_id": "ent_donald_j_trump",
                "name": "Donald J. Trump",
                "entity_type": "person",
                "total_claims": 3,
                "accurate_claims": 2,
                "current_credibility": 0.5,
            },
        ]
        graph = MockGraphStore(entities)

        # Make embedding vectors orthogonal so ONLY fuzzy tier can match them.
        embedder = _make_embedder_mock({
            "Donald Trump": _orthogonal_vec(1),
            "Donald J. Trump": _orthogonal_vec(50),
        })

        merger = EntityMerger(graph_store=graph, embedding_helper=embedder)
        summary = merger.reconcile()

        # One merge happened (the less-mentioned one merged into the more-mentioned one)
        assert summary["merges"] == 1
        assert summary["clusters"] == 1
        assert len(graph.merge_calls) == 1
        source, target = graph.merge_calls[0]
        # Representative should be the entity with more mentions (Donald Trump, 10 claims)
        assert target == "ent_donald_trump"
        assert source == "ent_donald_j_trump"

    def test_dissimilar_names_are_not_merged(self):
        """Unrelated entities should never cluster just because fuzzy runs on them."""
        entities = [
            {
                "entity_id": "ent_tesla",
                "name": "Tesla",
                "entity_type": "organization",
                "total_claims": 5,
                "accurate_claims": 4,
                "current_credibility": 0.5,
            },
            {
                "entity_id": "ent_apple",
                "name": "Apple",
                "entity_type": "organization",
                "total_claims": 3,
                "accurate_claims": 3,
                "current_credibility": 0.5,
            },
        ]
        graph = MockGraphStore(entities)
        embedder = _make_embedder_mock({
            "Tesla": _orthogonal_vec(1),
            "Apple": _orthogonal_vec(2),
        })

        merger = EntityMerger(graph_store=graph, embedding_helper=embedder)
        summary = merger.reconcile()

        assert summary["merges"] == 0
        assert graph.merge_calls == []

    def test_different_entity_types_never_merged(self):
        """Even if names are similar, different entity_types should not be merged."""
        # "Apple" (the product) vs "Apple" (the organization) — identical name,
        # but different types → must stay separate
        entities = [
            {
                "entity_id": "ent_apple_org",
                "name": "Apple",
                "entity_type": "organization",
                "total_claims": 8,
                "accurate_claims": 6,
                "current_credibility": 0.5,
            },
            {
                "entity_id": "ent_apple_prod",
                "name": "Apple",
                "entity_type": "product",
                "total_claims": 2,
                "accurate_claims": 2,
                "current_credibility": 0.5,
            },
        ]
        graph = MockGraphStore(entities)
        embedder = _make_embedder_mock({
            "Apple": _orthogonal_vec(1),  # Only asked per-type, so one vec is fine
        })

        merger = EntityMerger(graph_store=graph, embedding_helper=embedder)
        summary = merger.reconcile()

        assert summary["merges"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# SCENARIO 3 — Tier 3: embedding similarity + cluster + representative
# ──────────────────────────────────────────────────────────────────────────────


class TestEmbeddingSimilarity:
    """When fuzzy match doesn't catch a pair but their embeddings are close,
    Tier 3 (cosine similarity ≥ 0.85) should merge them.

    Cases:
      - cosine ≥ 0.85: merge
      - cosine < 0.85: no merge
    """

    def test_high_similarity_auto_merges(self):
        """cosine ≥ 0.85 should merge."""
        entities = [
            {
                "entity_id": "ent_a",
                "name": "FooCorp",
                "entity_type": "organization",
                "total_claims": 3,
                "accurate_claims": 2,
                "current_credibility": 0.5,
            },
            {
                "entity_id": "ent_b",
                "name": "FooCorporation",
                "entity_type": "organization",
                "total_claims": 1,
                "accurate_claims": 1,
                "current_credibility": 0.5,
            },
        ]
        graph = MockGraphStore(entities)

        # Build vectors with cosine = 0.90 (well above threshold)
        base_a = _orthogonal_vec(1)
        base_b = _orthogonal_vec(2)
        vec_a = base_a
        vec_b = _blend(base_a, base_b, target_cos=0.90)

        embedder = _make_embedder_mock({
            "FooCorp": vec_a,
            "FooCorporation": vec_b,
        })

        merger = EntityMerger(graph_store=graph, embedding_helper=embedder)
        summary = merger.reconcile()

        # Either Tier 2 (fuzzy) or Tier 3 (embedding) would catch this.
        assert summary["merges"] == 1

    def test_low_similarity_does_not_merge(self):
        """cosine < 0.85 should not merge, even for entities of same type."""
        entities = [
            {
                "entity_id": "ent_a",
                "name": "Tesla",
                "entity_type": "organization",
                "total_claims": 5,
                "accurate_claims": 4,
                "current_credibility": 0.5,
            },
            {
                "entity_id": "ent_b",
                "name": "Banana Republic",
                "entity_type": "organization",
                "total_claims": 2,
                "accurate_claims": 1,
                "current_credibility": 0.5,
            },
        ]
        graph = MockGraphStore(entities)

        # Orthogonal vectors ≈ cos 0
        embedder = _make_embedder_mock({
            "Tesla": _orthogonal_vec(1),
            "Banana Republic": _orthogonal_vec(2),
        })

        merger = EntityMerger(graph_store=graph, embedding_helper=embedder)
        summary = merger.reconcile()

        assert summary["merges"] == 0

    def test_moderate_similarity_does_not_merge(self):
        """Cosine in [0.70, 0.85) was previously LLM-verified. Now it stays unmerged.

        Trade-off: some semantic variants like "America" vs "United States" (which
        tend to sit around 0.75-0.80 cosine) will stay separate unless they hit
        the canonical list, get caught by fuzzy match, or accumulate enough
        mentions to be promoted to the canonical list.
        """
        entities = [
            {
                "entity_id": "ent_a",
                "name": "America",
                "entity_type": "country",
                "total_claims": 4,
                "accurate_claims": 3,
                "current_credibility": 0.5,
            },
            {
                "entity_id": "ent_b",
                "name": "United States",
                "entity_type": "country",
                "total_claims": 12,
                "accurate_claims": 10,
                "current_credibility": 0.5,
            },
        ]
        graph = MockGraphStore(entities)

        # cosine = 0.75 (in what used to be the ambiguous zone)
        base_a = _orthogonal_vec(1)
        base_b = _orthogonal_vec(2)
        vec_america = _blend(base_a, base_b, target_cos=0.75)
        vec_us = base_a

        embedder = _make_embedder_mock({
            "America": vec_america,
            "United States": vec_us,
        })

        merger = EntityMerger(graph_store=graph, embedding_helper=embedder)
        summary = merger.reconcile()

        assert summary["merges"] == 0


class TestClusteringAndRepresentative:
    """When 3+ entities are all linked via pairwise similarity, they should
    form a single cluster and a single representative is picked."""

    def test_three_way_cluster_picks_canonical_as_representative(self, monkeypatch):
        """If one of the cluster members is already a canonical name, it's the rep.

        Variants: ["United States", "USA-var", "North American USA"].
        All cosine-similar; "United States" is canonical → must be target.
        """
        # Patch canonical lookup to contain just "United States"
        import src.memory.entity_merger as em

        monkeypatch.setattr(em, "get_all_canonical_names", lambda: {("country", "United States")})

        entities = [
            {
                "entity_id": "ent_us",
                "name": "United States",
                "entity_type": "country",
                "total_claims": 5,
                "accurate_claims": 4,
                "current_credibility": 0.5,
            },
            {
                "entity_id": "ent_usa_var",
                "name": "USA-var",
                "entity_type": "country",
                "total_claims": 20,  # More claims, but NOT canonical
                "accurate_claims": 15,
                "current_credibility": 0.5,
            },
            {
                "entity_id": "ent_na_usa",
                "name": "North American USA",
                "entity_type": "country",
                "total_claims": 3,
                "accurate_claims": 2,
                "current_credibility": 0.5,
            },
        ]
        graph = MockGraphStore(entities)

        base_a = _orthogonal_vec(1)
        base_b = _orthogonal_vec(2)
        vec_us = base_a
        # All three share high cosine similarity
        vec_usa_var = _blend(base_a, base_b, target_cos=0.92)
        vec_na_usa = _blend(base_a, base_b, target_cos=0.90)

        embedder = _make_embedder_mock({
            "United States": vec_us,
            "USA-var": vec_usa_var,
            "North American USA": vec_na_usa,
        })

        merger = EntityMerger(graph_store=graph, embedding_helper=embedder)
        summary = merger.reconcile()

        # 3 entities in one cluster → 2 merges, 1 cluster
        assert summary["clusters"] == 1
        assert summary["merges"] == 2

        # Representative must be the canonical entity, NOT the one with more claims
        targets = {target for (_source, target) in graph.merge_calls}
        assert targets == {"ent_us"}

    def test_cluster_without_canonical_picks_highest_mentions(self, monkeypatch):
        """No canonical in cluster → rep is entity with highest total_claims."""
        import src.memory.entity_merger as em

        monkeypatch.setattr(em, "get_all_canonical_names", lambda: set())

        entities = [
            {
                "entity_id": "ent_a",
                "name": "NovelOrg Alpha",
                "entity_type": "organization",
                "total_claims": 2,
                "accurate_claims": 1,
                "current_credibility": 0.5,
            },
            {
                "entity_id": "ent_b",
                "name": "NovelOrg Beta",
                "entity_type": "organization",
                "total_claims": 9,  # highest
                "accurate_claims": 8,
                "current_credibility": 0.5,
            },
            {
                "entity_id": "ent_c",
                "name": "NovelOrg Gamma",
                "entity_type": "organization",
                "total_claims": 3,
                "accurate_claims": 2,
                "current_credibility": 0.5,
            },
        ]
        graph = MockGraphStore(entities)

        base_a = _orthogonal_vec(1)
        base_b = _orthogonal_vec(2)
        embedder = _make_embedder_mock({
            "NovelOrg Alpha": base_a,
            "NovelOrg Beta": _blend(base_a, base_b, target_cos=0.92),
            "NovelOrg Gamma": _blend(base_a, base_b, target_cos=0.91),
        })

        merger = EntityMerger(graph_store=graph, embedding_helper=embedder)
        summary = merger.reconcile()

        assert summary["clusters"] == 1
        assert summary["merges"] == 2
        targets = {target for (_source, target) in graph.merge_calls}
        assert targets == {"ent_b"}  # highest total_claims


# ──────────────────────────────────────────────────────────────────────────────
# Helper: UnionFind sanity check
# ──────────────────────────────────────────────────────────────────────────────


class TestUnionFind:
    def test_transitively_merges_three_items_via_two_unions(self):
        uf = UnionFind(["a", "b", "c"])
        uf.union("a", "b")
        uf.union("b", "c")
        groups = uf.groups()
        assert len(groups) == 1
        only_cluster = list(groups.values())[0]
        assert set(only_cluster) == {"a", "b", "c"}

    def test_singletons_remain_separate(self):
        uf = UnionFind(["x", "y", "z"])
        groups = uf.groups()
        assert len(groups) == 3
