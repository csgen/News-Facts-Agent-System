"""Post-pipeline entity clustering and merge.

After article ingestion, this module:
1. Fetches all Entity nodes from Neo4j
2. Clusters variants using two tiers:
   - Tier 2: fuzzy string match (rapidfuzz) — catches typos, abbreviations
   - Tier 3: embedding similarity (OpenAI text-embedding-3-small) — catches
     semantic variants ("America" vs "United States") that differ in surface form
3. Merges each cluster into a single representative entity (canonical if present)

Tier 1 (canonical list lookup) is already applied during extraction in
`canonical_names.canonicalize`. This pass catches the rest.
"""

import logging
from collections import defaultdict

import numpy as np
from rapidfuzz import fuzz

from src.memory.embeddings import EmbeddingHelper
from src.memory.graph_store import GraphStore
from src.preprocessing.canonical_names import get_all_canonical_names

logger = logging.getLogger(__name__)

FUZZY_THRESHOLD = 85     # rapidfuzz token_sort_ratio (0-100)
EMBED_THRESHOLD = 0.85   # cosine similarity for merge (conservative to avoid false positives)


class UnionFind:
    """Disjoint-set data structure for clustering entities by similarity."""

    def __init__(self, items: list):
        self._parent = {item: item for item in items}

    def find(self, item):
        # Path compression
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while item != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def groups(self) -> dict:
        clusters: dict = defaultdict(list)
        for item in self._parent:
            clusters[self.find(item)].append(item)
        return dict(clusters)


class EntityMerger:
    def __init__(
        self,
        graph_store: GraphStore,
        embedding_helper: EmbeddingHelper,
    ):
        self._graph = graph_store
        self._embeddings = embedding_helper

    def reconcile(self) -> dict:
        """Run the full reconciliation pass.

        Returns a summary dict with counts for logging.
        """
        entities = self._graph.get_all_entities()
        if len(entities) < 2:
            return {"total_entities": len(entities), "merges": 0, "clusters": 0}

        # Group by entity_type (we only merge within the same type)
        by_type: dict[str, list[dict]] = defaultdict(list)
        for e in entities:
            by_type[e["entity_type"]].append(e)

        canonical_names = get_all_canonical_names()
        total_merges = 0
        total_clusters = 0

        for entity_type, type_entities in by_type.items():
            if len(type_entities) < 2:
                continue

            merges, clusters = self._reconcile_type(
                entity_type, type_entities, canonical_names
            )
            total_merges += merges
            total_clusters += clusters

        summary = {
            "total_entities": len(entities),
            "merges": total_merges,
            "clusters": total_clusters,
        }
        logger.info("Entity reconciliation complete: %s", summary)
        return summary

    def _reconcile_type(
        self,
        entity_type: str,
        entities: list[dict],
        canonical_names: set[tuple[str, str]],
    ) -> tuple[int, int]:
        """Cluster + merge entities of a single type. Returns (merges, clusters)."""
        ids = [e["entity_id"] for e in entities]
        id_to_entity = {e["entity_id"]: e for e in entities}

        uf = UnionFind(ids)

        # Tier 2: fuzzy string match (all pairs)
        n = len(entities)
        for i in range(n):
            for j in range(i + 1, n):
                score = fuzz.token_sort_ratio(entities[i]["name"], entities[j]["name"])
                if score >= FUZZY_THRESHOLD:
                    uf.union(ids[i], ids[j])

        # Tier 3: embedding similarity (catches semantic matches fuzzy can't)
        try:
            names = [e["name"] for e in entities]
            vectors = self._embeddings.embed_batch(names)
            vec_matrix = np.array(vectors)
            norms = np.linalg.norm(vec_matrix, axis=1, keepdims=True)
            normalized = vec_matrix / np.clip(norms, 1e-10, None)
            similarity = normalized @ normalized.T

            for i in range(n):
                for j in range(i + 1, n):
                    sim = float(similarity[i, j])
                    if sim >= EMBED_THRESHOLD:
                        uf.union(ids[i], ids[j])
        except Exception as e:
            logger.warning("Embedding similarity step failed, skipping: %s", e)

        # Build clusters and merge
        clusters = uf.groups()
        merges = 0
        multi_member_clusters = 0

        for _root, member_ids in clusters.items():
            if len(member_ids) < 2:
                continue
            multi_member_clusters += 1

            members = [id_to_entity[mid] for mid in member_ids]
            target = self._pick_representative(members, entity_type, canonical_names)

            for member in members:
                if member["entity_id"] == target["entity_id"]:
                    continue
                try:
                    self._graph.merge_entity(
                        source_id=member["entity_id"],
                        target_id=target["entity_id"],
                    )
                    merges += 1
                    logger.info(
                        "Merged entity '%s' → '%s' (type=%s)",
                        member["name"], target["name"], entity_type,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to merge %s → %s: %s",
                        member["entity_id"], target["entity_id"], e,
                    )

        return merges, multi_member_clusters

    def _pick_representative(
        self,
        members: list[dict],
        entity_type: str,
        canonical_names: set[tuple[str, str]],
    ) -> dict:
        """Pick the representative entity from a cluster.

        Priority:
        1. Entity whose name is in the canonical list (for this type)
        2. Entity with the highest total_claims
        3. Tie-breaker: longest name (usually more specific: "Donald Trump" over "Trump")
        """
        # Tier 1: any canonical match
        for m in members:
            if (entity_type, m["name"]) in canonical_names:
                return m

        # Tier 2: highest total_claims, then longest name
        return max(
            members,
            key=lambda m: (m.get("total_claims", 0), len(m["name"])),
        )
