"""Memory Agent facade — the single public API for all agents.

Teammates import MemoryAgent and call its methods. They never interact
with VectorStore or GraphStore directly.

Usage:
    from src.memory.agent import MemoryAgent
    from src.config import settings

    memory = MemoryAgent(settings)
    memory.ingest_preprocessed(preprocessing_output)
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from src.config import Settings
from src.memory.canonical_promoter import CanonicalPromoter
from src.memory.embeddings import EmbeddingHelper
from src.memory.entity_merger import EntityMerger
from src.memory.graph_store import GraphStore
from src.memory.vector_store import VectorStore
from src.models.caption import ImageCaption
from src.models.credibility import CredibilitySnapshot, Prediction
from src.models.pipeline import PreprocessingOutput
from src.models.verdict import Verdict

logger = logging.getLogger(__name__)


class MemoryAgent:
    def __init__(self, settings: Settings):
        self._embeddings = EmbeddingHelper(
            api_key=settings.google_api_key,
            model=settings.embedding_model,
            output_dimensionality=settings.embedding_dim,
        )
        self._vector = VectorStore(
            api_key=settings.chroma_api_key,
            tenant=settings.chroma_tenant,
            database=settings.chroma_database,
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        self._graph = GraphStore(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
        self._entity_merger = EntityMerger(self._graph, self._embeddings)
        self._canonical_promoter = CanonicalPromoter(self._graph)
        logger.info("MemoryAgent initialized")

    def close(self) -> None:
        self._graph.close()

    def init_schema(self) -> None:
        """Initialize Neo4j constraints and indexes."""
        self._graph.init_schema()

    # ── Primary Write: Ingest Preprocessed Output ───────────────────────

    def ingest_preprocessed(self, output: PreprocessingOutput) -> bool:
        """Ingest a preprocessed article into both databases.

        Returns False if the article is a duplicate (already exists).
        """
        # Dedup check
        if self._vector.check_content_hash_exists(output.article.content_hash):
            logger.info("Duplicate article skipped: %s", output.article.article_id)
            return False

        # Collect all texts for batch embedding
        texts_to_embed = []
        text_labels = []

        # Article embedding
        texts_to_embed.append(output.article.body_snippet)
        text_labels.append(("article", 0))

        # Claim embeddings
        for i, claim in enumerate(output.claims):
            texts_to_embed.append(claim.claim_text)
            text_labels.append(("claim", i))

        # Caption embedding
        if output.image_caption:
            texts_to_embed.append(output.image_caption.vlm_caption)
            text_labels.append(("caption", 0))

        # Batch embed
        embeddings = self._embeddings.embed_batch(texts_to_embed)

        # Map embeddings back
        article_embedding = None
        claim_embeddings: list[list[float]] = []
        caption_embedding = None

        for idx, (label_type, _) in enumerate(text_labels):
            if label_type == "article":
                article_embedding = embeddings[idx]
            elif label_type == "claim":
                claim_embeddings.append(embeddings[idx])
            elif label_type == "caption":
                caption_embedding = embeddings[idx]

        # ── Write to Graph DB ───────────────────────────────────────────
        # Source (MERGE — idempotent)
        self._graph.merge_source(
            source_id=output.source.source_id,
            name=output.source.name,
            domain=output.source.domain,
            category=output.source.category,
            base_credibility=output.source.base_credibility,
        )

        # Article
        self._graph.create_article(
            article_id=output.article.article_id,
            title=output.article.title,
            url=output.article.url,
            source_id=output.article.source_id,
            published_at=output.article.published_at,
            ingested_at=output.article.ingested_at,
            content_hash=output.article.content_hash,
        )

        # Claims with entities
        claims_data = []
        for claim in output.claims:
            claims_data.append({
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "claim_type": claim.claim_type or "",
                "topic_text": claim.topic_text,
                "extracted_at": claim.extracted_at.isoformat(),
                "status": claim.status,
                "entities": [
                    {
                        "entity_id": e.entity_id,
                        "name": e.name,
                        "entity_type": e.entity_type,
                        "sentiment": e.sentiment,
                    }
                    for e in claim.entities
                ],
            })
        self._graph.create_claims_with_entities(claims_data, output.article.article_id)

        # Image caption
        if output.image_caption:
            self._graph.create_image_caption(
                caption_id=output.image_caption.caption_id,
                article_id=output.image_caption.article_id,
                image_url=output.image_caption.image_url,
                vlm_caption=output.image_caption.vlm_caption,
            )

        # ── Write to Vector DB ──────────────────────────────────────────
        # Article
        self._vector.upsert_article(
            article_id=output.article.article_id,
            embedding=article_embedding,
            document=output.article.body_snippet,
            source_id=output.article.source_id,
            domain=output.source.category,
            content_hash=output.article.content_hash,
            published_at=output.article.published_at.isoformat(),
        )

        # Claims
        for i, claim in enumerate(output.claims):
            self._vector.upsert_claim(
                claim_id=claim.claim_id,
                embedding=claim_embeddings[i],
                document=claim.claim_text,
                article_id=claim.article_id,
                source_id=output.article.source_id,
                status=claim.status,
                extracted_at=claim.extracted_at.isoformat(),
                topic_text=claim.topic_text,
            )

        # Image caption
        if output.image_caption and caption_embedding:
            self._vector.upsert_caption(
                caption_id=output.image_caption.caption_id,
                embedding=caption_embedding,
                document=output.image_caption.vlm_caption,
                article_id=output.image_caption.article_id,
                image_url=output.image_caption.image_url,
            )

        logger.info(
            "Ingested article %s with %d claims",
            output.article.article_id,
            len(output.claims),
        )
        return True

    # ── Teammate Write Methods ──────────────────────────────────────────

    def add_verdict(self, verdict: Verdict) -> None:
        """Write a fact-check verdict to both databases."""
        # Get the claim text for combined embedding
        claim_results = self._vector.get_claims_by_ids([verdict.claim_id])
        claim_text = ""
        if claim_results["documents"]:
            claim_text = claim_results["documents"][0]

        combined_text = f"{claim_text} [verdict: {verdict.label}]"
        embedding = self._embeddings.embed(combined_text)

        self._vector.upsert_verdict(
            verdict_id=verdict.verdict_id,
            embedding=embedding,
            document=verdict.evidence_summary,
            claim_id=verdict.claim_id,
            label=verdict.label,
            confidence=verdict.confidence,
            bias_score=verdict.bias_score,
            image_mismatch=verdict.image_mismatch,
            verified_at=verdict.verified_at.isoformat(),
        )

        self._graph.create_verdict(
            verdict_id=verdict.verdict_id,
            claim_id=verdict.claim_id,
            label=verdict.label,
            confidence=verdict.confidence,
            evidence_summary=verdict.evidence_summary,
            bias_score=verdict.bias_score,
            image_mismatch=verdict.image_mismatch,
            verified_at=verdict.verified_at,
        )

    def add_credibility_snapshot(self, snapshot: CredibilitySnapshot) -> None:
        """Write a credibility snapshot to Graph DB."""
        self._graph.create_snapshot(
            snapshot_id=snapshot.snapshot_id,
            entity_id=snapshot.entity_id,
            credibility_score=snapshot.credibility_score,
            sentiment_score=snapshot.sentiment_score,
            snapshot_at=snapshot.snapshot_at,
        )

    def update_entity(self, entity_id: str, **updates) -> None:
        """Update fields on an Entity node."""
        self._graph.update_entity(entity_id, updates)

    # ── Entity Reconciliation (post-ingestion) ──────────────────────────

    def reconcile_entities(self) -> dict:
        """Cluster variant entities and merge them.

        Uses 3-tier matching: fuzzy string match + embedding similarity.
        Safe to call multiple times (idempotent when no new variants exist).
        Returns summary dict for logging.
        """
        return self._entity_merger.reconcile()

    def promote_canonical_candidates(self, threshold: int = 10) -> list[dict]:
        """Auto-append entities with >= threshold mentions to the canonical list.

        Must be called AFTER reconcile_entities() so counts reflect merged variants.
        Returns list of promoted entries for logging.
        """
        return self._canonical_promoter.promote(threshold=threshold)

    def add_prediction(self, prediction: Prediction) -> None:
        """Write a prediction to Graph DB."""
        self._graph.create_prediction(
            prediction_id=prediction.prediction_id,
            entity_id=prediction.entity_id,
            prediction_text=prediction.prediction_text,
            confidence=prediction.confidence,
            predicted_at=prediction.predicted_at,
            deadline=prediction.deadline,
            outcome=prediction.outcome,
        )

    def resolve_prediction(self, prediction_id: str, outcome: str) -> None:
        """Resolve an expired prediction."""
        self._graph.resolve_prediction(prediction_id, outcome)

    # ── Query Methods ───────────────────────────────────────────────────

    def search_similar_claims(
        self, text: str, top_k: int = 5
    ) -> dict:
        """Find semantically similar claims in Vector DB."""
        embedding = self._embeddings.embed(text)
        return self._vector.search_similar_claims(embedding, top_k)

    def check_duplicate(self, content_hash: str) -> bool:
        """Check if an article with this hash already exists."""
        return self._vector.check_content_hash_exists(content_hash)

    def find_existing_claim_ids(self, content_hash: str) -> list[str]:
        """Return claim_ids of an article matching this content_hash, or [].

        Used by `decompose_input` to keep the public API idempotent: re-submitting
        the same query returns the same claim_ids instead of creating duplicates.
        """
        article_id = self._vector.get_article_id_by_content_hash(content_hash)
        if not article_id:
            return []
        return self._graph.get_claim_ids_for_article(article_id)

    def get_claims_by_ids(self, ids: list[str]) -> dict:
        return self._vector.get_claims_by_ids(ids)

    def get_caption_by_article(self, article_id: str) -> dict:
        return self._vector.get_caption_by_article(article_id)

    def get_verdict_by_claim(self, claim_id: str) -> dict:
        return self._vector.get_verdict_by_claim(claim_id)

    def get_entity_context(self, claim_id: str) -> list[dict]:
        return self._graph.get_entity_context(claim_id)

    def get_entity_claims(
        self, entity_id: str, since: Optional[datetime] = None
    ) -> list[dict]:
        return self._graph.get_entity_claims(entity_id, since)

    def get_entity_snapshots(
        self, entity_id: str, limit: int = 20
    ) -> list[dict]:
        return self._graph.get_entity_snapshots(entity_id, limit)

    def get_source_credibility(self, article_id: str) -> Optional[float]:
        return self._graph.get_source_credibility(article_id)

    def get_trending_entities(
        self, since: datetime, limit: int = 10
    ) -> list[dict]:
        return self._graph.get_trending_entities(since, limit)

    def get_expired_predictions(self) -> list[dict]:
        return self._graph.get_expired_predictions()

    # ── Extended API (used by FakeNewsAgent and PredictionAgent) ─────────

    def ensure_entity_exists(self, name: str) -> str:
        """Create entity node if it doesn't exist. Returns entity_id."""
        return self._graph.ensure_entity_exists(name)

    def backfill_mentions_for_entity(self, name: str) -> int:
        """Link existing claims that mention this entity by name. Returns link count."""
        entity_id = self._graph.ensure_entity_exists(name)
        return self._graph.backfill_mentions_for_entity(entity_id, name)

    def get_entity_by_name(
        self,
        name: str,
        fuzzy_threshold: int = 85,
    ) -> Optional[dict]:
        """Look up an Entity node by name with alias-aware multi-tier resolution.

        Tier 1 — exact case-insensitive match on Entity.name in Neo4j.
        Tier 2 — canonical alias resolution via data/canonical_entities.json
                 (e.g. "USA" → "United States", "DPRK" → "North Korea").
        Tier 3 — fuzzy string match across all Entity.name values using
                 rapidfuzz.token_sort_ratio with a configurable threshold
                 (default 85 — same threshold used by EntityMerger).

        Returns the matched entity dict (same shape as graph.get_entity_by_name)
        or None if no tier produces a match.

        Backward-compatible: callers that pass only `name` see strictly more
        matches than before; an exact match (Tier 1) returns immediately, so
        existing well-formed inputs incur zero extra cost.

        Notes
        -----
        - Names shorter than 3 chars skip Tier 3 (too short for reliable fuzzy
          matching — would generate many false positives).
        - Tier 3 fetches all Entity nodes from Neo4j once per call. For very
          large graphs (10k+ entities) consider caching upstream.
        """
        if not name or not name.strip():
            return None

        # Tier 1: exact case-insensitive match
        result = self._graph.get_entity_by_name(name)
        if result:
            return result

        # Tier 2: canonical alias resolution (no entity_type known at query time)
        from src.preprocessing.canonical_names import canonicalize_any_type
        canonical = canonicalize_any_type(name)
        if canonical and canonical.lower() != name.strip().lower():
            result = self._graph.get_entity_by_name(canonical)
            if result:
                logger.info("get_entity_by_name: aliased '%s' → '%s'", name, canonical)
                return result

        # Tier 3: fuzzy match against all entity names
        if len(name.strip()) < 3:
            return None

        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            logger.warning("rapidfuzz not available; skipping fuzzy entity lookup")
            return None

        all_entities = self._graph.get_all_entities()
        if not all_entities:
            return None

        # Build name → entity map. Multiple entities may share the same name
        # (rare after EntityMerger runs); we keep the one with the most claims.
        name_to_entity: dict[str, dict] = {}
        for e in all_entities:
            existing = name_to_entity.get(e["name"])
            if existing is None or (e.get("total_claims", 0) >
                                    existing.get("total_claims", 0)):
                name_to_entity[e["name"]] = e

        match = process.extractOne(
            name.strip(),
            list(name_to_entity.keys()),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=fuzzy_threshold,
        )
        if match is None:
            return None

        matched_name, score, _ = match
        result = name_to_entity[matched_name]
        logger.info(
            "get_entity_by_name: fuzzy '%s' → '%s' (score=%.0f)",
            name, matched_name, score,
        )
        return result

    def get_claim_count_for_entity(self, entity_id: str, since: datetime) -> int:
        return self._graph.get_claim_count_for_entity(entity_id, since)

    def get_predictions_for_entity(
        self, entity_id: str, include_resolved: bool = False
    ) -> list[dict]:
        return self._graph.get_predictions_for_entity(entity_id, include_resolved)

    def get_entity_ids_for_claims(self, claim_ids: list[str]) -> list[dict]:
        return self._graph.get_entity_ids_for_claims(claim_ids)

    def get_graph_claims_for_entities(self, entity_ids: list[str]) -> list[dict]:
        return self._graph.get_graph_claims_for_entities(entity_ids)

    def auto_store_claim_with_entities(
        self,
        claim_id: str,
        claim_text: str,
        article_id: str,
        entity_dicts: list[dict],
    ) -> None:
        """Store a claim in ChromaDB + create Claim/Entity nodes in Neo4j.

        Called after every frontend fact-check so entities are tracked
        even when claims come directly from the UI (no preprocessing pipeline).
        """
        try:
            embedding = self._embeddings.embed(claim_text)
            self._vector.upsert_claim(
                claim_id=claim_id,
                embedding=embedding,
                document=claim_text,
                article_id=article_id,
                source_id="src_frontend",
                status="verified",
                extracted_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
            logger.warning("ChromaDB claim upsert skipped: %s", e)
        self._graph.auto_store_claim_with_entities(
            claim_id, claim_text, article_id, entity_dicts
        )

    # ── Reflection Agent read/write (Task 2) ───────────────────────────

    def add_source_credibility_point(
        self,
        point_id: str,
        claim_text: str,
        topic_text: str,
        source_id: str,
        credibility: float,
        bias: float,
        verdict_label: str,
        verdict_id: str,
        created_at: str,
    ) -> None:
        """Append one (source, topic, credibility, bias) observation."""
        embedding = self._embeddings.embed(topic_text)
        self._vector.upsert_source_credibility_point(
            point_id=point_id,
            embedding=embedding,
            document=topic_text,
            source_id=source_id,
            credibility=credibility,
            bias=bias,
            verdict_label=verdict_label,
            verdict_id=verdict_id,
            created_at=created_at,
        )

    def query_source_credibility(
        self, claim_text: str, source_id: str, k: int = 20
    ) -> dict:
        """Retrieve k nearest (source, topic) credibility observations."""
        embedding = self._embeddings.embed(claim_text)
        return self._vector.query_source_credibility(embedding, source_id=source_id, k=k)
