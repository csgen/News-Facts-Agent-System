"""Neo4j wrapper for the Knowledge Graph."""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class GraphStore:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", uri)

    def close(self) -> None:
        self._driver.close()

    # ── Schema Initialization ───────────────────────────────────────────

    def init_schema(self) -> None:
        """Create uniqueness constraints and indexes. Idempotent."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.source_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.article_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Claim) REQUIRE c.claim_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Verdict) REQUIRE v.verdict_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ic:ImageCaption) REQUIRE ic.caption_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (cs:CredibilitySnapshot) REQUIRE cs.snapshot_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Prediction) REQUIRE p.prediction_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (c:Claim) ON (c.extracted_at)",
            "CREATE INDEX IF NOT EXISTS FOR (v:Verdict) ON (v.verified_at)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Prediction) ON (p.deadline)",
        ]
        with self._driver.session() as session:
            for stmt in constraints + indexes:
                session.run(stmt)
        logger.info("Neo4j schema initialized (9 constraints, 3 indexes)")

    # ── Write: Sources ──────────────────────────────────────────────────

    def merge_source(
        self,
        source_id: str,
        name: str,
        domain: str,
        category: str,
        base_credibility: float,
    ) -> None:
        """MERGE a source node (upsert — avoids duplicates for known outlets)."""
        with self._driver.session() as session:
            session.run(
                """
                MERGE (s:Source {source_id: $source_id})
                SET s.name = $name,
                    s.domain = $domain,
                    s.category = $category,
                    s.base_credibility = $base_credibility
                """,
                source_id=source_id,
                name=name,
                domain=domain,
                category=category,
                base_credibility=base_credibility,
            )

    # ── Write: Articles ─────────────────────────────────────────────────

    def create_article(
        self,
        article_id: str,
        title: str,
        url: str,
        source_id: str,
        published_at: datetime,
        ingested_at: datetime,
        content_hash: str,
    ) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (s:Source {source_id: $source_id})
                CREATE (a:Article {
                    article_id: $article_id,
                    title: $title,
                    url: $url,
                    source_id: $source_id,
                    published_at: datetime($published_at),
                    ingested_at: datetime($ingested_at),
                    content_hash: $content_hash
                })
                CREATE (s)-[:PUBLISHES]->(a)
                """,
                article_id=article_id,
                title=title,
                url=url,
                source_id=source_id,
                published_at=published_at.isoformat(),
                ingested_at=ingested_at.isoformat(),
                content_hash=content_hash,
            )

    # ── Write: Claims with Entities ─────────────────────────────────────

    def create_claims_with_entities(
        self,
        claims: list[dict[str, Any]],
        article_id: str,
    ) -> None:
        """Create Claim nodes, Entity nodes (MERGE), and relationship edges.

        Each claim dict should have: claim_id, claim_text, claim_type,
        extracted_at, status, and entities (list of {entity_id, name,
        entity_type, sentiment}).
        """
        with self._driver.session() as session:
            for claim in claims:
                # Create Claim node + CONTAINS edge from Article
                session.run(
                    """
                    MATCH (a:Article {article_id: $article_id})
                    CREATE (c:Claim {
                        claim_id: $claim_id,
                        article_id: $article_id,
                        claim_text: $claim_text,
                        claim_type: $claim_type,
                        topic_text: $topic_text,
                        extracted_at: datetime($extracted_at),
                        status: $status
                    })
                    CREATE (a)-[:CONTAINS]->(c)
                    """,
                    article_id=article_id,
                    claim_id=claim["claim_id"],
                    claim_text=claim["claim_text"],
                    claim_type=claim.get("claim_type", ""),
                    topic_text=claim.get("topic_text", ""),
                    extracted_at=claim["extracted_at"],
                    status=claim.get("status", "pending"),
                )

                # MERGE entities + create MENTIONS edges
                for entity in claim.get("entities", []):
                    session.run(
                        """
                        MATCH (c:Claim {claim_id: $claim_id})
                        MERGE (e:Entity {entity_id: $entity_id})
                        ON CREATE SET
                            e.name = $name,
                            e.entity_type = $entity_type,
                            e.current_credibility = 0.5,
                            e.total_claims = 0,
                            e.accurate_claims = 0,
                            e.first_seen = datetime($extracted_at),
                            e.last_seen = datetime($extracted_at)
                        ON MATCH SET
                            e.last_seen = datetime($extracted_at)
                        CREATE (c)-[:MENTIONS {sentiment: $sentiment}]->(e)
                        """,
                        claim_id=claim["claim_id"],
                        entity_id=entity["entity_id"],
                        name=entity["name"],
                        entity_type=entity["entity_type"],
                        sentiment=entity["sentiment"],
                        extracted_at=claim["extracted_at"],
                    )

    # ── Write: Image Captions ───────────────────────────────────────────

    def create_image_caption(
        self,
        caption_id: str,
        article_id: str,
        image_url: str,
        vlm_caption: str,
    ) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (a:Article {article_id: $article_id})
                CREATE (ic:ImageCaption {
                    caption_id: $caption_id,
                    article_id: $article_id,
                    image_url: $image_url,
                    vlm_caption: $vlm_caption
                })
                CREATE (a)-[:HAS_IMAGE]->(ic)
                """,
                caption_id=caption_id,
                article_id=article_id,
                image_url=image_url,
                vlm_caption=vlm_caption,
            )

    # ── Write: Verdicts (called by Fact-Check Agent) ────────────────────

    def create_verdict(
        self,
        verdict_id: str,
        claim_id: str,
        label: str,
        confidence: float,
        evidence_summary: str,
        image_mismatch: bool,
        verified_at: datetime,
    ) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MERGE (c:Claim {claim_id: $claim_id})
                ON CREATE SET
                    c.claim_text   = '',
                    c.extracted_at = datetime(),
                    c.status       = 'pending'
                MERGE (v:Verdict {verdict_id: $verdict_id})
                ON CREATE SET
                    v.claim_id         = $claim_id,
                    v.label            = $label,
                    v.confidence       = $confidence,
                    v.evidence_summary = $evidence_summary,
                    v.image_mismatch   = $image_mismatch,
                    v.verified_at      = datetime($verified_at)
                ON MATCH SET
                    v.label            = $label,
                    v.confidence       = $confidence,
                    v.evidence_summary = $evidence_summary
                MERGE (c)-[:VERIFIED_AS]->(v)
                SET c.status = 'verified'
                """,
                verdict_id=verdict_id,
                claim_id=claim_id,
                label=label,
                confidence=confidence,
                evidence_summary=evidence_summary,
                image_mismatch=image_mismatch,
                verified_at=verified_at.isoformat(),
            )

    def supersede_verdict(self, old_verdict_id: str, new_verdict_id: str) -> None:
        """Mark an old VERDICT node as superseded by a newer one.

        Sets old.status = 'superseded' and creates a (old)-[:SUPERSEDED_BY]->(new) edge.
        Idempotent — repeated calls leave the same end state thanks to MERGE on the edge.
        """
        with self._driver.session() as session:
            session.run(
                """
                MATCH (old:Verdict {verdict_id: $old_id})
                MATCH (new:Verdict {verdict_id: $new_id})
                SET old.status = 'superseded'
                MERGE (old)-[:SUPERSEDED_BY]->(new)
                """,
                old_id=old_verdict_id,
                new_id=new_verdict_id,
            )

    # ── Write: Credibility Snapshots (called by Entity Tracker) ─────────

    def create_snapshot(
        self,
        snapshot_id: str,
        entity_id: str,
        credibility_score: float,
        sentiment_score: float,
        snapshot_at: datetime,
    ) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (e:Entity {entity_id: $entity_id})
                CREATE (s:CredibilitySnapshot {
                    snapshot_id: $snapshot_id,
                    entity_id: $entity_id,
                    credibility_score: $credibility_score,
                    sentiment_score: $sentiment_score,
                    snapshot_at: datetime($snapshot_at)
                })
                CREATE (e)-[:TRACKED_OVER_TIME]->(s)
                """,
                snapshot_id=snapshot_id,
                entity_id=entity_id,
                credibility_score=credibility_score,
                sentiment_score=sentiment_score,
                snapshot_at=snapshot_at.isoformat(),
            )

    # ── Write: Entity updates (called by Entity Tracker) ────────────────

    def update_entity(self, entity_id: str, updates: dict[str, Any]) -> None:
        """Update specific fields on an Entity node."""
        set_clauses = ", ".join(f"e.{k} = ${k}" for k in updates)
        with self._driver.session() as session:
            session.run(
                f"MATCH (e:Entity {{entity_id: $entity_id}}) SET {set_clauses}",
                entity_id=entity_id,
                **updates,
            )

    # ── Entity reconciliation (called by entity_merger) ─────────────────

    def get_all_entities(self) -> list[dict]:
        """Fetch all Entity nodes with their aggregate fields.

        Used by the entity_merger to cluster variants and determine
        merge targets.
        """
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                RETURN e.entity_id AS entity_id,
                       e.name AS name,
                       e.entity_type AS entity_type,
                       coalesce(e.total_claims, 0) AS total_claims,
                       coalesce(e.accurate_claims, 0) AS accurate_claims,
                       coalesce(e.current_credibility, 0.5) AS current_credibility
                """
            )
            return [dict(record) for record in result]

    def count_entity_mentions(self, entity_id: str) -> int:
        """Count how many claims mention this entity (live count from MENTIONS edges)."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (:Claim)-[m:MENTIONS]->(e:Entity {entity_id: $entity_id})
                RETURN count(m) AS mention_count
                """,
                entity_id=entity_id,
            )
            record = result.single()
            return record["mention_count"] if record else 0

    def merge_entity(self, source_id: str, target_id: str) -> None:
        """Merge source Entity into target Entity.

        Rewires all edges from source to target, sums aggregate fields,
        updates first_seen/last_seen bounds, then deletes source.
        Safe no-op if source_id == target_id.
        """
        if source_id == target_id:
            return

        with self._driver.session() as session:
            # Rewire CLAIM-[:MENTIONS]->source to CLAIM-[:MENTIONS]->target
            # (preserving sentiment property on the edge)
            session.run(
                """
                MATCH (c:Claim)-[m:MENTIONS]->(s:Entity {entity_id: $source_id})
                MATCH (t:Entity {entity_id: $target_id})
                CREATE (c)-[:MENTIONS {sentiment: m.sentiment}]->(t)
                DELETE m
                """,
                source_id=source_id,
                target_id=target_id,
            )

            # Rewire source-[:TRACKED_OVER_TIME]->snapshot to target
            session.run(
                """
                MATCH (s:Entity {entity_id: $source_id})-[r:TRACKED_OVER_TIME]->(snap:CredibilitySnapshot)
                MATCH (t:Entity {entity_id: $target_id})
                CREATE (t)-[:TRACKED_OVER_TIME]->(snap)
                DELETE r
                SET snap.entity_id = $target_id
                """,
                source_id=source_id,
                target_id=target_id,
            )

            # Rewire source-[:SUBJECT_OF]->prediction to target
            session.run(
                """
                MATCH (s:Entity {entity_id: $source_id})-[r:SUBJECT_OF]->(p:Prediction)
                MATCH (t:Entity {entity_id: $target_id})
                CREATE (t)-[:SUBJECT_OF]->(p)
                DELETE r
                SET p.entity_id = $target_id
                """,
                source_id=source_id,
                target_id=target_id,
            )

            # Sum aggregate fields and update first_seen/last_seen bounds, then delete source
            session.run(
                """
                MATCH (s:Entity {entity_id: $source_id})
                MATCH (t:Entity {entity_id: $target_id})
                SET t.total_claims = coalesce(t.total_claims, 0) + coalesce(s.total_claims, 0),
                    t.accurate_claims = coalesce(t.accurate_claims, 0) + coalesce(s.accurate_claims, 0),
                    t.first_seen = CASE
                        WHEN s.first_seen IS NULL THEN t.first_seen
                        WHEN t.first_seen IS NULL THEN s.first_seen
                        WHEN s.first_seen < t.first_seen THEN s.first_seen
                        ELSE t.first_seen
                    END,
                    t.last_seen = CASE
                        WHEN s.last_seen IS NULL THEN t.last_seen
                        WHEN t.last_seen IS NULL THEN s.last_seen
                        WHEN s.last_seen > t.last_seen THEN s.last_seen
                        ELSE t.last_seen
                    END
                DETACH DELETE s
                """,
                source_id=source_id,
                target_id=target_id,
            )

    # ── Write: Predictions (called by Prediction Agent) ─────────────────

    def create_prediction(
        self,
        prediction_id: str,
        entity_id: str,
        prediction_text: str,
        confidence: float,
        predicted_at: datetime,
        deadline: datetime,
        outcome: Optional[str] = None,
    ) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (e:Entity {entity_id: $entity_id})
                CREATE (p:Prediction {
                    prediction_id: $prediction_id,
                    entity_id: $entity_id,
                    prediction_text: $prediction_text,
                    confidence: $confidence,
                    predicted_at: datetime($predicted_at),
                    deadline: datetime($deadline),
                    outcome: $outcome
                })
                CREATE (e)-[:SUBJECT_OF]->(p)
                """,
                prediction_id=prediction_id,
                entity_id=entity_id,
                prediction_text=prediction_text,
                confidence=confidence,
                predicted_at=predicted_at.isoformat(),
                deadline=deadline.isoformat(),
                outcome=outcome,
            )

    def resolve_prediction(self, prediction_id: str, outcome: str) -> None:
        with self._driver.session() as session:
            session.run(
                "MATCH (p:Prediction {prediction_id: $prediction_id}) SET p.outcome = $outcome",
                prediction_id=prediction_id,
                outcome=outcome,
            )

    # ── Read: Entity context ────────────────────────────────────────────

    def get_entity_context(self, claim_id: str) -> list[dict]:
        """Get entities mentioned in a claim with their credibility."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (c:Claim {claim_id: $claim_id})-[m:MENTIONS]->(e:Entity)
                RETURN e.entity_id AS entity_id,
                       e.name AS name,
                       e.entity_type AS entity_type,
                       e.current_credibility AS current_credibility,
                       m.sentiment AS sentiment
                """,
                claim_id=claim_id,
            )
            return [dict(record) for record in result]

    def get_entity_claims(
        self,
        entity_id: str,
        since: Optional[datetime] = None,
    ) -> list[dict]:
        """Get claims mentioning an entity, optionally filtered by time.

        Superseded verdicts are filtered out. Legacy verdicts (without a
        status property) are treated as active.
        """
        query = """
            MATCH (e:Entity {entity_id: $entity_id})<-[m:MENTIONS]-(c:Claim)
            -[:VERIFIED_AS]->(v:Verdict)
            WHERE (v.status IS NULL OR v.status = 'active')
        """
        params: dict[str, Any] = {"entity_id": entity_id}

        if since:
            query += " AND v.verified_at > datetime($since)"
            params["since"] = since.isoformat()

        query += """
            RETURN c.claim_id AS claim_id,
                   c.claim_text AS claim_text,
                   v.label AS verdict_label,
                   v.confidence AS verdict_confidence,
                   m.sentiment AS sentiment
            ORDER BY v.verified_at DESC
        """
        with self._driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    def get_entity_snapshots(
        self, entity_id: str, limit: int = 20
    ) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {entity_id: $entity_id})
                      -[:TRACKED_OVER_TIME]->(s:CredibilitySnapshot)
                RETURN s.snapshot_id AS snapshot_id,
                       s.credibility_score AS credibility_score,
                       s.sentiment_score AS sentiment_score,
                       s.snapshot_at AS snapshot_at
                ORDER BY s.snapshot_at DESC
                LIMIT $limit
                """,
                entity_id=entity_id,
                limit=limit,
            )
            return [dict(record) for record in result]

    def get_source_credibility(self, article_id: str) -> Optional[float]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (s:Source)-[:PUBLISHES]->(a:Article {article_id: $article_id})
                RETURN s.base_credibility AS base_credibility
                """,
                article_id=article_id,
            )
            record = result.single()
            return record["base_credibility"] if record else None

    def get_base_credibility(self, source_id: str) -> Optional[float]:
        """Return the static base_credibility of a Source node, or None if unknown."""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (s:Source {source_id: $source_id}) RETURN s.base_credibility AS v",
                source_id=source_id,
            )
            record = result.single()
            return float(record["v"]) if record and record["v"] is not None else None

    def update_claim_status(self, claim_id: str, status: str) -> None:
        """Update the status field on an existing Claim node."""
        with self._driver.session() as session:
            session.run(
                "MATCH (c:Claim {claim_id: $claim_id}) SET c.status = $status",
                claim_id=claim_id, status=status,
            )

    def get_topic_for_verdict(self, verdict_id: str) -> str:
        """Return the topic_text of the Claim linked to this Verdict, or '' if not found."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (c:Claim)-[:VERIFIED_AS]->(v:Verdict {verdict_id: $verdict_id})
                RETURN c.topic_text AS topic_text
                """,
                verdict_id=verdict_id,
            )
            record = result.single()
            return (record["topic_text"] or "") if record else ""

    def get_source_topic_credibility(self, source_id: str, topic: str) -> Optional[float]:
        """Return current credibility for a (source, topic) pair, or None if no record exists."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (s:Source {source_id: $source_id})-[r:HAS_CREDIBILITY]->(t:Topic {name: $topic})
                RETURN r.credibility AS credibility
                """,
                source_id=source_id, topic=topic,
            )
            record = result.single()
            return float(record["credibility"]) if record else None

    def upsert_source_topic_credibility(
        self, source_id: str, topic: str, credibility: float
    ) -> None:
        """Create or update the (Source)-[:HAS_CREDIBILITY]->(Topic) relationship.

        MERGE on Source and Topic ensures new topics are created automatically.
        The Source node is created bare if it doesn't exist yet (scraped sources
        will already have it; unknown sources get one without base_credibility).
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._driver.session() as session:
            session.run(
                """
                MERGE (s:Source {source_id: $source_id})
                MERGE (t:Topic {name: $topic})
                MERGE (s)-[r:HAS_CREDIBILITY]->(t)
                SET r.credibility = $credibility,
                    r.last_updated = datetime($now)
                """,
                source_id=source_id, topic=topic,
                credibility=credibility, now=now,
            )

    def get_trending_entities(
        self, since: datetime, limit: int = 10
    ) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)<-[:MENTIONS]-(c:Claim)
                WHERE c.extracted_at > datetime($since)
                RETURN e.entity_id AS entity_id,
                       e.name AS name,
                       e.entity_type AS entity_type,
                       e.current_credibility AS current_credibility,
                       count(c) AS mention_count
                ORDER BY mention_count DESC
                LIMIT $limit
                """,
                since=since.isoformat(),
                limit=limit,
            )
            return [dict(record) for record in result]

    def get_expired_predictions(self) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (p:Prediction)
                WHERE p.deadline < datetime() AND p.outcome IS NULL
                RETURN p.prediction_id AS prediction_id,
                       p.entity_id AS entity_id,
                       p.prediction_text AS prediction_text,
                       p.confidence AS confidence,
                       p.deadline AS deadline
                """
            )
            return [dict(record) for record in result]

    # ── Extended reads (added for Entity Tracker, Prediction Agent, frontend) ──

    def get_entity_by_name(self, name: str) -> Optional[dict]:
        """Look up an Entity node by its name field."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) = toLower($name)
                RETURN e.entity_id AS entity_id,
                       e.name AS name,
                       e.entity_type AS entity_type,
                       e.current_credibility AS current_credibility,
                       e.total_claims AS total_claims,
                       e.accurate_claims AS accurate_claims,
                       e.first_seen AS first_seen,
                       e.last_seen AS last_seen
                LIMIT 1
                """,
                name=name,
            )
            record = result.single()
            return dict(record) if record else None

    def ensure_entity_exists(self, name: str) -> str:
        """Create an Entity node if one with this name doesn't already exist.

        Returns the entity_id (either existing or newly created).
        Safe to call multiple times — MERGE is idempotent.
        """
        entity_id = f"ent_{name.strip().lower().replace(' ', '_')}"
        now = datetime.now(timezone.utc).isoformat()
        with self._driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {entity_id: $entity_id})
                ON CREATE SET
                    e.name                = $name,
                    e.entity_type         = 'unknown',
                    e.current_credibility = 0.5,
                    e.total_claims        = 0,
                    e.accurate_claims     = 0,
                    e.first_seen          = datetime($now),
                    e.last_seen           = datetime($now)
                ON MATCH SET
                    e.last_seen = datetime($now)
                """,
                entity_id=entity_id, name=name.strip(), now=now,
            )
        return entity_id

    def backfill_mentions_for_entity(self, entity_id: str, name: str) -> int:
        """Create MENTIONS links from any existing Claim whose text contains
        the entity name (case-insensitive). Returns the number of links created.
        """
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (c:Claim)
                WHERE toLower(replace(replace(c.claim_text, '_', ' '), '-', ' ')) CONTAINS toLower($name)
                MATCH (e:Entity {entity_id: $entity_id})
                MERGE (c)-[:MENTIONS {sentiment: 'neutral'}]->(e)
                RETURN count(c) AS linked
                """,
                entity_id=entity_id, name=name,
            )
            linked = result.single()["linked"]
            return linked

    def get_claim_count_for_entity(self, entity_id: str, since: datetime) -> int:
        """Count claims mentioning an entity since a given datetime."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {entity_id: $entity_id})<-[:MENTIONS]-(c:Claim)
                WHERE c.extracted_at > datetime($since)
                RETURN count(c) AS claim_count
                """,
                entity_id=entity_id, since=since.isoformat(),
            )
            record = result.single()
            return record["claim_count"] if record else 0

    def get_predictions_for_entity(self, entity_id: str, include_resolved: bool = False) -> list[dict]:
        """Get Prediction nodes linked to an entity."""
        where_clause = "" if include_resolved else "WHERE p.outcome IS NULL"
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (e:Entity {{entity_id: $entity_id}})-[:SUBJECT_OF]->(p:Prediction)
                {where_clause}
                RETURN p.prediction_id AS prediction_id,
                       p.prediction_text AS prediction_text,
                       p.confidence AS confidence,
                       p.predicted_at AS predicted_at,
                       p.deadline AS deadline,
                       p.outcome AS outcome
                ORDER BY p.predicted_at DESC
                """,
                entity_id=entity_id,
            )
            return [dict(record) for record in result]

    def get_entity_ids_for_claims(self, claim_ids: list[str]) -> list[dict]:
        """Return entity_id for all entities mentioned by a list of claims (for GraphRAG)."""
        if not claim_ids:
            return []
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (c:Claim)-[:MENTIONS]->(e:Entity)
                WHERE c.claim_id IN $claim_ids
                RETURN DISTINCT e.entity_id AS entity_id
                """,
                claim_ids=claim_ids,
            )
            return [dict(record) for record in result]

    def get_graph_claims_for_entities(self, entity_ids: list[str]) -> list[dict]:
        """Return verified claims mentioning a list of entities (for GraphRAG).

        Superseded verdicts are filtered out. Legacy verdicts (without a
        status property) are treated as active.
        """
        if not entity_ids:
            return []
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)<-[m:MENTIONS]-(c:Claim)-[:VERIFIED_AS]->(v:Verdict)
                WHERE e.entity_id IN $entity_ids
                  AND (v.status IS NULL OR v.status = 'active')
                RETURN c.claim_id AS claim_id,
                       c.claim_text AS claim_text,
                       v.label AS verdict_label,
                       v.confidence AS verdict_confidence,
                       v.verified_at AS verified_at,
                       0.0 AS distance
                ORDER BY v.verified_at DESC
                LIMIT 20
                """,
                entity_ids=entity_ids,
            )
            return [dict(record) for record in result]

    def get_claim_ids_for_article(self, article_id: str) -> list[str]:
        """Return all claim_ids attached to a given article (via CONTAINS edge)."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (a:Article {article_id: $article_id})-[:CONTAINS]->(c:Claim)
                RETURN c.claim_id AS claim_id
                """,
                article_id=article_id,
            )
            return [record["claim_id"] for record in result]

    def get_article_url_by_id(self, article_id: str) -> Optional[str]:
        """Return the URL stored on the Article node, or None if missing.

        Used by the fact-check adapter to populate FactCheckInput.source_url
        when reconstructing inputs from claim_ids alone.
        """
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (a:Article {article_id: $article_id})
                RETURN a.url AS url
                LIMIT 1
                """,
                article_id=article_id,
            )
            record = result.single()
            return record["url"] if record and record["url"] else None

    # ── Feature: Source Bias ────────────────────────────────────────────

    def get_source_bias_for_entity(self, entity_name: str) -> list[dict]:
        """Return per-source claim stats for a given entity."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (s:Source)-[:PUBLISHES]->(a:Article)
                      -[:CONTAINS]->(c:Claim)-[:MENTIONS]->(e:Entity)
                WHERE toLower(e.name) = toLower($entity_name)
                OPTIONAL MATCH (c)-[:VERIFIED_AS]->(v:Verdict)
                WITH s.name AS source_name,
                     count(DISTINCT c) AS claim_count,
                     avg(v.confidence)  AS avg_confidence,
                     sum(CASE WHEN v.label = 'supported'  THEN 1 ELSE 0 END) AS supported,
                     sum(CASE WHEN v.label = 'refuted'    THEN 1 ELSE 0 END) AS refuted,
                     sum(CASE WHEN v.label = 'misleading' THEN 1 ELSE 0 END) AS misleading
                WHERE claim_count > 0
                RETURN source_name, claim_count, avg_confidence,
                       supported, refuted, misleading
                ORDER BY claim_count DESC
                LIMIT 10
                """,
                entity_name=entity_name,
            )
            return [dict(r) for r in result]

    # ── Feature: Human Feedback / Verdict Override ──────────────────────

    def update_verdict_with_feedback(
        self,
        verdict_id: str,
        correct_label: str,
        correct_confidence: float,
        feedback_note: str = "",
    ) -> None:
        """Override a verdict with human-corrected label and confidence."""
        with self._driver.session() as session:
            session.run(
                """
                MATCH (v:Verdict {verdict_id: $verdict_id})
                SET v.label              = $label,
                    v.confidence         = $confidence,
                    v.human_feedback     = true,
                    v.feedback_note      = $note,
                    v.corrected_at       = datetime()
                """,
                verdict_id=verdict_id,
                label=correct_label,
                confidence=correct_confidence,
                note=feedback_note,
            )
            print(f"[graph_store] verdict {verdict_id} overridden → {correct_label} ({correct_confidence:.0%})")

    def auto_store_claim_with_entities(
        self,
        claim_id: str,
        claim_text: str,
        article_id: str,
        entity_dicts: list[dict],
    ) -> None:
        """Create/merge a Claim node and link it to entities.

        Used when claims arrive directly from the frontend (no Article node exists).
        Safe to call multiple times — MERGE is idempotent.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._driver.session() as session:
            session.run(
                """
                MERGE (c:Claim {claim_id: $claim_id})
                ON CREATE SET
                    c.claim_text  = $text,
                    c.article_id  = $article_id,
                    c.extracted_at = datetime($now),
                    c.status      = 'verified'
                ON MATCH SET
                    c.status = 'verified'
                """,
                claim_id=claim_id, text=claim_text,
                article_id=article_id, now=now,
            )
            for entity in entity_dicts:
                session.run(
                    """
                    MATCH (c:Claim {claim_id: $claim_id})
                    MERGE (e:Entity {entity_id: $entity_id})
                    ON CREATE SET
                        e.name               = $name,
                        e.entity_type        = $entity_type,
                        e.current_credibility = 0.5,
                        e.total_claims       = 0,
                        e.accurate_claims    = 0,
                        e.first_seen         = datetime($now),
                        e.last_seen          = datetime($now)
                    ON MATCH SET
                        e.last_seen = datetime($now)
                    MERGE (c)-[:MENTIONS {sentiment: 'neutral'}]->(e)
                    """,
                    claim_id=claim_id,
                    entity_id=entity["entity_id"],
                    name=entity["name"],
                    entity_type=entity["entity_type"],
                    now=now,
                )
