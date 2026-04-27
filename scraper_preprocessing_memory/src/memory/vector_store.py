"""ChromaDB wrapper managing all 4 collections."""

import logging

import chromadb

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        api_key: str = "",
        tenant: str = "",
        database: str = "",
        host: str = "",
        port: int = 8000,
    ):
        if api_key:
            # ChromaDB Cloud
            self._client = chromadb.CloudClient(
                api_key=api_key,
                tenant=tenant,
                database=database,
            )
        elif host:
            # Remote ChromaDB reachable over HTTP (e.g. local Docker container)
            self._client = chromadb.HttpClient(host=host, port=int(port))
        else:
            # Persistent local ChromaDB — survives process restarts
            import os
            _chroma_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "..", "chroma_data"
            )
            _chroma_path = os.path.normpath(_chroma_path)
            os.makedirs(_chroma_path, exist_ok=True)
            logger.info("ChromaDB using persistent local store at: %s", _chroma_path)
            self._client = chromadb.PersistentClient(path=_chroma_path)

        self._claims = self._client.get_or_create_collection("claims")
        self._articles = self._client.get_or_create_collection("articles")
        self._verdicts = self._client.get_or_create_collection("verdicts")
        self._image_captions = self._client.get_or_create_collection("image_captions")
        self._source_credibility = self._client.get_or_create_collection("source_credibility")

    # ── Claims ──────────────────────────────────────────────────────────

    def upsert_claim(
        self,
        claim_id: str,
        embedding: list[float],
        document: str,
        article_id: str,
        source_id: str,
        status: str,
        extracted_at: str,
        topic_text: str = "",
    ) -> None:
        self._claims.upsert(
            ids=[claim_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[{
                "article_id": article_id,
                "source_id": source_id,
                "status": status,
                "extracted_at": extracted_at,
                "topic_text": topic_text,
            }],
        )

    def update_claim_status(self, claim_id: str, status: str) -> None:
        """Patch the status metadata field on an existing claim — no re-embedding needed."""
        self._claims.update(ids=[claim_id], metadatas=[{"status": status}])

    def search_similar_claims(
        self, query_embedding: list[float], top_k: int = 5
    ) -> dict:
        """Cosine similarity search on claims collection."""
        return self._claims.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    def get_claims_by_ids(self, ids: list[str]) -> dict:
        if not ids:
            return {"ids": [], "documents": [], "metadatas": []}
        return self._claims.get(ids=ids)

    # ── Articles ────────────────────────────────────────────────────────

    def upsert_article(
        self,
        article_id: str,
        embedding: list[float],
        document: str,
        source_id: str,
        domain: str,
        content_hash: str,
        published_at: str,
    ) -> None:
        self._articles.upsert(
            ids=[article_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[{
                "source_id": source_id,
                "domain": domain,
                "content_hash": content_hash,
                "published_at": published_at,
            }],
        )

    def check_content_hash_exists(self, content_hash: str) -> bool:
        """Check if an article with this hash already exists."""
        results = self._articles.get(
            where={"content_hash": content_hash},
            limit=1,
        )
        return len(results["ids"]) > 0

    def get_article_id_by_content_hash(self, content_hash: str):
        """Return the article_id for an existing content_hash, or None."""
        results = self._articles.get(
            where={"content_hash": content_hash},
            limit=1,
        )
        if results["ids"]:
            return results["ids"][0]
        return None

    # ── Verdicts ────────────────────────────────────────────────────────

    def upsert_verdict(
        self,
        verdict_id: str,
        embedding: list[float],
        document: str,
        claim_id: str,
        label: str,
        confidence: float,
        image_mismatch: bool,
        verified_at: str,
    ) -> None:
        self._verdicts.upsert(
            ids=[verdict_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[{
                "claim_id": claim_id,
                "label": label,
                "confidence": confidence,
                "image_mismatch": image_mismatch,
                "verified_at": verified_at,
                "status": "active",
            }],
        )

    def supersede_verdict(self, old_verdict_id: str, new_verdict_id: str) -> None:
        """Mark an existing verdict as superseded by a newer one.

        Only the listed metadata keys are modified; existing fields (label,
        confidence, etc.) survive. Idempotent — calling twice with the same
        old/new pair leaves the same end state.
        """
        self._verdicts.update(
            ids=[old_verdict_id],
            metadatas=[{
                "status": "superseded",
                "superseded_by": new_verdict_id,
            }],
        )

    def get_verdict_by_claim(self, claim_id: str) -> dict:
        # Filter out superseded verdicts. The `$ne: "superseded"` clause matches
        # both rows that explicitly say active and legacy rows written before
        # the status field existed (where the field is absent).
        return self._verdicts.get(
            where={
                "$and": [
                    {"claim_id": claim_id},
                    {"status": {"$ne": "superseded"}},
                ]
            }
        )

    def update_verdict_metadata(self, verdict_id: str, label: str, confidence: float) -> None:
        """Patch label + confidence on an existing verdict in ChromaDB (human feedback)."""
        existing = self._verdicts.get(ids=[verdict_id], include=["metadatas", "embeddings", "documents"])
        if not existing["ids"]:
            return
        meta = dict(existing["metadatas"][0])
        meta["label"]          = label
        meta["confidence"]     = confidence
        meta["human_feedback"] = True
        self._verdicts.upsert(
            ids=[verdict_id],
            embeddings=[existing["embeddings"][0]],
            documents=[existing["documents"][0]],
            metadatas=[meta],
        )

    def find_human_verdict_by_embedding(self, embedding: list[float], threshold: float = 0.70) -> dict | None:
        """Search verdicts for a human-corrected entry similar to the given embedding.

        Searches top-5 nearest verdicts, then checks metadata for human_feedback flag.
        Avoids ChromaDB boolean `where` filters (unreliable across versions).
        Returns the best match dict (with label, confidence, claim_id) or None.
        """
        try:
            results = self._verdicts.query(
                query_embeddings=[embedding],
                n_results=5,
                include=["metadatas", "distances"],
            )
            if not results["ids"] or not results["ids"][0]:
                return None

            for i, (rid, meta, dist) in enumerate(zip(
                results["ids"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                # Check human_feedback — stored as bool True or string "true"
                hf = meta.get("human_feedback")
                is_human = hf is True or str(hf).lower() == "true"
                if not is_human:
                    continue

                # ChromaDB default uses L2 (squared Euclidean): 0=identical
                # Convert to 0-1 similarity: closer to 1 = more similar
                similarity = 1.0 / (1.0 + dist)
                if similarity >= threshold:
                    result = dict(meta)
                    result["verdict_id"] = rid
                    result["_similarity"] = round(similarity, 3)
                    return result

        except Exception as e:
            import logging as _log
            _log.getLogger(__name__).warning("find_human_verdict_by_embedding failed: %s", e)
        return None

    # ── Image Captions ──────────────────────────────────────────────────

    def upsert_caption(
        self,
        caption_id: str,
        embedding: list[float],
        document: str,
        article_id: str,
        image_url: str,
    ) -> None:
        self._image_captions.upsert(
            ids=[caption_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[{
                "article_id": article_id,
                "image_url": image_url,
            }],
        )

    def get_caption_by_article(self, article_id: str) -> dict:
        return self._image_captions.get(where={"article_id": article_id})

    # ── Source Credibility (HITL write path from frontend) ──────────────

    def upsert_source_credibility_point(
        self,
        point_id: str,
        embedding: list[float],
        document: str,
        source_id: str,
        credibility: float,
        verdict_label: str,
        verdict_id: str,
        created_at: str,
    ) -> None:
        """Append a (source, topic, credibility) observation from HITL feedback."""
        self._source_credibility.upsert(
            ids=[point_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[{
                "source_id": source_id,
                "credibility": credibility,
                "verdict_label": verdict_label,
                "verdict_id": verdict_id,
                "created_at": created_at,
            }],
        )

    def query_source_credibility(
        self,
        query_embedding: list[float],
        source_id: str,
        k: int = 20,
    ) -> dict:
        """Retrieve k nearest (source, topic) observations for a given source."""
        return self._source_credibility.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where={"source_id": source_id},
        )

