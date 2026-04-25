"""Two-stage entity extraction: spaCy NER + LLM refinement (batched)."""

import json
import logging

import spacy
from langfuse.decorators import observe
from langfuse.openai import OpenAI

from src.id_utils import make_entity_id
from src.preprocessing.canonical_names import canonicalize
from src.preprocessing.prompts import ENTITY_EXTRACTION_BATCH_PROMPT

logger = logging.getLogger(__name__)

BATCH_CHUNK_SIZE = 3


class EntityExtractor:
    def __init__(self, api_key: str, model: str):
        # langfuse.openai auto-instruments; no manual generation tagging needed.
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._nlp = spacy.load("en_core_web_sm")

    @observe(name="entity_extraction_batch")
    def extract_entities_batch(
        self,
        claims: list[dict],
        article_context: str,
    ) -> list[list[dict]]:
        """Extract entities for all claims, chunked into groups of 3 per LLM call.

        Args:
            claims: list of {"text": str, "type": str} dicts
            article_context: the full article body text

        Returns:
            list of entity lists, one per claim (same order as input).
            Each entity is {"entity_id", "name", "entity_type", "sentiment"}.
        """
        if not claims:
            return []

        # Stage 1: spaCy NER per claim (fast, free) — ONLY on claim text
        # Running NER on claim + article_context caused the LLM to extract
        # article-wide entities for every claim (>100 entities per claim).
        all_candidates = []
        for claim in claims:
            candidates = self._spacy_ner(claim["text"])
            all_candidates.append(candidates)

        # Stage 2: LLM refinement in chunks of BATCH_CHUNK_SIZE
        all_results: list[list[dict]] = []
        for i in range(0, len(claims), BATCH_CHUNK_SIZE):
            chunk_claims = claims[i : i + BATCH_CHUNK_SIZE]
            chunk_candidates = all_candidates[i : i + BATCH_CHUNK_SIZE]

            chunk_results = self._llm_refine_batch(
                chunk_claims, chunk_candidates, article_context
            )

            # If chunk fails, fall back to spaCy candidates
            if chunk_results is None:
                chunk_results = [
                    [{**c, "sentiment": "neutral"} for c in candidates]
                    for candidates in chunk_candidates
                ]

            all_results.extend(chunk_results)

        # Canonicalize names + generate deterministic entity IDs
        # This unifies known aliases (USA → United States) into a single entity_id
        # so Neo4j MERGE treats them as the same node.
        for claim_entities in all_results:
            for entity in claim_entities:
                canonical = canonicalize(entity["name"], entity["entity_type"])
                entity["name"] = canonical
                entity["entity_id"] = make_entity_id(
                    canonical, entity["entity_type"]
                )

        return all_results

    def _spacy_ner(self, text: str) -> list[dict]:
        """Run spaCy NER to get candidate entities."""
        doc = self._nlp(text)
        candidates = []
        seen = set()

        label_map = {
            "PERSON": "person",
            "ORG": "organization",
            "GPE": "country",
            "LOC": "location",
            "EVENT": "event",
            "PRODUCT": "product",
            "NORP": "organization",
            "FAC": "location",
        }

        for ent in doc.ents:
            if ent.label_ in label_map and ent.text.strip() not in seen:
                seen.add(ent.text.strip())
                candidates.append({
                    "name": ent.text.strip(),
                    "entity_type": label_map[ent.label_],
                })

        return candidates

    def _llm_refine_batch(
        self,
        claims: list[dict],
        all_candidates: list[list[dict]],
        article_context: str,
    ) -> list[list[dict]] | None:
        """Use a single LLM call to refine entities for all claims."""
        # Build the claims + candidates block for the prompt
        claims_block = []
        for i, (claim, candidates) in enumerate(zip(claims, all_candidates)):
            claims_block.append(
                f"Claim {i}: \"{claim['text']}\"\n"
                f"NER candidates: {json.dumps(candidates)}"
            )
        claims_text = "\n\n".join(claims_block)

        prompt = ENTITY_EXTRACTION_BATCH_PROMPT.format(
            claims_with_candidates=claims_text,
            article_context=article_context[:500],
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            claim_results = parsed.get("claims", [])

            # Build result list indexed by claim position
            results: list[list[dict]] = [[] for _ in claims]
            for item in claim_results:
                idx = item.get("claim_index", -1)
                if 0 <= idx < len(claims):
                    for e in item.get("entities", []):
                        if isinstance(e, dict) and "name" in e and "entity_type" in e:
                            results[idx].append({
                                "name": e["name"],
                                "entity_type": e["entity_type"],
                                "sentiment": e.get("sentiment", "neutral"),
                            })

            logger.info("Batch entity extraction: %d claims processed in 1 LLM call", len(claims))
            return results

        except Exception as e:
            logger.error("Batch LLM entity refinement failed: %s", e)
            return None
