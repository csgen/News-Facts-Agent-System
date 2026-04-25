"""Preprocessing Agent orchestrator — transforms raw articles into structured data."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import Settings
from src.id_utils import make_id
from src.models.article import Article, Source
from src.models.caption import ImageCaption
from src.models.claim import Claim, MentionSentiment
from src.models.pipeline import PreprocessingOutput
from src.preprocessing.caption_generator import CaptionGenerator
from src.preprocessing.claim_isolator import ClaimIsolator
from src.preprocessing.entity_extractor import EntityExtractor
from src.preprocessing.text_cleaner import clean_body_text
from src.scraper.fetchers.base import RawArticle

logger = logging.getLogger(__name__)


# ── SOURCE_CATEGORIES — loaded once from data/sources.json ──────────────────
# Single source of truth. The JSON is also used by scripts/seed_sources.py to
# MERGE Source nodes into Neo4j; here we re-use it for the per-ingest credibility
# lookup so seeding and ingestion can never disagree.
#
# Shape: { domain: (category, base_credibility) }
# Domains not in this dict default to ("unknown", 0.50) at lookup time.

_SOURCES_JSON_PATH = Path(__file__).resolve().parents[2] / "data" / "sources.json"


def _load_source_categories() -> dict[str, tuple[str, float]]:
    try:
        entries = json.loads(_SOURCES_JSON_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("sources.json not found at %s; SOURCE_CATEGORIES is empty",
                       _SOURCES_JSON_PATH)
        return {}
    except json.JSONDecodeError as e:
        logger.error("sources.json is malformed (%s); SOURCE_CATEGORIES is empty", e)
        return {}

    out: dict[str, tuple[str, float]] = {}
    for entry in entries:
        domain = entry.get("domain")
        if not domain:
            continue
        out[domain] = (
            entry.get("category", "unknown"),
            float(entry.get("base_credibility", 0.50)),
        )
    return out


SOURCE_CATEGORIES = _load_source_categories()


class PreprocessingAgent:
    def __init__(self, settings: Settings):
        self._claim_isolator = ClaimIsolator(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
        )
        self._entity_extractor = EntityExtractor(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
        )
        self._caption_generator = CaptionGenerator(
            api_key=settings.openai_api_key,
            model=settings.vision_model,
        )

    def process(self, raw: RawArticle) -> PreprocessingOutput:
        """Transform a RawArticle into structured PreprocessingOutput."""
        now = datetime.now(timezone.utc)

        # ── Clean body text ─────────────────────────────────────────────
        # Tavily's raw_content includes navigation/boilerplate Markdown. Strip
        # it here once so all downstream LLM calls use clean text.
        cleaned_body = clean_body_text(raw.body_text)

        # ── Build Source ────────────────────────────────────────────────
        category, base_credibility = SOURCE_CATEGORIES.get(
            raw.source_domain, ("unknown", 0.50)
        )
        source_id = f"src_{raw.source_domain.replace('.', '_')}"
        source = Source(
            source_id=source_id,
            name=raw.source_name or raw.source_domain,
            domain=raw.source_domain,
            category=category,
            base_credibility=base_credibility,
        )

        # ── Build Article ───────────────────────────────────────────────
        article_id = make_id("art_")
        body_snippet = f"{raw.title}. {cleaned_body[:500]}"
        article = Article(
            article_id=article_id,
            title=raw.title,
            url=raw.url,
            source_id=source_id,
            published_at=raw.published_at or now,
            ingested_at=now,
            content_hash=raw.content_hash,
            body_snippet=body_snippet,
        )

        # ── Extract Claims ──────────────────────────────────────────────
        raw_claims = self._claim_isolator.extract_claims(raw.title, cleaned_body)

        # ── Extract Entities (batched — single LLM call for all claims) ─
        all_entities = self._entity_extractor.extract_entities_batch(
            claims=raw_claims,
            article_context=cleaned_body,
        )

        claims: list[Claim] = []
        for i, raw_claim in enumerate(raw_claims):
            entities_data = all_entities[i] if i < len(all_entities) else []

            mentions = [
                MentionSentiment(
                    entity_id=e["entity_id"],
                    name=e["name"],
                    entity_type=e["entity_type"],
                    sentiment=e["sentiment"],
                )
                for e in entities_data
            ]

            claim = Claim(
                claim_id=make_id("clm_"),
                article_id=article_id,
                claim_text=raw_claim["text"],
                claim_type=raw_claim.get("type"),
                extracted_at=now,
                status="pending",
                entities=mentions,
            )
            claims.append(claim)

        # ── Generate Image Caption ──────────────────────────────────────
        image_caption = None
        if raw.image_urls:
            caption_text = self._caption_generator.generate_caption(
                raw.image_urls[0]
            )
            if caption_text:
                image_caption = ImageCaption(
                    caption_id=make_id("cap_"),
                    article_id=article_id,
                    image_url=raw.image_urls[0],
                    vlm_caption=caption_text,
                )

        logger.info(
            "Preprocessed article '%s': %d claims, %s caption",
            raw.title[:60],
            len(claims),
            "with" if image_caption else "no",
        )

        return PreprocessingOutput(
            source=source,
            article=article,
            claims=claims,
            image_caption=image_caption,
        )
