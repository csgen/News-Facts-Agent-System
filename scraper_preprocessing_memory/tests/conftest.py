"""Shared test fixtures."""

from datetime import datetime, timezone

import pytest

from src.models.article import Article, Source
from src.models.caption import ImageCaption
from src.models.claim import Claim, MentionSentiment
from src.models.pipeline import PreprocessingOutput
from src.scraper.fetchers.base import RawArticle


@pytest.fixture
def sample_source():
    return Source(
        source_id="src_reuters_com",
        name="Reuters",
        domain="reuters.com",
        category="wire_service",
        base_credibility=0.95,
    )


@pytest.fixture
def sample_article():
    now = datetime.now(timezone.utc)
    return Article(
        article_id="art_test12345678",
        title="Tesla Recalls 500,000 Vehicles Over Brake Defects",
        url="https://reuters.com/test-article",
        source_id="src_reuters_com",
        published_at=now,
        ingested_at=now,
        content_hash="sha256_abc123def456",
        body_snippet="Tesla Recalls 500,000 Vehicles Over Brake Defects. Tesla Inc has issued a recall...",
    )


@pytest.fixture
def sample_claim():
    return Claim(
        claim_id="clm_test12345678",
        article_id="art_test12345678",
        claim_text="500,000 Tesla vehicles were recalled due to brake defects",
        claim_type="statistical",
        extracted_at=datetime.now(timezone.utc),
        status="pending",
        entities=[
            MentionSentiment(
                entity_id="ent_aabbccdd1122",
                name="Tesla",
                entity_type="organization",
                sentiment="negative",
            ),
            MentionSentiment(
                entity_id="ent_eeff00112233",
                name="NHTSA",
                entity_type="organization",
                sentiment="neutral",
            ),
        ],
    )


@pytest.fixture
def sample_caption():
    return ImageCaption(
        caption_id="cap_test12345678",
        article_id="art_test12345678",
        image_url="https://example.com/tesla-recall.jpg",
        vlm_caption="A parking lot filled with white Tesla Model 3 vehicles parked in rows.",
    )


@pytest.fixture
def sample_preprocessing_output(sample_source, sample_article, sample_claim, sample_caption):
    return PreprocessingOutput(
        source=sample_source,
        article=sample_article,
        claims=[sample_claim],
        image_caption=sample_caption,
    )


@pytest.fixture
def sample_raw_article():
    return RawArticle(
        url="https://reuters.com/test-article",
        title="Tesla Recalls 500,000 Vehicles Over Brake Defects",
        body_text="Tesla Inc has issued a voluntary recall of approximately 500,000 vehicles...",
        image_urls=["https://example.com/tesla-recall.jpg"],
        source_name="Reuters",
        source_domain="reuters.com",
        published_at=datetime.now(timezone.utc),
        content_hash="sha256_abc123def456",
    )
