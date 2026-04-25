"""Article and Source data models for Graph DB and Vector DB."""

from datetime import datetime

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A news source (e.g. Reuters, BBC, Reddit)."""

    source_id: str
    name: str
    domain: str
    category: str  # "wire_service", "social_media", "news_outlet", etc.
    base_credibility: float = Field(ge=0.0, le=1.0)


class Article(BaseModel):
    """A scraped news article."""

    article_id: str
    title: str
    url: str
    source_id: str
    published_at: datetime
    ingested_at: datetime
    content_hash: str  # SHA-256 for exact deduplication
    body_snippet: str = ""  # title + first 500 chars for Vector DB document
