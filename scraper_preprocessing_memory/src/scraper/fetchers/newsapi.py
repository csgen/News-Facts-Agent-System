"""Tavily Search API fetcher for news articles."""

import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

from src.scraper.dedup import compute_content_hash
from src.scraper.fetchers.base import BaseFetcher, RawArticle

logger = logging.getLogger(__name__)

TAVILY_BASE = "https://api.tavily.com"


class TavilyFetcher(BaseFetcher):
    def __init__(self, api_key: str, query: str = "latest news technology politics economy"):
        self._api_key = api_key
        self._query = query

    def fetch(self, max_results: int = 10) -> list[RawArticle]:
        articles: list[RawArticle] = []

        try:
            with httpx.Client(timeout=30) as client:
                response = client.post(
                    f"{TAVILY_BASE}/search",
                    json={
                        "query": self._query,
                        "topic": "news",
                        "search_depth": "basic",
                        "max_results": min(max_results, 20),
                        "include_raw_content": True,
                        "include_images": True,
                        "time_range": "week",
                    },
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                data = response.json()

            for item in data.get("results", []):
                title = item.get("title", "").strip()
                url = item.get("url", "")
                body = item.get("raw_content") or item.get("content") or ""

                if not title or not body:
                    continue

                # Extract image URLs
                image_urls = []
                if item.get("images"):
                    image_urls = [img for img in item["images"] if isinstance(img, str)]

                # published_date is available when topic="news"
                published_at = None
                if item.get("published_date"):
                    try:
                        published_at = datetime.fromisoformat(
                            item["published_date"].replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass
                if published_at is None:
                    published_at = datetime.now(timezone.utc)

                # Derive source info from URL
                source_domain = ""
                source_name = ""
                if url:
                    parsed = urlparse(url)
                    source_domain = parsed.netloc.lstrip("www.")
                    source_name = source_domain.split(".")[0].capitalize()

                raw = RawArticle(
                    url=url,
                    title=title,
                    body_text=body,
                    image_urls=image_urls,
                    source_name=source_name,
                    source_domain=source_domain,
                    published_at=published_at,
                    content_hash=compute_content_hash(title, body),
                )
                articles.append(raw)

        except httpx.HTTPStatusError as e:
            logger.error("Tavily HTTP error: %s", e.response.status_code)
        except Exception as e:
            logger.error("Tavily fetch failed: %s", e)

        logger.info("Tavily fetched %d articles", len(articles))
        return articles
