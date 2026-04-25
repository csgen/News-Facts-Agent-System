"""Telegram fetcher — thin HTTP client calling a remote scraper API.

The actual Telegram scraping + image upload logic lives in Google Cloud. This fetcher just calls that
API and converts the JSON response to list[RawArticle].
"""

import logging
from datetime import datetime

import httpx

from src.scraper.fetchers.base import BaseFetcher, RawArticle

logger = logging.getLogger(__name__)


class TelegramFetcher(BaseFetcher):
    def __init__(self, api_url: str, api_key: str = ""):
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key

    def fetch(self, max_results: int = 20) -> list[RawArticle]:
        articles: list[RawArticle] = []

        try:
            with httpx.Client(timeout=60) as client:
                response = client.post(
                    f"{self._api_url}/scrape",
                    json={"max_results": max_results},
                    headers={"X-API-Key": self._api_key},
                )
                response.raise_for_status()
                data = response.json()

            for item in data.get("articles", []):
                title = item.get("title", "").strip()
                body = item.get("body_text", "")

                if not title or not body:
                    continue

                published_at = None
                if item.get("published_at"):
                    try:
                        published_at = datetime.fromisoformat(item["published_at"])
                    except ValueError:
                        pass

                raw = RawArticle(
                    url=item.get("url", ""),
                    title=title,
                    body_text=body,
                    image_urls=item.get("image_urls", []),
                    source_name=item.get("source_name", ""),
                    source_domain=item.get("source_domain", "t.me"),
                    published_at=published_at,
                    content_hash=item.get("content_hash", ""),
                )
                articles.append(raw)

        except httpx.HTTPStatusError as e:
            logger.error("Telegram API HTTP error: %s", e.response.status_code)
        except httpx.ConnectError:
            logger.error("Telegram API unreachable at %s", self._api_url)
        except Exception as e:
            logger.error("Telegram fetch failed: %s", e)

        logger.info("Telegram API returned %d articles", len(articles))
        return articles
