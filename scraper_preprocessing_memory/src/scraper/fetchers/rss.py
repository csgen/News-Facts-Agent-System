"""RSS/Atom feed fetcher for high-credibility sources (BBC, Reuters, AP)."""

import logging
from email.utils import parsedate_to_datetime

import feedparser
import httpx

from src.scraper.dedup import compute_content_hash
from src.scraper.fetchers.base import BaseFetcher, RawArticle

logger = logging.getLogger(__name__)

# Default RSS feeds — high-credibility baseline sources
DEFAULT_FEEDS = [
    {
        "url": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "name": "BBC News",
        "domain": "bbc.co.uk",
    },
    {
        "url": "https://feeds.reuters.com/reuters/topNews",
        "name": "Reuters",
        "domain": "reuters.com",
    },
    {
        "url": "https://rss.app/feeds/v1.1/ts45Yh3f6oLbREhx.xml",
        "name": "AP News",
        "domain": "apnews.com",
    },
]


class RSSFetcher(BaseFetcher):
    def __init__(self, feeds: list[dict] | None = None):
        self._feeds = feeds or DEFAULT_FEEDS

    def fetch(self, max_results: int = 10) -> list[RawArticle]:
        articles: list[RawArticle] = []
        per_feed = max(max_results // len(self._feeds), 3)

        for feed_config in self._feeds:
            try:
                feed_articles = self._fetch_feed(feed_config, per_feed)
                articles.extend(feed_articles)
            except Exception as e:
                logger.error("RSS fetch failed for %s: %s", feed_config["name"], e)

        logger.info("RSS fetched %d articles total", len(articles))
        return articles[:max_results]

    def _fetch_feed(self, feed_config: dict, max_items: int) -> list[RawArticle]:
        parsed = feedparser.parse(feed_config["url"])
        articles: list[RawArticle] = []

        for entry in parsed.entries[:max_items]:
            title = entry.get("title", "")
            url = entry.get("link", "")
            if not title or not url:
                continue

            # Get body from feed summary
            body = entry.get("summary", "") or entry.get("description", "")

            # Try to extract full text via httpx if body is too short
            if len(body) < 100:
                body = self._extract_body(url) or body

            if not body:
                continue

            # Parse published date
            published_at = None
            if entry.get("published"):
                try:
                    published_at = parsedate_to_datetime(entry["published"])
                except Exception:
                    pass

            # Extract image URLs from media content
            image_urls = []
            for media in entry.get("media_content", []):
                if media.get("url") and media.get("medium") == "image":
                    image_urls.append(media["url"])
            # Also check enclosures
            for enc in entry.get("enclosures", []):
                if enc.get("type", "").startswith("image/") and enc.get("href"):
                    image_urls.append(enc["href"])

            raw = RawArticle(
                url=url,
                title=title,
                body_text=body,
                image_urls=image_urls,
                source_name=feed_config["name"],
                source_domain=feed_config["domain"],
                published_at=published_at,
                content_hash=compute_content_hash(title, body),
            )
            articles.append(raw)

        return articles

    def _extract_body(self, url: str) -> str:
        """Try to extract article body using newspaper3k, fallback to httpx."""
        try:
            from newspaper import Article as NewspaperArticle

            article = NewspaperArticle(url)
            article.download()
            article.parse()
            return article.text
        except Exception:
            pass

        try:
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                resp = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                # Return raw text — will be truncated by preprocessing
                return resp.text[:2000]
        except Exception:
            return ""
