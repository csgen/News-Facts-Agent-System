"""Scraper Agent orchestrator — runs all fetchers, deduplicates, returns novel articles."""

import logging

from src.config import Settings
from src.scraper.fetchers.base import BaseFetcher, RawArticle
from src.scraper.fetchers.newsapi import TavilyFetcher
from src.scraper.fetchers.reddit import RedditFetcher
from src.scraper.fetchers.rss import RSSFetcher
from src.scraper.fetchers.telegram import TelegramFetcher

logger = logging.getLogger(__name__)


class ScraperAgent:
    def __init__(self, settings: Settings):
        self._fetchers: list[BaseFetcher] = []

        # Tavily Search API
        if settings.tavily_api_key:
            self._fetchers.append(TavilyFetcher(api_key=settings.tavily_api_key))

        # RSS feeds (no API key needed)
        self._fetchers.append(RSSFetcher())

        # Telegram (remote API — credentials isolated)
        if settings.telegram_scraper_api_url:
            self._fetchers.append(
                TelegramFetcher(
                    api_url=settings.telegram_scraper_api_url,
                    api_key=settings.telegram_scraper_api_key,
                )
            )

        # Reddit via EnsembleData (subreddit-posts endpoint)
        if settings.ensembledata_api_token:
            self._fetchers.append(
                RedditFetcher(api_token=settings.ensembledata_api_token)
            )

    def scrape(self, max_per_source: int = 20) -> list[RawArticle]:
        """Run all fetchers and return deduplicated articles.

        Each fetcher is run independently — one failure does not block others.
        Deduplication is done via content_hash within this batch.
        """
        all_articles: list[RawArticle] = []
        seen_hashes: set[str] = set()

        for fetcher in self._fetchers:
            fetcher_name = type(fetcher).__name__
            try:
                articles = fetcher.fetch(max_results=max_per_source)
                for article in articles:
                    if article.content_hash not in seen_hashes:
                        seen_hashes.add(article.content_hash)
                        all_articles.append(article)
            except Exception as e:
                logger.error("Fetcher %s failed: %s", fetcher_name, e)

        logger.info(
            "Scraper completed: %d unique articles from %d fetchers",
            len(all_articles),
            len(self._fetchers),
        )
        return all_articles
