"""Base fetcher interface and RawArticle intermediate type."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class RawArticle:
    """Raw article data before preprocessing.

    This is the intermediate format between Scraper and Preprocessing.
    """

    url: str
    title: str
    body_text: str
    image_urls: list[str] = field(default_factory=list)
    source_name: str = ""
    source_domain: str = ""
    published_at: Optional[datetime] = None
    content_hash: str = ""  # Set after hashing


class BaseFetcher(ABC):
    """Abstract base class for news source fetchers."""

    @abstractmethod
    def fetch(self, max_results: int = 20) -> list[RawArticle]:
        """Fetch articles from the source.

        Returns a list of RawArticle objects. Implementations should
        handle their own errors and return partial results on failure.
        """
        ...
