"""Reddit fetcher — calls EnsembleData's subreddit-posts endpoint.

Replaces the Telegram fetcher as the project's "unverified UGC / rumor channel"
source. Scope decisions (validated empirically in the exploratory test script):

- Only self-posts are kept (is_self=True). Link posts duplicate Tavily/RSS and
  create source-attribution ambiguity.
- selftext must exceed MIN_SELFTEXT_LEN so the claim-extraction step has real
  content to work with.
- Per subreddit we keep only the top TOP_N_PER_SUBREDDIT posts ranked by
  (upvote_ratio, score) — scale-invariant and deterministic across subreddit
  sizes.
- Images are dropped (image_urls = []). Reddit thumbnails add little over the
  article's own images and re-hosting adds complexity.
"""

import logging
from datetime import datetime, timezone

import httpx

from src.scraper.dedup import compute_content_hash
from src.scraper.fetchers.base import BaseFetcher, RawArticle

logger = logging.getLogger(__name__)


class RedditFetcher(BaseFetcher):
    ENDPOINT = "https://ensembledata.com/apis/reddit/subreddit/posts"
    DEFAULT_SUBREDDITS = ["AskConservatives"]
    TOP_N_PER_SUBREDDIT = 5
    MIN_SELFTEXT_LEN = 100

    def __init__(self, api_token: str, subreddits: list[str] | None = None):
        self._api_token = api_token
        self._subreddits = subreddits or list(self.DEFAULT_SUBREDDITS)

    def fetch(self, max_results: int = 20) -> list[RawArticle]:
        """Fetch top self-posts from each configured subreddit.

        The ``max_results`` parameter is accepted to satisfy the BaseFetcher
        contract but intentionally ignored — this fetcher caps output at
        ``TOP_N_PER_SUBREDDIT`` per subreddit by design.
        """
        articles: list[RawArticle] = []

        for subreddit in self._subreddits:
            try:
                posts = self._fetch_subreddit(subreddit)
            except httpx.HTTPStatusError as e:
                # Include a snippet of the response body so it's obvious what the
                # API rejected (bad token, invalid subreddit, rate limit, etc.).
                logger.error(
                    "Reddit API HTTP %s for r/%s: %s",
                    e.response.status_code,
                    subreddit,
                    e.response.text[:200],
                )
                continue
            except httpx.ConnectError as e:
                logger.error("Reddit API unreachable for r/%s: %s", subreddit, e)
                continue
            except Exception as e:
                logger.error("Reddit fetch failed for r/%s: %s", subreddit, e)
                continue

            qualifying = [
                p for p in posts
                if p.get("is_self") is True
                and len(p.get("selftext") or "") > self.MIN_SELFTEXT_LEN
            ]
            qualifying.sort(
                key=lambda p: (p.get("upvote_ratio", 0.0), p.get("score", 0)),
                reverse=True,
            )

            kept_before = len(articles)
            for post in qualifying[: self.TOP_N_PER_SUBREDDIT]:
                raw = self._to_raw_article(post)
                if raw is not None:
                    articles.append(raw)
            logger.info(
                "Reddit r/%s: fetched=%d qualifying=%d kept=%d",
                subreddit,
                len(posts),
                len(qualifying),
                len(articles) - kept_before,
            )

        logger.info(
            "Reddit fetcher returned %d articles across %d subreddit(s)",
            len(articles),
            len(self._subreddits),
        )
        return articles

    def _fetch_subreddit(self, subreddit: str) -> list[dict]:
        """Call the subreddit-posts endpoint and unwrap the posts list."""
        with httpx.Client(timeout=30) as client:
            response = client.get(
                self.ENDPOINT,
                params={
                    "name": subreddit,
                    "sort": "top",
                    "period": "day",
                    "cursor": "",
                    "token": self._api_token,
                },
            )
            response.raise_for_status()
            payload = response.json()

        data = payload.get("data") or {}
        wrapped_posts = data.get("posts") or []
        # Each post is {"kind": "t3", "data": {...}} — unwrap to the inner dict.
        return [item.get("data", {}) for item in wrapped_posts if isinstance(item, dict)]

    @staticmethod
    def _to_raw_article(post: dict) -> RawArticle | None:
        title = (post.get("title") or "").strip()
        body = post.get("selftext") or ""
        if not title or not body:
            return None

        permalink = post.get("permalink") or ""
        url = f"https://reddit.com{permalink}" if permalink else ""

        published_at = None
        created_utc = post.get("created_utc")
        if isinstance(created_utc, (int, float)):
            try:
                published_at = datetime.fromtimestamp(created_utc, tz=timezone.utc)
            except (ValueError, OSError):
                pass

        subreddit = post.get("subreddit") or ""
        source_name = f"r/{subreddit}" if subreddit else "reddit"

        return RawArticle(
            url=url,
            title=title,
            body_text=body,
            image_urls=[],
            source_name=source_name,
            source_domain="reddit.com",
            published_at=published_at,
            content_hash=compute_content_hash(title, body),
        )
