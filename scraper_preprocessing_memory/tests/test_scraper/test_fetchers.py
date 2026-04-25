"""Tests for news fetchers with mocked HTTP responses."""

import httpx
import respx

from src.scraper.fetchers.newsapi import TavilyFetcher
from src.scraper.fetchers.reddit import RedditFetcher


class TestTavilyFetcher:
    @respx.mock
    def test_fetch_success(self):
        mock_response = {
            "results": [
                {
                    "title": "Test Article Title",
                    "url": "https://reuters.com/test-article",
                    "content": "Short description of the article.",
                    "raw_content": "Full article content here for testing purposes.",
                    "score": 0.95,
                    "images": ["https://example.com/image.jpg"],
                }
            ],
        }

        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        fetcher = TavilyFetcher(api_key="tvly-test_key")
        articles = fetcher.fetch(max_results=10)

        assert len(articles) == 1
        assert articles[0].title == "Test Article Title"
        assert articles[0].source_domain == "reuters.com"
        assert articles[0].body_text == "Full article content here for testing purposes."
        assert len(articles[0].image_urls) == 1
        assert articles[0].content_hash.startswith("sha256_")

    @respx.mock
    def test_fetch_falls_back_to_content(self):
        """When raw_content is missing, should use content field."""
        mock_response = {
            "results": [
                {
                    "title": "Fallback Article",
                    "url": "https://bbc.co.uk/news/test",
                    "content": "Short content used as fallback.",
                    "score": 0.8,
                }
            ],
        }

        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        fetcher = TavilyFetcher(api_key="tvly-test_key")
        articles = fetcher.fetch(max_results=10)

        assert len(articles) == 1
        assert articles[0].body_text == "Short content used as fallback."

    @respx.mock
    def test_fetch_skips_empty_results(self):
        mock_response = {
            "results": [
                {"title": "", "url": "https://example.com", "content": "body"},
                {"title": "No Body", "url": "https://example.com", "content": ""},
            ],
        }

        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        fetcher = TavilyFetcher(api_key="tvly-test_key")
        articles = fetcher.fetch(max_results=10)
        assert len(articles) == 0

    @respx.mock
    def test_fetch_handles_api_error(self):
        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(429, json={"error": "rate limited"})
        )

        fetcher = TavilyFetcher(api_key="tvly-test_key")
        articles = fetcher.fetch(max_results=10)
        assert len(articles) == 0


# --- RedditFetcher helpers --------------------------------------------------


REDDIT_ENDPOINT = "https://ensembledata.com/apis/reddit/subreddit/posts"


def _make_reddit_post(
    *,
    title: str,
    selftext: str,
    upvote_ratio: float = 0.7,
    score: int = 10,
    is_self: bool = True,
    subreddit: str = "AskConservatives",
    permalink: str | None = None,
    created_utc: float = 1_776_948_435.0,
) -> dict:
    """Build a single Reddit post in EnsembleData's nested envelope format."""
    slug = title.lower().replace(" ", "_")[:30]
    return {
        "kind": "t3",
        "data": {
            "title": title,
            "selftext": selftext,
            "is_self": is_self,
            "upvote_ratio": upvote_ratio,
            "score": score,
            "subreddit": subreddit,
            "permalink": permalink or f"/r/{subreddit}/comments/abc123/{slug}/",
            "created_utc": created_utc,
        },
    }


def _make_reddit_payload(posts: list[dict]) -> dict:
    return {"data": {"posts": posts, "nextCursor": None}}


class TestRedditFetcher:
    @respx.mock
    def test_fetch_success(self):
        long_body = "a" * 150  # > MIN_SELFTEXT_LEN (100)
        posts = [
            _make_reddit_post(title=f"Post {i}", selftext=long_body,
                              upvote_ratio=0.5 + 0.05 * i, score=i)
            for i in range(6)
        ]
        respx.get(REDDIT_ENDPOINT).mock(
            return_value=httpx.Response(200, json=_make_reddit_payload(posts))
        )

        fetcher = RedditFetcher(api_token="test_token")
        articles = fetcher.fetch()

        # Top-5 cap per subreddit, ordered by upvote_ratio desc.
        assert len(articles) == 5
        assert articles[0].title == "Post 5"  # highest upvote_ratio
        assert articles[0].source_domain == "reddit.com"
        assert articles[0].source_name == "r/AskConservatives"
        assert articles[0].image_urls == []
        assert articles[0].url.startswith("https://reddit.com/r/AskConservatives/")
        assert articles[0].content_hash.startswith("sha256_")
        assert articles[0].published_at is not None

    @respx.mock
    def test_fetch_filters_out_non_self_posts(self):
        long_body = "b" * 200
        posts = [
            _make_reddit_post(title="Self post", selftext=long_body, is_self=True),
            _make_reddit_post(title="Link post", selftext=long_body, is_self=False),
        ]
        respx.get(REDDIT_ENDPOINT).mock(
            return_value=httpx.Response(200, json=_make_reddit_payload(posts))
        )

        fetcher = RedditFetcher(api_token="test_token")
        articles = fetcher.fetch()

        assert len(articles) == 1
        assert articles[0].title == "Self post"

    @respx.mock
    def test_fetch_filters_out_short_selftext(self):
        posts = [
            _make_reddit_post(title="Long enough", selftext="c" * 150),
            _make_reddit_post(title="Exactly at threshold", selftext="d" * 100),  # not > 100
            _make_reddit_post(title="Too short", selftext="short"),
        ]
        respx.get(REDDIT_ENDPOINT).mock(
            return_value=httpx.Response(200, json=_make_reddit_payload(posts))
        )

        fetcher = RedditFetcher(api_token="test_token")
        articles = fetcher.fetch()

        assert len(articles) == 1
        assert articles[0].title == "Long enough"

    @respx.mock
    def test_fetch_caps_at_top_5_per_subreddit(self):
        long_body = "e" * 150
        # 10 qualifying posts; only top 5 by upvote_ratio should survive.
        posts = [
            _make_reddit_post(title=f"Post {i}", selftext=long_body,
                              upvote_ratio=0.1 * i, score=i)
            for i in range(1, 11)
        ]
        respx.get(REDDIT_ENDPOINT).mock(
            return_value=httpx.Response(200, json=_make_reddit_payload(posts))
        )

        fetcher = RedditFetcher(api_token="test_token")
        articles = fetcher.fetch()

        assert len(articles) == 5
        titles = [a.title for a in articles]
        assert titles == ["Post 10", "Post 9", "Post 8", "Post 7", "Post 6"]

    @respx.mock
    def test_fetch_handles_api_error(self):
        respx.get(REDDIT_ENDPOINT).mock(
            return_value=httpx.Response(500, json={"error": "server error"})
        )

        fetcher = RedditFetcher(api_token="test_token")
        articles = fetcher.fetch()

        assert articles == []

    @respx.mock
    def test_fetch_tiebreaks_on_score(self):
        long_body = "f" * 150
        posts = [
            _make_reddit_post(title="Low score", selftext=long_body,
                              upvote_ratio=0.8, score=5),
            _make_reddit_post(title="High score", selftext=long_body,
                              upvote_ratio=0.8, score=50),
        ]
        respx.get(REDDIT_ENDPOINT).mock(
            return_value=httpx.Response(200, json=_make_reddit_payload(posts))
        )

        fetcher = RedditFetcher(api_token="test_token")
        articles = fetcher.fetch()

        assert len(articles) == 2
        assert articles[0].title == "High score"
        assert articles[1].title == "Low score"
