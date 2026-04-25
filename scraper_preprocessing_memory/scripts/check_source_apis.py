"""Live health check for all configured scraper source APIs.

Unlike the unit tests in `tests/test_scraper/test_fetchers.py` (which mock HTTP
with respx), this script hits the *real* APIs and reports which ones are
healthy right now. Use it when:

- You suspect a source API is down or credentials expired.
- You just changed something in the pipeline and want a quick confidence check.
- You want to confirm a newly-added fetcher actually works end-to-end.

Not run in CI — requires live API keys and costs quota. Run on demand:

    docker-compose exec app python scripts/check_source_apis.py

Each fetcher is classified as:
    HEALTHY  — articles returned, no ERROR logs during the call
    DEGRADED — no articles returned, no errors (empty result set)
    FAILED   — ERROR log emitted or an exception escaped
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Make `src.*` importable regardless of where the script is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.scraper.fetchers.base import BaseFetcher  # noqa: E402
from src.scraper.fetchers.newsapi import TavilyFetcher  # noqa: E402
from src.scraper.fetchers.reddit import RedditFetcher  # noqa: E402
from src.scraper.fetchers.rss import RSSFetcher  # noqa: E402
from src.scraper.fetchers.telegram import TelegramFetcher  # noqa: E402


MAX_RESULTS = 2


class _RecordCapture(logging.Handler):
    """Collects log records emitted during a fetcher call so we can see any
    errors the fetcher reported internally (each fetcher swallows its own
    exceptions and logs them rather than raising)."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _build_fetchers() -> list[tuple[str, BaseFetcher | None, str | None]]:
    """Return (display_name, fetcher_or_None, skip_reason) per source.

    A fetcher is None + skip_reason set when the required credentials are
    missing — we still report the source so it's obvious why it's absent.
    """
    tavily = (
        TavilyFetcher(api_key=settings.tavily_api_key)
        if settings.tavily_api_key else None
    )
    # RSS hits public feeds — no credentials to gate on, always runnable.
    rss = RSSFetcher()
    telegram = (
        TelegramFetcher(
            api_url=settings.telegram_scraper_api_url,
            api_key=settings.telegram_scraper_api_key,
        ) if settings.telegram_scraper_api_url else None
    )
    reddit = (
        RedditFetcher(api_token=settings.ensembledata_api_token)
        if settings.ensembledata_api_token else None
    )

    return [
        ("TavilyFetcher",   tavily,   None if tavily else "TAVILY_API_KEY not set"),
        ("RSSFetcher",      rss,      None),
        ("TelegramFetcher", telegram, None if telegram else "TELEGRAM_SCRAPER_API_URL not set"),
        ("RedditFetcher",   reddit,   None if reddit else "ENSEMBLEDATA_API_TOKEN not set"),
    ]


def _check_one(name: str, fetcher: BaseFetcher) -> tuple[str, int, float, list[str]]:
    """Run the fetcher, capture any error logs, return (status, count, seconds, messages)."""
    capture = _RecordCapture()
    # Attach to the fetcher's own logger (`src.scraper.fetchers.<name>`) AND the
    # root logger as a safety net — whichever the fetcher actually logs to,
    # we'll catch it.
    root = logging.getLogger()
    root.addHandler(capture)
    previous_level = root.level
    root.setLevel(logging.WARNING)

    start = time.monotonic()
    exception_msg: str | None = None
    articles: list = []
    try:
        articles = fetcher.fetch(max_results=MAX_RESULTS)
    except Exception as e:  # defence in depth — fetchers shouldn't raise, but just in case
        exception_msg = f"uncaught {type(e).__name__}: {e}"
    elapsed = time.monotonic() - start

    root.removeHandler(capture)
    root.setLevel(previous_level)

    error_msgs = [r.getMessage() for r in capture.records if r.levelno >= logging.ERROR]
    if exception_msg:
        error_msgs.append(exception_msg)

    if error_msgs:
        status = "FAILED"
    elif len(articles) == 0:
        status = "DEGRADED"
    else:
        status = "HEALTHY"

    return status, len(articles), elapsed, error_msgs


_STATUS_MARK = {"HEALTHY": "[OK]  ", "DEGRADED": "[WARN]", "FAILED": "[FAIL]", "SKIPPED": "[SKIP]"}


def main() -> int:
    print("Source API health check")
    print("=" * 60)

    results: list[tuple[str, str]] = []  # (name, status)

    for name, fetcher, skip_reason in _build_fetchers():
        if fetcher is None:
            print(f"{_STATUS_MARK['SKIPPED']} | {name:<18} | skipped   | ({skip_reason})")
            results.append((name, "SKIPPED"))
            continue

        status, count, elapsed, errors = _check_one(name, fetcher)
        print(f"{_STATUS_MARK[status]} | {name:<18} | {count:>2} articles | {elapsed:>5.1f}s")
        for msg in errors:
            # Keep error messages short — full traceback would already be in the logs.
            print(f"    |- {msg[:180]}")
        results.append((name, status))

    print("=" * 60)
    healthy = sum(1 for _, s in results if s == "HEALTHY")
    total_runnable = sum(1 for _, s in results if s != "SKIPPED")
    print(f"Summary: {healthy}/{total_runnable} sources healthy "
          f"({sum(1 for _, s in results if s == 'SKIPPED')} skipped)")

    # Exit code: 0 if all runnable sources are HEALTHY, else 1 — useful if you
    # ever want to wire this into a cron/Slack alert.
    return 0 if healthy == total_runnable and total_runnable > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
