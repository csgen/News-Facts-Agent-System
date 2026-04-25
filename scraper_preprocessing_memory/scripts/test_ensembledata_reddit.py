"""Exploratory test script for EnsembleData's Reddit keyword-search endpoint.

Goal: empirically discover the response schema and decide whether the data is
sufficient to populate our RawArticle contract (url, title, body_text,
image_urls, source_name, source_domain, published_at).

Usage:
    # Keyword mode (default): run with default keywords:
    python scripts/test_ensembledata_reddit.py

    # Keyword mode with a custom keyword:
    python scripts/test_ensembledata_reddit.py "ukraine"

    # Subreddit mode with defaults (r/news, r/worldnews):
    python scripts/test_ensembledata_reddit.py --subreddit

    # Subreddit mode with specific subreddits:
    python scripts/test_ensembledata_reddit.py --subreddit news worldnews technology

Requires ENSEMBLEDATA_API_TOKEN set in .env.

This is NOT a pytest test — it makes real HTTP calls and writes sample JSON
under scripts/ensembledata_samples/ (gitignored) for manual inspection.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# Make `src.config` importable when the script is run from either the repo root
# or the scraper_preprocessing_memory directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import settings  # noqa: E402

KEYWORD_ENDPOINT = "https://ensembledata.com/apis/reddit/keyword/search"
SUBREDDIT_ENDPOINT = "https://ensembledata.com/apis/reddit/subreddit/posts"
DEFAULT_KEYWORDS = ["technology"]
# News-ish subreddits worth probing for fact-checkable content.
DEFAULT_SUBREDDITS = ["AskConservatives"]
SAMPLES_DIR = _PROJECT_ROOT / "scripts" / "ensembledata_samples"

# Candidate keys to probe for each RawArticle field. Ordered by likelihood —
# the first match wins in the fit report.
RAW_ARTICLE_CANDIDATES: dict[str, list[str]] = {
    "url": ["url", "permalink", "link", "post_url"],
    "title": ["title", "post_title"],
    "body_text": ["selftext", "body", "text", "content", "self_text"],
    "image_urls": ["preview", "media", "thumbnail", "images", "url_overridden_by_dest"],
    "source_name": ["subreddit", "subreddit_name_prefixed", "subreddit_prefixed", "author"],
    # source_domain is always "reddit.com" for Reddit posts — static.
    "published_at": ["created_utc", "created", "created_at", "timestamp"],
}


def fetch_keyword(keyword: str, token: str) -> dict[str, Any]:
    """Call the EnsembleData Reddit keyword-search endpoint."""
    with httpx.Client(timeout=30) as client:
        response = client.get(
            KEYWORD_ENDPOINT,
            params={"name": keyword,
                    "sort": "top",
                    "period": "day",
                    "cursor": "",
                    "token": token},
        )
        response.raise_for_status()
        return response.json()


def fetch_subreddit(subreddit: str, token: str) -> dict[str, Any]:
    """Call the EnsembleData Reddit subreddit-posts endpoint."""
    with httpx.Client(timeout=30) as client:
        response = client.get(
            SUBREDDIT_ENDPOINT,
            params={"name": subreddit,
                    "sort": "top",
                    "period": "day",
                    "cursor": "",
                    "token": token},
        )
        response.raise_for_status()
        return response.json()


def save_sample(label: str, payload: dict[str, Any], mode: str = "kw") -> Path:
    """Dump the full raw response to disk for offline inspection.

    `mode` is a short prefix ("kw" or "sub") so keyword and subreddit runs
    don't overwrite each other in the samples directory.
    """
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = label.replace(" ", "_").replace("/", "_")
    out_path = SAMPLES_DIR / f"{mode}_{safe_label}_{stamp}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def extract_result_list(payload: dict[str, Any]) -> list[Any]:
    """EnsembleData typically wraps results under 'data' — probe a few shapes."""
    if isinstance(payload, list):
        return payload
    for key in ("data", "results", "posts", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        # Sometimes the payload is nested: {"data": {"posts": [...]}}
        if isinstance(value, dict):
            for inner_key in ("posts", "results", "items", "data", "children"):
                inner = value.get(inner_key)
                if isinstance(inner, list):
                    return inner
    return []


def truncate(value: Any, width: int = 60) -> str:
    """Produce a single-line truncated preview of a value."""
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    text = text.replace("\n", " ").replace("\r", " ")
    return text if len(text) <= width else text[: width - 3] + "..."


def find_candidate(record: dict[str, Any], candidates: list[str]) -> tuple[str | None, Any]:
    """Return (first_matching_key, value) or (None, None) if no candidate is present."""
    for key in candidates:
        if key in record and record[key] not in (None, "", [], {}):
            return key, record[key]
    return None, None


def print_fit_report(record: dict[str, Any]) -> None:
    """Print a RawArticle-field-by-field report against the first result."""
    header_field = "RawArticle field"
    header_candidate = "Matched key"
    header_sample = "Sample value (truncated)"
    print(f"  {header_field:<18} | {header_candidate:<28} | {header_sample}")
    print(f"  {'-' * 18} | {'-' * 28} | {'-' * 40}")

    for field, candidates in RAW_ARTICLE_CANDIDATES.items():
        matched_key, value = find_candidate(record, candidates)
        if matched_key is None:
            print(f"  {field:<18} | {'<NOT FOUND>':<28} | (missing — tried: {', '.join(candidates)})")
        else:
            print(f"  {field:<18} | {matched_key:<28} | {truncate(value)}")

    # source_domain is static for Reddit.
    print(f"  {'source_domain':<18} | {'(static)':<28} | reddit.com")


def print_pagination_hints(payload: dict[str, Any]) -> None:
    """Surface any cursor / has_more / next-page hints."""
    hints = {}
    for key in ("nextCursor", "next_cursor", "cursor", "has_more", "has_next", "next"):
        if key in payload:
            hints[key] = payload[key]
    if hints:
        print(f"  pagination hints: {hints}")
    else:
        print("  pagination hints: none found at top level")


def summarize(label: str, payload: dict[str, Any], mode: str = "keyword") -> None:
    print(f"\n=== {mode.capitalize()}: {label!r} ===")

    if isinstance(payload, dict):
        top_keys = sorted(payload.keys())
        print(f"Top-level keys ({len(top_keys)}): {top_keys}")
        print(f"Top-level types: {[f'{k}:{type(payload[k]).__name__}' for k in top_keys]}")
    else:
        print(f"Top-level type: {type(payload).__name__}")

    results = extract_result_list(payload)
    print(f"Result count: {len(results)}")

    if not results:
        print("No results extracted — inspect the saved JSON manually.")
        return

    first = results[0]
    # Reddit-style payloads often wrap each post under {"kind": "t3", "data": {...}}.
    if isinstance(first, dict) and "data" in first and isinstance(first["data"], dict):
        print("Detected nested 'data' wrapper on each result (Reddit 'listing' convention).")
        first = first["data"]

    if not isinstance(first, dict):
        print(f"First result is not a dict ({type(first).__name__}). Inspect raw JSON.")
        return

    result_keys = sorted(first.keys())
    print(f"First-result keys ({len(result_keys)}): {result_keys}")

    print("\nRawArticle fit report:")
    print_fit_report(first)

    print_pagination_hints(payload if isinstance(payload, dict) else {})


def main() -> int:
    token = settings.ensembledata_api_token.strip()
    if not token:
        print(
            "ERROR: ENSEMBLEDATA_API_TOKEN is not set.\n"
            "Add it to scraper_preprocessing_memory/.env and retry.",
            file=sys.stderr,
        )
        return 1

    # Mode switch: `--subreddit [name ...]` hits the subreddit-posts endpoint;
    # anything else is treated as keyword-search input.
    argv = sys.argv[1:]
    if argv and argv[0] == "--subreddit":
        mode = "subreddit"
        fetch_fn = fetch_subreddit
        labels = argv[1:] or DEFAULT_SUBREDDITS
        file_mode = "sub"
    else:
        mode = "keyword"
        fetch_fn = fetch_keyword
        labels = argv or DEFAULT_KEYWORDS
        file_mode = "kw"

    print(f"Testing EnsembleData Reddit {mode} endpoint with: {labels}")
    print(f"Samples will be written to: {SAMPLES_DIR}")

    exit_code = 0
    for label in labels:
        try:
            payload = fetch_fn(label, token)
        except httpx.HTTPStatusError as e:
            print(
                f"\n[{label}] HTTP {e.response.status_code}: {e.response.text[:200]}",
                file=sys.stderr,
            )
            exit_code = 2
            continue
        except httpx.HTTPError as e:
            print(f"\n[{label}] request failed: {e}", file=sys.stderr)
            exit_code = 2
            continue

        out_path = save_sample(label, payload, mode=file_mode)
        print(f"\nSaved raw response: {out_path}")
        summarize(label, payload, mode=mode)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
