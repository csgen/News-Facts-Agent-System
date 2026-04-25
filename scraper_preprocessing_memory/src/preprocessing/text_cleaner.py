"""Clean raw article body text before sending to LLM.

Tavily's `raw_content` returns full parsed page content as Markdown, which
includes navigation menus, subscribe banners, and footer links alongside the
actual article. Sending that straight to the LLM wastes tokens and degrades
claim extraction quality.

This module strips that boilerplate while preserving article prose.
"""

import re

# Markdown link syntax: [visible text](url) — possibly with extra whitespace
# Captures the visible text for optional preservation
_MD_LINK = re.compile(r"\[([^\[\]]*)\]\(([^()\s]+(?:\s+\"[^\"]*\")?)\)")

# When a line is dominated by link characters, it's almost certainly navigation
_LINK_DENSITY_DROP_THRESHOLD = 0.40

# Short lines containing these keywords are usually navigation/footer
_BOILERPLATE_KEYWORDS = (
    "subscribe",
    "newsletter",
    "sign up",
    "sign in",
    "log in",
    "follow us",
    "privacy policy",
    "terms of service",
    "cookie policy",
    "my account",
    "e-paper",
    "ebooks",
    "see all newsletters",
    "download our app",
    "back to top",
    "copyright",
    "all rights reserved",
)

# If a boilerplate keyword appears AND the line is shorter than this,
# treat it as junk. Longer lines (actual prose) are kept even if they
# mention "subscribe" in passing.
_BOILERPLATE_MAX_LEN = 120

# Hard cap on total cleaned length to avoid runaway prompts.
_DEFAULT_MAX_CHARS = 10_000


def clean_body_text(text: str, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
    """Return a cleaned version of an article body.

    Strategy (per line):
    1. If the line is dominated by Markdown links (>40% of chars), drop it
       — this removes navigation menus like `[Home](/) [News](/news) ...`
    2. If the line is short (<120 chars) and contains boilerplate keywords
       like "subscribe" or "newsletter", drop it.
    3. Otherwise, strip remaining Markdown link syntax but keep the visible
       text — so an inline reference like `[Jamie Dimon](https://...)` becomes
       `Jamie Dimon`.
    4. Drop now-empty lines, collapse runs of 3+ newlines into 2.
    5. Truncate to max_chars.

    Returns empty string for empty input.
    """
    if not text:
        return ""

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            # Preserve paragraph breaks; we'll collapse runs later
            cleaned_lines.append("")
            continue

        # Rule 1: high link density → navigation, drop
        link_chars = sum(len(m.group(0)) for m in _MD_LINK.finditer(stripped))
        if link_chars > 0 and link_chars / len(stripped) > _LINK_DENSITY_DROP_THRESHOLD:
            continue

        # Rule 2: short boilerplate line → drop
        lower = stripped.lower()
        if len(stripped) < _BOILERPLATE_MAX_LEN and any(
            kw in lower for kw in _BOILERPLATE_KEYWORDS
        ):
            continue

        # Rule 3: remove remaining link markup, keep the visible text
        cleaned = _MD_LINK.sub(lambda m: m.group(1), line)

        # If stripping links emptied the line, skip it
        if not cleaned.strip():
            continue

        cleaned_lines.append(cleaned)

    # Rule 4: collapse 3+ consecutive blank lines into at most one blank line
    collapsed: list[str] = []
    blank_run = 0
    for line in cleaned_lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 1:
                collapsed.append(line)
        else:
            blank_run = 0
            collapsed.append(line)

    # Trim leading/trailing blank lines
    while collapsed and collapsed[0].strip() == "":
        collapsed.pop(0)
    while collapsed and collapsed[-1].strip() == "":
        collapsed.pop()

    result = "\n".join(collapsed)

    # Rule 5: hard cap for safety
    if len(result) > max_chars:
        result = result[:max_chars]

    return result
