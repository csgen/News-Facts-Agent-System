"""Tests for text_cleaner.clean_body_text.

The cleaner's job is to strip navigation/boilerplate from Tavily's
`raw_content` Markdown output without removing article prose.
"""

from src.preprocessing.text_cleaner import clean_body_text


class TestLinkDensityDrop:
    """Lines that are dominated by Markdown links are navigation menus."""

    def test_navigation_menu_with_many_links_is_dropped(self):
        text = (
            "[News](https://www.thehindu.com/news/)   "
            "[India](https://www.thehindu.com/news/national/)   "
            "[World](https://www.thehindu.com/news/international/)"
        )
        assert clean_body_text(text) == ""

    def test_line_with_single_link_and_no_other_text_is_dropped(self):
        text = "[Account](https://www.thehindu.com/myaccount/)"
        assert clean_body_text(text) == ""

    def test_line_with_mostly_links_is_dropped(self):
        text = "[eBooks](https://www.thehindu.com/premium/ebook/) [Subscribe](https://www.thehindu.com/subscription/)"
        assert clean_body_text(text) == ""

    def test_short_date_followed_by_link_is_dropped(self):
        """'April 15, 2026[e-Paper](...)' — mostly link chars → navigation."""
        text = "April 15, 2026[e-Paper](https://epaper.thehindu.com/reader?utm_source=Hindu)"
        result = clean_body_text(text)
        # Link portion dominates → whole line dropped
        assert result == ""


class TestInlineLinksArePreserved:
    """Prose that happens to contain a link should keep the visible text."""

    def test_prose_with_one_inline_link_keeps_text(self):
        text = (
            "JPMorgan's CEO [Jamie Dimon](https://www.jpmorgan.com/ceo) said "
            "credit card loans were up 7% from a year ago."
        )
        result = clean_body_text(text)
        assert "Jamie Dimon" in result
        assert "https://" not in result
        assert "credit card loans" in result
        assert "7%" in result

    def test_plain_prose_without_links_is_unchanged(self):
        text = "500,000 Tesla vehicles were recalled due to brake defects."
        assert clean_body_text(text) == text


class TestBoilerplateKeywordDrop:
    """Short lines matching footer/nav keywords are dropped."""

    def test_short_subscribe_line_dropped(self):
        text = "Subscribe to our newsletter"
        assert clean_body_text(text) == ""

    def test_long_line_mentioning_subscribe_is_kept(self):
        """Articles may legitimately discuss subscriptions."""
        text = (
            "Netflix reported that its subscribe-to-watch model grew by 12% "
            "last quarter, with 8 million new paid subscribers added across "
            "North America, Europe, and Asia markets combined this period."
        )
        # >120 chars AND contains "subscribe" but it's real content → keep
        result = clean_body_text(text)
        assert "Netflix" in result
        assert "12%" in result

    def test_copyright_footer_dropped(self):
        text = "© 2026 The Hindu. All rights reserved."
        assert clean_body_text(text) == ""


class TestMultilineCleanup:
    """Full-document scenarios — the common case."""

    def test_real_world_hindu_header_stripped(self):
        """Simulates the user-reported Tavily raw_content header."""
        text = (
            "April 15, 2026[e-Paper](https://epaper.thehindu.com/reader?utm_source=Hindu)\n"
            "\n"
            "[Account](https://www.thehindu.com/myaccount/)\n"
            "\n"
            "[eBooks](https://www.thehindu.com/premium/ebook/) [Subscribe](https://www.thehindu.com/subscription/)\n"
            "\n"
            "[Live Now](/topic/live-news/)\n"
            "\n"
            "[News](https://www.thehindu.com/news/)   [India](https://www.thehindu.com/news/national/)   "
            "[World](https://www.thehindu.com/news/international/)   [States](https://www.thehindu.com/news/states/)\n"
            "\n"
            "India's central bank announced a 25-basis-point rate cut on Tuesday, "
            "bringing the repo rate to 6.25%.\n"
            "\n"
            "The decision was unanimous among the six-member monetary policy committee."
        )
        result = clean_body_text(text)
        # Navigation stripped
        assert "e-Paper" not in result
        assert "Account" not in result
        assert "eBooks" not in result
        assert "Live Now" not in result
        # Actual article content kept
        assert "India's central bank" in result
        assert "25-basis-point" in result
        assert "6.25%" in result
        assert "monetary policy committee" in result

    def test_consecutive_blank_lines_collapsed(self):
        text = "First paragraph.\n\n\n\n\nSecond paragraph."
        result = clean_body_text(text)
        # No more than one blank line between paragraphs
        assert "\n\n\n" not in result
        assert "First paragraph." in result
        assert "Second paragraph." in result

    def test_leading_and_trailing_blanks_removed(self):
        text = "\n\n\nActual content here.\n\n\n"
        assert clean_body_text(text) == "Actual content here."


class TestEdgeCases:
    def test_empty_input_returns_empty(self):
        assert clean_body_text("") == ""

    def test_whitespace_only_returns_empty(self):
        assert clean_body_text("   \n\n\t  \n") == ""

    def test_max_chars_is_enforced(self):
        text = "A" * 20_000
        assert len(clean_body_text(text, max_chars=5000)) == 5000

    def test_malformed_markdown_link_is_left_alone(self):
        """Unclosed or weird Markdown shouldn't cause errors."""
        text = "This article has [an unclosed link and then some text."
        # Should not raise; content preserved since no valid link matched
        result = clean_body_text(text)
        assert "unclosed link" in result
