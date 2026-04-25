"""Tests for Pydantic data models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.models.article import Article, Source
from src.models.claim import Claim, MentionSentiment
from src.models.credibility import CredibilitySnapshot, Prediction
from src.models.verdict import Verdict


class TestSource:
    def test_valid_source(self, sample_source):
        assert sample_source.source_id == "src_reuters_com"
        assert sample_source.base_credibility == 0.95

    def test_credibility_bounds(self):
        with pytest.raises(ValidationError):
            Source(
                source_id="src_bad",
                name="Bad",
                domain="bad.com",
                category="unknown",
                base_credibility=1.5,
            )


class TestArticle:
    def test_valid_article(self, sample_article):
        assert sample_article.article_id.startswith("art_")

    def test_body_snippet_default(self):
        now = datetime.now(timezone.utc)
        article = Article(
            article_id="art_test",
            title="Test",
            url="https://test.com",
            source_id="src_test",
            published_at=now,
            ingested_at=now,
            content_hash="sha256_test",
        )
        assert article.body_snippet == ""


class TestClaim:
    def test_valid_claim(self, sample_claim):
        assert len(sample_claim.entities) == 2
        assert sample_claim.status == "pending"

    def test_default_entities_empty(self):
        claim = Claim(
            claim_id="clm_test",
            article_id="art_test",
            claim_text="Test claim",
            extracted_at=datetime.now(timezone.utc),
        )
        assert claim.entities == []
        assert claim.claim_type is None


class TestVerdict:
    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Verdict(
                verdict_id="vrd_test",
                claim_id="clm_test",
                label="supported",
                confidence=1.5,
                evidence_summary="test",
                bias_score=0.1,
                verified_at=datetime.now(timezone.utc),
            )


class TestMentionSentiment:
    def test_valid_mention(self):
        mention = MentionSentiment(
            entity_id="ent_test",
            name="Tesla",
            entity_type="organization",
            sentiment="negative",
        )
        assert mention.sentiment == "negative"
