"""Composite output model for the Preprocessing Agent pipeline."""

from typing import Optional

from pydantic import BaseModel

from src.models.article import Article, Source
from src.models.caption import ImageCaption
from src.models.claim import Claim


class PreprocessingOutput(BaseModel):
    """The contract between Preprocessing Agent and Memory Agent.

    This is what the Preprocessing Agent produces for each article,
    and what MemoryAgent.ingest_preprocessed() consumes.
    """

    source: Source
    article: Article
    claims: list[Claim]
    image_caption: Optional[ImageCaption] = None
