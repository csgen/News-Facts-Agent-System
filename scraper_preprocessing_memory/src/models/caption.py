"""Image caption data model."""

from pydantic import BaseModel


class ImageCaption(BaseModel):
    """VLM-generated objective description of an article's image.

    Separate from Article because Fact-Check Agent needs independent access
    for cross-modal consistency checks.
    """

    caption_id: str
    article_id: str
    image_url: str  # Original URL from the news article
    vlm_caption: str  # Objective description from Visual Language Model
