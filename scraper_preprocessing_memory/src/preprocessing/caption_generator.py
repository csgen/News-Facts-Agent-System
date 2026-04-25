"""GPT-4o Vision image captioning for cross-modal consistency checks."""

import logging

from langfuse.decorators import observe
from langfuse.openai import OpenAI

from src.preprocessing.prompts import CAPTION_PROMPT

logger = logging.getLogger(__name__)


class CaptionGenerator:
    def __init__(self, api_key: str, model: str):
        self._client = OpenAI(api_key=api_key)
        self._model = model

    @observe(name="image_captioning")
    def generate_caption(self, image_url: str) -> str:
        """Generate an objective description of an image from its URL.

        GPT-4o vision accepts image URLs directly — no download needed.
        Returns empty string on failure.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": CAPTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url, "detail": "low"},
                            },
                        ],
                    }
                ],
                max_tokens=300,
                temperature=0.1,
            )
            caption = response.choices[0].message.content.strip()
            logger.info("Generated caption for image: %s...", caption[:80])
            return caption

        except Exception as e:
            logger.error("Caption generation failed for %s: %s", image_url, e)
            return ""
