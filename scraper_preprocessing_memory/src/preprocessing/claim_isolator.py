"""LLM-based claim isolation from article text (OpenAI gpt-4o-mini)."""

import json
import logging

from langfuse.decorators import observe
from langfuse.openai import OpenAI

from src.preprocessing.prompts import CLAIM_ISOLATION_PROMPT

logger = logging.getLogger(__name__)


class ClaimIsolator:
    def __init__(self, api_key: str, model: str):
        # langfuse.openai auto-instruments: every .chat.completions.create call
        # becomes a child generation span under the current @observe trace, with
        # model name, input/output tokens, and latency captured automatically.
        self._client = OpenAI(api_key=api_key)
        self._model = model

    @observe(name="claim_isolation")
    def extract_claims(self, title: str, body_text: str) -> list[dict]:
        """Extract falsifiable claims from article text.

        Returns a list of {"text": str, "type": str} dicts.
        """
        prompt = CLAIM_ISOLATION_PROMPT.format(title=title, body_text=body_text)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            claims = parsed.get("claims", [])

            valid_claims = []
            for claim in claims:
                if isinstance(claim, dict) and "text" in claim:
                    valid_claims.append({
                        "text": claim["text"],
                        "type": claim.get("type", ""),
                        "topic_text": claim.get("topic_text", ""),
                    })

            logger.info("Extracted %d claims from article", len(valid_claims))
            return valid_claims

        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON: %s", e)
            return []
        except Exception as e:
            logger.error("Claim isolation failed: %s", e)
            return []
