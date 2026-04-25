"""Langfuse helper for attaching google-genai embedding usage metadata to the
current observation span.

Langfuse v2 auto-instruments `langfuse.openai.OpenAI` — no helper is needed
for OpenAI calls. For embeddings (still on `google.genai.Client`), this helper
does manually what the OpenAI wrapper did automatically: set model name +
input token count so Langfuse renders the call as a "generation".

Usage (inside a function already decorated with `@observe(as_type="generation")`):

    response = client.models.embed_content(...)
    log_embedding_usage(model=self._model, response=response)

All failures are swallowed — Langfuse instrumentation must never break the
business logic it observes.
"""

import logging
from typing import Any

from langfuse.decorators import langfuse_context

logger = logging.getLogger(__name__)


def log_embedding_usage(
    *,
    model: str,
    response: Any,
) -> None:
    """Push model name + input token count from a google-genai
    EmbedContentResponse into the current Langfuse observation. Embeddings
    have no output tokens."""
    try:
        um = getattr(response, "usage_metadata", None)
        prompt_tokens = (
            getattr(um, "prompt_token_count", 0) if um is not None else 0
        ) or 0

        langfuse_context.update_current_observation(
            model=model,
            usage={"input": prompt_tokens, "output": 0, "total": prompt_tokens},
        )
    except Exception as e:
        logger.debug("log_embedding_usage swallowed: %s", e)
