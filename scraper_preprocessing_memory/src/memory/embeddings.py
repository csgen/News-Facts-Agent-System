"""Google gemini-embedding-001 helper with batching, retry, and rate limiting."""

import logging
import time

from google import genai
from google.genai import types
from langfuse.decorators import observe

from src.utils.langfuse_utils import log_embedding_usage
from src.utils.rate_limiter import EMBED_LIMITER

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
MAX_RETRIES = 3

# Embeddings are fast (<1s typical) but give them a safety margin; a stuck
# connection shouldn't hang the pipeline.
_API_TIMEOUT_MS = 60_000


class EmbeddingHelper:
    def __init__(
        self,
        api_key: str,
        model: str,
        output_dimensionality: int,
    ):
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=_API_TIMEOUT_MS),
        )
        self._model = model
        self._output_dim = output_dimensionality

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed_batch([text])[0]

    @observe(name="embedding_batch")
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts with batching and exponential backoff."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            embeddings = self._embed_with_retry(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    @observe(name="embedding_call", as_type="generation")
    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(MAX_RETRIES):
            try:
                EMBED_LIMITER.wait()
                response = self._client.models.embed_content(
                    model=self._model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self._output_dim,
                    ),
                )
                # Tag the Langfuse generation span with model + input token
                # count (embeddings have no output tokens).
                log_embedding_usage(model=self._model, response=response)
                return [emb.values for emb in response.embeddings]
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = 2**attempt
                logger.warning(
                    f"Embedding request failed (attempt {attempt + 1}): {e}. Retrying in {wait}s."
                )
                time.sleep(wait)
        raise RuntimeError("Unreachable")
