"""Google gemini-embedding-001 helper with batching, retry, and rate limiting."""

import logging
import re
import time

from google import genai
from google.genai import types
from langfuse.decorators import observe

from src.utils.langfuse_utils import log_embedding_usage
from src.utils.rate_limiter import EMBED_LIMITER

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
# Bumped from 3 to 5 so multiple consecutive 429s during entity reconcile
# each get a full Google-suggested cooldown (typically ~25 s).
MAX_RETRIES = 5

# Pulls Google's RetryInfo.retryDelay (e.g. "'retryDelay': '25s'") from the
# stringified error. The python-genai SDK doesn't expose RetryInfo as a typed
# field, so we parse the message body. Quote style varies between SDK
# versions, hence the [\"'] alternation.
_RETRY_DELAY_RE = re.compile(r"""['"]retryDelay['"]\s*:\s*['"](\d+(?:\.\d+)?)s['"]""")


def _parse_retry_delay(error: Exception, default: float) -> float:
    """Extract Google's suggested retry delay (seconds) from a 429 message.

    Falls back to `default` if the error isn't a 429 or has no RetryInfo.
    """
    match = _RETRY_DELAY_RE.search(str(error))
    return float(match.group(1)) if match else default

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
                # Honour Google's RetryInfo.retryDelay on 429 if present;
                # otherwise back off exponentially. +1s jitter so multiple
                # processes don't wake up exactly when the window opens.
                wait = _parse_retry_delay(e, default=2 ** attempt) + 1.0
                logger.warning(
                    "Embedding request failed (attempt %d/%d): %s. Retrying in %.1fs.",
                    attempt + 1, MAX_RETRIES, e, wait,
                )
                time.sleep(wait)
        raise RuntimeError("Unreachable")
