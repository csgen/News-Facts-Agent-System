"""Cross-modal consistency tool — checks for conflicts between claim text and image.

Three modes (selected in priority order when image_url is provided):
  1. SigLIP mode: local embedding similarity via google/siglip-base-patch16-224.
     Fast, deterministic, no API calls. Activated by `use_siglip=True` in settings.
  2. Vision LLM mode: sends image to Gemma 4 via Ollama for reasoning-based check.
     Activated when `llm_provider == "ollama"` and SigLIP is disabled.
  3. Caption mode (fallback): sends claim + text caption to any LLM.
     Used when no image_url is available.

SigLIP scores the probability that a (image, text) pair is a match.
Low probability → claim doesn't describe the image → potential conflict.
"""

import base64
import io
import json
import logging
from functools import lru_cache
from typing import Optional

from openai import OpenAI

import fact_check_agent.src.llm_factory as _llm_factory
from fact_check_agent.src.config import settings
from fact_check_agent.src.prompts import CROSS_MODAL_PROMPT, CROSS_MODAL_VISION_PROMPT

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_siglip(model_name: str):
    """Load SigLIP model and processor once, cache for reuse."""
    from transformers import AutoModel, AutoProcessor

    logger.info("Loading SigLIP model: %s", model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return processor, model


def _decode_image(image_url: str):
    """Return a PIL Image from a base64 data URI or an https:// URL."""
    from PIL import Image as PILImage

    if image_url.startswith("data:"):
        # data:image/jpeg;base64,<b64>
        header, b64 = image_url.split(",", 1)
        return PILImage.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    import urllib.request

    req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return PILImage.open(io.BytesIO(r.read())).convert("RGB")


def _siglip_check(claim_text: str, image_url: str) -> dict:
    """Compute SigLIP image-text similarity.

    Returns conflict=True when the sigmoid probability that (image, claim)
    is a matching pair falls below settings.siglip_threshold.
    """
    import torch

    try:
        processor, model = _load_siglip(settings.siglip_model)
        try:
            image = _decode_image(image_url)
        except Exception as fetch_err:
            logger.warning("SigLIP: could not fetch image %s: %s — skipping", image_url, fetch_err)
            return {"conflict": False, "explanation": None, "siglip_score": None}

        inputs = processor(
            text=[claim_text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)
            # logits_per_image shape: (n_images, n_texts)
            prob = torch.sigmoid(outputs.logits_per_image[0, 0]).item()

        conflict = prob < settings.siglip_threshold
        explanation = (
            f"SigLIP match probability {prob:.3f} is below threshold "
            f"{settings.siglip_threshold} — image likely does not match claim."
            if conflict
            else None
        )
        logger.debug(
            "SigLIP score=%.3f threshold=%.3f conflict=%s",
            prob,
            settings.siglip_threshold,
            conflict,
        )
        return {"conflict": conflict, "explanation": explanation, "siglip_score": prob}

    except Exception as e:
        logger.error("SigLIP check failed: %s", e)
        return {"conflict": False, "explanation": None, "siglip_score": None}


def _openai_vision_check(claim_text: str, image_url: str, api_key: str, model: str) -> dict:
    """Send image URL + claim to GPT-4o / GPT-4o-mini vision API.

    Uses the model's native vision capability — no base64 encoding needed.
    Falls back to caption-text LLM check if the call fails.
    """
    # Use a vision-capable model — upgrade mini to full vision if needed
    vision_model = model if model in ("gpt-4o", "gpt-4o-mini") else "gpt-4o-mini"
    prompt = (
        "You are a fact-checking assistant verifying whether a news article image "
        "is consistent with the article's claim.\n\n"
        f"Claim: {claim_text}\n\n"
        "Look at the image and answer:\n"
        "1. What does the image show? (1-2 sentences)\n"
        "2. Does the image support, contradict, or is it unrelated to the claim?\n\n"
        "Respond in JSON: "
        '{"conflict": true/false, "explanation": "what the image shows and whether it matches the claim"}'
    )
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)
        logger.debug(
            "OpenAI vision cross-modal: conflict=%s explanation=%s",
            result.get("conflict"),
            result.get("explanation"),
        )
        return result
    except Exception as e:
        logger.warning("OpenAI vision check failed (%s) — skipping cross-modal", e)
        return {"conflict": False, "explanation": None}


def check_cross_modal(
    claim_text: str,
    image_caption: Optional[str],
    api_key: str,
    model: str,
    image_url: Optional[str] = None,
) -> dict:
    """Check for logical conflicts between claim text and image/caption.

    Priority order:
      1. SigLIP (local, fast) — if use_siglip=True and image_url provided
      2. Gemma 4 vision via Ollama — if llm_provider=ollama and image_url provided
      3. GPT-4o vision — if llm_provider=openai and image_url provided  ← NEW
      4. LLM caption text check (fallback when no image_url)

    Returns:
        {"flag": bool, "explanation": str | None, "siglip_score": float | None}
    """
    if not image_url and not image_caption:
        logger.debug("No image data — skipping cross-modal check")
        return {"flag": False, "explanation": None, "siglip_score": None}

    siglip_score = None

    if image_url and settings.use_siglip:
        result = _siglip_check(claim_text, image_url)
        siglip_score = result.get("siglip_score")
    elif image_url and settings.llm_provider == "ollama":
        result = _vision_check(claim_text, image_url)
        if result is None:
            result = _llm_check(claim_text, image_caption or "", api_key, model)
    elif image_url and settings.llm_provider == "openai":
        # Use GPT-4o vision to actually look at the image
        result = _openai_vision_check(claim_text, image_url, api_key, model)
    else:
        result = _llm_check(claim_text, image_caption or "", api_key, model)

    return {
        "flag": result.get("conflict", False),
        "explanation": result.get("explanation"),
        "siglip_score": siglip_score,
    }


def _ensure_base64_uri(image_url: str) -> Optional[str]:
    """Convert an https:// image URL to a base64 data URI.

    Returns None if the image cannot be fetched (403, timeout, etc.) so the
    caller can fall back to caption mode instead of passing a raw URL to Ollama.
    """
    if image_url.startswith("data:"):
        return image_url
    try:
        import urllib.request

        req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            content_type = r.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
            raw = r.read()
        b64 = base64.b64encode(raw).decode()
        return f"data:{content_type};base64,{b64}"
    except Exception as e:
        logger.warning("Could not fetch image %s: %s", image_url, e)
        return None


def _vision_check(claim_text: str, image_url: str) -> Optional[dict]:
    """Send image + claim to Gemma 4 via Ollama vision API.

    Returns None when the image cannot be fetched so the caller can fall back
    to caption mode.
    """
    image_data_uri = _ensure_base64_uri(image_url)
    if image_data_uri is None:
        return None

    client = _llm_factory.make_vlm_client()
    prompt = CROSS_MODAL_VISION_PROMPT.format(claim_text=claim_text)
    try:
        response = client.chat.completions.create(
            model=_llm_factory.vlm_model_name(),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        logger.warning("Vision cross-modal check failed (%s) — falling back to caption mode", e)
        return None


def _llm_check(claim_text: str, image_caption: str, api_key: str, model: str) -> dict:
    client = _llm_factory.make_llm_client()
    prompt = CROSS_MODAL_PROMPT.format(
        claim_text=claim_text,
        image_caption=image_caption,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error("Cross-modal LLM check failed: %s", e)
        return {"conflict": False, "explanation": None}
