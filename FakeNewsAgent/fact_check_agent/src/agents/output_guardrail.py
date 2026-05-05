"""Output guardrail — runs before write_memory to catch bad LLM output.

Three checks (in order):
  1. Toxicity / injection scan  — regex on reasoning text.
     Blocks if any pattern matches (after stripping attempt leaves empty text).

  2. Hallucination heuristics   — fast, zero-cost signals.
     Flags: reasoning too short, LLM refusal phrases, no keyword overlap with
     claim.  Warns only — does not block by itself.

  3. Grounding check (LLM)      — asks the LLM whether the reasoning is
     supported by the supplied evidence.  Blocks if ungrounded.

When blocked: caller receives blocked=True. The node calls log_failure() and
returns output=None, causing fact_check_agent.py to skip adding the result —
the frontend then shows an error banner to the user.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# ── 1. Toxicity / injection patterns ─────────────────────────────────────────

_TOXICITY_PATTERNS = [
    r"ignore\s+(previous|all|above|prior)\s+instructions?",
    r"disregard\s+(previous|all|above|prior)\s+instructions?",
    r"override\s+(previous|all|above|prior)\s+instructions?",
    r"you\s+are\s+now\s+(a\s+)?(?!fact)",
    r"act\s+as\s+(if\s+you\s+are|a)\s+(?!fact)",
    r"jailbreak",
    r"(show|print|repeat|reveal|leak)\s+(me\s+)?(your\s+)?(system\s+prompt|instructions?)",
    r"###\s*(instruction|system|human|assistant)",
    r"\[INST\]",
    r"<\|im_start\|>",
    r"<\|system\|>",
    r"\bkill\s+all\b",
    r"\bexterminate\b",
    r"\bhow\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|poison|malware|virus)\b",
]

_TOXICITY_RE = [re.compile(p, re.IGNORECASE) for p in _TOXICITY_PATTERNS]


def _check_toxicity(text: str) -> bool:
    """Return True if any toxicity/injection pattern matches."""
    sample = text[:4000].lower()
    for rx in _TOXICITY_RE:
        if rx.search(sample):
            logger.warning("output_guardrail: toxicity pattern matched: %r", rx.pattern)
            return True
    return False


# ── 2. Hallucination heuristics ───────────────────────────────────────────────

_REFUSAL_PHRASES = [
    r"as an ai",
    r"i (cannot|can't|am unable to)",
    r"i don't have (access|information|knowledge)",
    r"i'm not able to",
    r"my (knowledge|training) (cutoff|data)",
]
_REFUSAL_RE = [re.compile(p, re.IGNORECASE) for p in _REFUSAL_PHRASES]

_MIN_REASONING_LENGTH = 50


def _hallucination_heuristics(reasoning: str, claim_text: str) -> list[str]:
    """Return warning strings for each heuristic that fires. Does not block."""
    warnings = []

    if len(reasoning.strip()) < _MIN_REASONING_LENGTH:
        warnings.append(f"reasoning too short ({len(reasoning)} chars)")

    for rx in _REFUSAL_RE:
        if rx.search(reasoning):
            warnings.append(f"LLM refusal phrase detected: {rx.pattern!r}")
            break

    claim_words = {
        w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", claim_text)
    } - {"that", "this", "with", "from", "have", "been", "were", "they", "their"}
    if claim_words and not any(w in reasoning.lower() for w in claim_words):
        warnings.append("reasoning shares no keywords with the claim")

    return warnings


# ── 3. LLM grounding check ────────────────────────────────────────────────────

_GROUNDING_PROMPT = """\
You are a quality-control reviewer for a fact-checking system.

CLAIM: {claim_text}

EVIDENCE USED:
{evidence_block}

GENERATED REASONING:
{reasoning}

Task: Decide whether the reasoning is genuinely grounded in the evidence above.
Grounded means: the reasoning references or is consistent with at least some of
the evidence items. Ungrounded means: the reasoning makes claims that contradict
or are entirely absent from the evidence, or appears fabricated.

Respond ONLY in this exact JSON format:
{{"grounded": true/false, "explanation": "one sentence"}}"""


def _format_evidence_block(context_claims: list[dict]) -> str:
    lines = []
    for i, c in enumerate(context_claims, 1):
        lines.append(f"{i}. [{c.get('source', 'unknown')}] {c.get('content', '')[:200]}")
    return "\n".join(lines) if lines else "(no evidence)"


def check_grounding(
    claim_text: str,
    reasoning: str,
    context_claims: list[dict],
    client,
    model: str,
) -> tuple[bool, str]:
    """Return (is_grounded, explanation) via LLM call.
    Fails open (grounded=True) if the LLM call itself errors.
    """
    prompt = _GROUNDING_PROMPT.format(
        claim_text=claim_text,
        evidence_block=_format_evidence_block(context_claims),
        reasoning=reasoning,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=150,
        )
        result = json.loads(response.choices[0].message.content or "{}")
        grounded = bool(result.get("grounded", True))
        explanation = result.get("explanation", "")
        return grounded, explanation
    except Exception as exc:
        logger.warning("output_guardrail: grounding check failed (%s) — assuming grounded", exc)
        return True, ""


# ── Public entry point ────────────────────────────────────────────────────────

def run_output_guardrail(
    reasoning: str,
    claim_text: str,
    context_claims: list[dict],
    client,
    model: str,
) -> dict:
    """Run all three checks.

    Returns:
        {
          "blocked":               bool,   # True → node must clear output and call log_failure
          "block_reason":          str,    # "toxicity" | "ungrounded" | ""
          "hallucination_warnings": list[str],  # non-blocking warnings
          "grounding_explanation": str,
        }
    """
    # 1. Toxicity — blocks immediately
    if _check_toxicity(reasoning):
        return {
            "blocked": True,
            "block_reason": "toxicity",
            "hallucination_warnings": [],
            "grounding_explanation": "",
        }

    # 2. Hallucination heuristics — warn only
    hallucination_warnings = _hallucination_heuristics(reasoning, claim_text)
    for w in hallucination_warnings:
        logger.warning("output_guardrail: hallucination signal — %s", w)

    # 3. Grounding check — blocks if ungrounded
    grounded, grounding_explanation = check_grounding(
        claim_text, reasoning, context_claims, client, model
    )
    if not grounded:
        return {
            "blocked": True,
            "block_reason": "ungrounded",
            "hallucination_warnings": hallucination_warnings,
            "grounding_explanation": grounding_explanation,
        }

    return {
        "blocked": False,
        "block_reason": "",
        "hallucination_warnings": hallucination_warnings,
        "grounding_explanation": grounding_explanation,
    }
