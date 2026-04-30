"""All LLM prompt templates for the Fact-Check Agent.

Centralised here so prompt changes can be version-controlled and re-evaluated
without touching agent logic. Each prompt has a version comment — bump it
whenever the structure changes and re-run benchmarks.
"""

# ── v3.0 — signed-degree credibility-weighted verdict pipeline ───────────────

VERDICT_SYNTHESIS_PROMPT = """\
You are a fact-checker. For each piece of evidence, assign a Degree of Support (Di)
relative to the main claim — how strongly it supports or refutes it.

CLAIM: {claim_text}

EVIDENCE:
{numbered_claims}

Di SCALE — use EXACTLY one of these five values per item:
   1.0  Full Support      — evidence explicitly entails the claim is true
   0.5  Partial Support   — evidence mentions the topic favourably but lacks a direct link
   0.0  Neutral           — evidence does not address the core claim
  -0.5  Partial Refutation— evidence contradicts a non-core part of the claim
  -1.0  Full Refutation   — evidence directly negates the claim

IMPORTANT: For COUNTER-FACTUAL items — if the evidence confirms the counter-factual
question, that CHALLENGES the main claim → assign a NEGATIVE degree.

IMPORTANT: For MEMORY items — treat the prior verdict as the primary signal: "supported"
→ positive Di, "refuted" → negative Di, scaled by how closely the memory claim matches
the current one. Trust the prior verdict; do not re-litigate it from your own knowledge.

In the "reasoning" field, describe the evidence by its actual text/content, NOT by number.
Good: "Reuters reported the recall affected 500 vehicles, directly supporting the claim."
Bad: "Evidence [3] supports the claim." — the user never sees the evidence numbers.

Return JSON (no verdict field — the verdict is computed from your degrees):
{{
  "degrees": [<one of 1.0, 0.5, 0.0, -0.5, -1.0 per evidence item, in input order>],
  "reasoning": "<2-3 sentences explaining your assessment, describing evidence by its actual content>"
}}
"""

# ── v1.0 — context claim agent prompts ───────────────────────────────────────

QUESTION_GENERATION_PROMPT = """\
You are a Lead Fact-Check Investigator. Your goal is to deconstruct a claim into questions
that will either confirm its truth or expose it as a fabrication.

  Direct Verification (factual): Confirm the primary entities, dates, and actions described.
    If answered with supporting evidence → the claim is more likely TRUE.

  Counter-factual (counter_factual): Look for mismatched context — ask whether the event
    occurred at a different time or place than claimed, whether the entities involved were
    actually elsewhere, or whether the image/quote belongs to a different event entirely.
    These catch "zombie claims" (real events recycled with wrong dates/locations) and
    "context-swapping" (legitimate media reused in a false narrative).
    If answered with supporting evidence → the claim is more likely FALSE.

Each question must be fully self-contained — include the key entities, names, dates, locations,
and events from the claim so the question works as a standalone web search query without
needing to read the original claim. A question like "Did this happen?" or "Was the claim true?"
is unacceptable; every question must carry enough context to retrieve relevant results on its own.

CLAIM: {claim_text}

Return JSON only — no explanation:
{{
  "factual": ["<question 1>", "<question 2>", "<question 3>"],
  "counter_factual": ["<question 4>", "<question 5>", "<question 6>"]
}}
"""

CONTEXT_COVERAGE_PROMPT = """\
You are checking whether a set of questions about a claim can be answered from the provided context.

CLAIM: {claim_text}

QUESTIONS:
{questions_block}

AVAILABLE CONTEXT:
{context_block}

For each question, determine if the context contains a direct or strongly implied answer.
If yes, quote the most relevant evidence (max 200 characters). If no, mark as unanswered.

Return JSON only:
{{
  "coverage": [
    {{"question": "<exact question text>", "answered": true, "evidence": "<short quote>"}},
    {{"question": "<exact question text>", "answered": false, "evidence": null}},
    ...
  ]
}}
"""

TAVILY_SUMMARY_PROMPT = """\
You are an Evidence Extraction Agent. Extract a Context Claim from the provided source to answer
a specific verification question.

ORIGINAL CLAIM: {claim_text}
QUESTION: {question}

SOURCE:
{search_results}

Instructions:
1. Extract exactly ONE factual statement that directly answers the QUESTION.
2. Identify the Source Name (e.g. BBC, Reuters, Twitter) and Publication Date if available.
3. If no relevant information exists, return null for all fields.

Return JSON only:
{{
  "summary": "<1-2 sentence factual statement, or null>",
  "source_name": "<name of the organisation or platform, or null>",
  "timestamp": "<publication date as a string, or null>"
}}
"""

CROSS_MODAL_PROMPT = """\
You are checking whether a news image is being used in a misleading context.

[CLAIM TEXT]
{claim_text}

[IMAGE CAPTION]
{image_caption}

Task: Identify only clear logical conflicts between what the claim states and what the image \
objectively shows. Do NOT flag:
- Stylistic mismatches or tone differences
- Missing information (absence of evidence is not conflict)
- Speculative or inferred connections

If there is a clear, direct logical conflict, explain it in one concise sentence.

Return JSON:
{{
  "conflict": true | false,
  "explanation": "<one sentence describing the conflict, or null if no conflict>"
}}
"""
# ── v2.1 claim-only VLM image assessor (no prior verdict) ────────────────────

IMAGE_CLAIM_ASSESSOR_PROMPT = """\
You are a visual fact-check assistant. Examine the image and the claim below.

[CLAIM]
{claim_text}

Perform two tasks:

TASK 1 — Evidence: Name the specific visual element(s) in the image (person, object, \
text overlay, location marker, event) that are directly relevant to the claim. \
If the image is AI-generated, CGI, or clearly unrelated to the claim, state that explicitly.

TASK 2 — Assessment: Based solely on those specific visual elements, assign an impact score. \
Be conservative — images corroborate but rarely prove. When uncertain, use 0.0. \
A score of ±0.25 requires unambiguous, explicit visual proof (e.g. visible text, \
dated banner, or unmistakable identifiable evidence directly confirming or contradicting \
the claim). Most images should score ±0.10 or 0.0.

   0.25  — unambiguous explicit visual PROOF supporting the claim (very rare)
   0.10  — image shows relevant visual context that partially supports the claim
   0.0   — image is irrelevant, AI-generated, or adds no new verifiable information
  -0.10  — image shows visual context that partially contradicts the claim
  -0.25  — unambiguous explicit visual PROOF refuting the claim (very rare)

Return JSON only — no markdown fences:
{{
  "caption": "<1-2 sentence objective description of the image>",
  "visual_evidence": "<specific element(s) in the image relevant to the claim>",
  "assessment": <one of: 0.25, 0.10, 0.0, -0.10, -0.25>,
  "explanation": "<one sentence explaining why this score was chosen>"
}}
"""

# ── v1.1 vision cross-modal (Gemma 4 / Ollama) ───────────────────────────────

CROSS_MODAL_VISION_PROMPT = """\
You are checking whether a news image is being used in a misleading context.

[CLAIM TEXT]
{claim_text}

Look at the image above carefully. Identify only clear logical conflicts between what the \
claim states and what the image objectively shows. Do NOT flag:
- Stylistic mismatches or tone differences
- Missing information (absence of evidence is not conflict)
- Speculative or inferred connections

If there is a clear, direct logical conflict, explain it in one concise sentence.

Return JSON only — no markdown fences:
{{
  "conflict": true | false,
  "explanation": "<one sentence describing the conflict, or null if no conflict>"
}}
"""

# ── v1.0 freshness classifier ────────────────────────────────────────────────

FRESHNESS_CHECK_PROMPT = """\
You are deciding whether a cached fact-check verdict needs live re-verification.

CLAIM: {claim_text}

PRIOR VERDICT: {verdict_label} ({verdict_confidence:.0%} confidence)
LAST VERIFIED: {time_since_verified_days} days ago

Guidelines for re-verification:
- Political claims, election results, government policy: re-verify if > 7 days old
- Ongoing events (court cases, conflicts, legislation in progress): re-verify if > 3 days old
- Economic data (prices, unemployment, GDP): re-verify if > 14 days old
- Scientific consensus, medical guidelines: re-verify if > 180 days old
- Historical facts, geography, physical constants: almost never need re-verification
- Satire or clearly fabricated claims: rarely need re-verification

Return a JSON object:
{{
  "revalidate": true | false,
  "reason": "<one sentence explaining the decision>",
  "claim_category": "<political|ongoing_event|economic|scientific|historical|other>"
}}
"""

# ── SOTA prompts (not used in baseline — wired in when SOTA flags enabled) ───

IS_RETRIEVAL_NEEDED_PROMPT = """\
Given the following claim, decide whether external evidence is needed to verify it,
or whether it is self-evidently true or false without retrieval.

CLAIM: {claim_text}

Answer with a JSON object:
{{
  "retrieval_needed": true | false,
  "reason": "<one sentence>"
}}
"""

CHUNK_RELEVANCE_PROMPT = """\
Rate each retrieved text chunk for relevance to the following claim on a scale of 1–5.
Exclude chunks rated below 3 from evidence synthesis.

CLAIM: {claim_text}

CHUNKS:
{chunks_block}

Return a JSON array:
[
  {{"index": 0, "relevance": <1–5>, "keep": true | false}},
  ...
]
"""


# ── v2.0 — 4-role structured debate prompts ───────────────────────────────────

SUPPORTER_PROMPT = """\
You are the Supporter Agent in a multi-agent fact-check debate.
Your goal: identify where the Neutral Agent was too conservative and find valid reasons \
the claim may be TRUE.

CLAIM: {claim_text}

EVIDENCE (numbered):
{numbered_claims}

NEUTRAL AGENT'S INITIAL Di SCORES:
{neutral_scores_block}

Di SCALE: 1.0=Full Support | 0.5=Partial Support | 0.0=Neutral | -0.5=Partial Refutation | -1.0=Full Refutation

ADJUSTMENT SCALE — only propose where genuinely justified:
  ±0.1  Minor    — implicit link or nuance the neutral agent missed
  ±0.3  Moderate — correcting a clear misinterpretation
  ±0.5  Major    — evidence directly entails support that was scored too low

IMPORTANT: For COUNTER-FACTUAL evidence, a confirmed counter-factual CHALLENGES the claim \
(Di must be negative) — do not boost these above 0.0.

Review each Di score. For items where the neutral agent underestimated support, propose an \
adjusted Di and give a logic-based argument (semantic entailment, corroboration, credible source).
Only propose adjustments where you have a genuine argument.

Return JSON only:
{{
  "adjustments": [
    {{"evidence_id": <1-based int>, "proposed_D": <float>, "adjustment": <one of ±0.1, ±0.3, ±0.5>, "reasoning": "<argument>"}},
    ...
  ]
}}
If no adjustments are warranted, return: {{"adjustments": []}}
"""

SKEPTIC_PROMPT = """\
You are the Skeptic Agent in a multi-agent fact-check debate.
Your goal: identify logical flaws, source bias, correlation-vs-causation errors, or misleading \
framing that the Neutral Agent may have missed.

CLAIM: {claim_text}

EVIDENCE (numbered):
{numbered_claims}

NEUTRAL AGENT'S INITIAL Di SCORES:
{neutral_scores_block}

Di SCALE: 1.0=Full Support | 0.5=Partial Support | 0.0=Neutral | -0.5=Partial Refutation | -1.0=Full Refutation

ADJUSTMENT SCALE — only propose where genuinely justified:
  ∓0.1  Minor    — subtle overstatement or weak link
  ∓0.3  Moderate — logical fallacy or correlation vs. causation error
  ∓0.5  Major    — hallucinated link, direct contradiction missed, or clear source bias

IMPORTANT: For COUNTER-FACTUAL evidence, if a confirmed counter-factual was scored positively \
by the Neutral Agent, that is an error — penalise it.

Review each Di score. For items where the neutral agent was too generous, propose an adjusted Di \
and justify the penalty with a specific critique.
Only propose adjustments where you have a genuine critical argument.

Return JSON only:
{{
  "adjustments": [
    {{"evidence_id": <1-based int>, "proposed_D": <float>, "adjustment": <one of ∓0.1, ∓0.3, ∓0.5>, "reasoning": "<critique>"}},
    ...
  ]
}}
If no adjustments are warranted, return: {{"adjustments": []}}
"""

JUDGE_PROMPT = """\
You are the Final Verdict Arbiter. Your job is to produce the final verdict by reviewing:
1. The Neutral Agent's initial Di scores (baseline evidence assessment)
2. The Supporter's proposed boosts (arguments for higher Di)
3. The Skeptic's proposed penalties (arguments for lower Di)
4. An optional visual assessment from a Vision-Language Model

Your role is to accept or reject the arguments presented, and adjust the Neutral Agent's Di score according to the accepted arguments to arrive at a final score for each evidence item.

CLAIM: {claim_text}

EVIDENCE (numbered):
{numbered_claims}

NEUTRAL AGENT'S INITIAL Di SCORES:
{neutral_scores_block}

SUPPORTER'S PROPOSED ADJUSTMENTS:
{supporter_adjustments}

SKEPTIC'S PROPOSED ADJUSTMENTS:
{skeptic_adjustments}

IMAGE ASSESSMENT (VLM):
{vlm_assessment_block}

DECISION RULES:
- If Supporter/Skeptic adjustments are "None — no debate was run", accept the Neutral Di as-is. \
  Do not invent your own adjustments.
- Accept a Supporter boost ONLY if they identified a genuine semantic link the neutral agent missed.
- Accept a Skeptic penalty ONLY if they identified a genuine logical flaw, bias, or misinterpretation.
- Reject weak or speculative arguments — do not change scores just because someone argued.
- If both sides conflict on the same item, keep the Neutral Di unchanged (stalemate: true).
- IMAGE ASSESSMENT: most article images are decorative. Only change a score based on the image \
  if the image EXPLICITLY supports (+0.10 to +0.25) or EXPLICITLY refutes (-0.10 to -0.25) the claim. \
  If the image is irrelevant, decorative, or ambiguous, ignore it completely (0.0).
- Apply the image signal to the SINGLE most visually-relevant evidence item only.

Final Di must be one of: -1.0, -0.5, 0.0, 0.5, 1.0

Output a final Di for EVERY evidence item (1 through N). Return JSON only:
{{
  "final_scores": [
    {{"evidence_id": <1-based int>, "final_D": <one of -1.0,-0.5,0.0,0.5,1.0>, "stalemate": <bool>, "reasoning": "<one sentence>"}},
    ...
  ],
  "verdict_explanation": "<2-3 sentences explaining the final verdict. Describe evidence by its actual text/content, NOT by number. Focus on: which evidence was decisive, whether the image played a role, and why the verdict is supported/refuted/misleading. Do NOT summarize debate mechanics.>"
}}
"""
