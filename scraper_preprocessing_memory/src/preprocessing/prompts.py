"""Centralized LLM prompt templates for the Preprocessing Agent."""

CLAIM_ISOLATION_PROMPT = """\
You are a fact-checking assistant. Given a news article, extract all falsifiable claims.

A falsifiable claim is a concrete statement that can be verified as true or false using evidence.
If there is explicit expression of the statement source, it should also be included in the claim.
Exclude opinions, questions, and vague statements.

For each claim, classify its type:
- "statistical": involves numbers, percentages, quantities
- "attribution": attributes an action or statement to a specific entity
- "causal": asserts a cause-effect relationship
- "predictive": makes a prediction about the future

Article title: {title}

Article text:
{body_text}

Return a JSON object with this structure:
{{
  "claims": [
    {{"text": "the exact falsifiable claim", "type": "statistical|attribution|causal|predictive"}}
  ]
}}

If no falsifiable claims are found, return {{"claims": []}}.
"""

ENTITY_EXTRACTION_PROMPT = """\
You are an NLP specialist. Given a news claim and candidate entities from NER, refine the entity list.

Tasks:
1. Correct any misclassified entity types
2. Merge duplicates (e.g., "Elon Musk" and "Musk" should be one entity)
3. Normalize entity names to their canonical form (full name for people, official name for orgs)
4. Assign entity_type from: "person", "organization", "country", "location", "event", "product"
5. Determine the sentiment of this claim toward each entity: "positive", "negative", or "neutral"

Claim: {claim_text}

NER candidates: {ner_candidates}

Article context: {article_context}

Return a JSON object:
{{
  "entities": [
    {{
      "name": "canonical entity name",
      "entity_type": "person|organization|country|location|event|product",
      "sentiment": "positive|negative|neutral"
    }}
  ]
}}
"""

ENTITY_EXTRACTION_BATCH_PROMPT = """\
You are an NLP specialist. Given multiple news claims and candidate entities from NER, refine the entity list for EACH claim.

CRITICAL RULE: Only include entities that are ACTUALLY MENTIONED in the specific claim text. Do NOT include entities from the article context unless they appear in the claim itself. The article context is provided ONLY for disambiguation (e.g., to know which "Apple" is meant).

Tasks for each claim:
1. Identify entities that appear in THAT claim's text
2. Correct any misclassified entity types from the NER candidates
3. Merge duplicates (e.g., "Elon Musk" and "Musk" should be one entity)
4. Normalize entity names to their canonical form (full name for people, official name for orgs)
5. Assign entity_type from: "person", "organization", "country", "location", "event", "product"
6. Determine the sentiment of that specific claim toward each entity: "positive", "negative", or "neutral"

Claims and their NER candidates:
{claims_with_candidates}

Article context (for disambiguation only, NOT for entity extraction): {article_context}

Return a JSON object with entities grouped by claim index (0-based):
{{
  "claims": [
    {{
      "claim_index": 0,
      "entities": [
        {{
          "name": "canonical entity name",
          "entity_type": "person|organization|country|location|event|product",
          "sentiment": "positive|negative|neutral"
        }}
      ]
    }}
  ]
}}
"""

CAPTION_PROMPT = """\
Describe this image in purely objective, factual terms. Focus on:
- Physical objects visible in the image
- Actions being performed
- Setting and environment
- Text visible in the image (signs, labels, etc.)
- Number of people and their apparent activities

Do NOT:
- Speculate about emotions, intentions, or context
- Make subjective judgments
- Reference any news article or story
- Infer information not directly visible

Provide a single paragraph description.
"""
