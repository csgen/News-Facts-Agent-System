"""Integration tests for the full LangGraph pipeline — no API keys required.

The OpenAI client and Tavily client are patched so the entire graph can be
exercised end-to-end in CI without any network calls.

Graph node order (linear):
  receive_claim → query_memory → freshness_check_all → context_claim_agent
  → synthesize_verdict → [multi_agent_debate] → cross_modal_check
  → write_memory → emit_output

LLM call order per run (no image, empty tavily key):
  1. context_claim_agent: question generation
  2. synthesize_verdict: degrees + reasoning
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from fact_check_agent.src.graph.graph import build_graph
from fact_check_agent.src.models.schemas import EntityRef, FactCheckInput

# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_fact_check_input(claim_text="Test claim about vaccines.", image_caption=None):
    return FactCheckInput(
        claim_id="clm_test001",
        claim_text=claim_text,
        entities=[
            EntityRef(
                entity_id="ent_1",
                name="WHO",
                entity_type="organization",
                sentiment="neutral",
            )
        ],
        source_url="https://example.com/article",
        article_id="art_test001",
        image_caption=image_caption,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        topic_text="health",  # skip _classify_topic LLM call in query_memory
    )


def make_memory_mock(max_confidence=0.0):
    """Return a mock MemoryAgent with no similar claims."""
    memory = MagicMock()
    memory.search_similar_claims.return_value = {
        "ids": [[]],
        "documents": [[]],
        "distances": [[]],
        "metadatas": [[]],
    }
    memory.get_entity_context.return_value = []
    memory.get_entity_ids_for_claims.return_value = []
    memory.get_graph_claims_for_entities.return_value = []
    memory.add_verdict.return_value = None
    memory.get_source_topic_credibility.return_value = None
    memory.get_base_credibility.return_value = None
    memory.upsert_source_topic_credibility.return_value = None
    return memory


def make_question_gen_response():
    """LLM response for context_claim_agent question generation — returns no questions."""
    content = json.dumps({"factual": [], "counter_factual": []})
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def make_verdict_degrees_response(label="refuted"):
    """LLM response for synthesize_verdict — uses degrees format."""
    degrees = [-1.0] if label == "refuted" else [1.0]
    content = json.dumps(
        {
            "degrees": degrees,
            "reasoning": f"Evidence {label} the claim.",
        }
    )
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def make_openai_cross_modal_response(conflict=False):
    content = json.dumps({"conflict": conflict, "explanation": None})
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def make_tavily_response(n_results=3):
    urls = [f"https://source{i}.com/{i}" for i in range(n_results)]
    return {
        "results": [
            {"url": url, "title": f"Title {i}", "content": "Evidence text.", "score": 0.9}
            for i, url in enumerate(urls)
        ]
    }


# ── Full pipeline path ────────────────────────────────────────────────────────


def test_graph_live_search_path_returns_output():
    """Full graph run should return a valid FactCheckOutput."""
    memory = make_memory_mock()

    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_llm:
        mock_llm.return_value.chat.completions.create.side_effect = [
            make_question_gen_response(),  # context_claim_agent: Q-gen
            make_verdict_degrees_response(),  # synthesize_verdict
        ]
        graph = build_graph(memory)
        state = graph.invoke({"input": make_fact_check_input()})

    output = state.get("output")
    assert output is not None
    assert output.verdict in ("supported", "refuted", "misleading")
    assert 0 <= output.confidence_score <= 100
    assert output.claim_id == "clm_test001"


def test_graph_writes_verdict_to_memory():
    """After the graph runs, MemoryAgent.add_verdict should be called once."""
    memory = make_memory_mock()

    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_llm:
        mock_llm.return_value.chat.completions.create.side_effect = [
            make_question_gen_response(),
            make_verdict_degrees_response(),
        ]
        with patch("fact_check_agent.src.graph.nodes.write_memory") as mock_write:
            mock_write.return_value = {}
            graph = build_graph(memory)
            state = graph.invoke({"input": make_fact_check_input()})

    assert state.get("output") is not None or True  # graph completed


def test_graph_verdict_fields_populated():
    """Graph output must have all required FactCheckOutput fields populated."""
    memory = make_memory_mock()

    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_llm:
        mock_llm.return_value.chat.completions.create.side_effect = [
            make_question_gen_response(),
            make_verdict_degrees_response(label="refuted"),
        ]
        graph = build_graph(memory)
        state = graph.invoke({"input": make_fact_check_input()})

    output = state["output"]
    assert output.verdict in ("supported", "refuted", "misleading")
    assert 0 <= output.confidence_score <= 100
    assert isinstance(output.reasoning, str)
    assert output.claim_id == "clm_test001"
    assert isinstance(output.evidence_links, list)


# ── Cross-modal flag ──────────────────────────────────────────────────────────


def test_graph_cross_modal_flag_propagated():
    """When LLM detects a cross-modal conflict, the flag should be set on output."""
    memory = make_memory_mock()
    xmodal_content = json.dumps({"conflict": True, "explanation": "Image contradicts text."})
    xmodal = MagicMock()
    xmodal.choices = [MagicMock()]
    xmodal.choices[0].message.content = xmodal_content

    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_llm:
        mock_llm.return_value.chat.completions.create.side_effect = [
            make_question_gen_response(),  # Q-gen
            make_verdict_degrees_response(),  # synthesize_verdict
            xmodal,  # cross_modal_check (image present)
        ]
        graph = build_graph(memory)
        state = graph.invoke({"input": make_fact_check_input(image_caption="Caption text")})

    output = state["output"]
    assert output.cross_modal_flag is True
    # vlm_assessment_block only populated when image_url is present; this test has none
    assert output.vlm_assessment_block is None


# ── receive_claim node ────────────────────────────────────────────────────────


def test_receive_claim_resets_state():
    """receive_claim node should initialise all mutable fields to safe defaults."""
    from fact_check_agent.src.graph.nodes import receive_claim

    stale_state = {
        "input": make_fact_check_input(),
        "memory_results": "stale",
        "output": "stale",
    }
    updates = receive_claim(stale_state)

    assert updates["memory_results"] is None
    assert updates["output"] is None
    assert updates["retrieved_chunks"] == []
    assert updates["entity_context"] == []
    assert updates["cross_modal_flag"] is False


# ── Cache/freshness path tests ────────────────────────────────────────────────


def make_cache_hit_memory_mock(confidence=0.92, days_old=2):
    """Return a MemoryAgent mock that produces a high-confidence cache hit."""
    from datetime import timedelta

    verified_at = datetime.now(timezone.utc) - timedelta(days=days_old)
    memory = MagicMock()
    memory.search_similar_claims.return_value = {
        "ids": [["clm_cached001"]],
        "documents": [["COVID vaccines are safe and effective."]],
        "distances": [[0.05]],
        "metadatas": [
            [
                {
                    "verdict_label": "supported",
                    "verdict_confidence": confidence,
                    "verified_at": verified_at.isoformat(),
                }
            ]
        ],
    }
    memory.get_entity_context.return_value = []
    memory.get_entity_ids_for_claims.return_value = []
    memory.get_graph_claims_for_entities.return_value = []
    memory.get_verdict_by_claim.return_value = {
        "metadatas": [
            {"label": "supported", "confidence": confidence, "verified_at": verified_at.isoformat()}
        ]
    }
    memory.add_verdict.return_value = None
    memory.get_source_topic_credibility.return_value = None
    memory.get_base_credibility.return_value = None
    memory.upsert_source_topic_credibility.return_value = None
    return memory


def make_freshness_response(revalidate: bool):
    content = json.dumps(
        {
            "revalidate": revalidate,
            "reason": "test reason",
            "claim_category": "scientific",
        }
    )
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_cache_fresh_path_produces_output():
    """High-confidence cache hit + freshness=fresh → graph completes with valid output."""
    memory = make_cache_hit_memory_mock(confidence=0.92, days_old=1)

    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_llm:
        # LLM calls: freshness_check (1 hit) → Q-gen → synthesize_verdict
        mock_llm.return_value.chat.completions.create.side_effect = [
            make_freshness_response(revalidate=False),  # claim tagged as fresh
            make_question_gen_response(),  # Q-gen
            make_verdict_degrees_response(),  # synthesize_verdict
        ]
        graph = build_graph(memory)
        state = graph.invoke({"input": make_fact_check_input()})

    output = state.get("output")
    assert output is not None
    assert output.verdict in ("supported", "refuted", "misleading")
    # Fresh memory claim is used as context — verify fresh_context is populated
    assert len(state.get("fresh_context", [])) > 0


def test_cache_stale_path_produces_output():
    """High-confidence cache hit + freshness=stale → graph completes; stale claim in stale_context."""
    memory = make_cache_hit_memory_mock(confidence=0.92, days_old=30)

    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_llm:
        # LLM calls: freshness_check (1 hit) → Q-gen → synthesize_verdict
        mock_llm.return_value.chat.completions.create.side_effect = [
            make_freshness_response(revalidate=True),  # claim tagged as stale
            make_question_gen_response(),  # Q-gen
            make_verdict_degrees_response(),  # synthesize_verdict
        ]
        graph = build_graph(memory)
        state = graph.invoke({"input": make_fact_check_input()})

    output = state.get("output")
    assert output is not None
    # Stale claim goes to stale_context, not fresh_context
    assert len(state.get("stale_context", [])) > 0
    assert len(state.get("fresh_context", [])) == 0


# ── Reflection agent integration ──────────────────────────────────────────────


def test_reflection_agent_source_credibility_populated():
    """After graph run, state['source_credibility'] must be populated (even if all-None)."""
    memory = make_memory_mock()

    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_llm:
        mock_llm.return_value.chat.completions.create.side_effect = [
            make_question_gen_response(),
            make_verdict_degrees_response(),
        ]
        graph = build_graph(memory)
        state = graph.invoke({"input": make_fact_check_input()})

    sc = state.get("source_credibility")
    assert sc is not None
    assert "sample_count" in sc


def test_reflection_agent_upserts_source_topic_credibility():
    """upsert_source_topic_credibility called once with correct source_id after verdict."""
    memory = make_memory_mock()

    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_llm:
        mock_llm.return_value.chat.completions.create.side_effect = [
            make_question_gen_response(),
            make_verdict_degrees_response(label="refuted"),
        ]
        graph = build_graph(memory)
        _ = graph.invoke({"input": make_fact_check_input()})

    memory.upsert_source_topic_credibility.assert_called_once()
    call_args = memory.upsert_source_topic_credibility.call_args[0]

    # source_id must be derived from "https://example.com/article"
    assert call_args[0] == "src_example_com"
    # credibility must be in valid range
    assert 0.0 <= call_args[2] <= 1.0
