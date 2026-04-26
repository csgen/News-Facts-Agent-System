"""
FactGuard - Agentic AI Fact Checking Dashboard
Task 3: Full-Stack & Evaluation Engineer

Run with: streamlit run frontend/app.py
"""

import os
import sys

# Ensure the project root is on sys.path so imports like `from agents...` work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FactGuard | AI Fact Checker",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

#MainMenu, footer, header {visibility: hidden;}

.main .block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
}

.verdict-card {
    background: linear-gradient(135deg, #1e2535 0%, #151b2e 100%);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.verdict-supported  { border-left: 4px solid #10b981; box-shadow: 0 0 20px rgba(16,185,129,0.08); }
.verdict-refuted    { border-left: 4px solid #ef4444; box-shadow: 0 0 20px rgba(239,68,68,0.08); }
.verdict-misleading { border-left: 4px solid #f59e0b; box-shadow: 0 0 20px rgba(245,158,11,0.08); }

.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-supported  { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-refuted    { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-misleading { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2535;
}

.metric-box   { background: #1e2535; border-radius: 12px; padding: 1rem 1.2rem; text-align: center; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: #a78bfa; }
.metric-label { font-size: 0.75rem; color: #64748b; margin-top: 2px; }

.bias-text { background: #1e2535; border-radius: 12px; padding: 1.2rem; line-height: 2; font-size: 0.95rem; }
.highlight-high { background: rgba(239,68,68,0.25);  border-radius: 4px; padding: 1px 4px; }
.highlight-med  { background: rgba(245,158,11,0.2);  border-radius: 4px; padding: 1px 4px; }
.highlight-low  { background: rgba(99,102,241,0.15); border-radius: 4px; padding: 1px 4px; }

.source-pill {
    display: inline-block;
    background: #1e2535;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.8rem;
    margin: 4px;
    color: #94a3b8;
}

.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #1e2535; padding: 6px; border-radius: 12px; }
.stTabs [data-baseweb="tab"]      { border-radius: 8px; font-family: 'Space Mono', monospace; font-size: 0.75rem; letter-spacing: 0.05em; color: #64748b; }
.stTabs [aria-selected="true"]    { background: #2d3748 !important; color: #a78bfa !important; }

.stTextArea textarea, .stTextInput input {
    background: #1e2535 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white; border: none; border-radius: 10px;
    font-family: 'Space Mono', monospace; font-size: 0.8rem;
    font-weight: 700; letter-spacing: 0.05em;
    padding: 0.6rem 2rem; transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.img-mismatch-warning { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); border-radius: 10px; padding: 0.8rem 1rem; color: #fca5a5; font-size: 0.85rem; }
.img-match-ok         { background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); border-radius: 10px; padding: 0.8rem 1rem; color: #6ee7b7; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# BACKEND INITIALIZATION (lazy — avoids startup crash if DBs not running)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _init_memory():
    """Initialize MemoryAgent singleton once per Streamlit session."""
    try:
        from agents.memory_agent import get_memory
        return get_memory(), None
    except Exception as e:
        return None, str(e)


def _get_memory():
    memory, err = _init_memory()
    return memory


# ─────────────────────────────────────────────
# REAL BACKEND FUNCTIONS
# ─────────────────────────────────────────────

def _run_entity_tracker_background(claim_text_or_name: str, direct_name: str = "") -> None:
    """
    Run the Entity Tracker in a background thread after every fact-check.

    Accepts either a claim text (extracts entities via spaCy) or a direct
    entity name string. Always includes `direct_name` if provided.
    """
    def _worker():
        try:
            from agents.entity_tracker import run_entity_tracker
            from agents.memory_agent import get_memory

            print(f"\n[bg_tracker] thread started  claim_text_or_name={claim_text_or_name!r}  direct_name={direct_name!r}")
            entity_names: set[str] = set()

            # Include direct entity name if given (e.g. from entity search box)
            # Also ensure the entity node exists in Neo4j — spaCy won't create
            # nodes for common nouns like "apple" or "bitcoin".
            if direct_name:
                clean = direct_name.strip()
                entity_names.add(clean)
                print(f"[bg_tracker] direct_name provided → adding {clean!r}")
                try:
                    memory = get_memory()
                    memory.ensure_entity_exists(clean)
                    print(f"[bg_tracker] ensure_entity_exists OK for {clean!r}")
                    linked = memory.backfill_mentions_for_entity(clean)
                    print(f"[bg_tracker] backfill_mentions linked {linked} existing claim(s) → entity {clean!r}")
                except Exception as _ex:
                    print(f"[bg_tracker] ensure/backfill FAILED: {_ex}")

            # Also extract from claim text via spaCy
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm")
                doc = nlp(claim_text_or_name)
                spacy_hits = [(e.text, e.label_) for e in doc.ents
                              if e.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "NORP", "FAC"}
                              and len(e.text.strip()) > 1]
                print(f"[bg_tracker] spaCy NER on claim text → {spacy_hits}")
                for ent_text, _ in spacy_hits:
                    entity_names.add(ent_text.strip())
            except Exception as _ex:
                print(f"[bg_tracker] spaCy failed: {_ex}")

            print(f"[bg_tracker] entity_names to track: {entity_names}")
            if not entity_names:
                print("[bg_tracker] nothing to track — exiting thread")
                return

            memory = get_memory()
            for name in entity_names:
                print(f"[bg_tracker] → run_entity_tracker({name!r})")
                run_entity_tracker(name, window_hours=720, memory=memory)
            print("[bg_tracker] thread done")
        except Exception as _ex:
            import traceback
            print(f"[bg_tracker] UNHANDLED ERROR: {_ex}")
            traceback.print_exc()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def get_real_verdict(query: str) -> dict:
    """
    Call Task 2's Fact-Check Agent (LangGraph pipeline) for a raw claim.

    Falls back to an error result if the pipeline is unavailable.
    """
    try:
        from agents.fact_check_agent import fact_check_claim
        output = fact_check_claim(query)

        return {
            "label":            output.verdict,
            "confidence":       output.confidence_score / 100,
            "claim_text":       query[:200],
            "evidence_summary": output.reasoning,
            "image_mismatch":   output.cross_modal_flag,
            "image_url":        None,
            "vlm_caption":      output.cross_modal_explanation or "",
            "sources":          [{"name": url.split("/")[2] if len(url.split("/")) > 2 else url,
                                  "url": url, "credibility": 0.5}
                                 for url in (output.evidence_links or [])[:4]],
            "charged_phrases":  [],  # TODO: integrate bias phrase detection
        }
    except Exception as e:
        st.error(f"Fact-Check Agent unavailable: {e}")
        return {
            "label":            "misleading",
            "confidence":       0.0,
            "claim_text":       query[:200],
            "evidence_summary": f"Pipeline error: {e}",
            "image_mismatch":   False,
            "image_url":        None,
            "vlm_caption":      "",
            "sources":          [],
            "charged_phrases":  [],
        }


def get_real_entity_history(entity_name: str) -> pd.DataFrame:
    """
    Query Graph DB for CredibilitySnapshot history for a given entity.

    Returns a DataFrame with columns: date, credibility_score, sentiment_score.
    Falls back to an empty DataFrame if the entity is not found.
    """
    memory = _get_memory()
    if memory is None:
        return _empty_entity_df()

    try:
        print(f"\n[get_entity_history] looking up entity {entity_name!r}")
        entity_dict = memory.get_entity_by_name(entity_name)
        print(f"[get_entity_history] get_entity_by_name result: {entity_dict}")
        if entity_dict is None:
            print("[get_entity_history] entity NOT found in Neo4j")
            st.info(f"No data yet for '{entity_name}'. Fact-check a claim mentioning them first — the tracker runs automatically.")
            return _empty_entity_df()

        print(f"[get_entity_history] entity found: id={entity_dict['entity_id']!r}")
        snapshots = memory.get_entity_snapshots(entity_dict["entity_id"], limit=60)
        print(f"[get_entity_history] snapshots returned: {len(snapshots)}")
        if not snapshots:
            print("[get_entity_history] entity exists but has 0 snapshots")
            st.info(f"'{entity_name}' was found but has no credibility snapshots yet. Check a claim mentioning them and the chart will populate.")
            return _empty_entity_df()

        rows = []
        for snap in snapshots:
            snap_at = snap["snapshot_at"]
            # Neo4j returns neo4j.time.DateTime — convert to Python datetime
            if hasattr(snap_at, "to_native"):
                snap_at = snap_at.to_native()
            rows.append({
                "date":             snap_at,
                "credibility_score": snap["credibility_score"],
                "sentiment_score":   snap["sentiment_score"],
            })

        return pd.DataFrame(rows)

    except Exception as e:
        st.warning(f"Could not load entity history: {e}")
        return _empty_entity_df()


def _empty_entity_df() -> pd.DataFrame:
    return pd.DataFrame({"date": [], "credibility_score": [], "sentiment_score": []})


def get_real_predictions(entity_name: str) -> list:
    """
    Query Graph DB for pending Prediction nodes for the given entity.

    Falls back to an empty list if not available.
    """
    memory = _get_memory()
    if memory is None:
        return []

    try:
        entity_dict = memory.get_entity_by_name(entity_name)
        if entity_dict is None:
            return []

        preds = memory.get_predictions_for_entity(entity_dict["entity_id"], include_resolved=False)
        result = []
        for p in preds:
            deadline = p.get("deadline")
            if hasattr(deadline, "to_native"):
                deadline = deadline.to_native()
            deadline_str = deadline.strftime("%Y-%m-%d") if deadline else "—"
            result.append({
                "text":       p["prediction_text"],
                "confidence": float(p.get("confidence") or 0.0),
                "deadline":   deadline_str,
                "outcome":    p.get("outcome"),
            })
        return result

    except Exception as e:
        st.warning(f"Could not load predictions: {e}")
        return []


# ─────────────────────────────────────────────
# HELPER FUNCTIONS — original UI helpers preserved
# ─────────────────────────────────────────────

def render_verdict_badge(label: str) -> str:
    icons = {"supported": "✓", "refuted": "✗", "misleading": "⚠"}
    icon = icons.get(label, "?")
    return f'<span class="badge badge-{label}">{icon} {label.upper()}</span>'


def render_confidence_gauge(confidence: float, label: str):
    color_map = {"supported": "#10b981", "refuted": "#ef4444", "misleading": "#f59e0b"}
    color = color_map.get(label, "#a78bfa")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={"suffix": "%", "font": {"size": 32, "color": "#e2e8f0", "family": "Space Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#475569", "size": 10}},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "#1e2535",
            "bordercolor": "#2d3748",
            "steps": [
                {"range": [0, 40], "color": "rgba(239,68,68,0.08)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.08)"},
                {"range": [70, 100], "color": "rgba(16,185,129,0.08)"},
            ],
            "threshold": {"line": {"color": color, "width": 2}, "thickness": 0.8, "value": confidence * 100}
        }
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e2e8f0"}
    )
    return fig


def render_credibility_chart(df: pd.DataFrame, entity_name: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["credibility_score"],
        mode="lines+markers", name="Credibility",
        line=dict(color="#a78bfa", width=2.5, shape="spline"),
        marker=dict(size=4, color="#a78bfa"),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.06)"
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["sentiment_score"],
        mode="lines", name="Sentiment",
        line=dict(color="#38bdf8", width=1.5, dash="dot", shape="spline"),
        yaxis="y2"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="DM Sans"),
        title=dict(text=f"Credibility Drift — {entity_name}", font=dict(color="#e2e8f0", size=14)),
        xaxis=dict(gridcolor="#1e2535", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#1e2535", showgrid=True, zeroline=False,
                   range=[0, 1], title=dict(text="Credibility", font=dict(color="#a78bfa"))),
        yaxis2=dict(overlaying="y", side="right", range=[-1, 1],
                    title=dict(text="Sentiment", font=dict(color="#38bdf8")), showgrid=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2d3748"),
        height=300, margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified"
    )
    return fig




# ─────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────

# ── Centered hero header ──────────────────────
st.markdown("""
<div style="text-align:center; margin-bottom:0.5rem; padding-top:0.5rem;">
  <div style="font-size:3rem; margin-bottom:0.3rem; line-height:1;">🛡️</div>
  <div style="font-family:'Space Mono',monospace; font-size:2.6rem; font-weight:700;
              color:#f0f4ff; letter-spacing:-1px; margin-bottom:0.35rem; line-height:1.1;">FactGuard</div>
  <div style="font-size:0.98rem; color:#64748b; margin-bottom:1.2rem;">
    Multi-Agent AI Fact-Checking System &middot; Powered by LLMs + Knowledge Graph
  </div>
  <div style="display:flex; justify-content:center; max-width:640px; margin:0 auto 1.6rem auto;">
    <div style="flex:1; text-align:center; font-size:0.82rem; font-weight:500;
                color:#94a3b8; padding:0.15rem 1rem; border-right:1px solid #2d3748;">Shantam Sharma</div>
    <div style="flex:1; text-align:center; font-size:0.82rem; font-weight:500;
                color:#94a3b8; padding:0.15rem 1rem; border-right:1px solid #2d3748;">Chen Sigen</div>
    <div style="flex:1; text-align:center; font-size:0.82rem; font-weight:500;
                color:#94a3b8; padding:0.15rem 1rem;">Ahmed Abdul Wasae</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Architecture Diagram ──────────────────────
import streamlit.components.v1 as _components

_components.html("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    background: #0a0e1a;
    font-family: 'DM Sans', sans-serif;
    overflow: hidden;
  }
  body { padding: 12px 20px 8px 20px; }
  .lbl {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 12px;
  }
  .row {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: nowrap;
    gap: 0;
  }
  .box {
    background: #1e2535;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 8px 10px;
    text-align: center;
    min-width: 90px;
    max-width: 108px;
    flex-shrink: 0;
  }
  .box-fc   { border-color: rgba(56,189,248,0.45); background: rgba(56,189,248,0.07); }
  .box-pred { border-color: rgba(52,211,153,0.45); background: rgba(52,211,153,0.07); }
  .icon { font-size: 1.25rem; margin-bottom: 3px; line-height: 1.3; }
  .name { font-family: 'Space Mono', monospace; font-size: 9px; font-weight: 700; color: #a78bfa; line-height: 1.4; }
  .sub  { font-size: 8px; color: #475569; margin-top: 3px; }
  .sub-fc   { color: #38bdf8; }
  .sub-pred { color: #34d399; }
  .arr { display: flex; align-items: center; flex-shrink: 0; padding: 0 1px; }
  @keyframes flow { from { stroke-dashoffset: 20; } to { stroke-dashoffset: 0; } }
  .fp { stroke-dasharray: 5 4; animation: flow 0.8s linear infinite; }
</style>
</head>
<body>
  <div class="lbl">&#128295;&nbsp; System Architecture — Agent Pipeline</div>
  <div class="row">

    <div class="box">
      <div class="icon">🛡️</div>
      <div class="name">Input<br>Guardrail</div>
      <div class="sub">Layer A+B</div>
    </div>
    <div class="arr"><svg width="40" height="16" viewBox="0 0 40 16">
      <path class="fp" d="M0,8 L32,8" stroke="#a78bfa" stroke-width="2" fill="none"/>
      <polygon points="32,3 40,8 32,13" fill="#a78bfa"/>
    </svg></div>

    <div class="box">
      <div class="icon">⚙️</div>
      <div class="name">Preprocessing<br>Agent</div>
      <div class="sub">NER · Claims</div>
    </div>
    <div class="arr"><svg width="40" height="16" viewBox="0 0 40 16">
      <path class="fp" d="M0,8 L32,8" stroke="#a78bfa" stroke-width="2" fill="none"/>
      <polygon points="32,3 40,8 32,13" fill="#a78bfa"/>
    </svg></div>

    <div class="box box-fc">
      <div class="icon">🔍</div>
      <div class="name">Fact-Check<br>Agent</div>
      <div class="sub sub-fc">LangGraph</div>
    </div>
    <div class="arr"><svg width="40" height="16" viewBox="0 0 40 16">
      <path class="fp" d="M0,8 L32,8" stroke="#a78bfa" stroke-width="2" fill="none"/>
      <polygon points="32,3 40,8 32,13" fill="#a78bfa"/>
    </svg></div>

    <div class="box">
      <div class="icon">🧠</div>
      <div class="name">Memory<br>Agent</div>
      <div class="sub">Neo4j · Chroma</div>
    </div>
    <div class="arr"><svg width="40" height="16" viewBox="0 0 40 16">
      <path class="fp" d="M0,8 L32,8" stroke="#a78bfa" stroke-width="2" fill="none"/>
      <polygon points="32,3 40,8 32,13" fill="#a78bfa"/>
    </svg></div>

    <div class="box">
      <div class="icon">📊</div>
      <div class="name">Entity<br>Tracker</div>
      <div class="sub">Credibility</div>
    </div>
    <div class="arr"><svg width="40" height="16" viewBox="0 0 40 16">
      <path class="fp" d="M0,8 L32,8" stroke="#a78bfa" stroke-width="2" fill="none"/>
      <polygon points="32,3 40,8 32,13" fill="#a78bfa"/>
    </svg></div>

    <div class="box box-pred">
      <div class="icon">🔮</div>
      <div class="name">Prediction<br>Agent</div>
      <div class="sub sub-pred">Trend · Forecast</div>
    </div>

  </div>
</body>
</html>""", height=148, scrolling=False)

st.divider()

# ── Input Section ──
col_input, col_btn = st.columns([5, 1])
with col_input:
    user_input = st.text_area(
        "Paste a news claim, article URL, or text snippet",
        placeholder="e.g. '500,000 Tesla vehicles were recalled due to brake defects' or paste a full article URL...",
        height=100,
        label_visibility="collapsed"
    )
with col_btn:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    run_btn = st.button("⚡ VERIFY", use_container_width=True)

# ── Entity search ──
# _auto_entity holds the last auto-detected entity name.
# We pass it as value= so Streamlit doesn't complain about modifying
# a keyed widget after instantiation.
if "_auto_entity" not in st.session_state:
    st.session_state["_auto_entity"] = ""

entity_col, _ = st.columns([3, 7])
with entity_col:
    entity_query = st.text_input(
        "Track an entity",
        value=st.session_state["_auto_entity"],
        placeholder="e.g. Elon Musk, Tesla, WHO...",
        label_visibility="visible",
    )
    # Sync whatever the user typed back into session_state so the background
    # tracker can read it when the Verify button fires.
    if entity_query.strip():
        st.session_state["_auto_entity"] = entity_query.strip()
        print(f"[entity_box] user typed entity: {entity_query.strip()!r}")

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RESULTS (shown after clicking VERIFY)
# ─────────────────────────────────────────────
if run_btn and user_input.strip():
    # ── Layer A + B Input Guardrail — runs BEFORE the pipeline ───────────
    try:
        from agents.input_guardrail import check_input
        _guard = check_input(user_input)
        if _guard["blocked"]:
            st.error(
                f"⚠️ **Input blocked** [{_guard['risk']} risk]\n\n"
                f"{_guard['reason']}\n\n"
                f"*Layer {'A — rule-based filter' if _guard['layer'] == 'A' else 'B — AI safety classifier'} triggered.*"
            )
            st.stop()
    except Exception as _ge:
        pass  # guardrail failure never blocks the pipeline

    with st.spinner("🔍 Agents working: Scraping → Preprocessing → Fact-Checking..."):
        result = get_real_verdict(user_input)

    # ── Auto-fill entity box + run tracker ──────────────────────────────
    # Priority 1: whatever the user already typed in the entity search box
    # Priority 2: first named entity spaCy finds in the claim text
    # Priority 3: first meaningful word in a URL path (e.g. /story/can-apple-seeds → "apple")
    _auto_entity = st.session_state.get("_auto_entity", "").strip()

    if not _auto_entity:
        # Try spaCy NER on the claim text
        try:
            import spacy as _spacy
            _nlp = _spacy.load("en_core_web_sm")
            _doc = _nlp(user_input)
            _valid_labels = {"PERSON", "ORG", "GPE", "PRODUCT", "NORP", "FAC"}
            _found = [e.text.strip() for e in _doc.ents
                      if e.label_ in _valid_labels and len(e.text.strip()) > 1]
            if _found:
                _auto_entity = _found[0]
        except Exception:
            pass

    if not _auto_entity and user_input.strip().startswith("http"):
        # Extract first meaningful word from URL path
        try:
            from urllib.parse import urlparse
            _path = urlparse(user_input).path          # e.g. /story/can-apple-seeds-kill-you
            _words = [w for w in _path.replace("/", "-").split("-")
                      if len(w) > 3 and w.isalpha()]   # skip short words like "can", "a"
            if _words:
                _auto_entity = _words[0].capitalize()  # "apple" → "Apple"
        except Exception:
            pass

    if _auto_entity:
        st.session_state["_auto_entity"] = _auto_entity
        # Run entity tracker for this entity in the background
        _run_entity_tracker_background("", direct_name=_auto_entity)
        st.caption(f"Entity tracker running for **{_auto_entity}** in background — switch to the Entity & Trend tab in ~5 seconds to see results.")

    # ── 3 TABS ──
    tab1, tab2, tab3 = st.tabs([
        "📋  Fact Verdict",
        "🎭  Perspective & Bias",
        "📈  Entity & Trend"
    ])

    # ─────────────────────
    # TAB 1: FACT VERDICT
    # ─────────────────────
    with tab1:
        st.markdown('<div class="section-header">Verification Result</div>', unsafe_allow_html=True)

        left, right = st.columns([3, 2])

        with left:
            label = result["label"]
            st.markdown(f"""
            <div class="verdict-card verdict-{label}">
                <div style="margin-bottom:0.8rem">{render_verdict_badge(label)}</div>
                <div style="font-size:0.85rem; color:#94a3b8; margin-bottom:0.8rem; font-style:italic;">
                    "{result['claim_text']}"
                </div>
                <div style="font-size:0.9rem; color:#cbd5e1; line-height:1.7;">
                    {result['evidence_summary']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Verified sources
            st.markdown('<div class="section-header" style="margin-top:1rem">Verified Sources</div>', unsafe_allow_html=True)
            if result["sources"]:
                source_html = ""
                for src in result["sources"]:
                    cred_color = "#10b981" if src["credibility"] > 0.7 else "#ef4444"
                    source_html += f'<span class="source-pill">🔗 {src["name"]} <span style="color:{cred_color}; font-weight:600;">{src["credibility"]:.0%}</span></span>'
                st.markdown(source_html, unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#475569; font-size:0.85rem;">No external sources retrieved.</span>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-header">Confidence Score</div>', unsafe_allow_html=True)
            st.plotly_chart(render_confidence_gauge(result["confidence"], label),
                            use_container_width=True, config={"displayModeBar": False})

            st.markdown('<div class="section-header" style="margin-top:0.5rem">Image Cross-Check</div>', unsafe_allow_html=True)
            if result["image_mismatch"]:
                explanation = result.get("vlm_caption") or "The article image does not match the described event context."
                st.markdown(f"""
                <div class="img-mismatch-warning">
                    ⚠️ <strong>Image Mismatch Detected</strong><br>
                    {explanation}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="img-match-ok">
                    ✓ <strong>Image Consistent</strong><br>
                    Image content aligns with the article claim.
                </div>
                """, unsafe_allow_html=True)

            if result.get("image_url"):
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                st.image(result["image_url"],
                         caption=f"VLM Caption: {result['vlm_caption']}" if result.get("vlm_caption") else None,
                         use_container_width=True)

    # ─────────────────────
    # TAB 2: BIAS
    # ─────────────────────
    with tab2:
        st.markdown('<div class="section-header">Bias Analysis</div>', unsafe_allow_html=True)

        b_left, b_right = st.columns([3, 2])

        with b_left:
            st.markdown('<div class="section-header">Emotionally Charged Text</div>', unsafe_allow_html=True)
            highlighted = user_input
            for phrase in result.get("charged_phrases", []):
                css_class = f"highlight-{phrase['intensity']}"
                highlighted = highlighted.replace(
                    phrase["text"],
                    f'<span class="{css_class}">{phrase["text"]}</span>'
                )
            st.markdown(f'<div class="bias-text">{highlighted}</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="margin-top:0.8rem; font-size:0.78rem; color:#64748b; display:flex; gap:16px;">
                <span><span class="highlight-high">■</span> High intensity</span>
                <span><span class="highlight-med">■</span> Medium</span>
                <span><span class="highlight-low">■</span> Low</span>
            </div>
            """, unsafe_allow_html=True)

        with b_right:
            st.markdown('<div class="section-header">Source Credibility</div>', unsafe_allow_html=True)
            confidence = result.get("confidence", 0.0)
            conf_label = "High" if confidence >= 0.65 else ("Moderate" if confidence >= 0.35 else "Low")
            conf_color = "#10b981" if confidence >= 0.65 else ("#f59e0b" if confidence >= 0.35 else "#ef4444")
            st.markdown(f"""
            <div class="metric-box" style="margin-top:0.5rem">
                <div class="metric-value" style="color:{conf_color};">{confidence:.0%}</div>
                <div class="metric-label">Verdict Confidence — {conf_label}</div>
            </div>
            """, unsafe_allow_html=True)

    # ─────────────────────
    # TAB 3: ENTITY & TREND
    # ─────────────────────
    with tab3:
        entity_name = entity_query.strip() if entity_query.strip() else "Tesla"
        st.markdown(f'<div class="section-header">Entity Profile — {entity_name}</div>', unsafe_allow_html=True)

        df = get_real_entity_history(entity_name)

        if df.empty:
            st.info(f"No credibility history yet for '{entity_name}'. Check a claim mentioning them and it will appear here automatically.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            current_cred = df["credibility_score"].iloc[-1]
            cred_change  = df["credibility_score"].iloc[-1] - df["credibility_score"].iloc[0]
            avg_sent     = df["sentiment_score"].mean()

            with m1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color:{'#10b981' if current_cred > 0.6 else '#ef4444'};">
                        {current_cred:.0%}
                    </div>
                    <div class="metric-label">Current Credibility</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                arrow = "↓" if cred_change < 0 else "↑"
                col = "#ef4444" if cred_change < 0 else "#10b981"
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color:{col};">{arrow} {abs(cred_change):.0%}</div>
                    <div class="metric-label">Drift ({len(df)}-snapshot span)</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                sent_label = "Positive" if avg_sent > 0.2 else ("Negative" if avg_sent < -0.1 else "Neutral")
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value" style="font-size:1.4rem;">{sent_label}</div>
                    <div class="metric-label">Avg Sentiment</div>
                </div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{len(df)}</div>
                    <div class="metric-label">Snapshots Tracked</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.plotly_chart(render_credibility_chart(df, entity_name),
                            use_container_width=True, config={"displayModeBar": False})

        # Predictions
        st.markdown('<div class="section-header" style="margin-top:0.5rem">AI Predictions</div>', unsafe_allow_html=True)
        predictions = get_real_predictions(entity_name)

        if not predictions:
            st.info(f"No predictions yet for '{entity_name}'. They generate automatically after credibility history builds up.")
        else:
            for pred in predictions:
                conf_color = "#10b981" if pred["confidence"] > 0.7 else "#f59e0b"
                st.markdown(f"""
                <div class="verdict-card" style="margin-bottom:0.7rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                        <span style="font-size:0.75rem; color:#64748b; font-family:'Space Mono',monospace;">
                            📅 Deadline: {pred['deadline']}
                        </span>
                        <span style="font-size:0.75rem; font-weight:700; color:{conf_color}; font-family:'Space Mono',monospace;">
                            {pred['confidence']:.0%} confidence
                        </span>
                    </div>
                    <div style="color:#cbd5e1; font-size:0.9rem;">{pred['text']}</div>
                </div>
                """, unsafe_allow_html=True)

elif run_btn and not user_input.strip():
    st.warning("Please enter a news claim or article URL to verify.")

# ─────────────────────────────────────────────
# EMPTY STATE (no query yet)
# ─────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div style="text-align:center; padding:4rem 0; color:#334155;">
        <div style="font-size:4rem; margin-bottom:1rem;">🛡️</div>
        <div style="font-family:'Space Mono',monospace; font-size:1rem; color:#475569;">
            Paste a claim above and click VERIFY to begin fact-checking
        </div>
        <div style="margin-top:1rem; font-size:0.8rem; color:#334155;">
            Powered by 6 specialized AI agents · Real-time web search · Knowledge Graph memory
        </div>
    </div>
    """, unsafe_allow_html=True)
