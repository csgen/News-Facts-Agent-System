"""
FactGuard - Agentic AI Fact Checking Dashboard
Task 3: Full-Stack & Evaluation Engineer

Run with: streamlit run frontend/app.py
"""

import hashlib
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on sys.path so imports like `from agents...` work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# SECOPS — LOGGING SETUP
# ─────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_guardrail_log = logging.getLogger("guardrail.blocked")
_guardrail_handler = logging.FileHandler(_LOG_DIR / "guardrail_blocked.log")
_guardrail_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
_guardrail_log.addHandler(_guardrail_handler)
_guardrail_log.setLevel(logging.WARNING)

_hitl_log = logging.getLogger("hitl.audit")
_hitl_handler = logging.FileHandler(_LOG_DIR / "hitl_audit.log")
_hitl_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
_hitl_log.addHandler(_hitl_handler)
_hitl_log.setLevel(logging.INFO)


# ── Rate limiter ──────────────────────────────
_RATE_LIMIT = 10  # max claims per session


def _check_rate_limit() -> bool:
    """Return True if the user is within the rate limit, False if exceeded."""
    if "_request_count" not in st.session_state:
        st.session_state["_request_count"] = 0
    return st.session_state["_request_count"] < _RATE_LIMIT


def _increment_rate_counter() -> None:
    st.session_state["_request_count"] = st.session_state.get("_request_count", 0) + 1


# ── Output schema validator ───────────────────
_VALID_LABELS = {"supported", "misleading", "refuted"}


def _validate_verdict(result: dict) -> tuple[bool, str]:
    """Validate verdict dict against expected schema.

    Returns (is_valid, error_message).
    Checks: required keys present, label is one of the 3 valid values,
    confidence is a float in [0, 1].
    """
    required = {"verdict_id", "label", "confidence", "claim_text", "evidence_summary"}
    missing = required - result.keys()
    if missing:
        return False, f"Missing fields: {missing}"
    if result["label"] not in _VALID_LABELS:
        return False, f"Invalid label '{result['label']}' — must be one of {_VALID_LABELS}"
    try:
        conf = float(result["confidence"])
        if not 0.0 <= conf <= 1.0:
            return False, f"Confidence {conf} out of range [0, 1]"
    except (TypeError, ValueError):
        return False, f"Confidence is not a number: {result['confidence']}"
    return True, ""


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


def _scrape_article(url: str) -> dict:
    """
    Fetch an article URL and extract title, body snippet, and og:image.
    Returns dict with keys: title, body, image_url, claim_text
    """
    import re
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # ── Title ──
        title = ""
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        elif soup.find("title"):
            title = soup.find("title").get_text(strip=True)

        # ── Description / lede ──
        description = ""
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            description = og_desc["content"].strip()
        else:
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                description = meta_desc["content"].strip()

        # ── Body text (first ~600 chars of article paragraphs) ──
        body = ""
        for tag in soup.find_all(["article", "main", "div"], limit=5):
            paras = tag.find_all("p")
            if len(paras) >= 3:
                body = " ".join(p.get_text(strip=True) for p in paras[:6])
                body = re.sub(r"\s+", " ", body)[:600]
                break
        if not body:
            paras = soup.find_all("p")
            body = " ".join(p.get_text(strip=True) for p in paras[:6])
            body = re.sub(r"\s+", " ", body)[:600]

        # ── Image URL ──
        image_url = ""
        og_img = soup.find("meta", property="og:image")
        if og_img and og_img.get("content"):
            image_url = og_img["content"].strip()

        # ── Build claim text for the pipeline ──
        parts = [p for p in [title, description] if p]
        claim_text = ". ".join(parts) if parts else (body[:300] if body else url)

        return {
            "title":      title,
            "body":       body,
            "image_url":  image_url,
            "claim_text": claim_text,
            "error":      None,
        }

    except Exception as e:
        return {
            "title":      "",
            "body":       "",
            "image_url":  "",
            "claim_text": url,
            "error":      str(e),
        }


def get_real_verdict(query: str) -> dict:
    """
    Call Task 2's Fact-Check Agent (LangGraph pipeline) for a raw claim or URL.

    If query is a URL, scrapes the article first and fact-checks the extracted text.
    Falls back to an error result if the pipeline is unavailable.
    """
    # ── URL detection: scrape article first ──────────────────────────────
    _image_url   = ""
    _article_url = ""
    _display_claim = query[:200]

    if query.strip().startswith(("http://", "https://")):
        _article_url = query.strip()
        with st.status("🌐 Fetching article...", expanded=False) as _s:
            scraped = _scrape_article(_article_url)
            if scraped["error"]:
                _s.update(label=f"⚠️ Could not fetch article: {scraped['error']}", state="error")
            else:
                _s.update(label=f"✅ Article loaded: {scraped['title'] or _article_url}", state="complete")

        _image_url     = scraped["image_url"]
        _display_claim = scraped["claim_text"][:200]
        claim_for_pipeline = scraped["claim_text"]

        # ── SecOps: scan scraped content for indirect prompt injection ────────
        # The URL itself may be clean but the article body could contain injections.
        try:
            from agents.input_guardrail import layer_a_check
            _content_guard = layer_a_check(claim_for_pipeline)
            if _content_guard["blocked"]:
                _content_hash = hashlib.sha256(claim_for_pipeline.encode()).hexdigest()[:16]
                _guardrail_log.warning(
                    "BLOCKED_URL_CONTENT | hash=%s | url=%s | risk=%s | reason=%s",
                    _content_hash, _article_url[:80], _content_guard["risk"], _content_guard["reason"],
                )
                return {
                    "verdict_id":       "blocked",
                    "label":            "misleading",
                    "confidence":       0.0,
                    "claim_text":       claim_for_pipeline[:200],
                    "evidence_summary": (
                        f"⚠️ Article content blocked [{_content_guard['risk']} risk]: "
                        f"{_content_guard['reason']}"
                    ),
                    "sources": [],
                    "image_url": "",
                    "entities": [],
                }
        except Exception:
            pass  # content scan failure never blocks the pipeline

        # Show what was extracted so user knows the pipeline is working on real content
        if scraped["title"]:
            st.info(f"**Article:** {scraped['title']}\n\n**Fact-checking:** {scraped['claim_text'][:200]}")
    else:
        claim_for_pipeline = query

    # ── Human-correction cache: skip pipeline if we already have a human verdict ──
    try:
        _mem = _get_memory()
        if _mem:
            print(f"[human_cache] searching for corrected verdict for: {claim_for_pipeline[:80]!r}")
            _human = _mem.find_human_verdict_for_claim(claim_for_pipeline)
            print(f"[human_cache] result: {_human}")
            if _human:
                print(f"[human_cache] HIT — returning human-corrected verdict: {_human.get('label')} {_human.get('confidence')}")
                return {
                    "verdict_id":       _human.get("verdict_id"),
                    "label":            _human.get("label", "misleading"),
                    "confidence":       float(_human.get("confidence", 0.5)),
                    "claim_text":       _display_claim,
                    "evidence_summary": "✅ This verdict was previously corrected by a human reviewer.",
                    "bias_score":       float(_human.get("bias_score", 0.5)),
                    "image_mismatch":   bool(_human.get("image_mismatch", False)),
                    "image_url":        _image_url,
                    "vlm_caption":      "",
                    "sources":          [],
                    "charged_phrases":  [],
                }
            else:
                print("[human_cache] MISS — running pipeline")
    except Exception as _hce:
        print(f"[human_cache] ERROR (falling through to pipeline): {_hce}")

    try:
        from agents.fact_check_agent import fact_check_claim
        output = fact_check_claim(
            claim_for_pipeline,
            source_url=_article_url or "https://unknown.source",
            image_url=_image_url or None,
        )

        # ── Store Claim + MENTIONS edges so entity tracker can count this claim ──
        try:
            import spacy as _spacy_cm
            _nlp_cm  = _spacy_cm.load("en_core_web_sm")
            _doc_cm  = _nlp_cm(claim_for_pipeline)
            _valid   = {"PERSON", "ORG", "GPE", "PRODUCT", "NORP", "FAC", "EVENT"}
            _ent_dicts = [
                {
                    "entity_id": "ent_" + e.text.lower().strip().replace(" ", "_"),
                    "name":      e.text.strip(),
                    "entity_type": e.label_.lower(),
                    "sentiment": "neutral",
                }
                for e in _doc_cm.ents if e.label_ in _valid and len(e.text.strip()) > 1
            ]
            # Save detected entity names so the UI can show them
            import streamlit as _st_inner
            _st_inner.session_state["_auto_detected_entities"] = [e["name"] for e in _ent_dicts]

            _mem_cm = _get_memory()
            if _mem_cm and _ent_dicts:
                # Use output.claim_id — same ID the pipeline used to create
                # the Claim node and [:VERIFIED_AS]→Verdict edge in Neo4j.
                # This ensures [:MENTIONS] edges land on the SAME Claim node.
                _article_id_cm = _article_url or f"art_{output.verdict_id}"
                _mem_cm.auto_store_claim_with_entities(
                    claim_id=output.claim_id,
                    claim_text=claim_for_pipeline,
                    article_id=_article_id_cm,
                    entity_dicts=_ent_dicts,
                )
                print(f"[claim_store] claim_id={output.claim_id} — stored {len(_ent_dicts)} entities: {[e['name'] for e in _ent_dicts]}")
        except Exception as _cse:
            print(f"[claim_store] skipped: {_cse}")
            import streamlit as _st_inner
            _st_inner.session_state["_auto_detected_entities"] = []

        return {
            "verdict_id":       output.verdict_id,
            "label":            output.verdict,
            "confidence":       output.confidence_score / 100,
            "claim_text":       _display_claim,
            "evidence_summary": output.reasoning,
            "image_mismatch":   output.cross_modal_flag,
            "image_url":        _image_url,
            "vlm_caption":      output.cross_modal_explanation or "",
            "sources":          [{"name": url.split("/")[2] if len(url.split("/")) > 2 else url,
                                  "url": url, "credibility": 0.5}
                                 for url in (output.evidence_links or [])[:4]],
            "charged_phrases":  [],
        }
    except Exception as e:
        st.error(f"Fact-Check Agent unavailable: {e}")
        return {
            "verdict_id":       None,
            "label":            "misleading",
            "confidence":       0.0,
            "claim_text":       _display_claim,
            "evidence_summary": f"Pipeline error: {e}",
            "image_mismatch":   False,
            "image_url":        _image_url,
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
            df = _empty_entity_df()
            df.attrs["entity_found"] = False
            df.attrs["snapshot_count"] = 0
            return df

        print(f"[get_entity_history] entity found: id={entity_dict['entity_id']!r}")
        snapshots = memory.get_entity_snapshots(entity_dict["entity_id"], limit=60)
        print(f"[get_entity_history] snapshots returned: {len(snapshots)}")
        if not snapshots:
            print("[get_entity_history] entity exists but has 0 snapshots")
            # Return empty df with a sentinel so Tab 3 can show a specific message
            df = _empty_entity_df()
            df.attrs["entity_found"] = True
            df.attrs["snapshot_count"] = 0
            return df

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
    # ── SecOps: Rate Limiting ─────────────────────────────────────────────
    if not _check_rate_limit():
        st.error(
            f"⚠️ **Rate limit reached** — you have submitted {_RATE_LIMIT} claims this session. "
            "Please refresh the page to start a new session."
        )
        st.stop()

    # ── Layer A + B Input Guardrail — runs BEFORE the pipeline ───────────
    try:
        from agents.input_guardrail import check_input
        _guard = check_input(user_input)
        if _guard["blocked"]:
            # SecOps: log blocked input (hashed for privacy)
            _input_hash = hashlib.sha256(user_input.encode()).hexdigest()[:16]
            _guardrail_log.warning(
                "BLOCKED | hash=%s | layer=%s | risk=%s | reason=%s",
                _input_hash, _guard["layer"], _guard["risk"], _guard["reason"],
            )
            st.error(
                f"⚠️ **Input blocked** [{_guard['risk']} risk]\n\n"
                f"{_guard['reason']}\n\n"
                f"*Layer {'A — rule-based filter' if _guard['layer'] == 'A' else 'B — AI safety classifier'} triggered.*"
            )
            st.stop()
    except Exception as _ge:
        pass  # guardrail failure never blocks the pipeline

    # ── Increment rate counter AFTER guardrail passes ────────────────────
    _increment_rate_counter()

    with st.spinner("🔍 Agents working: Scraping → Preprocessing → Fact-Checking..."):
        result = get_real_verdict(user_input)

    # ── SecOps: Output Schema Validation ─────────────────────────────────
    _valid, _err = _validate_verdict(result)
    if not _valid:
        logging.warning("[schema] Invalid verdict returned: %s — %s", result, _err)
        result["label"]      = "misleading"
        result["confidence"] = 0.0
        result["evidence_summary"] = (
            f"[Schema validation failed: {_err}] "
            + result.get("evidence_summary", "")
        )

    # ── Persist result so widget interactions (slider, radio) don't wipe it ──
    st.session_state["_last_result"]       = result
    st.session_state["_last_user_input"]   = user_input
    st.session_state["_last_claim_text"]   = result.get("claim_text", user_input)

# ── Restore result on reruns caused by widget interactions ───────────────────
if not run_btn and "last_result" not in st.session_state:
    # first load, nothing to show
    pass

_cached_result     = st.session_state.get("_last_result")
_cached_user_input = st.session_state.get("_last_user_input", "")

if _cached_result and (run_btn or not run_btn):
    result     = _cached_result
    user_input = _cached_user_input if not run_btn else user_input

    # ── Auto-fill entity box + run tracker ──────────────────────────────
    # Priority 1 (highest): whatever the user typed in the entity search box — NEVER override this
    # Priority 2: first auto-detected entity from spaCy (only when box is empty)
    # Priority 3: spaCy NER on scraped claim text (only when box is empty)
    # Priority 4: first meaningful word from URL path (last resort)

    _user_typed_entity = entity_query.strip()  # what the user actually typed right now

    if _user_typed_entity:
        # User typed something — always use that, never overwrite
        _auto_entity = _user_typed_entity
    else:
        # Box is empty — try auto-detection in priority order
        _auto_entity = ""

        # From spaCy run inside get_real_verdict (fastest, already done)
        _pre_detected = st.session_state.get("_auto_detected_entities", [])
        if _pre_detected:
            _auto_entity = _pre_detected[0]

        if not _auto_entity:
            # Re-run spaCy on the scraped claim text as fallback
            _claim_text_for_ner = st.session_state.get("_last_claim_text", "")
            if _claim_text_for_ner:
                try:
                    import spacy as _spacy
                    _nlp = _spacy.load("en_core_web_sm")
                    _doc = _nlp(_claim_text_for_ner)
                    _valid_labels = {"PERSON", "ORG", "GPE", "PRODUCT", "NORP", "FAC"}
                    _found = [e.text.strip() for e in _doc.ents
                              if e.label_ in _valid_labels and len(e.text.strip()) > 1]
                    if _found:
                        _auto_entity = _found[0]
                except Exception:
                    pass

        if not _auto_entity and user_input.strip().startswith("http"):
            try:
                from urllib.parse import urlparse
                _path = urlparse(user_input).path
                _words = [w for w in _path.replace("/", "-").split("-")
                          if len(w) > 3 and w.isalpha()]
                if _words:
                    _auto_entity = _words[0].capitalize()
            except Exception:
                pass

    if _auto_entity and run_btn:
        st.session_state["_auto_entity"] = _auto_entity
        # Sync into the entity search box so Tab 3 shows this entity
        if not entity_query.strip():
            st.session_state["_auto_entity"] = _auto_entity
        _claim_for_tracker = st.session_state.get("_last_claim_text", "")
        _run_entity_tracker_background(_claim_for_tracker, direct_name=_auto_entity)

    # ── Show tracked entities banner (typed + auto-detected) ────────────
    if run_btn:
        _typed      = st.session_state.get("_auto_entity", "").strip()
        _auto_ents  = st.session_state.get("_auto_detected_entities", [])
        # Merge: typed entity first, then auto-detected (deduplicated, case-insensitive)
        _seen       = set()
        _all_tracked = []
        for _n in ([_typed] if _typed else []) + _auto_ents:
            if _n and _n.lower() not in _seen:
                _seen.add(_n.lower())
                _all_tracked.append(_n)

        if _all_tracked:
            _pills = "".join(
                f'<span style="display:inline-block; background:#1e2535; border:1px solid #2d3748; '
                f'border-radius:999px; padding:2px 10px; font-size:0.75rem; color:#a78bfa; '
                f'margin:2px 4px 2px 0; font-family:\'Space Mono\',monospace;">'
                f'📊 {n}</span>'
                for n in _all_tracked
            )
            st.markdown(
                f'<div style="margin-bottom:0.6rem; font-size:0.78rem; color:#64748b;">'
                f'Entity tracker running for: {_pills}'
                f'<span style="color:#475569; margin-left:6px;">— switch to '
                f'<strong style="color:#94a3b8;">Entity &amp; Trend</strong> tab in ~5s</span></div>',
                unsafe_allow_html=True,
            )

    # ── 2 TABS ──
    tab1, tab2 = st.tabs([
        "📋  Fact Verdict",
        "📈  Entity & Trend"
    ])

    # ─────────────────────
    # TAB 1: FACT VERDICT
    # ─────────────────────
    with tab1:
        # Show one-shot feedback toast after correction rerun
        if "_feedback_toast" in st.session_state:
            st.success(st.session_state.pop("_feedback_toast"))

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
            _vlm_caption = (result.get("vlm_caption") or "").strip()
            _image_url   = (result.get("image_url") or "").strip()

            if result["image_mismatch"]:
                explanation = _vlm_caption or "The article image does not match the described event context."
                st.markdown(f"""
                <div class="img-mismatch-warning">
                    ⚠️ <strong>Image Mismatch Detected</strong><br>
                    <span style="font-size:0.82rem;">{explanation}</span>
                </div>
                """, unsafe_allow_html=True)
            elif _image_url:
                # Image was scraped — check if vision model ran
                if _vlm_caption:
                    st.markdown(f"""
                    <div class="img-match-ok">
                        ✓ <strong>Image Consistent</strong><br>
                        <span style="font-size:0.82rem; color:#94a3b8; font-style:italic;">
                            What the image shows:<br>{_vlm_caption}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Image found but USE_CROSS_MODAL=false — show image, note vision check is off
                    st.markdown("""
                    <div style="background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.25);
                                border-radius:10px; padding:0.8rem 1rem; font-size:0.82rem; color:#94a3b8;">
                        🖼️ <strong style="color:#38bdf8;">Article image found</strong><br>
                        Vision cross-check is disabled (<code>USE_CROSS_MODAL=false</code>).
                        Enable it in <code>.env</code> to get an AI description of this image and verify it matches the claim.
                    </div>
                    """, unsafe_allow_html=True)
                st.image(_image_url,
                         caption=f"VLM Caption: {_vlm_caption}" if _vlm_caption else "Article image (vision analysis disabled)",
                         use_container_width=True)
            else:
                # No image URL was present in this claim/article
                st.markdown("""
                <div style="background:rgba(71,85,105,0.15); border:1px solid rgba(71,85,105,0.3);
                            border-radius:10px; padding:0.8rem 1rem; font-size:0.82rem; color:#64748b;">
                    🖼️ <strong style="color:#94a3b8;">No image detected</strong><br>
                    No image was found in this article — cross-check was skipped.
                    Image analysis runs automatically when an article image is present.
                </div>
                """, unsafe_allow_html=True)

        # ── Human Feedback form (only shown when confidence is low) ──────
        _confidence = result["confidence"]
        _verdict_id = result.get("verdict_id")
        if _confidence < 0.75 and _verdict_id:
            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.25);
                        border-radius:12px; padding:1rem 1.2rem; margin-top:0.5rem;">
                <div style="font-family:'Space Mono',monospace; font-size:0.65rem;
                            letter-spacing:0.15em; text-transform:uppercase; color:#f59e0b;
                            margin-bottom:0.6rem;">⚠️ Confidence below 75% — Submit Correct Verdict</div>
                <div style="font-size:0.82rem; color:#94a3b8;">
                    The AI is uncertain about this claim. If you know the correct verdict,
                    submit it below to improve the system.
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("✏️  Correct this verdict", expanded=True):
                fb_col1, fb_col2 = st.columns([2, 3])
                with fb_col1:
                    fb_label = st.radio(
                        "Correct verdict",
                        ["supported", "misleading", "refuted"],
                        index=["supported", "misleading", "refuted"].index(result["label"])
                        if result["label"] in ["supported", "misleading", "refuted"] else 1,
                        key="fb_label"
                    )
                    fb_conf = st.slider(
                        "Correct confidence", 0, 100,
                        value=int(_confidence * 100), step=5, key="fb_conf"
                    )
                with fb_col2:
                    fb_note = st.text_area(
                        "Reason / evidence (optional)",
                        placeholder="e.g. 'Reuters reported this as confirmed on 2024-03-15'",
                        height=100, key="fb_note"
                    )

                if st.button("✅  Submit Correction", key="fb_submit"):
                    try:
                        mem = _get_memory()
                        if mem:
                            _correct_conf = fb_conf / 100
                            _old_label    = result.get("label", "unknown")
                            _old_conf     = result.get("confidence", 0.0)

                            # 1. Patch the verdict in Neo4j + ChromaDB
                            mem.update_verdict_with_feedback(
                                verdict_id=_verdict_id,
                                correct_label=fb_label,
                                correct_confidence=_correct_conf,
                                feedback_note=fb_note,
                            )

                            # SecOps: HITL audit log
                            _hitl_log.info(
                                "CORRECTION | verdict_id=%s | old_label=%s | new_label=%s "
                                "| old_conf=%.2f | new_conf=%.2f | note=%s",
                                _verdict_id, _old_label, fb_label,
                                _old_conf, _correct_conf,
                                fb_note.replace("\n", " ") if fb_note else "",
                            )

                            # 2. Log a source credibility observation so the
                            #    Reflection Agent picks up the human signal
                            try:
                                from id_utils import make_id
                                _claim_text = st.session_state.get("_last_claim_text", result.get("claim_text", ""))
                                _sources    = result.get("sources", [])
                                _source_id  = (
                                    "src_" + _sources[0]["url"].split("/")[2].replace(".", "_")
                                    if _sources else "src_frontend"
                                )
                                mem.add_source_credibility_point(
                                    point_id     = make_id("scp_"),
                                    claim_text   = _claim_text,
                                    topic_text   = _claim_text,
                                    source_id    = _source_id,
                                    credibility  = _correct_conf,
                                    bias         = result.get("bias_score", 0.5),
                                    verdict_label= fb_label,
                                    verdict_id   = _verdict_id,
                                    created_at   = datetime.now(timezone.utc).isoformat(),
                                )
                                print(f"[hitl] source credibility point written → source={_source_id}  cred={_correct_conf:.2f}  label={fb_label}")
                            except Exception as _sce:
                                print(f"[hitl] source credibility update skipped: {_sce}")

                            # 3. Update cached result so verdict card reflects correction immediately
                            if "_last_result" in st.session_state:
                                _updated = dict(st.session_state["_last_result"])
                                _updated["label"]      = fb_label
                                _updated["confidence"] = _correct_conf
                                st.session_state["_last_result"] = _updated
                            st.session_state["_feedback_toast"] = f"✅ Verdict updated to **{fb_label.upper()}** ({fb_conf}% confidence)"
                            st.rerun()
                        else:
                            st.warning("Memory not available — correction not saved.")
                    except Exception as _fbe:
                        st.error(f"Could not save correction: {_fbe}")

    # ─────────────────────
    # TAB 2: ENTITY & TREND
    # ─────────────────────
    with tab2:
        # Priority: typed box → auto-detected (only if box empty) → "Tesla" demo fallback
        _typed_in_box = entity_query.strip()
        entity_name = (
            _typed_in_box
            or (st.session_state.get("_auto_entity", "").strip() if not _typed_in_box else "")
            or "Tesla"
        )
        st.markdown(f'<div class="section-header">Entity Profile — {entity_name}</div>', unsafe_allow_html=True)

        df = get_real_entity_history(entity_name)

        if df.empty:
            _snap_count   = df.attrs.get("snapshot_count", 0)
            _entity_found = df.attrs.get("entity_found", False)
            if not _entity_found:
                st.markdown(f"""
                <div style="background:rgba(71,85,105,0.15); border:1px solid #2d3748;
                            border-radius:12px; padding:1.2rem 1.4rem;">
                    <div style="font-family:'Space Mono',monospace; font-size:0.7rem;
                                letter-spacing:0.15em; color:#475569; margin-bottom:0.5rem;">
                        ENTITY NOT TRACKED YET
                    </div>
                    <div style="font-size:0.9rem; color:#94a3b8;">
                        <strong style="color:#e2e8f0;">'{entity_name}'</strong> has not been seen in any fact-checked claim yet.
                    </div>
                    <div style="margin-top:0.8rem; font-size:0.82rem; color:#64748b;">
                        📌 Fact-check a claim that mentions <strong>{entity_name}</strong> and the tracker will
                        automatically build their credibility profile.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:rgba(71,85,105,0.15); border:1px solid #2d3748;
                            border-radius:12px; padding:1.2rem 1.4rem;">
                    <div style="font-family:'Space Mono',monospace; font-size:0.7rem;
                                letter-spacing:0.15em; color:#475569; margin-bottom:0.5rem;">
                        SNAPSHOTS FOUND
                    </div>
                    <div style="display:flex; align-items:baseline; gap:0.6rem; margin-bottom:0.6rem;">
                        <span style="font-family:'Space Mono',monospace; font-size:2rem;
                                     font-weight:700; color:#a78bfa;">{_snap_count}</span>
                        <span style="font-size:0.9rem; color:#64748b;">/ 3 needed for credibility graph</span>
                    </div>
                    <div style="background:#1e2535; border-radius:6px; height:6px; width:100%; margin-bottom:0.8rem;">
                        <div style="background:#a78bfa; border-radius:6px; height:6px;
                                    width:{min(100, int(_snap_count/3*100))}%;"></div>
                    </div>
                    <div style="font-size:0.82rem; color:#64748b;">
                        Each time you fact-check a claim mentioning
                        <strong style="color:#94a3b8;">{entity_name}</strong>,
                        the Entity Tracker adds one snapshot.
                        Check <strong>{max(0, 3 - _snap_count)}</strong> more claim(s) to unlock the trend chart.
                    </div>
                </div>
                """, unsafe_allow_html=True)
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

elif run_btn and not user_input.strip() and not _cached_result:
    st.warning("Please enter a news claim or article URL to verify.")

# ─────────────────────────────────────────────
# EMPTY STATE (no query yet)
# ─────────────────────────────────────────────
if not run_btn and not _cached_result:
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
