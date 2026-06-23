"""
app.py — Legal Intelligence RAG Assistant (Hugging Face Spaces edition).

Privacy model
-------------
* Each browser session gets its own VectorStore (in-memory ChromaDB).
* The Groq API key lives ONLY in st.session_state — never env vars, never disk.
* Uploaded files are written to a per-session /tmp subdirectory and deleted
  immediately after indexing.
* The "Clear Session Data" button wipes everything: key, vector store, chat,
  temp files — and triggers st.rerun() so the UI resets cleanly.

st.session_state keys
---------------------
  api_key           str        — Groq key (RAM only)
  api_key_status    str        — "unset" | "validating" | "valid" | "invalid"
  api_key_error     str        — last validation error message
  vector_store      VectorStore
  retriever         Retriever  — has cross-encoder reranker injected
  pipeline          RAGPipeline — has RAGAS guard injected
  chat_history      list[dict]
  indexed_docs      list[str]
  filename_filter   str|None
  session_tmp_dir   str
"""

from __future__ import annotations

import sys
import os

# Force the project root directory into the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Legal Intelligence Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #0f1117;
    color: #e8eaf0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] * { color: #c8ccd8 !important; }

/* ── Header ── */
.lia-header {
    display: flex; align-items: center; gap: 12px;
    padding: 1.2rem 0 0.6rem;
    border-bottom: 1px solid #1e2535;
    margin-bottom: 1.4rem;
}
.lia-header h1 { font-size:1.5rem; font-weight:700; letter-spacing:-0.02em; color:#e8eaf0; margin:0; }
.lia-tagline { font-size:0.78rem; color:#6b7280; margin-top:2px; letter-spacing:0.03em; text-transform:uppercase; }

/* ── Chat bubbles ── */
.chat-bubble { padding:0.85rem 1.1rem; border-radius:12px; margin-bottom:0.9rem; line-height:1.65; font-size:0.92rem; max-width:86%; }
.chat-user   { background:#1a2744; border:1px solid #233060; margin-left:auto; color:#c8d8ff; }
.chat-assistant { background:#141920; border:1px solid #1e2535; margin-right:auto; color:#dde2ec; }
.chat-unanswerable { background:#1f1418; border:1px solid #3d1f28; color:#f49090; }

/* ── Source badge ── */
.source-badge { display:inline-block; background:#1e2d4a; border:1px solid #2a3f6a; color:#7fa8e8; border-radius:6px; padding:2px 8px; font-size:0.72rem; margin:3px 3px 3px 0; font-family:monospace; }

/* ── Metric pills ── */
.metric-row { display:flex; gap:10px; margin-top:6px; flex-wrap:wrap; }
.metric-pill { background:#0d1a2e; border:1px solid #1b2f4e; border-radius:20px; padding:3px 10px; font-size:0.72rem; color:#6b8ab8; }

/* ── Doc chip ── */
.doc-chip { display:inline-flex; align-items:center; gap:5px; background:#131c2e; border:1px solid #1e2d45; border-radius:8px; padding:4px 10px; font-size:0.78rem; color:#8aa4cc; margin:3px 3px 3px 0; }

/* ── Banners ── */
.warn-box    { background:#1f1a10; border:1px solid #4a3a10; border-radius:8px; padding:0.7rem 1rem; color:#d4a843; font-size:0.84rem; margin-bottom:1rem; }
.info-box    { background:#101c2e; border:1px solid #1e3a5a; border-radius:8px; padding:0.7rem 1rem; color:#6b9fcc; font-size:0.84rem; margin-bottom:1rem; }
.success-box { background:#0d1f18; border:1px solid #1a4030; border-radius:8px; padding:0.7rem 1rem; color:#5cc89a; font-size:0.84rem; margin-bottom:1rem; }
.error-box   { background:#1f0d0d; border:1px solid #4a1515; border-radius:8px; padding:0.7rem 1rem; color:#f49090; font-size:0.84rem; margin-bottom:1rem; }

/* ── API key status badge ── */
.key-badge {
    display:inline-flex; align-items:center; gap:6px;
    border-radius:20px; padding:4px 12px; font-size:0.78rem; font-weight:600;
    margin-top:6px;
}
.key-badge-unset       { background:#1a1f2e; border:1px solid #2a3050; color:#5a6480; }
.key-badge-validating  { background:#101c2e; border:1px solid #1e3a5a; color:#6b9fcc; }
.key-badge-valid       { background:#0d1f18; border:1px solid #1a4030; color:#5cc89a; }
.key-badge-invalid     { background:#1f0d0d; border:1px solid #4a1515; color:#f49090; }

/* ── Pulse dot animation ── */
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
.dot-pulse { display:inline-block; width:8px; height:8px; border-radius:50%; animation:pulse 1.2s ease-in-out infinite; }
.dot-blue  { background:#6b9fcc; }
.dot-green { background:#5cc89a; }
.dot-red   { background:#f49090; }
.dot-gray  { background:#5a6480; }

/* ── Instruction cards ── */
.step-card {
    background:#131c2e; border:1px solid #1e2d45; border-radius:12px;
    padding:1.2rem 1.4rem; margin-bottom:1rem;
}
.step-number {
    display:inline-flex; align-items:center; justify-content:center;
    width:28px; height:28px; border-radius:50%;
    background:#1e3a5a; color:#7fa8e8; font-size:0.85rem; font-weight:700;
    margin-right:10px; flex-shrink:0;
}
.step-title { font-size:1rem; font-weight:600; color:#c8d8ff; }
.step-body  { font-size:0.84rem; color:#8a9ab8; margin-top:0.5rem; line-height:1.6; }
.feature-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:1rem; }
.feature-card {
    background:#0d1220; border:1px solid #1a2840; border-radius:10px;
    padding:1rem 1.1rem;
}
.feature-icon { font-size:1.4rem; margin-bottom:6px; }
.feature-title { font-size:0.84rem; font-weight:600; color:#c8d8ff; }
.feature-desc  { font-size:0.76rem; color:#6b7a96; margin-top:3px; line-height:1.5; }
.privacy-table { width:100%; border-collapse:collapse; font-size:0.82rem; margin-top:0.5rem; }
.privacy-table th { color:#7fa8e8; text-align:left; padding:6px 10px; border-bottom:1px solid #1e2d45; font-weight:600; }
.privacy-table td { color:#8a9ab8; padding:6px 10px; border-bottom:1px solid #131c2e; }
.privacy-table tr:last-child td { border-bottom:none; }
.tick { color:#5cc89a; font-weight:700; }


</style>
""", unsafe_allow_html=True)


# ── Cached model loaders — process-level, shared across all sessions ──────────

@st.cache_resource(show_spinner="Loading embedding model…")
def _load_embedder():
    """mpnet-base-v2, loaded once per process cold start."""
    from src.embedding.embedder import Embedder
    return Embedder()


@st.cache_resource(show_spinner="Loading re-ranker model…")
def _load_reranker():
    """
    ms-marco-MiniLM-L-6-v2 cross-encoder — ~80 MB.
    Read-only after init so safe to share across sessions.
    """
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _make_ragas_guard():
    """RAGASGuard is stateless — create one per session via _init_session()."""
    from src.evaluation.ragas_guard import RAGASGuard
    return RAGASGuard(_load_embedder())


# ── Session init ──────────────────────────────────────────────────────────────
def _init_session() -> None:
    from src.embedding.embedder import VectorStore
    from src.retrieval.retriever import Retriever
    from src.rag.rag_pipeline import RAGPipeline

    defaults = {
        "session_id":      str(uuid.uuid4())[:8],
        "api_key":         "",
        "api_key_status":  "unset",   # unset | validating | valid | invalid
        "api_key_error":   "",
        "chat_history":    [],
        "indexed_docs":    [],
        "filename_filter": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()

    if "retriever" not in st.session_state:
        retriever = Retriever(_load_embedder(), st.session_state.vector_store)
        # Inject the process-cached cross-encoder so it's never loaded twice
        retriever.set_reranker(_load_reranker())
        st.session_state.retriever = retriever

    if "pipeline" not in st.session_state:
        pipeline = RAGPipeline(st.session_state.retriever)
        # Inject RAGAS guard — shares the same embedder as the retriever
        pipeline.set_ragas_guard(_make_ragas_guard())
        st.session_state.pipeline = pipeline

    if "session_tmp_dir" not in st.session_state:
        st.session_state.session_tmp_dir = str(Path(tempfile.mkdtemp(prefix="lia_")))


# ── API key validation ────────────────────────────────────────────────────────
def _validate_groq_key(key: str) -> tuple[bool, str]:
    """
    Makes a minimal Groq API call to confirm the key is valid.
    Returns (is_valid, error_message).
    Uses the cheapest/fastest model with max_tokens=1 to minimise cost.
    """
    try:
        from openai import OpenAI, AuthenticationError, RateLimitError
        client = OpenAI(
            api_key=key,
            base_url="https://api.groq.com/openai/v1",
            timeout=10,
        )
        client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        return True, ""
    except AuthenticationError:
        return False, "Invalid API key — authentication failed."
    except RateLimitError:
        # Key is valid but rate-limited — treat as valid
        return True, ""
    except Exception as exc:
        return False, f"Validation error: {str(exc)[:120]}"


# ── Session clear ─────────────────────────────────────────────────────────────
def _clear_session() -> None:
    if "vector_store" in st.session_state:
        try:
            st.session_state.vector_store.reset()
        except Exception:
            pass
    tmp_dir = st.session_state.get("session_tmp_dir", "")
    if tmp_dir and Path(tmp_dir).exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## ⚖️ Legal Intelligence")
        st.markdown("---")

        # ────────────────────────────────────────────────────────────────
        # API KEY SECTION
        # ────────────────────────────────────────────────────────────────
        st.markdown("### 🔑 Groq API Key")
        st.markdown(
            '<div style="font-size:0.78rem;color:#6b7280;margin-bottom:8px;">'
            'Stored only in session memory — never written to disk or logs.'
            '</div>',
            unsafe_allow_html=True,
        )

        raw_key = st.text_input(
            "Groq API Key",
            value=st.session_state.api_key,
            type="password",
            placeholder="gsk_…",
            label_visibility="collapsed",
            key="api_key_input",
        )

        # Detect manual edits — reset validation state when key changes
        if raw_key.strip() != st.session_state.api_key:
            st.session_state.api_key = raw_key.strip()
            st.session_state.api_key_status = "unset" if not raw_key.strip() else "unset"
            st.session_state.api_key_error = ""

        # ── Validate / Clear buttons ──────────────────────────────────
        btn_col1, btn_col2 = st.columns([3, 2])
        with btn_col1:
            validate_clicked = st.button(
                "✔ Validate Key",
                use_container_width=True,
                disabled=(not st.session_state.api_key or st.session_state.api_key_status == "validating"),
                key="validate_key_btn",
            )
        with btn_col2:
            clear_key_clicked = st.button(
                "✕ Clear Key",
                use_container_width=True,
                disabled=not st.session_state.api_key,
                key="clear_key_btn",
            )

        # Handle clear key
        if clear_key_clicked:
            st.session_state.api_key = ""
            st.session_state.api_key_status = "unset"
            st.session_state.api_key_error = ""
            st.rerun()

        # Handle validate — show spinner inline then rerun with result
        if validate_clicked and st.session_state.api_key:
            st.session_state.api_key_status = "validating"
            with st.spinner("Validating key with Groq…"):
                ok, err = _validate_groq_key(st.session_state.api_key)
            st.session_state.api_key_status = "valid" if ok else "invalid"
            st.session_state.api_key_error = err
            st.rerun()

        # ── Status badge ──────────────────────────────────────────────
        status = st.session_state.api_key_status
        if status == "unset":
            badge_cls, dot_cls, label = "key-badge-unset",      "dot-gray",  "No key entered"
        elif status == "validating":
            badge_cls, dot_cls, label = "key-badge-validating",  "dot-blue",  "Validating…"
        elif status == "valid":
            badge_cls, dot_cls, label = "key-badge-valid",       "dot-green", "Key validated ✓"
        else:
            badge_cls, dot_cls, label = "key-badge-invalid",     "dot-red",   "Invalid key"

        st.markdown(
            f'<div class="key-badge {badge_cls}">'
            f'<span class="dot-pulse {dot_cls}"></span>{label}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if status == "invalid" and st.session_state.api_key_error:
            st.markdown(
                f'<div class="error-box" style="margin-top:6px;">'
                f'⚠ {st.session_state.api_key_error}<br>'
                f'<a href="https://console.groq.com" target="_blank" style="color:#f49090;">'
                f'Get a free key →</a></div>',
                unsafe_allow_html=True,
            )
        elif status == "unset" and not st.session_state.api_key:
            st.markdown(
                '<div style="font-size:0.76rem;color:#4a5568;margin-top:4px;">'
                '<a href="https://console.groq.com" target="_blank" style="color:#6b7280;">'
                '↗ Get a free Groq key</a></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ────────────────────────────────────────────────────────────────
        # DOCUMENT UPLOAD
        # ────────────────────────────────────────────────────────────────
        st.markdown("### 📄 Upload Documents")
        st.markdown(
            '<div style="font-size:0.78rem;color:#6b7280;margin-bottom:8px;">'
            'PDF, TXT, CSV, JSON · Deleted from disk after indexing.'
            '</div>',
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "Upload",
            type=["pdf", "txt", "text", "csv", "json"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            _handle_uploads(uploaded_files)

        indexed = st.session_state.get("indexed_docs", [])
        if indexed:
            st.markdown("**Indexed documents:**")
            for fname in indexed:
                st.markdown(f'<div class="doc-chip">📎 {fname}</div>', unsafe_allow_html=True)
            filter_opts = ["All documents"] + indexed
            chosen = st.selectbox("Query scope", filter_opts, key="doc_filter")
            st.session_state.filename_filter = None if chosen == "All documents" else chosen
        else:
            st.session_state.filename_filter = None

        st.markdown("---")

        # ────────────────────────────────────────────────────────────────
        # SESSION CONTROL
        # ────────────────────────────────────────────────────────────────
        st.markdown("### 🗑️ Session Control")
        st.markdown(
            '<div style="font-size:0.78rem;color:#6b7280;margin-bottom:8px;">'
            'Wipes all documents, embeddings, chat history, and API key from memory.'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("🗑️ Clear Session Data", use_container_width=True, type="secondary"):
            _clear_session()

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.72rem;color:#3d4455;text-align:center;">'
            'Session-isolated · Zero persistence'
            '</div>',
            unsafe_allow_html=True,
        )


# ── Upload handler ────────────────────────────────────────────────────────────
def _handle_uploads(uploaded_files) -> None:
    from src.embedding.indexer import index_file

    embedder    = _load_embedder()
    vector_store = st.session_state.vector_store
    tmp_dir     = Path(st.session_state.session_tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    already = set(st.session_state.indexed_docs)
    new_files = [f for f in uploaded_files if f.name not in already]
    if not new_files:
        return

    bar   = st.sidebar.progress(0, text="Indexing…")
    slot  = st.sidebar.empty()

    for i, uf in enumerate(new_files):
        tmp_path = tmp_dir / uf.name
        try:
            tmp_path.write_bytes(uf.read())
            slot.markdown(f'<div class="info-box">⏳ Indexing: {uf.name}…</div>', unsafe_allow_html=True)

            def _cb(msg: str, _slot=slot):
                _slot.markdown(f'<div class="info-box">{msg}</div>', unsafe_allow_html=True)

            chunks = index_file(tmp_path, embedder, vector_store, progress_callback=_cb)
            st.session_state.indexed_docs.append(uf.name)
            slot.markdown(
                f'<div class="success-box">✓ {uf.name} — {len(chunks)} chunks indexed</div>',
                unsafe_allow_html=True,
            )
        except Exception as exc:
            slot.markdown(
                f'<div class="error-box">✗ {uf.name}: {exc}</div>',
                unsafe_allow_html=True,
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        bar.progress((i + 1) / len(new_files))
        time.sleep(0.3)

    bar.empty()
    time.sleep(1.0)
    slot.empty()


# ── Citation formatter ────────────────────────────────────────────────────────
def _format_citations(text: str) -> str:
    def _rep(m):
        return f'<span class="source-badge">[S{m.group(1)}]</span>'
    return re.sub(r"\[SOURCE\s*(\d+)\]", _rep, text, flags=re.IGNORECASE)


# ── Chat tab ──────────────────────────────────────────────────────────────────
def _render_chat_tab() -> None:
    no_docs = not st.session_state.get("indexed_docs")
    # Allow querying if key exists in field even if not yet validated
    no_key  = not st.session_state.get("api_key")
    key_invalid = st.session_state.get("api_key_status") == "invalid"

    # ── Guidance banners ──
    if no_docs or no_key or key_invalid:
        cols = st.columns(2) if (no_docs and no_key) else [st.container()]
        messages = []
        if no_docs:
            messages.append('📄 <strong>Step 1:</strong> Upload a legal document in the sidebar.')
        if no_key:
            messages.append('🔑 <strong>Step 2:</strong> Enter & validate your Groq API key in the sidebar.')
        elif key_invalid:
            messages.append('🔑 Your API key is invalid. Please enter a valid Groq key.')

        if len(messages) == 2:
            for col, msg in zip(st.columns(2), messages):
                with col:
                    st.markdown(f'<div class="warn-box">{msg}</div>', unsafe_allow_html=True)
        elif messages:
            st.markdown(f'<div class="warn-box">{messages[0]}</div>', unsafe_allow_html=True)

    # ── Chat history ──
    history = st.session_state.get("chat_history", [])
    if not history:
        st.markdown(
            '<div class="info-box" style="text-align:center;padding:2rem 1rem;">'
            '⚖️ Upload a legal document and ask any question about it.<br>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        for msg in history:
            role, content = msg["role"], msg["content"]
            if role == "user":
                st.markdown(
                    f'<div class="chat-bubble chat-user">🧑‍💼 {content}</div>',
                    unsafe_allow_html=True,
                )
            else:
                extra = "chat-unanswerable" if content.startswith("UNANSWERABLE") else ""
                st.markdown(
                    f'<div class="chat-bubble chat-assistant {extra}">⚖️ {_format_citations(content)}</div>',
                    unsafe_allow_html=True,
                )
                if "meta" in msg:
                    m = msg["meta"]
                    faith = m.get("faithfulness", 0)
                    ans_rel = m.get("answer_relevancy", 0)
                    ctx_rel = m.get("context_relevancy", 0)
                    overall = m.get("ragas_overall", 0)
                    ragas_ok = m.get("ragas_passed", True)
                    gate = m.get("gate_failed", "")

                    def _score_color(v: float) -> str:
                        if v >= 0.7: return "#5cc89a"
                        if v >= 0.45: return "#d4a843"
                        return "#f49090"

                    ragas_badge = (
                        '<span class="metric-pill" style="color:#5cc89a;">✓ RAGAS passed</span>'
                        if ragas_ok and not gate else
                        f'<span class="metric-pill" style="color:#f49090;">✗ RAGAS failed</span>'
                    )
                    gate_badge = (
                        f'<span class="metric-pill" style="color:#f49090;font-size:0.68rem;">{gate}</span>'
                        if gate else ""
                    )
                    st.markdown(
                        f'<div class="metric-row">'
                        f'<span class="metric-pill">⏱ {m.get("latency_ms",0):.0f} ms</span>'
                        f'<span class="metric-pill" style="color:{_score_color(faith)}">🎯 faith {faith:.2f}</span>'
                        f'<span class="metric-pill" style="color:{_score_color(ans_rel)}">💬 ans_rel {ans_rel:.2f}</span>'
                        f'<span class="metric-pill" style="color:{_score_color(ctx_rel)}">🔍 ctx_rel {ctx_rel:.2f}</span>'
                        f'<span class="metric-pill" style="color:{_score_color(overall)}">📊 overall {overall:.2f}</span>'
                        f'<span class="metric-pill">📚 {m.get("sources",0)} src</span>'
                        f'{ragas_badge}{gate_badge}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                if msg.get("sources"):
                    with st.expander(f"📎 {len(msg['sources'])} retrieved sources", expanded=False):
                        for j, src in enumerate(msg["sources"], 1):
                            st.markdown(
                                f"**[SOURCE {j}]** `{src['filename']}`"
                                + (f" · *{src['section']}*" if src.get("section") else "")
                                + f" · similarity **{src['similarity']:.3f}**"
                            )
                            st.markdown(
                                f'<div style="background:#0d1220;border:1px solid #1a2840;border-radius:6px;'
                                f'padding:8px 12px;font-size:0.82rem;color:#9aadcc;margin-bottom:8px;">'
                                f'{src["text_preview"]}</div>',
                                unsafe_allow_html=True,
                            )

    # ── Summarise button ──
    if not no_docs and not no_key and not key_invalid:
        indexed = st.session_state.indexed_docs
        scope   = st.session_state.get("filename_filter")
        target  = scope or (indexed[0] if len(indexed) == 1 else None)
        if target:
            if st.button(f"📋 Summarise '{target}'"):
                with st.spinner("Generating summary…"):
                    summary = st.session_state.pipeline.summarise_document(
                        target, api_key=st.session_state.api_key
                    )
                st.session_state.chat_history += [
                    {"role": "user", "content": f"Summarise: {target}"},
                    {"role": "assistant", "content": summary, "sources": [],
                     "meta": {"latency_ms": 0, "faithfulness": 0, "sources": 0, "model": "groq"}},
                ]
                st.rerun()

    # ── Chat input ──
    question = st.chat_input(
        "Ask a question about your legal documents…",
        disabled=no_docs or no_key or key_invalid,
    )
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Retrieving and generating answer…"):
            resp = st.session_state.pipeline.run(
                question=question,
                api_key=st.session_state.api_key,
                filename_filter=st.session_state.get("filename_filter"),
            )
        sources_display = [
            {"filename": c.filename, "section": c.section, "similarity": c.similarity,
             "text_preview": c.text[:350] + "…" if len(c.text) > 350 else c.text}
            for c in resp.sources
        ]
        st.session_state.chat_history.append({
            "role": "assistant", "content": resp.answer,
            "sources": sources_display,
            "meta": {
                "latency_ms":         resp.latency_ms,
                "faithfulness":       resp.faithfulness_score,
                "answer_relevancy":   resp.answer_relevancy_score,
                "context_relevancy":  resp.context_relevancy_score,
                "ragas_overall":      resp.ragas_overall,
                "ragas_passed":       resp.ragas_passed,
                "gate_failed":        resp.gate_failed,
                "sources":            len(resp.sources),
                "model":              resp.model_used,
            },
        })
        st.rerun()


# ── Instructions tab ──────────────────────────────────────────────────────────
def _render_instructions_tab() -> None:
    st.markdown(
        '<div style="max-width:860px;">'

        # ── Hero ──
        '<div style="text-align:center;padding:1.5rem 0 2rem;">'
        '<div style="font-size:2.8rem;margin-bottom:0.4rem;">⚖️</div>'
        '<div style="font-size:1.3rem;font-weight:700;color:#c8d8ff;">Legal Intelligence Assistant</div>'
        '<div style="font-size:0.82rem;color:#6b7280;margin-top:4px;">'
        'Session-isolated RAG · Citation-grounded answers · Zero data persistence'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Quick-start steps ──
    st.markdown("#### 🚀 Quick Start")

    steps = [
        ("Get a Groq API Key",
         'Visit <a href="https://console.groq.com" target="_blank" style="color:#7fa8e8;">console.groq.com</a>, '
         "sign up for free, and create an API key. Groq provides fast LLM inference — the free tier is "
         "sufficient for most legal document queries."),
        ("Enter & Validate Your Key",
         "Paste the key into the <strong>🔑 Groq API Key</strong> field in the sidebar, then click "
         "<strong>✔ Validate Key</strong>. A green badge confirms the key works. "
         "Your key is stored only in this browser session — it is never written to disk."),
        ("Upload Legal Documents",
         "Click <strong>📄 Upload Documents</strong> in the sidebar and select one or more files. "
         "Supported formats: <code>PDF</code>, <code>TXT</code>, <code>CSV</code>, <code>JSON</code>. "
         "Documents are chunked, embedded, and stored in an in-memory vector database. "
         "The original file is deleted from the server immediately after indexing."),
        ("Ask Questions",
         "Type your question in the chat bar at the bottom of the <strong>💬 Chat</strong> tab. "
         "The assistant retrieves the most relevant passages and generates a cited answer. "
         "Every claim references a <code>[SOURCE N]</code> from your documents."),
        ("Summarise a Document",
         "After indexing, click <strong>📋 Summarise</strong> above the chat to get a structured "
         "overview covering parties, obligations, termination clauses, and key risks."),
        ("Clear Your Session",
         "Click <strong>🗑️ Clear Session Data</strong> in the sidebar at any time to instantly wipe "
         "all embeddings, chat history, and your API key from memory. "
         "Closing the browser tab also destroys all session data automatically."),
    ]

    for i, (title, body) in enumerate(steps, 1):
        st.markdown(
            f'<div class="step-card">'
            f'<div style="display:flex;align-items:flex-start;gap:10px;">'
            f'<span class="step-number">{i}</span>'
            f'<div>'
            f'<div class="step-title">{title}</div>'
            f'<div class="step-body">{body}</div>'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )

    # ── Feature grid ──
    st.markdown("#### ✨ Features")
    st.markdown(
        '<div class="feature-grid">'

        '<div class="feature-card">'
        '<div class="feature-icon">🔍</div>'
        '<div class="feature-title">Semantic Retrieval</div>'
        '<div class="feature-desc">mpnet-base-v2 embeddings + ChromaDB cosine search '
        'surface the most relevant passages — not just keyword matches.</div>'
        '</div>'

        '<div class="feature-card">'
        '<div class="feature-icon">📎</div>'
        '<div class="feature-title">Citation-Grounded Answers</div>'
        '<div class="feature-desc">Every claim in the answer cites [SOURCE N], '
        'linking back to the exact passage retrieved from your document.</div>'
        '</div>'

        '<div class="feature-card">'
        '<div class="feature-icon">📂</div>'
        '<div class="feature-title">Multi-Document Support</div>'
        '<div class="feature-desc">Upload several documents at once and query across all of them, '
        'or use the scope selector to restrict queries to a single file.</div>'
        '</div>'

        '<div class="feature-card">'
        '<div class="feature-icon">📋</div>'
        '<div class="feature-title">Document Summarisation</div>'
        '<div class="feature-desc">One-click structured summary covering parties, obligations, '
        'liability, termination clauses, and notable risks.</div>'
        '</div>'

        '<div class="feature-card">'
        '<div class="feature-icon">🔀</div>'
        '<div class="feature-title">Cross-Encoder Re-Ranking</div>'
        '<div class="feature-desc">After ANN retrieval, a ms-marco-MiniLM cross-encoder '
        're-scores every (query, passage) pair jointly — far more accurate than cosine '
        'similarity alone. Passages below the relevance threshold are dropped before '
        'the LLM even sees them.</div>'
        '</div>'

        '<div class="feature-card">'
        '<div class="feature-icon">🧪</div>'
        '<div class="feature-title">RAGAS Quality Guard</div>'
        '<div class="feature-desc">After generation, three RAGAS-style metrics are computed: '
        '<strong>Faithfulness</strong> (are all claims grounded in sources?), '
        '<strong>Answer Relevancy</strong> (does the answer address the question?), and '
        '<strong>Context Relevancy</strong> (were the right passages retrieved?). '
        'Any failure forces an UNANSWERABLE override.</div>'
        '</div>'

        '<div class="feature-card">'
        '<div class="feature-icon">⚡</div>'
        '<div class="feature-title">Fast Inference</div>'
        '<div class="feature-desc">Groq&#39;s hardware delivers gpt-oss-120b responses in '
        '2–6 seconds — even on complex multi-clause legal questions.</div>'
        '</div>'

        '</div>',
        unsafe_allow_html=True,
    )

    # ── Privacy table ──
    st.markdown("#### 🔒 Privacy Guarantees")
    st.markdown(
        '<table class="privacy-table">'
        '<thead><tr><th>Data</th><th>Storage</th><th>Deleted when</th></tr></thead>'
        '<tbody>'
        '<tr><td>Uploaded documents</td><td>Temp file (<code>/tmp/</code>)</td>'
        '<td><span class="tick">✓</span> Immediately after indexing</td></tr>'
        '<tr><td>Embeddings / vector store</td><td>In-memory only (ChromaDB EphemeralClient)</td>'
        '<td><span class="tick">✓</span> Session ends or Clear button</td></tr>'
        '<tr><td>Chat history</td><td>Browser session state</td>'
        '<td><span class="tick">✓</span> Session ends or Clear button</td></tr>'
        '<tr><td>Groq API key</td><td>Session RAM only</td>'
        '<td><span class="tick">✓</span> Clear Key, Clear Session, or tab close</td></tr>'
        '<tr><td>Cross-user data</td><td>Never shared</td>'
        '<td><span class="tick">✓</span> Fully isolated per session</td></tr>'
        '</tbody></table>',
        unsafe_allow_html=True,
    )

    # ── Tips ──
    st.markdown("#### 💡 Tips for Best Results")
    tips = [
        ("Ask specific, targeted questions",
         "\"What is the notice period for termination?\" works better than \"Tell me about the contract.\""),
        ("Use the document scope selector",
         "When multiple documents are indexed, narrow the query scope in the sidebar to avoid cross-document noise."),
        ("Check the source passages",
         "Expand the 📎 retrieved sources panel below each answer to verify the cited text directly."),
        ("Low RAGAS scores or UNANSWERABLE?",
         "A faithfulness score below 0.5 means claims are not fully grounded in the retrieved passages. "
         "An answer relevancy score below 0.4 means the answer drifted off-topic. "
         "A context relevancy score below 0.3 means the retrieved passages were weakly matched. "
         "In all cases, try rephrasing your question or uploading a more complete document."),
        ("Re-index updated documents",
         "Click Clear Session Data, then re-upload the revised file — this ensures stale embeddings are purged."),
    ]
    for title, body in tips:
        st.markdown(
            f'<div style="background:#0d1220;border-left:3px solid #2a3f6a;border-radius:0 8px 8px 0;'
            f'padding:0.65rem 1rem;margin-bottom:0.6rem;">'
            f'<div style="font-size:0.84rem;font-weight:600;color:#c8d8ff;">{title}</div>'
            f'<div style="font-size:0.80rem;color:#6b7a96;margin-top:3px;">{body}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Supported formats ──
    st.markdown("#### 📁 Supported File Formats")
    formats = [
        ("PDF", "Legal contracts, agreements, briefs, court filings"),
        ("TXT / TEXT", "Plain-text agreements, terms and conditions, policies"),
        ("CSV", "Structured legal data — each row becomes a retrievable chunk"),
        ("JSON", "Exported legal records; expects a <code>text</code> field or list of objects"),
    ]
    for fmt, desc in formats:
        st.markdown(
            f'<div style="display:flex;gap:12px;align-items:baseline;margin-bottom:6px;">'
            f'<code style="background:#1e2d4a;color:#7fa8e8;border-radius:4px;padding:2px 8px;'
            f'font-size:0.78rem;white-space:nowrap;">{fmt}</code>'
            f'<span style="font-size:0.82rem;color:#6b7a96;">{desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    _init_session()
    _render_sidebar()

    # Header
    st.markdown(
        '<div class="lia-header">'
        '<span style="font-size:2rem;">⚖️</span>'
        '<div>'
        '<h1>Legal Intelligence Assistant</h1>'
        '<div class="lia-tagline">Session-isolated · Citation-grounded answers</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Tabs
    chat_tab, instructions_tab = st.tabs(["💬 Chat", "📖 How to Use"])

    with chat_tab:
        _render_chat_tab()

    with instructions_tab:
        _render_instructions_tab()


if __name__ == "__main__":
    main()
