"""
app.py — Streamlit UI for the Legal RAG Assistant.

This is the main entry point for the web interface.  Run with:
  streamlit run app.py

Architecture note: Streamlit re-runs the entire script on every user
interaction.  Expensive objects (model, DB connection) must be cached using
@st.cache_resource so they are only created once per server session.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import tempfile

import streamlit as st

# ── Page config must be the first Streamlit call ──────────────────────────
st.set_page_config(
    page_title="Legal Intelligence Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0; }
    .sub-header  { color: #666; margin-top: 0; font-size: 0.95rem; }
    .metric-card {
        background: #f8f9ff; border: 1px solid #e0e4ff;
        border-radius: 10px; padding: 16px; text-align: center;
    }
    .answer-box {
        background: #fafbff; border-left: 4px solid #4f6ef7;
        border-radius: 6px; padding: 16px; margin: 8px 0;
    }
    .source-card {
        background: #fff8f0; border: 1px solid #ffe0b2;
        border-radius: 8px; padding: 12px; margin: 6px 0; font-size: 0.88rem;
    }
    .warning-box {
        background: #fff3e0; border-left: 4px solid #ff9800;
        border-radius: 6px; padding: 12px; color: #e65100;
    }
    .success-box {
        background: #e8f5e9; border-left: 4px solid #4caf50;
        border-radius: 6px; padding: 12px; color: #1b5e20;
    }
    .stProgress > div > div { background-color: #4f6ef7; }
</style>
""", unsafe_allow_html=True)


# ── Lazy imports inside cached functions (prevents slow import at startup) ─

@st.cache_resource(show_spinner="Loading AI models...")
def get_components():
    """
    Initialise and cache the RAG pipeline components.

    @st.cache_resource ensures these are created ONCE per Streamlit server
    process — subsequent page reloads reuse the same objects.  This is
    critical because loading the embedding model takes ~5 seconds.
    """
    from src.embedding.embedder import Embedder, VectorStore
    from src.retrieval.retriever import Retriever
    from src.rag.rag_pipeline import RAGPipeline

    embedder     = Embedder()
    vector_store = VectorStore()
    retriever    = Retriever(embedder, vector_store)
    pipeline     = RAGPipeline(retriever)
    return embedder, vector_store, retriever, pipeline


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialise Streamlit session state keys on first load."""
    defaults = {
        "chat_history": [],     # list of (question, RAGResponse)
        "active_tab": "chat",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def format_faithfulness(score: float) -> str:
    """Return a human-readable faithfulness label."""
    if score >= 0.8:
        return "🟢 High"
    if score >= 0.5:
        return "🟡 Medium"
    return "🔴 Low"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(vector_store) -> tuple[str | None, str]:
    """
    Render the sidebar with document upload and management controls.

    Returns:
        (selected_filename, active_mode) where active_mode is one of
        "chat", "summarise", "evaluate".
    """
    from src.embedding.indexer import index_file

    embedder, vector_store, _, _ = get_components()

    st.sidebar.markdown("## ⚖️ Legal Intelligence")
    st.sidebar.markdown("---")

    # ── Document upload ────────────────────────────────────────────────────
    st.sidebar.markdown("### 📂 Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload legal documents (PDF, TXT, JSON)",
        type=["pdf", "txt", "json"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if st.sidebar.button(f"Index: {uploaded_file.name}", key=f"idx_{uploaded_file.name}"):
                with st.sidebar:
                    progress_placeholder = st.empty()
                    messages: list[str] = []

                    def callback(msg: str) -> None:
                        messages.append(msg)
                        progress_placeholder.info("\n".join(messages[-3:]))

                    # Save to temp file so PyMuPDF can open it
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = Path(tmp.name)

                    try:
                        chunks = index_file(tmp_path, embedder, vector_store, callback)
                        progress_placeholder.success(f"✅ Indexed {len(chunks)} chunks")
                    except Exception as e:
                        progress_placeholder.error(f"❌ Error: {e}")
                    finally:
                        tmp_path.unlink(missing_ok=True)

    st.sidebar.markdown("---")

    # ── Indexed documents list ─────────────────────────────────────────────
    st.sidebar.markdown("### 📚 Indexed Documents")
    docs = vector_store.list_documents()

    selected_filename: str | None = None
    if docs:
        st.sidebar.markdown(f"*{len(docs)} document(s) indexed*")
        doc_names = ["All documents"] + [d["filename"] for d in docs]
        selection = st.sidebar.selectbox("Filter by document:", doc_names)
        if selection != "All documents":
            selected_filename = selection

        # Delete button
        if selected_filename:
            if st.sidebar.button(f"🗑️ Remove '{selected_filename}'", type="secondary"):
                doc_to_delete = next(
                    (d for d in docs if d["filename"] == selected_filename), None
                )
                if doc_to_delete:
                    vector_store.delete_document(doc_to_delete["doc_id"])
                    st.sidebar.success(f"Removed '{selected_filename}'")
                    st.rerun()
    else:
        st.sidebar.info("No documents indexed yet. Upload a file above.")

    st.sidebar.markdown("---")

    # ── Mode selection ────────────────────────────────────────────────────
    st.sidebar.markdown("### 🎯 Mode")
    mode = st.sidebar.radio(
        "Select mode:",
        ["💬 Chat / Q&A", "📋 Summarise", "📊 Evaluate"],
        label_visibility="collapsed",
    )
    active_mode = mode.split()[1].lower() if mode else "chat"

    # ── Stats ─────────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 System Stats")
    st.sidebar.metric("Total chunks indexed", vector_store.count)
    st.sidebar.metric("Documents indexed", len(docs))

    return selected_filename, active_mode


# ---------------------------------------------------------------------------
# Chat / Q&A tab
# ---------------------------------------------------------------------------

def render_chat_tab(pipeline, filename_filter: str | None) -> None:
    """Render the main Q&A interface."""
    st.markdown('<h1 class="main-header">Legal Q&A Assistant</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask questions about your uploaded legal documents. '
        'All answers are grounded in the document text and cite their sources.</p>',
        unsafe_allow_html=True,
    )

    # ── Chat history ──────────────────────────────────────────────────────
    for q, resp in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant", avatar="⚖️"):
            _render_rag_response(resp)

    # ── Input ─────────────────────────────────────────────────────────────
    if question := st.chat_input("Ask a legal question about your documents..."):
        _, vector_store, _, _ = get_components()

        if vector_store.count == 0:
            st.warning("⚠️ No documents indexed yet. Please upload a document first.")
            return

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant", avatar="⚖️"):
            with st.spinner("Searching documents and generating answer..."):
                t0 = time.perf_counter()
                resp = pipeline.run(question, filename_filter=filename_filter)

            _render_rag_response(resp)

        # Store in history (keep last 20 exchanges to avoid memory bloat)
        st.session_state.chat_history.append((question, resp))
        if len(st.session_state.chat_history) > 20:
            st.session_state.chat_history = st.session_state.chat_history[-20:]

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()


def _render_rag_response(resp) -> None:
    """Render a RAGResponse with answer, metrics, and source viewer."""
    # ── Main answer ───────────────────────────────────────────────────────
    if resp.is_answerable:
        st.markdown(
            f'<div class="answer-box">{resp.answer}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="warning-box">{resp.answer}</div>',
            unsafe_allow_html=True,
        )

    # ── Quality metrics ───────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⏱️ Latency", f"{resp.latency_ms:.0f} ms")
    col2.metric("📚 Sources retrieved", len(resp.sources))
    col3.metric("🔗 Sources cited", len(resp.cited_source_indices))
    col4.metric("🎯 Faithfulness", format_faithfulness(resp.faithfulness_score))

    # ── Source viewer ─────────────────────────────────────────────────────
    if resp.sources:
        with st.expander(f"📖 View {len(resp.sources)} retrieved passages", expanded=False):
            for i, chunk in enumerate(resp.sources, 1):
                is_cited = i in resp.cited_source_indices
                cited_label = " ✅ **Cited in answer**" if is_cited else ""
                st.markdown(
                    f"""<div class="source-card">
                    <strong>[SOURCE {i}]</strong>{cited_label}<br>
                    📄 <em>{chunk.filename}</em>
                    {f"| 📌 {chunk.section}" if chunk.section else ""}
                    | 🎯 Similarity: {chunk.similarity:.3f}
                    <hr style="margin: 8px 0; border-color: #eee;">
                    {chunk.text[:400]}{"..." if len(chunk.text) > 400 else ""}
                    </div>""",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Summarise tab
# ---------------------------------------------------------------------------

def render_summarise_tab(pipeline, filename_filter: str | None) -> None:
    """Render the document summarisation interface."""
    st.markdown("## 📋 Document Summariser")

    _, vector_store, _, _ = get_components()
    docs = vector_store.list_documents()

    if not docs:
        st.info("Upload and index a document first.")
        return

    doc_names = [d["filename"] for d in docs]
    target = filename_filter or st.selectbox("Choose a document to summarise:", doc_names)

    if st.button("Generate Summary", type="primary"):
        with st.spinner(f"Analysing '{target}'..."):
            summary = pipeline.summarise_document(target)
        st.markdown("### Summary")
        st.markdown(summary)


# ---------------------------------------------------------------------------
# Evaluate tab
# ---------------------------------------------------------------------------

def render_evaluate_tab(pipeline) -> None:
    """Render the evaluation metrics interface."""
    st.markdown("## 📊 Pipeline Evaluation")
    st.markdown(
        "Run a benchmark to measure retrieval quality, hallucination rate, and answer relevance."
    )

    from src.evaluation.evaluator import Evaluator

    evaluator = Evaluator(pipeline)

    col1, col2 = st.columns(2)
    with col1:
        uploaded_eval = st.file_uploader(
            "Upload eval questions (JSON) — optional",
            type=["json"],
            key="eval_uploader",
        )

    with col2:
        st.markdown("**JSON format:**")
        st.code(
            '[{"question": "What is the termination clause?", '
            '"expected_answer": "30 days notice"}]',
            language="json",
        )

    if st.button("▶️ Run Evaluation", type="primary"):
        questions = None
        if uploaded_eval:
            data = json.loads(uploaded_eval.read().decode("utf-8"))
            from src.evaluation.evaluator import EvalQuestion
            questions = [EvalQuestion(**item) for item in data]

        with st.spinner("Running evaluation..."):
            report = evaluator.run(questions=questions, save_results=True)

        st.markdown("### 📊 Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Questions", report.total_questions)
        c2.metric("Answerable", f"{report.answerable_rate:.0%}")
        c3.metric("Faithfulness", f"{report.avg_faithfulness:.3f}")
        c4.metric("Avg Latency", f"{report.avg_latency_ms:.0f} ms")

        st.markdown("### Per-question Breakdown")
        import pandas as pd
        df = pd.DataFrame([
            {
                "Question": r.question[:60] + "..." if len(r.question) > 60 else r.question,
                "Answerable": "✅" if r.is_answerable else "❌",
                "Faithfulness": r.faithfulness_score,
                "Sources Retrieved": r.num_sources_retrieved,
                "Sources Cited": r.num_sources_cited,
                "Latency (ms)": f"{r.latency_ms:.0f}",
            }
            for r in report.results
        ])
        st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    init_session_state()

    try:
        embedder, vector_store, retriever, pipeline = get_components()
    except Exception as e:
        st.error(f"Failed to initialise RAG components: {e}")
        st.stop()

    filename_filter, mode = render_sidebar(vector_store)

    if "chat" in mode:
        render_chat_tab(pipeline, filename_filter)
    elif "summar" in mode:
        render_summarise_tab(pipeline, filename_filter)
    elif "evaluat" in mode:
        render_evaluate_tab(pipeline)


if __name__ == "__main__":
    main()
