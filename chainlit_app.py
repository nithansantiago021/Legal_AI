"""
chainlit_app.py — Chainlit frontend for the Legal RAG Assistant.

Chainlit gives us a polished ChatGPT-style interface for free:
  • File upload directly in the chat input
  • Step-by-step reasoning display (retrieval steps visible)
  • Chat profiles (switch between Q&A / Summarise mode)
  • Markdown rendering with source cards
  • Session-scoped state (each browser tab is isolated)

Run with:
    chainlit run chainlit_app.py --watch

The --watch flag hot-reloads on file changes, just like Streamlit's --reload.

Architecture note
-----------------
Chainlit uses an async event loop.  All our RAG components are synchronous
(ChromaDB, sentence-transformers).  We run them in a thread-pool executor
via asyncio.to_thread() so we never block the event loop — this keeps the
UI responsive during long embedding or LLM calls.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import chainlit as cl

# ── Lazy imports — only pulled in when the server starts, not at module load ──
# This keeps `chainlit run` startup fast.
from config.settings import cfg
from src.embedding.embedder import Embedder, VectorStore
from src.embedding.indexer import index_file
from src.retrieval.retriever import Retriever
from src.rag.rag_pipeline import RAGPipeline, RAGResponse
from src.utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons — created once when the Chainlit server starts.
# Chainlit shares these across all sessions (safe because they are read-only
# after initialisation).
# ─────────────────────────────────────────────────────────────────────────────
_embedder: Embedder | None = None
_vector_store: VectorStore | None = None
_retriever: Retriever | None = None
_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    """
    Initialise RAG components lazily on first call.

    Using a module-level singleton avoids reloading the BGE model
    (130 MB) on every new chat session.
    """
    global _embedder, _vector_store, _retriever, _pipeline
    if _pipeline is None:
        log.info("Initialising RAG pipeline...")
        _embedder     = Embedder()
        _vector_store = VectorStore()
        _retriever    = Retriever(_embedder, _vector_store)
        _pipeline     = RAGPipeline(_retriever)
        log.info("RAG pipeline ready. Chunks in store: %d", _vector_store.count)
    return _pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Chat profiles — shown as a dropdown at the top of a new conversation.
# Each profile sets a different mode for the session.
# ─────────────────────────────────────────────────────────────────────────────
@cl.set_chat_profiles
async def set_chat_profiles():
    """
    Define the two modes the user can pick at the start of a session.

    Chainlit renders these as selectable cards before the first message.
    The selected profile name is stored in cl.user_session so we can
    branch behaviour inside on_message.
    """
    return [
        cl.ChatProfile(
            name="Legal Q&A",
            markdown_description=(
                "**Ask questions** about your uploaded legal documents.\n\n"
                "Upload PDFs, TXTs, or CSVs directly in the chat input. "
                "Every answer is grounded in document text with citations."
            ),
            icon="⚖️",
        ),
        cl.ChatProfile(
            name="Summarise",
            markdown_description=(
                "**Generate a structured summary** of an indexed legal document.\n\n"
                "Type the filename of a document you have already uploaded "
                "and I will produce a clause-by-clause breakdown."
            ),
            icon="📋",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Session start
# ─────────────────────────────────────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    """
    Called once when a user opens a new chat tab.

    We:
    1. Warm up the pipeline (no-op if already initialised)
    2. Store per-session state in cl.user_session
    3. Send a welcome message appropriate to the selected profile
    """
    # Warm up in a thread so we don't block the async loop
    await asyncio.to_thread(_get_pipeline)

    # Per-session state — isolated between browser tabs
    cl.user_session.set("doc_filter", None)   # filename to restrict search to
    cl.user_session.set("profile", cl.user_session.get("chat_profile", "Legal Q&A"))

    profile = cl.user_session.get("chat_profile", "Legal Q&A")
    vs = _vector_store

    # Build indexed-documents list for the welcome message
    docs = vs.list_documents() if vs else []
    doc_list_md = "\n".join(f"- 📄 `{d['filename']}`" for d in docs) if docs else "_No documents indexed yet._"

    if profile == "Legal Q&A":
        welcome = f"""## ⚖️ Legal Q&A Assistant

I answer questions grounded exclusively in your uploaded legal documents.
Every answer includes **[SOURCE N]** citations so you can verify each claim.

**Currently indexed documents:**
{doc_list_md}

**You can:**
- 📎 Upload a document directly in the chat input (PDF, TXT, CSV, JSON)
- 💬 Ask any legal question once a document is indexed
- 🔍 Type `filter: filename.pdf` to restrict search to one document
- 📋 Type `list docs` to see all indexed documents
"""
    else:
        welcome = f"""## 📋 Document Summariser

I generate a structured legal summary covering parties, obligations,
key dates, termination clauses, and notable risks.

**Currently indexed documents:**
{doc_list_md}

**Usage:** Type the exact filename you want summarised, e.g.:
> `summarise sample_contract.pdf`

Or upload a new document first, then ask to summarise it.
"""

    await cl.Message(content=welcome, author="Lex").send()


# ─────────────────────────────────────────────────────────────────────────────
# File upload handler — triggered when user attaches a file to a message
# ─────────────────────────────────────────────────────────────────────────────
async def handle_file_upload(files: list[cl.File]) -> list[str]:
    """
    Index uploaded files and return a list of status strings.

    Chainlit passes uploaded files as cl.File objects with a .path
    attribute pointing to a temp file on disk.  We pass this directly
    to our indexer, then report back.

    Args:
        files: List of cl.File objects from the message.

    Returns:
        List of status messages to include in the response.
    """
    pipeline = _get_pipeline()
    statuses: list[str] = []

    for f in files:
        file_path = Path(f.path)
        supported = {".pdf", ".txt", ".text", ".json", ".csv"}

        if file_path.suffix.lower() not in supported:
            statuses.append(f"⚠️ `{f.name}` — unsupported type (use PDF, TXT, CSV, JSON)")
            continue

        # Show a step so the user sees what's happening in real time
        async with cl.Step(name=f"Indexing {f.name}", type="tool") as step:
            step.input = f"File: {f.name} ({f.size / 1024:.1f} KB)"

            try:
                progress_messages: list[str] = []

                def progress_cb(msg: str) -> None:
                    """Collect progress messages from the indexer."""
                    progress_messages.append(msg)

                # Run the synchronous indexer in a thread
                chunks = await asyncio.to_thread(
                    index_file, file_path, _embedder, _vector_store, progress_cb
                )

                step.output = f"✅ Indexed {len(chunks)} chunks\n" + "\n".join(progress_messages[-3:])
                statuses.append(f"✅ **`{f.name}`** → {len(chunks)} chunks indexed")

            except Exception as exc:
                step.output = f"❌ Failed: {exc}"
                statuses.append(f"❌ **`{f.name}`** — failed: {str(exc)[:120]}")
                log.error("Upload indexing failed for '%s': %s", f.name, exc, exc_info=True)

    return statuses


# ─────────────────────────────────────────────────────────────────────────────
# Main message handler
# ─────────────────────────────────────────────────────────────────────────────
@cl.on_message
async def on_message(message: cl.Message):
    """
    Called on every user message.

    Routing logic:
    1.  If files are attached → index them first, then answer the text
    2.  If profile is Summarise → run summarisation
    3.  Special commands (list docs, filter:) → handle inline
    4.  Otherwise → run the RAG pipeline
    """
    pipeline  = _get_pipeline()
    profile   = cl.user_session.get("chat_profile", "Legal Q&A")
    text      = message.content.strip()
    files     = message.elements   # cl.File objects if any were attached

    # ── 1. Handle file uploads ────────────────────────────────────────────
    if files:
        # Filter to actual file elements (Chainlit can also pass other element types)
        uploads = [e for e in files if isinstance(e, cl.File)]
        if uploads:
            statuses = await handle_file_upload(uploads)
            status_block = "\n".join(statuses)

            # If the user also typed a question, answer it after indexing
            if not text or text.lower() in ("", "upload", "index"):
                docs = _vector_store.list_documents() if _vector_store else []
                doc_list = "\n".join(f"- `{d['filename']}`" for d in docs)
                await cl.Message(
                    content=f"{status_block}\n\n**All indexed documents:**\n{doc_list}",
                    author="Lex"
                ).send()
                return
            # else: fall through and answer the question too

    # ── 2. Special commands ───────────────────────────────────────────────

    # list docs
    if text.lower() in ("list docs", "list documents", "show docs", "what documents"):
        docs = _vector_store.list_documents() if _vector_store else []
        if docs:
            lines = "\n".join(f"- 📄 `{d['filename']}` (id: `{d['doc_id'][:8]}…`)" for d in docs)
            content = f"**{len(docs)} indexed document(s):**\n\n{lines}\n\n_Total chunks: {_vector_store.count}_"
        else:
            content = "No documents indexed yet. Upload a file using the 📎 attachment button."
        await cl.Message(content=content, author="Lex").send()
        return

    # filter: filename.pdf
    if text.lower().startswith("filter:"):
        filename = text[7:].strip()
        cl.user_session.set("doc_filter", filename if filename else None)
        msg = f"🔍 Search now restricted to `{filename}`." if filename else "🔍 Filter cleared — searching all documents."
        await cl.Message(content=msg, author="Lex").send()
        return

    # clear filter
    if text.lower() in ("clear filter", "remove filter", "all docs"):
        cl.user_session.set("doc_filter", None)
        await cl.Message(content="🔍 Filter cleared — searching all documents.", author="Lex").send()
        return

    # ── 3. Summarise mode ─────────────────────────────────────────────────
    if profile == "Summarise":
        await _handle_summarise(text, pipeline)
        return

    # ── 4. Q&A mode — full RAG pipeline ──────────────────────────────────
    if not text:
        await cl.Message(content="Please type a question or upload a document.", author="Lex").send()
        return

    await _handle_qa(text, pipeline)


# ─────────────────────────────────────────────────────────────────────────────
# Q&A handler
# ─────────────────────────────────────────────────────────────────────────────
async def _handle_qa(question: str, pipeline: RAGPipeline) -> None:
    """
    Run the full RAG pipeline and stream the response.

    Uses cl.Step to show the retrieval step as a collapsible sub-step
    so the user can inspect which passages were found before reading
    the final answer.
    """
    doc_filter = cl.user_session.get("doc_filter")
    if _vector_store and _vector_store.count == 0:
        await cl.Message(
            content="⚠️ No documents indexed yet. Upload a document using the 📎 button.",
            author="Lex"
        ).send()
        return

    # ── Retrieval step — shown as collapsible in the UI ──────────────────
    async with cl.Step(name="🔍 Retrieving relevant passages", type="retrieval") as step:
        step.input = question

        # Run synchronous retrieval in a thread
        chunks = await asyncio.to_thread(
            _retriever.retrieve, question, None, doc_filter
        )

        if chunks:
            # Format retrieved chunks as a numbered list inside the step
            chunk_lines = []
            for i, c in enumerate(chunks, 1):
                chunk_lines.append(
                    f"**[SOURCE {i}]** `{c.filename}`"
                    + (f" › {c.section}" if c.section else "")
                    + f" (sim: {c.similarity:.3f})\n"
                    + f"> {c.text[:200].strip()}{'…' if len(c.text) > 200 else ''}"
                )
            step.output = "\n\n".join(chunk_lines)
        else:
            step.output = "No relevant passages found above the similarity threshold."

    # ── LLM generation — shown as a second step ───────────────────────────
    async with cl.Step(name="🤖 Generating answer", type="llm") as step:
        step.input = f"Question: {question}\nContext chunks: {len(chunks)}"

        # Run the full pipeline (includes prompt build + LLM call) in a thread
        resp: RAGResponse = await asyncio.to_thread(
            pipeline.run, question, doc_filter
        )

        step.output = f"Faithfulness: {resp.faithfulness_score:.3f} | Latency: {resp.latency_ms:.0f}ms"

    # ── Final answer message ───────────────────────────────────────────────
    answer_md = _format_answer(resp)
    await cl.Message(content=answer_md, author="Lex").send()


def _format_answer(resp: RAGResponse) -> str:
    """
    Format a RAGResponse as rich Markdown for Chainlit.

    Chainlit renders Markdown natively, so we can use:
    - Bold, italic, code spans
    - Blockquotes for source cards
    - Tables for metrics
    """
    # Colour the [SOURCE N] tags in the answer text
    answer_text = resp.answer.replace(
        "[SOURCE", "**[SOURCE"
    ).replace("]", "]**") if resp.is_answerable else resp.answer

    # Build source cards as blockquotes
    source_cards = ""
    if resp.sources:
        cited = set(resp.cited_source_indices)
        cards: list[str] = []
        for i, chunk in enumerate(resp.sources, 1):
            cited_badge = " ✅ _cited_" if i in cited else ""
            section_part = f" › `{chunk.section}`" if chunk.section else ""
            cards.append(
                f"> **[SOURCE {i}]**{cited_badge}  \n"
                f"> 📄 `{chunk.filename}`{section_part} | sim: `{chunk.similarity:.3f}`  \n"
                f"> {chunk.text[:300].strip().replace(chr(10), '  ') }{'…' if len(chunk.text) > 300 else ''}"
            )
        source_cards = "\n\n---\n### 📖 Retrieved Sources\n\n" + "\n\n".join(cards)

    # Metrics footer
    faith_emoji = "🟢" if resp.faithfulness_score >= 0.8 else "🟡" if resp.faithfulness_score >= 0.5 else "🔴"
    metrics = (
        f"\n\n---\n"
        f"_{faith_emoji} Faithfulness: `{resp.faithfulness_score:.2f}` · "
        f"⏱ `{resp.latency_ms:.0f}ms` · "
        f"📚 {len(resp.sources)} sources · "
        f"🔗 {len(resp.cited_source_indices)} cited · "
        f"🤖 `{resp.model_used}`_"
    )

    return answer_text + source_cards + metrics


# ─────────────────────────────────────────────────────────────────────────────
# Summarise handler
# ─────────────────────────────────────────────────────────────────────────────
async def _handle_summarise(text: str, pipeline: RAGPipeline) -> None:
    """
    Handle summarisation requests in the Summarise profile.

    Accepts:
    - "summarise filename.pdf"
    - "summarize contract.txt"
    - Just the filename: "contract.pdf"
    """
    # Parse the filename from the message
    filename = text
    for prefix in ("summarise ", "summarize ", "summary of ", "summary "):
        if text.lower().startswith(prefix):
            filename = text[len(prefix):].strip()
            break

    if not filename:
        docs = _vector_store.list_documents() if _vector_store else []
        if docs:
            names = "\n".join(f"- `{d['filename']}`" for d in docs)
            await cl.Message(
                content=f"Please specify a filename to summarise:\n\n{names}",
                author="Lex"
            ).send()
        else:
            await cl.Message(
                content="No documents indexed yet. Upload a file first.",
                author="Lex"
            ).send()
        return

    # Check the filename exists in the store
    docs = _vector_store.list_documents() if _vector_store else []
    known = [d["filename"] for d in docs]
    if filename not in known:
        # Try a fuzzy match — the user may have omitted the extension
        matches = [n for n in known if filename.lower() in n.lower()]
        if len(matches) == 1:
            filename = matches[0]
        elif matches:
            names = "\n".join(f"- `{m}`" for m in matches)
            await cl.Message(
                content=f"Multiple matches found for `{filename}`:\n\n{names}\n\nPlease be more specific.",
                author="Lex"
            ).send()
            return
        else:
            names = "\n".join(f"- `{n}`" for n in known) if known else "_none_"
            await cl.Message(
                content=f"Document `{filename}` not found.\n\n**Indexed documents:**\n{names}",
                author="Lex"
            ).send()
            return

    async with cl.Step(name=f"📋 Summarising {filename}", type="tool") as step:
        step.input = filename
        summary = await asyncio.to_thread(pipeline.summarise_document, filename)
        step.output = f"Generated {len(summary)} chars"

    # Format citations in the summary
    summary_md = summary.replace("[SOURCE", "**[SOURCE").replace("]", "]**")

    await cl.Message(
        content=f"## 📋 Summary — `{filename}`\n\n{summary_md}",
        author="Lex"
    ).send()
