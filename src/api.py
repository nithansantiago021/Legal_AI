"""
api.py — FastAPI REST backend for the Legal RAG Assistant.

Exposes endpoints for:
  POST /index      — upload and index a legal document
  POST /query      — ask a question (RAG pipeline)
  POST /summarise  — summarise an indexed document
  GET  /documents  — list all indexed documents
  DELETE /documents/{doc_id} — remove a document

Run with:
  uvicorn src.api.api:app --reload --port 8000
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import cfg
from src.embedding.embedder import Embedder, VectorStore
from src.embedding.indexer import index_file
from src.retrieval.retriever import Retriever
from src.rag.rag_pipeline import RAGPipeline
from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# App init — dependencies are singletons, created once at startup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Legal RAG Assistant API",
    description="AI-powered legal document question-answering using RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# These are expensive to create (model loading, DB connection), so we
# create them once and share across requests.
_embedder: Embedder | None = None
_vector_store: VectorStore | None = None
_retriever: Retriever | None = None
_pipeline: RAGPipeline | None = None


@app.on_event("startup")
async def startup_event() -> None:
    """Initialise all shared components when the server starts."""
    global _embedder, _vector_store, _retriever, _pipeline
    log.info("Initialising RAG components...")
    _embedder     = Embedder()
    _vector_store = VectorStore()
    _retriever    = Retriever(_embedder, _vector_store)
    _pipeline     = RAGPipeline(_retriever)
    log.info("API ready. Vector store has %d chunks.", _vector_store.count)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    filename_filter: str | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    is_answerable: bool
    faithfulness_score: float
    latency_ms: float
    sources: list[dict]


class SummariseRequest(BaseModel):
    filename: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/index", summary="Upload and index a legal document")
async def index_document(file: UploadFile = File(...)) -> dict:
    """
    Accept a file upload, save it temporarily, index it, then delete the temp file.

    We write to a temp file (not memory) because our PDF extractor (PyMuPDF)
    requires a file path, not a byte stream.
    """
    if _embedder is None or _vector_store is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    allowed_exts = {".pdf", ".txt", ".text", ".json"}
    suffix = Path(file.filename or "doc").suffix.lower()
    if suffix not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed_exts}",
        )

    # Write to a temporary file so PyMuPDF can open it by path
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        chunks = index_file(tmp_path, _embedder, _vector_store)
        return {
            "filename": file.filename,
            "chunks_indexed": len(chunks),
            "message": f"Successfully indexed '{file.filename}' into {len(chunks)} chunks.",
        }
    except Exception as exc:
        log.error("Indexing failed for '%s': %s", file.filename, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)  # always clean up temp file


@app.post("/query", response_model=QueryResponse, summary="Ask a question")
async def query(request: QueryRequest) -> QueryResponse:
    """Run the RAG pipeline for a user question."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    resp = _pipeline.run(
        question=request.question,
        filename_filter=request.filename_filter,
    )

    return QueryResponse(
        question=resp.question,
        answer=resp.answer,
        is_answerable=resp.is_answerable,
        faithfulness_score=resp.faithfulness_score,
        latency_ms=resp.latency_ms,
        sources=[
            {
                "filename": c.filename,
                "section":  c.section,
                "similarity": c.similarity,
                "text_preview": c.text[:300] + "..." if len(c.text) > 300 else c.text,
            }
            for c in resp.sources
        ],
    )


@app.post("/summarise", summary="Summarise an indexed document")
async def summarise(request: SummariseRequest) -> dict:
    """Generate a structured summary of a previously indexed document."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    summary = _pipeline.summarise_document(request.filename)
    return {"filename": request.filename, "summary": summary}


@app.get("/documents", summary="List all indexed documents")
async def list_documents() -> dict:
    """Return a list of all documents currently in the vector store."""
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    docs = _vector_store.list_documents()
    return {"total": len(docs), "documents": docs}


@app.delete("/documents/{doc_id}", summary="Delete an indexed document")
async def delete_document(doc_id: str) -> dict:
    """Remove all chunks belonging to a document from the vector store."""
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    _vector_store.delete_document(doc_id)
    return {"message": f"Document '{doc_id}' removed from index."}


@app.get("/health")
async def health() -> dict:
    """Health check endpoint for load balancers / monitoring."""
    return {
        "status": "ok",
        "chunks_indexed": _vector_store.count if _vector_store else 0,
    }
