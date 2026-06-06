"""
retriever.py — Semantic retrieval with optional metadata filtering.

The retriever is the bridge between the user's question and the relevant
passages in the vector store.  It wraps ChromaDB search and adds:

  1. Metadata-based filtering (restrict to a specific uploaded document)
  2. Optional cross-encoder re-ranking (improves precision at the cost of
     latency; disabled by default since it requires an extra model download)
  3. Deduplication of near-identical chunks (can appear due to overlapping
     windows in the same document)

Design decisions
----------------
* Two-stage retrieval (retrieve more → re-rank → keep top-n) is preferred
  over just increasing top_k for the LLM, because passing too many chunks
  dilutes the LLM's attention and increases cost / latency.
* The cross-encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2) is loaded
  lazily — only when use_reranker=True — to keep startup time fast.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Optional

from config.settings import cfg
from src.embedding.embedder import Embedder, VectorStore
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A single chunk returned by the retriever, ready for LLM consumption."""
    chunk_id: str
    text: str
    filename: str
    section: str
    chunk_index: int
    similarity: float
    doc_id: str


class Retriever:
    """
    Performs semantic search against the vector store.

    Typical flow
    ------------
    1. embed the query
    2. run top-k ANN search in ChromaDB
    3. filter by similarity threshold
    4. optionally re-rank with a cross-encoder
    5. deduplicate near-identical results
    6. return top-n chunks
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = None  # loaded lazily

    def _load_reranker(self) -> None:
        """
        Lazily load the cross-encoder re-ranker.

        Cross-encoders jointly encode (query, passage) pairs and produce a
        relevance score that is much more accurate than bi-encoder cosine
        similarity — but require O(k) forward passes (one per candidate),
        so we only use them when enabled.
        """
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            log.info("Loading cross-encoder re-ranker...")
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _deduplicate(self, chunks: list[RetrievedChunk], threshold: float = 0.85) -> list[RetrievedChunk]:
        """
        Remove near-duplicate chunks using character-level sequence ratio.

        Duplicates arise when the same passage is covered by two overlapping
        windows from the same document.  Presenting both to the LLM wastes
        context tokens without adding information.

        Args:
            chunks:    Retrieved chunks, sorted by relevance.
            threshold: SequenceMatcher ratio above which two chunks are
                       considered duplicates.  The lower-ranked duplicate
                       is dropped.

        Returns:
            Deduplicated list, preserving original order.
        """
        unique: list[RetrievedChunk] = []
        for candidate in chunks:
            is_dup = any(
                difflib.SequenceMatcher(
                    None, candidate.text, kept.text, autojunk=False
                ).ratio() >= threshold
                for kept in unique
            )
            if not is_dup:
                unique.append(candidate)
        return unique

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filename_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query:           The user's natural-language question.
            top_k:           Override the default cfg.retrieval.top_k.
            filename_filter: If provided, restrict search to chunks from
                             this specific filename.

        Returns:
            A ranked list of RetrievedChunk objects (most relevant first).
        """
        if self._vector_store.count == 0:
            log.warning("Vector store is empty — cannot retrieve anything")
            return []

        # ── 1. Embed the query ────────────────────────────────────────────
        query_vec = self._embedder.encode_query(query)

        # ── 2. Build optional ChromaDB metadata filter ────────────────────
        where = None
        if filename_filter:
            where = {"filename": filename_filter}

        # ── 3. ANN search ─────────────────────────────────────────────────
        k = top_k or cfg.retrieval.top_k
        raw_hits = self._vector_store.similarity_search(
            query_embedding=query_vec,
            top_k=k,
            where=where,
        )

        if not raw_hits:
            log.info("No hits above similarity threshold for query: '%s'", query[:80])
            return []

        # ── 4. Convert to typed objects ───────────────────────────────────
        chunks = [
            RetrievedChunk(
                chunk_id=h["id"],
                text=h["text"],
                filename=h["metadata"].get("filename", "unknown"),
                section=h["metadata"].get("section", ""),
                chunk_index=int(h["metadata"].get("chunk_index", 0)),
                similarity=h["similarity"],
                doc_id=h["metadata"].get("doc_id", ""),
            )
            for h in raw_hits
        ]

        # ── 5. Optional cross-encoder re-ranking ──────────────────────────
        if cfg.retrieval.use_reranker and len(chunks) > 1:
            self._load_reranker()
            pairs = [(query, c.text) for c in chunks]
            scores = self._reranker.predict(pairs)
            # Sort by re-ranker score (descending), keep top-n
            chunks = [
                c for _, c in sorted(
                    zip(scores, chunks), key=lambda x: x[0], reverse=True
                )
            ][: cfg.retrieval.rerank_top_n]
        else:
            chunks = chunks[: cfg.retrieval.rerank_top_n]

        # ── 6. Deduplicate ────────────────────────────────────────────────
        chunks = self._deduplicate(chunks)

        log.info(
            "Retrieved %d chunks for query '%s...' (top sim=%.3f)",
            len(chunks), query[:50], chunks[0].similarity if chunks else 0,
        )
        return chunks
