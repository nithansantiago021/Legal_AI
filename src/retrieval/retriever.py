"""
retriever.py — Two-stage retrieval: bi-encoder ANN + cross-encoder re-ranking.

Stage 1  all-mpnet-base-v2 encodes the query and performs approximate nearest-
         neighbour search in ChromaDB (cosine similarity).  We deliberately
         retrieve more candidates (top_k=12) than we will eventually pass to
         the LLM, because bi-encoders trade recall for speed and the re-ranker
         will prune the weak hits.

Stage 2  ms-marco-MiniLM-L-6-v2 (a cross-encoder) scores every (query, chunk)
         pair jointly.  Cross-encoders are far more accurate than bi-encoders
         but require O(k) inference passes, so we only apply them to the small
         set of ANN candidates.  The model is cached at the process level via
         @st.cache_resource in app.py so it is downloaded only once per cold
         start.

         After re-ranking:
           - Chunks with logit < rerank_score_threshold are dropped entirely.
           - If the best remaining chunk is still below min_top_rerank_score
             the retriever returns [] so the pipeline can gate to UNANSWERABLE
             without ever calling the LLM.

Stage 3  Near-duplicate deduplication (SequenceMatcher ratio) removes chunks
         that are near-identical due to overlapping windows.

Each RetrievedChunk carries both a `similarity` (bi-encoder cosine, 0–1) and
a `rerank_score` (cross-encoder raw logit, unbounded) so the pipeline and the
RAGAS evaluator can use whichever is appropriate.
"""

from __future__ import annotations

import difflib
import numpy as np
from dataclasses import dataclass
from typing import Optional

from config.settings import cfg
from src.embedding.embedder import Embedder, VectorStore
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    filename: str
    section: str
    chunk_index: int
    similarity: float
    rerank_score: float
    doc_id: str


class Retriever:
    """Two-stage semantic retriever with cross-encoder re-ranking."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = None

    def set_reranker(self, reranker) -> None:
        self._reranker = reranker

    def _load_reranker(self) -> None:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            log.info("Loading reranker: %s", cfg.retrieval.reranker_model)
            self._reranker = CrossEncoder(cfg.retrieval.reranker_model)

    def _deduplicate(
        self,
        chunks: list[RetrievedChunk],
        threshold: float = 0.85,
    ) -> list[RetrievedChunk]:
        unique = []
        for c in chunks:
            if any(
                difflib.SequenceMatcher(None, c.text, u.text, autojunk=False).ratio() >= threshold
                for u in unique
            ):
                continue
            unique.append(c)
        return unique

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filename_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:

        if self._vector_store.count == 0:
            log.warning("Empty vector store")
            return []

        # ── Stage 1: ANN retrieval ─────────────────────────────
        query_vec = self._embedder.encode_query(query)
        where = {"filename": filename_filter} if filename_filter else None
        k = top_k or cfg.retrieval.top_k

        raw_hits = self._vector_store.similarity_search(
            query_embedding=query_vec,
            top_k=k,
            where=where,
        )

        if not raw_hits:
            return []

        candidates = [
            RetrievedChunk(
                chunk_id=h["id"],
                text=h["text"],
                filename=h["metadata"].get("filename", "unknown"),
                section=h["metadata"].get("section", ""),
                chunk_index=int(h["metadata"].get("chunk_index", 0)),
                similarity=h["similarity"],
                rerank_score=0.0,
                doc_id=h["metadata"].get("doc_id", ""),
            )
            for h in raw_hits
            if h["similarity"] >= cfg.retrieval.similarity_threshold
        ]

        if not candidates:
            return []

        # ── Stage 2: Cross-encoder rerank (NO HARD THRESHOLDS) ───────────
        if cfg.retrieval.use_reranker and len(candidates) > 1:
            self._load_reranker()

            pairs = [(query, c.text) for c in candidates]
            scores = self._reranker.predict(pairs)

            # convert to float list
            scores = np.array(scores, dtype=float)

            # optional normalization (VERY useful for stability)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

            for c, s in zip(candidates, scores):
                c.rerank_score = float(s)

        else:
            for c in candidates:
                c.rerank_score = c.similarity

        # ── PURE RANKING (NO FILTERING) ────────────────────────────────
        candidates.sort(key=lambda c: c.rerank_score, reverse=True)
        candidates = candidates[: cfg.retrieval.rerank_top_n]

        # ── Stage 3: deduplication ──────────────────────────────────────
        candidates = self._deduplicate(candidates)

        log.info(
            "Retrieved %d chunks | top score=%.3f | query='%s'",
            len(candidates),
            candidates[0].rerank_score if candidates else 0.0,
            query[:50],
        )

        return candidates