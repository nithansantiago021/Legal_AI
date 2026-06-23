"""
embedder.py — Embedding + ephemeral vector store for HF Spaces.

Key changes from the local version:
  * VectorStore uses chromadb.EphemeralClient() (pure in-memory) so that
    NOTHING is written to disk.  Each Streamlit session creates its own
    VectorStore instance stored in st.session_state, giving full isolation.
  * The expensive SentenceTransformer load is cached at the *process* level
    via @st.cache_resource so the model is loaded only once per cold start,
    not once per user session — but the VectorStore is never cached (it is
    always session-scoped).
  * BGE-small-en-v1.5 runs comfortably on CPU (33 MB, ~180ms per batch of 32).
"""

from __future__ import annotations

import time
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from config.settings import cfg
from src.preprocessing.preprocessor import DocumentChunk
from src.utils.logger import get_logger

log = get_logger(__name__)

_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    """
    Wraps BGE-small-en-v1.5 for asymmetric query/document encoding.
    Thread-safe and shareable across Streamlit sessions (read-only after init).
    """

    def __init__(self) -> None:
        log.info("Loading embedding model: %s", cfg.embedding.model_name)
        t0 = time.perf_counter()
        self._model = SentenceTransformer(
            cfg.embedding.model_name,
            device=cfg.embedding.device,   # "cpu" on HF free tier
        )
        log.info("Model loaded in %.2fs", time.perf_counter() - t0)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            batch_size=cfg.embedding.batch_size,
            normalize_embeddings=cfg.embedding.normalize,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def encode_query(self, query: str) -> list[float]:
        instructed = _BGE_QUERY_INSTRUCTION + query
        vec = self._model.encode(
            [instructed],
            normalize_embeddings=cfg.embedding.normalize,
        )
        return vec[0].tolist()


class VectorStore:
    """
    Ephemeral ChromaDB vector store — pure in-memory, zero disk writes.

    One instance per Streamlit session (stored in st.session_state).
    When the session ends or the user resets, the instance is garbage-collected
    and all data disappears automatically.
    """

    def __init__(self) -> None:
        # EphemeralClient = pure RAM, no sqlite, no disk, process-local only
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=cfg.vectorstore.collection_name,
            metadata={"hnsw:space": cfg.vectorstore.distance_metric},
        )
        log.info("Ephemeral VectorStore created (session-scoped, in-memory only)")

    @property
    def count(self) -> int:
        return self._collection.count()

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        batch_size: int = 64,
    ) -> None:
        assert len(chunks) == len(embeddings)
        for i in range(0, len(chunks), batch_size):
            bc = chunks[i : i + batch_size]
            be = embeddings[i : i + batch_size]
            self._collection.upsert(
                ids=[c.chunk_id for c in bc],
                embeddings=be,
                documents=[c.text for c in bc],
                metadatas=[c.to_chroma_metadata() for c in bc],
            )
        log.info("Upserted %d chunks (total: %d)", len(chunks), self.count)

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: Optional[int] = None,
        where: Optional[dict] = None,
    ) -> list[dict]:
        k = top_k or cfg.retrieval.top_k
        if self._collection.count() == 0:
            return []

        kw: dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, self._collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kw["where"] = where

        results = self._collection.query(**kw)
        hits = []
        for doc, meta, dist, cid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            similarity = 1.0 - dist
            if similarity >= cfg.retrieval.similarity_threshold:
                hits.append({
                    "id": cid,
                    "text": doc,
                    "metadata": meta,
                    "similarity": round(similarity, 4),
                })
        return hits

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})
        log.info("Deleted chunks for doc_id='%s'", doc_id)

    def list_documents(self) -> list[dict]:
        if self.count == 0:
            return []
        all_meta = self._collection.get(include=["metadatas"])["metadatas"]
        seen: set[str] = set()
        docs: list[dict] = []
        for m in all_meta:
            if m["doc_id"] not in seen:
                seen.add(m["doc_id"])
                docs.append({"doc_id": m["doc_id"], "filename": m["filename"]})
        return docs

    def reset(self) -> None:
        """Nuke all data in this store — called on manual session clear."""
        self._client.delete_collection(cfg.vectorstore.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=cfg.vectorstore.collection_name,
            metadata={"hnsw:space": cfg.vectorstore.distance_metric},
        )
        log.info("VectorStore reset: all chunks deleted")
