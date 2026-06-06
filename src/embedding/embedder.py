"""
embedder.py — Embedding generation and vector store management.

We wrap the Sentence-Transformers model and ChromaDB client in thin,
testable classes so that the rest of the codebase doesn't depend on
implementation details of either library.

Design decisions
----------------
* BGE-small-en-v1.5 prepends a task-specific instruction to every *query*
  (but NOT to documents).  This asymmetric encoding is a known technique
  from the BGE paper that significantly improves retrieval recall.
* ChromaDB is used in persistent mode so the index survives restarts.
  We use the new chromadb >= 0.4 API (HttpClient vs. PersistentClient).
* Batched insertion prevents OOM on large corpora — we never load all
  embeddings into RAM at the same time.
"""

from __future__ import annotations

import time
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config.settings import cfg
from src.preprocessing.preprocessor import DocumentChunk
from src.utils.logger import get_logger

log = get_logger(__name__)

# BGE-small instruction prefix for query-side encoding only.
# This is the exact string recommended in the BGE model card.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    """
    Wraps the Sentence-Transformers model to provide encode() methods
    for both documents and queries with the correct BGE instruction handling.
    """

    def __init__(self) -> None:
        log.info("Loading embedding model: %s", cfg.embedding.model_name)
        t0 = time.perf_counter()
        self._model = SentenceTransformer(
            cfg.embedding.model_name,
            device=cfg.embedding.device,
        )
        log.info("Model loaded in %.2fs", time.perf_counter() - t0)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Produce embeddings for a batch of document chunks.

        Documents are encoded WITHOUT any instruction prefix — that is the
        BGE convention.  The model was fine-tuned with this asymmetry.

        Args:
            texts: List of chunk text strings.

        Returns:
            List of embedding vectors (each a list of 384 floats for BGE-small).
        """
        vecs = self._model.encode(
            texts,
            batch_size=cfg.embedding.batch_size,
            normalize_embeddings=cfg.embedding.normalize,
            show_progress_bar=len(texts) > 32,  # only show bar for large batches
        )
        return vecs.tolist()

    def encode_query(self, query: str) -> list[float]:
        """
        Produce an embedding for a single user query.

        The BGE instruction prefix is prepended HERE (not in encode_documents)
        because retrieval is asymmetric: the query instruction improves
        alignment with the document embedding space.

        Args:
            query: The user's natural-language question.

        Returns:
            A single embedding vector (384 floats).
        """
        instructed = _BGE_QUERY_INSTRUCTION + query
        vec = self._model.encode(
            [instructed],
            normalize_embeddings=cfg.embedding.normalize,
        )
        return vec[0].tolist()


class VectorStore:
    """
    Thin wrapper around ChromaDB that handles collection management,
    batched upsert, and similarity search.
    """

    def __init__(self) -> None:
        log.info("Connecting to ChromaDB at: %s", cfg.vectorstore.persist_directory)
        # PersistentClient auto-saves to disk after every mutation.
        self._client = chromadb.PersistentClient(
            path=cfg.vectorstore.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # get_or_create: idempotent — safe to call on every startup
        self._collection = self._client.get_or_create_collection(
            name=cfg.vectorstore.collection_name,
            metadata={"hnsw:space": cfg.vectorstore.distance_metric},
        )
        log.info(
            "Collection '%s' contains %d documents",
            cfg.vectorstore.collection_name,
            self._collection.count(),
        )

    @property
    def count(self) -> int:
        """Current number of chunks in the collection."""
        return self._collection.count()

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        batch_size: int = 128,
    ) -> None:
        """
        Insert or update chunks in ChromaDB.

        We use UPSERT (not insert) so that re-indexing the same document
        updates existing vectors rather than creating duplicates.  ChromaDB
        deduplicates on the 'ids' field.

        The batch_size cap prevents HTTP payload limits when using the
        ChromaDB HTTP server and avoids RAM spikes during large ingestions.

        Args:
            chunks:     DocumentChunk objects to store.
            embeddings: Corresponding embedding vectors.
            batch_size: Number of records per upsert call.
        """
        assert len(chunks) == len(embeddings), "Chunks and embeddings must align 1:1"

        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start : batch_start + batch_size]
            batch_embeds = embeddings[batch_start : batch_start + batch_size]

            self._collection.upsert(
                ids=[c.chunk_id for c in batch_chunks],
                embeddings=batch_embeds,
                documents=[c.text for c in batch_chunks],
                metadatas=[c.to_chroma_metadata() for c in batch_chunks],
            )
            log.debug(
                "Upserted batch %d-%d (%d records)",
                batch_start, batch_start + len(batch_chunks), len(batch_chunks)
            )

        log.info("Upserted %d chunks; collection now has %d", len(chunks), self.count)

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: Optional[int] = None,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Return the top-k most similar chunks for a query embedding.

        Args:
            query_embedding: The embedded user query (from Embedder.encode_query).
            top_k:           How many results to return; defaults to cfg.retrieval.top_k.
            where:           Optional ChromaDB metadata filter dict, e.g.
                             {"filename": "contract_v3.pdf"} to restrict search to
                             a specific document.

        Returns:
            List of result dicts with keys:
                id, document (text), metadata, distance (cosine distance, lower = more similar)
        """
        k = top_k or cfg.retrieval.top_k

        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, self._collection.count()),  # can't request more than we have
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        # ChromaDB returns nested lists (one sub-list per query).
        # We asked for one query, so we unpack [0].
        hits = []
        for doc, meta, dist, chunk_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            # Convert cosine distance → cosine similarity (1 - dist for normalised vecs)
            similarity = 1.0 - dist
            if similarity >= cfg.retrieval.similarity_threshold:
                hits.append({
                    "id":         chunk_id,
                    "text":       doc,
                    "metadata":   meta,
                    "similarity": round(similarity, 4),
                })

        log.debug("Query returned %d / %d hits above threshold", len(hits), k)
        return hits

    def delete_document(self, doc_id: str) -> None:
        """
        Remove all chunks belonging to a single document.

        Useful when a user wants to re-upload a corrected version of a contract.
        """
        self._collection.delete(where={"doc_id": doc_id})
        log.info("Deleted all chunks for doc_id='%s'", doc_id)

    def list_documents(self) -> list[dict]:
        """
        Return a deduplicated list of indexed documents with their metadata.

        Since each document is split into many chunks, we query all metadata
        and deduplicate by doc_id.
        """
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
