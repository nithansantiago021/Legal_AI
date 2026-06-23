"""
indexer.py — End-to-end document indexing pipeline.

Orchestrates the sequence: load → clean → chunk → embed → store.
Exposed as a single function so the Streamlit UI and the CLI can both
call it with identical behaviour.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from src.preprocessing.preprocessor import (
    DocumentChunk,
    load_document,
    chunk_document,
    process_directory,
)
from src.embedding.embedder import Embedder, VectorStore
from src.utils.logger import get_logger

log = get_logger(__name__)


def index_file(
    file_path: Path,
    embedder: Embedder,
    vector_store: VectorStore,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> list[DocumentChunk]:
    """
    Index a single legal document file.

    Args:
        file_path:         Path to the file to index.
        embedder:          Shared Embedder instance (avoid reloading the model).
        vector_store:      Shared VectorStore instance.
        progress_callback: Optional callable that receives progress messages
                           (used by the Streamlit UI to update a status widget).

    Returns:
        The list of DocumentChunk objects that were indexed.
    """
    def _progress(msg: str) -> None:
        log.info(msg)
        if progress_callback:
            progress_callback(msg)

    _progress(f"Loading: {file_path.name}")
    doc = load_document(file_path)

    _progress(f"Chunking: {file_path.name}")
    chunks = chunk_document(doc)

    if not chunks:
        log.warning("No chunks produced from '%s' — skipping", file_path.name)
        return []

    _progress(f"Embedding {len(chunks)} chunks...")
    t0 = time.perf_counter()
    embeddings = embedder.encode_documents([c.text for c in chunks])
    elapsed = time.perf_counter() - t0
    _progress(f"Embedded {len(chunks)} chunks in {elapsed:.1f}s")

    _progress(f"Storing in vector database...")
    vector_store.upsert_chunks(chunks, embeddings)
    _progress(f"Indexed '{file_path.name}': {len(chunks)} chunks stored")

    return chunks


def index_directory(
    directory: Path,
    embedder: Embedder,
    vector_store: VectorStore,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Recursively index all supported files in a directory.

    Returns:
        Total number of chunks indexed.
    """
    supported_exts = {".pdf", ".txt", ".text", ".json"}
    files = [
        f for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() in supported_exts
    ]

    if not files:
        log.warning("No supported files found in '%s'", directory)
        return 0

    log.info("Indexing %d files from '%s'", len(files), directory)
    total_chunks = 0

    for file_path in files:
        try:
            chunks = index_file(file_path, embedder, vector_store, progress_callback)
            total_chunks += len(chunks)
        except Exception as exc:
            log.error("Failed to index '%s': %s", file_path.name, exc, exc_info=True)

    log.info("Indexing complete. Total chunks stored: %d", total_chunks)
    return total_chunks
