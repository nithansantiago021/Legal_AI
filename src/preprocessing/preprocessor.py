"""
preprocessor.py — Legal document cleaning and chunking pipeline.

Legal texts arrive in many raw forms: PDFs with scanned OCR artefacts,
plain-text contracts with inconsistent formatting, and structured HTML
court judgments.  This module normalises all of them into clean text,
then splits the text into retrieval-friendly chunks with rich metadata.

Design decisions
----------------
* We preserve section headers as metadata (not inline text) so that the
  retrieval system can filter by clause type later.
* RecursiveCharacterTextSplitter (from LangChain) is preferred over token-
  level splitters for legal text because legal sentences can be very long
  and we want chunks to align with natural paragraph breaks, not byte counts.
* We keep the original character offsets so we can show exact document
  locations in the UI.
"""

import re
import json
import hashlib
from pathlib import Path
from typing import Iterator
from dataclasses import dataclass, field, asdict

import fitz            # PyMuPDF — for PDF text extraction
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RawDocument:
    """Represents a single source file before any splitting."""
    doc_id: str          # deterministic SHA-256 of the file path
    filename: str        # basename of the source file
    file_type: str       # "pdf" | "txt" | "json"
    raw_text: str        # full extracted text (may be very long)
    metadata: dict = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """
    A single retrieval unit produced by splitting a RawDocument.

    Every field is intentionally serialisable (str / int / dict) so
    that chunks can be stored as ChromaDB metadata without conversion.
    """
    chunk_id: str        # SHA-256 of (doc_id + chunk_index)
    doc_id: str          # links back to the source RawDocument
    filename: str
    chunk_index: int     # ordinal position within the document
    text: str            # the actual chunk content sent to the LLM
    char_start: int      # byte offset in the original raw_text
    char_end: int
    section: str = ""    # detected section header (e.g. "CLAUSE 7 — TERMINATION")
    metadata: dict = field(default_factory=dict)

    def to_chroma_metadata(self) -> dict:
        """
        Flatten to a ChromaDB-compatible metadata dict.

        ChromaDB only accepts str/int/float/bool values, so we JSON-encode
        nested dicts and drop the full text (stored separately as 'document').
        """
        return {
            "doc_id":      self.doc_id,
            "filename":    self.filename,
            "chunk_index": self.chunk_index,
            "char_start":  self.char_start,
            "char_end":    self.char_end,
            "section":     self.section,
            **{k: str(v) for k, v in self.metadata.items()},
        }


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_text_from_pdf(path: Path) -> str:
    """
    Extract plain text from a PDF using PyMuPDF (fitz).

    PyMuPDF is chosen over pdfplumber because it handles scanned PDFs with
    embedded OCR layers and is significantly faster on large court judgments
    (100+ pages is common in legal corpora).

    Args:
        path: Absolute path to the PDF file.

    Returns:
        Concatenated text of all pages, with form-feed separators.
    """
    doc = fitz.open(str(path))
    pages: list[str] = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")   # "text" = plain text; "blocks" = layout-aware
        pages.append(text)
    doc.close()
    log.debug("PDF '%s' extracted: %d pages, %d chars", path.name, len(pages), sum(len(p) for p in pages))
    return "\f".join(pages)   # \f = form-feed, one per page boundary


def _extract_text_from_txt(path: Path) -> str:
    """Read a plain-text file, trying UTF-8 then Latin-1 as a fallback."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Latin-1 (ISO-8859-1) never raises UnicodeDecodeError — every byte
        # is a valid character.  Some older contract files use this encoding.
        log.warning("UTF-8 decode failed for '%s'; falling back to latin-1", path.name)
        return path.read_text(encoding="latin-1")


def _extract_text_from_json(path: Path) -> str:
    """
    Extract text from the Legal Text Classification Dataset JSON format.

    The dataset stores each case as {"text": "...", "label": "..."}.
    We concatenate the text fields with clear separators.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return "\n\n---\n\n".join(item.get("text", "") for item in data)
    if isinstance(data, dict):
        return data.get("text", str(data))
    return str(data)


def load_document(path: Path) -> RawDocument:
    """
    Dispatch to the correct extractor based on file extension.

    Args:
        path: Path to a legal document file.

    Returns:
        A RawDocument with raw_text and basic metadata filled in.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = path.suffix.lower()
    log.info("Loading document: %s", path.name)

    # Use the file's absolute path as the seed for a deterministic ID.
    # This means re-indexing the same file produces the same doc_id, which
    # allows us to detect and skip already-indexed documents.
    doc_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]

    extractors = {
        ".pdf":  _extract_text_from_pdf,
        ".txt":  _extract_text_from_txt,
        ".text": _extract_text_from_txt,
        ".json": _extract_text_from_json,
    }

    if ext not in extractors:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(extractors.keys())}")

    raw_text = extractors[ext](path)

    return RawDocument(
        doc_id=doc_id,
        filename=path.name,
        file_type=ext.lstrip("."),
        raw_text=raw_text,
        metadata={"source": str(path), "file_size_bytes": path.stat().st_size},
    )


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

# Pre-compiled patterns for speed — compiled once at import, reused for
# every document.
_NOISE_PATTERNS = [
    (re.compile(r"\f"),                    " "),     # form-feeds → space
    (re.compile(r"[ \t]+"),               " "),      # collapse horizontal whitespace
    (re.compile(r"\n{3,}"),              "\n\n"),    # max 2 consecutive newlines
    (re.compile(r"[-–—]{3,}"),           "---"),     # normalise dashes
    (re.compile(r"\x00"),                 ""),       # null bytes (rare but fatal for ChromaDB)
    (re.compile(r"[\u200b-\u200d\ufeff]"), ""),      # zero-width spaces / BOM
]

# Pattern to detect section headings in legal documents.
# Matches numbered clauses ("1.", "1.2", "CLAUSE 7") and all-caps headings.
_SECTION_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*\.?\s+[A-Z]"           # "1.2 Termination" or "7. Force Majeure"
    r"|CLAUSE\s+\d+"                        # "CLAUSE 7"
    r"|ARTICLE\s+[IVXLC\d]+"              # "ARTICLE IV" (Roman numerals)
    r"|(?:[A-Z][A-Z ]{4,})\s*$"           # ALL-CAPS HEADING ON OWN LINE
    r")",
    re.MULTILINE,
)


def clean_text(raw_text: str) -> str:
    """
    Normalise raw extracted text into clean, consistent plain text.

    We do NOT remove all whitespace or collapse paragraphs because legal
    meaning often depends on paragraph structure (e.g. a new paragraph can
    introduce a new obligation).

    Args:
        raw_text: Unprocessed text from a PDF or text file.

    Returns:
        Cleaned text suitable for embedding and chunking.
    """
    text = raw_text
    for pattern, replacement in _NOISE_PATTERNS:
        text = pattern.sub(replacement, text)
    # Strip leading/trailing whitespace from every line but keep blank lines
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def detect_section(text_chunk: str) -> str:
    """
    Attempt to identify the section / clause heading relevant to a chunk.

    We look at the first few lines of the chunk because headings appear at
    the start of a section, not buried in the middle.

    Args:
        text_chunk: A single text chunk.

    Returns:
        The detected heading string, or "" if no heading is found.
    """
    first_lines = "\n".join(text_chunk.splitlines()[:4])
    match = _SECTION_RE.search(first_lines)
    if match:
        # Return just the matched heading, stripped to one line
        return match.group(0).strip().split("\n")[0][:120]  # 120 char safety cap
    return ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_document(doc: RawDocument) -> list[DocumentChunk]:
    """
    Split a RawDocument into retrieval-optimised chunks.

    Strategy
    --------
    1. Clean the raw text.
    2. Use LangChain's RecursiveCharacterTextSplitter with legal-document
       separators.  The recursive strategy tries larger separators first
       (\n\n for paragraph breaks) before falling back to smaller ones,
       which keeps semantic units intact.
    3. Compute character offsets so the UI can highlight source passages.
    4. Detect section headers for each chunk to enable metadata filtering.

    Args:
        doc: A loaded RawDocument.

    Returns:
        A list of DocumentChunk objects ready for embedding.
    """
    clean = clean_text(doc.raw_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
        separators=cfg.chunking.separators,
        # length_function defaults to len(), which counts characters.
        # We use characters (not tokens) because our embedding model's
        # effective context window is ~512 tokens ≈ 1500–2000 chars for
        # legal prose, so character-based chunks at 512 are conservative
        # and will never exceed the model's limit.
        length_function=len,
        add_start_index=True,   # LangChain records char_start per chunk
    )

    lc_chunks = splitter.create_documents(
        texts=[clean],
        metadatas=[{"filename": doc.filename, "doc_id": doc.doc_id}],
    )

    chunks: list[DocumentChunk] = []
    for i, lc_chunk in enumerate(lc_chunks):
        char_start = lc_chunk.metadata.get("start_index", 0)
        chunk_text = lc_chunk.page_content

        chunk_id = hashlib.sha256(
            f"{doc.doc_id}:{i}".encode()
        ).hexdigest()[:16]

        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            filename=doc.filename,
            chunk_index=i,
            text=chunk_text,
            char_start=char_start,
            char_end=char_start + len(chunk_text),
            section=detect_section(chunk_text),
            metadata={
                "total_chunks": len(lc_chunks),
                "doc_file_type": doc.file_type,
            },
        ))

    log.info("Document '%s' → %d chunks", doc.filename, len(chunks))
    return chunks


def process_directory(directory: Path) -> Iterator[DocumentChunk]:
    """
    Walk a directory and yield DocumentChunk objects for all supported files.

    This is a generator to avoid loading all documents into memory at once —
    important when processing large legal corpora (hundreds of PDFs).

    Args:
        directory: Path to a folder containing legal documents.

    Yields:
        DocumentChunk objects from each file in the directory.
    """
    supported_exts = {".pdf", ".txt", ".text", ".json"}
    files = [
        f for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() in supported_exts
    ]
    log.info("Found %d files in '%s'", len(files), directory)

    for file_path in files:
        try:
            doc = load_document(file_path)
            chunks = chunk_document(doc)
            yield from chunks
        except Exception as exc:
            # Log and continue — one bad file should not abort the whole batch
            log.error("Failed to process '%s': %s", file_path.name, exc, exc_info=True)
