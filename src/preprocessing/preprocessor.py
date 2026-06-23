import sys
import re
import json
import csv
import hashlib
from pathlib import Path
from typing import Iterator
from dataclasses import dataclass, field

import fitz 
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RawDocument:
    doc_id: str          
    filename: str        
    file_type: str       
    raw_text: str        
    metadata: dict = field(default_factory=dict)


@dataclass
class DocumentChunk:
    chunk_id: str        
    doc_id: str          
    filename: str
    chunk_index: int     
    text: str            
    char_start: int      
    char_end: int
    section: str = ""    
    metadata: dict = field(default_factory=dict)

    def to_chroma_metadata(self) -> dict:
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
    doc = fitz.open(str(path))
    pages: list[str] = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")   
        pages.append(text)
    doc.close()
    log.debug("PDF '%s' extracted: %d pages, %d chars", path.name, len(pages), sum(len(p) for p in pages))
    return "\f".join(pages)   


def _extract_text_from_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        log.warning("UTF-8 decode failed for '%s'; falling back to latin-1", path.name)
        return path.read_text(encoding="latin-1")


def _extract_text_from_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return "\n\n---\n\n".join(item.get("text", "") for item in data)
    if isinstance(data, dict):
        return data.get("text", str(data))
    return str(data)


def _extract_text_from_csv(path: Path) -> str:
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)
    rows = []
    with open(path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text_lines = [f"{col}: {val}" for col, val in row.items() if val]
            rows.append("\n".join(text_lines))
    return "\n\n---\n\n".join(rows)


def load_document(path: Path) -> RawDocument:
    ext = path.suffix.lower()
    log.info("Loading document: %s", path.name)

    doc_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]

    extractors = {
        ".pdf":  _extract_text_from_pdf,
        ".txt":  _extract_text_from_txt,
        ".text": _extract_text_from_txt,
        ".json": _extract_text_from_json,
        ".csv":  _extract_text_from_csv,
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

_NOISE_PATTERNS = [
    (re.compile(r"\f"),                    " "),     
    (re.compile(r"[ \t]+"),               " "),      
    (re.compile(r"\n{3,}"),              "\n\n"),    
    (re.compile(r"[-–—]{3,}"),           "---"),     
    (re.compile(r"\x00"),                  ""),       
    (re.compile(r"[\u200b-\u200d\ufeff]"), ""),      
]

_SECTION_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*\.?\s+[A-Z]"           
    r"|CLAUSE\s+\d+"                        
    r"|ARTICLE\s+[IVXLC\d]+"              
    r"|(?:[A-Z][A-Z ]{4,})\s*$"           
    r")",
    re.MULTILINE,
)


def clean_text(raw_text: str) -> str:
    text = raw_text
    for pattern, replacement in _NOISE_PATTERNS:
        text = pattern.sub(replacement, text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def detect_section(text_chunk: str) -> str:
    first_lines = "\n".join(text_chunk.splitlines()[:4])
    match = _SECTION_RE.search(first_lines)
    if match:
        return match.group(0).strip().split("\n")[0][:120]  
    return ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_document(doc: RawDocument) -> list[DocumentChunk]:
    if doc.file_type == "csv":
        chunks: list[DocumentChunk] = []
        raw_chunks = doc.raw_text.split("\n\n---\n\n")
        
        current_offset = 0
        for i, chunk_text in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue
                
            chunk_id = hashlib.sha256(f"{doc.doc_id}:{i}".encode()).hexdigest()[:16]
            
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                filename=doc.filename,
                chunk_index=i,
                text=chunk_text,
                char_start=current_offset,
                char_end=current_offset + len(chunk_text),
                section="",
                metadata={
                    "total_chunks": len(raw_chunks),
                    "doc_file_type": doc.file_type,
                },
            ))
            current_offset += len(chunk_text) + 9 
            
        log.info("Document '%s' → %d chunks", doc.filename, len(chunks))
        return chunks

    clean = clean_text(doc.raw_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
        separators=cfg.chunking.separators,
        length_function=len,
        add_start_index=True,   
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
    supported_exts = {".pdf", ".txt", ".text", ".json", ".csv"}
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
            log.error("Failed to process '%s': %s", file_path.name, exc, exc_info=True)