"""
settings.py — Centralised configuration for the Legal RAG Assistant.

All environment-driven settings live here so that the rest of the codebase
never imports `os` or reads `.env` directly. This gives us a single source
of truth and makes testing easy (just patch this module).
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from the project root. ok=True means "don't crash if missing" —
# useful in CI where variables are injected via the environment directly.
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)

# ---------------------------------------------------------------------------
# Path constants — computed once so every module can do `from config.settings
# import PATHS` instead of re-deriving the same paths independently.
# ---------------------------------------------------------------------------
ROOT_DIR       = Path(__file__).parent.parent
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
VECTORSTORE_DIR= DATA_DIR / "vectorstore"
LOG_DIR        = ROOT_DIR / "logs"

# Create directories eagerly — avoids cryptic "file not found" errors later
for _d in (RAW_DIR, PROCESSED_DIR, VECTORSTORE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Embedding model settings.
# BGE-small-en-v1.5 is chosen because:
#   • Only 33M parameters — fast on CPU
#   • Trained specifically on retrieval tasks (not just general NLP)
#   • Produces 384-dim vectors, a good balance between quality and RAM
# ---------------------------------------------------------------------------
@dataclass
class EmbeddingConfig:
    model_name: str  = "BAAI/bge-small-en-v1.5"
    device: str      = "cpu"           # set to "cuda" if a GPU is available
    batch_size: int  = 32              # number of chunks to embed in one call
    normalize: bool  = True            # L2-normalise so cosine ≡ dot-product


# ---------------------------------------------------------------------------
# ChromaDB vector-store settings.
# Using a persistent (disk-backed) client so the index survives restarts.
# ---------------------------------------------------------------------------
@dataclass
class VectorStoreConfig:
    collection_name: str  = "legal_documents"
    persist_directory: str = str(VECTORSTORE_DIR)
    distance_metric: str   = "cosine"   # aligns with normalised BGE embeddings


# ---------------------------------------------------------------------------
# Text-chunking settings.
# Legal documents have very long sentences and dense paragraphs, so we use
# larger chunks than is typical for web content.  The overlap ensures that
# a clause that straddles a chunk boundary is still retrievable from either
# side.
# ---------------------------------------------------------------------------
@dataclass
class ChunkConfig:
    chunk_size: int    = 512           # characters per chunk (not tokens)
    chunk_overlap: int = 64            # characters repeated across boundaries
    separators: list   = field(default_factory=lambda: [
        "\n\n", "\n", ". ", " ", ""   # ordered from most to least preferred
    ])


# ---------------------------------------------------------------------------
# LLM settings.
# The model string must match what the Groq / Hugging-Face / Ollama endpoint
# returns.  Keeping temperature low (0.1) reduces creative hallucination —
# crucial for legal contexts where factual fidelity matters most.
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    provider: str       = os.getenv("LLM_PROVIDER", "ollama")
    model_name: str     = os.getenv("LLM_MODEL", "qwen2.5:7b")  
    api_key: str        = os.getenv("GROQ_API_KEY", "ollama")
    base_url: str       = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    temperature: float  = 0.1
    max_tokens: int     = 1024
    timeout: int        = 180      # ← increase from 60 to 120; local models are slower

# ---------------------------------------------------------------------------
# Retrieval settings.
# top_k — how many chunks are fetched from ChromaDB.
# rerank_top_n — how many of those are passed to the LLM after optional
#   cross-encoder re-ranking.  Fewer = faster inference, more focused context.
# ---------------------------------------------------------------------------
@dataclass
class RetrievalConfig:
    top_k: int          = 10           # initial candidate pool
    rerank_top_n: int   = 4            # after re-ranking, pass this many to LLM
    similarity_threshold: float = 0.3  # discard chunks below this cosine score
    use_reranker: bool  = False        # flip to True when cross-encoder is available


# ---------------------------------------------------------------------------
# Evaluation settings (RAGAS).
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    test_dataset_path: str = str(DATA_DIR / "eval_questions.json")
    results_path: str      = str(DATA_DIR / "eval_results.json")
    metrics: list          = field(default_factory=lambda: [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ])


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
@dataclass
class LogConfig:
    level: str    = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = str(LOG_DIR / "legal_rag.log")
    format: str   = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


# ---------------------------------------------------------------------------
# Top-level config object that groups everything.
# Import pattern: `from config.settings import cfg`
# ---------------------------------------------------------------------------
@dataclass
class AppConfig:
    embedding:    EmbeddingConfig    = field(default_factory=EmbeddingConfig)
    vectorstore:  VectorStoreConfig  = field(default_factory=VectorStoreConfig)
    chunking:     ChunkConfig        = field(default_factory=ChunkConfig)
    llm:          LLMConfig          = field(default_factory=LLMConfig)
    retrieval:    RetrievalConfig    = field(default_factory=RetrievalConfig)
    evaluation:   EvalConfig         = field(default_factory=EvalConfig)
    logging:      LogConfig          = field(default_factory=LogConfig)


cfg = AppConfig()
