"""
settings.py — Configuration for Hugging Face Spaces deployment.

Key differences from local config:
- CPU-only (no CUDA)
- all-mpnet-base-v2 embedder + ms-marco-MiniLM-L-6-v2 cross-encoder re-ranker
- ChromaDB EphemeralClient (pure in-memory, zero disk writes)
- Groq-only LLM backend
- RAGAS-style post-generation quality gates
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True


@dataclass
class VectorStoreConfig:
    persist_directory: str = ""        # empty = EphemeralClient (no disk)
    collection_name: str = "legal_docs"
    distance_metric: str = "cosine"


@dataclass
class ChunkingConfig:
    chunk_size: int = 1024
    chunk_overlap: int = 128
    separators: list = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])


@dataclass
class RetrievalConfig:
    # Stage 1 — bi-encoder ANN (fetch more than needed; re-ranker prunes)
    top_k: int = 12
    similarity_threshold: float = 0.25     # permissive; cross-encoder tightens

    # Stage 2 — cross-encoder re-ranking
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 5
    # ms-marco logit scale: >0 = strong match, -2..0 = moderate, <-3 = weak
    rerank_score_threshold: float = -3.0   # drop chunks below this after re-ranking
    min_top_rerank_score: float = -2.5     # if best chunk is below this, go UNANSWERABLE


@dataclass
class LLMConfig:
    provider: str = "groq"
    model_name: str = "llama-3.3-70b-versatile"
    judge_model: str = "llama-3.1-8b-instant"   # fast model used as RAGAS NLI judge
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: int = 60
    base_url: str = ""


@dataclass
class RAGASConfig:
    enabled: bool = True
    min_faithfulness: float = 0.5          # fraction of statements grounded in context
    min_answer_relevancy: float = 0.40     # cosine sim of answer embedding vs question
    min_context_relevancy: float = 0.30    # avg normalised re-rank score of retrieved chunks
    max_statements_to_check: int = 8       # cap NLI decomposition for latency


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    log_file: str = "/tmp/legal_rag.log"


@dataclass
class EvaluationConfig:
    test_dataset_path: str = ""
    results_path: str = "/tmp/eval_results.json"


@dataclass
class Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    ragas: RAGASConfig = field(default_factory=RAGASConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


cfg = Config()
