# ⚖️ Legal Intelligence Assistant — RAG Pipeline

> An end-to-end **Retrieval-Augmented Generation** system for legal document Q&A, clause retrieval, summarisation, and hallucination-aware evaluation.

---

## Architecture Overview

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                                │
│                                                                     │
│  ┌──────────┐   embed    ┌────────────┐   top-k   ┌────────────┐  │
│  │  Query   │──────────▶│  ChromaDB  │──────────▶│  Retriever │  │
│  │  (BGE)   │           │ VectorStore│           │ (+ rerank) │  │
│  └──────────┘           └────────────┘           └─────┬──────┘  │
│                                                         │          │
│                                                   top-n chunks     │
│                                                         │          │
│  ┌───────────┐   prompt  ┌────────────┐   answer  ┌────▼───────┐ │
│  │ Response  │◀─────────│    LLM     │◀──────────│  Prompt    │ │
│  │ + Citations│          │(Groq/Ollama)│           │  Builder   │ │
│  └───────────┘           └────────────┘           └────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
legal_rag/
├── app.py                        # Streamlit UI (main entry point)
├── cli.py                        # Command-line interface
├── requirements.txt
├── .env.example                  # Copy to .env and fill in API keys
│
├── config/
│   └── settings.py               # Centralised configuration (all env vars)
│
├── src/
│   ├── preprocessing/
│   │   └── preprocessor.py       # PDF/TXT loading, cleaning, chunking
│   ├── embedding/
│   │   ├── embedder.py           # BGE-small embedding model + ChromaDB wrapper
│   │   └── indexer.py            # Orchestrates load → chunk → embed → store
│   ├── retrieval/
│   │   └── retriever.py          # Semantic search + optional cross-encoder rerank
│   ├── rag/
│   │   └── rag_pipeline.py       # Prompt building, LLM call, citation extraction
│   ├── evaluation/
│   │   └── evaluator.py          # RAGAS + custom faithfulness metrics
│   └── utils/
│       └── logger.py             # Rotating file + console logger
│
├── data/
│   ├── raw/                      # Drop source documents here
│   ├── processed/                # Intermediate artefacts
│   ├── vectorstore/              # ChromaDB persistent index
│   └── eval_questions.json       # Benchmark questions (optional)
│
└── tests/
    └── test_pipeline.py          # pytest unit tests
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Get a free Groq API key at https://console.groq.com (no credit card required).

**Or use Ollama (local, fully offline):**
```bash
# Install Ollama: https://ollama.ai
ollama pull mistral:7b-instruct
# Then edit .env:
# LLM_PROVIDER=ollama
# LLM_MODEL=mistral:7b-instruct
# LLM_BASE_URL=http://localhost:11434/v1
```

### 3. Launch the Streamlit UI

```bash
streamlit run app.py
```

### 4. Or use the CLI

```bash
# Index a directory of documents
python cli.py index --dir data/raw/

# Index a single file
python cli.py index --file my_contract.pdf

# Query
python cli.py query "What is the termination clause?"

# List indexed documents
python cli.py list-docs

# Run evaluation
python cli.py evaluate
```

### 5. Or start the REST API

```bash
uvicorn src.api:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

---

## Datasets

### Primary: Legal Text Classification Dataset
- Source: https://huggingface.co/datasets/legal_text_classification
- Format: JSON with `text` and `label` fields
- Usage: Place downloaded JSON files in `data/raw/`

### Optional: US Court Cases
- Source: https://huggingface.co/datasets/pile-of-law/pile-of-law
- Any subset works (court_listener, eur_parl, etc.)

---

## Tech Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Embeddings | BGE-small-en-v1.5 | Best retrieval accuracy per parameter for legal English |
| Vector DB | ChromaDB | Local-first, no server required, excellent Python API |
| LLM | Groq / Ollama | Groq: fast inference; Ollama: fully local for confidential documents |
| Framework | LangChain | Mature text splitting; avoids reinventing chunking logic |
| API | FastAPI | High-performance async, auto-generated OpenAPI docs |
| UI | Streamlit | Fast iteration, suited to data science prototypes |
| Evaluation | RAGAS | Framework-native RAG metrics; no ground truth required |

---

## Evaluation Metrics

| Metric | Goal | Description |
|--------|------|-------------|
| Faithfulness | ≥ 0.8 | Answer contains only claims from context |
| Answer Relevancy | ≥ 0.7 | Answer addresses the actual question |
| Context Precision | ≥ 0.7 | Retrieved chunks are relevant |
| Context Recall | ≥ 0.6 | Context covers what's needed |
| Answerable Rate | ≥ 80% | % of questions the system attempts |

---

## Key Design Decisions

**Why BGE-small over all-MiniLM-L6-v2?**
BGE models are fine-tuned specifically for retrieval (not just similarity), and use an asymmetric query instruction technique that significantly boosts recall. all-MiniLM is a good general-purpose model but BGE outperforms it on retrieval benchmarks.

**Why character-based chunking instead of token-based?**
Legal prose is dense and contains very long sentences. Character-based chunks at ~512 chars are safely within BGE-small's 512-token context limit (English legal text averages ~4 chars/token), and don't require a tokenizer in the preprocessing step.

**Why set temperature to 0.1 for the LLM?**
Legal answers must be reproducible and factually grounded. Higher temperature introduces creative variation that can drift from the source text. 0.1 keeps outputs near-deterministic while allowing natural phrasing variation.

**Why UNANSWERABLE as a first-class response?**
Silently hallucinating an answer is far more dangerous in a legal context than admitting ignorance. The system is explicitly prompted to declare UNANSWERABLE when the context is insufficient.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Deliverables Checklist

- [x] Source code (modular, documented)
- [x] Streamlit application (`app.py`)
- [x] FastAPI backend (`src/api.py`)
- [x] Retrieval evaluation (`src/evaluation/evaluator.py`)
- [x] CLI (`cli.py`)
- [x] Unit tests (`tests/`)
- [ ] Demo video *(record with OBS or Loom after running locally)*
- [ ] Architecture diagram *(export from draw.io or Lucidchart)*
- [ ] Evaluation report *(generated automatically at `data/eval_results.json`)*
