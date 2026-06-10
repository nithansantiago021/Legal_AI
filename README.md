# Legal Intelligence Assistant

A production-grade Retrieval-Augmented Generation (RAG) system for legal document question-answering, clause retrieval, document summarisation, and hallucination-aware response evaluation.

---

## Overview

This system allows legal professionals and researchers to upload legal documents — contracts, court judgments, compliance records, NDAs — and query them using natural language. All answers are grounded exclusively in the uploaded document text, with inline source citations and a faithfulness score on every response.

The system supports three frontend interfaces: a Streamlit web application, a Node.js/Express UI, and a Chainlit chat interface. A FastAPI backend and a command-line interface are also included.

---

## Architecture

```
User Query
    |
    v
Query Embedder (BGE-small-en-v1.5)
    |
    v
ChromaDB Vector Store  <----  Document Indexing Pipeline
    |                              |
    v                         Preprocessor (PDF / TXT / JSON)
Retriever (top-k ANN + dedup)     |
    |                         Embedder (BGE-small-en-v1.5)
    v
RAG Pipeline
    |-- Prompt Builder
    |-- LLM Call (Ollama local / Groq API)
    |-- Citation Extractor
    |-- Faithfulness Scorer
    |
    v
Response with [SOURCE N] citations
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Embedding model | BAAI/bge-small-en-v1.5 (384-dim, retrieval-tuned) |
| Vector database | ChromaDB (persistent, disk-backed) |
| LLM backend | Ollama (local) or Groq API (cloud, free tier) |
| RAG framework | LangChain (text splitting only) |
| Backend API | FastAPI |
| Streamlit UI | Streamlit |
| Chat UI | Chainlit |
| Premium UI | Node.js + Express |
| Evaluation | RAGAS |
| Language | Python 3.10 |

---

## Project Structure

```
legal_rag/
|
|-- app.py                        Streamlit UI (main entry point)
|-- chainlit_app.py               Chainlit chat UI
|-- cli.py                        Command-line interface
|-- requirements.txt
|-- .env.example                  Environment variable template
|
|-- config/
|   |-- settings.py               Centralised configuration (all env vars)
|
|-- src/
|   |-- preprocessing/
|   |   |-- preprocessor.py       Document loading, cleaning, chunking
|   |-- embedding/
|   |   |-- embedder.py           BGE model wrapper + ChromaDB client
|   |   |-- indexer.py            Indexing orchestration
|   |-- retrieval/
|   |   |-- retriever.py          Semantic search + deduplication
|   |-- rag/
|   |   |-- rag_pipeline.py       Prompt building, LLM call, citation extraction
|   |-- evaluation/
|   |   |-- evaluator.py          RAGAS metrics + faithfulness scoring
|   |-- api.py                    FastAPI REST backend
|   |-- utils/
|       |-- logger.py             Rotating file + console logger
|
|-- data/
|   |-- raw/                      Place source documents here
|   |-- vectorstore/              ChromaDB index (auto-generated)
|
|-- tests/
|   |-- test_pipeline.py          pytest unit tests
|
|-- legal_rag_ui/                 Node.js UI (separate folder)
|   |-- server.js                 Express proxy server
|   |-- public/index.html         Full browser UI
|   |-- package.json
```

---

## Supported File Formats

- PDF (`.pdf`) — via PyMuPDF
- Plain text (`.txt`)
- JSON (`.json`) — Legal Text Classification Dataset format

---

## Setup

### Prerequisites

- Python 3.10
- Conda or virtualenv
- Ollama (for local LLM) or a Groq API key (for cloud LLM)
- Node.js 18+ (only if using the Node.js UI)

### 1. Clone the repository

```bash
git clone https://github.com/nithansantiago021/Legal_AI.git
cd legal_rag
```

### 2. Create the Python environment

```bash
conda create -n rag python=3.10 -y
conda activate rag
```

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Open `.env` and set your LLM backend:

**Ollama (local — recommended):**
```
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
LLM_BASE_URL=http://localhost:11434
GROQ_API_KEY=ollama
LOG_LEVEL=INFO
```

**Groq API (cloud):**
```
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
LLM_BASE_URL=https://api.groq.com/openai/v1
GROQ_API_KEY=gsk_your_key_here
LOG_LEVEL=INFO
```

### 5. Install Ollama (if using local LLM)

Download from https://ollama.com/download and run the installer. Then pull the required model:

```bash
ollama pull qwen2.5:7b
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

---

## Running the Application

### Streamlit UI

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Chainlit UI

```bash
pip install chainlit
chainlit run chainlit_app.py --watch
# Opens at http://localhost:8000
```

### FastAPI + Node.js UI

Open two terminals:

```bash
# Terminal 1 — FastAPI backend
uvicorn src.api:app --port 8000 --reload

# Terminal 2 — Node.js UI
cd legal_rag_ui
npm install
node server.js
# Opens at http://localhost:3000
```

### Command-line interface

```bash
# Index a single file
python cli.py index --file data/raw/contract.pdf

# Index all files in a directory
python cli.py index --dir data/raw/

# Ask a question
python cli.py query "What is the termination clause?"

# List all indexed documents
python cli.py list-docs

# Run evaluation benchmark
python cli.py evaluate
```

---

## Evaluation

Run the built-in benchmark:

```bash
python cli.py evaluate
```

Results are saved to `data/eval_results.json`.

| Metric | Description | Target |
|---|---|---|
| Faithfulness | Answer contains only claims from context | >= 0.8 |
| Answer Relevancy | Answer addresses the question asked | >= 0.7 |
| Context Precision | Retrieved chunks are relevant | >= 0.7 |
| Context Recall | Context covers what is needed | >= 0.6 |
| Answerable Rate | Percentage of questions the system attempts | >= 80% |

Full RAGAS evaluation requires a Groq or OpenAI API key. The built-in benchmark runs without any external API.

---

## Recommended Models (Ollama)

| Use Case | Model | VRAM | Notes |
|---|---|---|---|
| RAG / Legal Q&A | qwen2.5:7b | ~4.8 GB | Best citation following |
| RAG / Legal Q&A | mistral | ~4.5 GB | Reliable, fast |
| Coding | qwen2.5-coder:7b | ~4.8 GB | Best at 8 GB VRAM |
| Agents / Tool use | qwen3:8b | ~5 GB | Native tool calling |
| Heavy reasoning | phi4 | ~8 GB | 14B quality at Q4 |

---

## Key Design Decisions

**BGE-small over all-MiniLM:** BGE models are fine-tuned specifically for retrieval tasks, not general similarity. They use an asymmetric query instruction technique that significantly improves recall. The instruction prefix is applied to queries only, not to document chunks — this asymmetry is intentional and matches the BGE training procedure.

**Character-based chunking:** Legal sentences are long and dense. Character-based chunks at 512 characters are safely within BGE-small's 512-token context limit without requiring a tokenizer in the preprocessing step. English legal prose averages approximately 4 characters per token.

**Temperature 0.1:** Legal answers must be reproducible and factually grounded. Higher temperature introduces variation that can drift from the source text.

**UNANSWERABLE as a first-class signal:** The system prompt explicitly instructs the model to return `UNANSWERABLE: [reason]` rather than speculate when the context is insufficient. A hallucinated legal clause is more harmful than a declared inability to answer.

**Native Ollama API over OpenAI wrapper:** The system calls Ollama's `/api/chat` endpoint directly rather than the OpenAI-compatibility layer at `/v1/chat/completions`. This avoids version-dependent compatibility issues with local Ollama installations.

---
