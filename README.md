---
title: Legal Intelligence Assistant
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
short_description: RAG assistant for legal
---

# Legal Intelligence Assistant

A **Retrieval-Augmented Generation (RAG)** system for legal document analysis. Upload contracts, agreements, or any legal PDF and ask questions — all grounded in cited source passages.

## Privacy Guarantees

| What | Guarantee |
|------|-----------|
| Uploaded documents | Deleted from disk immediately after indexing |
| Embeddings / vector store | In-memory only (ChromaDB EphemeralClient) — gone on session end |
| Chat history | Browser session only — never persisted |
| Groq API key | Session RAM only — never written to disk, env vars, or logs |
| Cross-user isolation | Each session has its own independent vector store |

## Quick Start

1. **Get a free Groq API key** at [console.groq.com](https://console.groq.com)
2. Enter the key in the sidebar (stored in session memory only)
3. Upload one or more legal documents (PDF, TXT, CSV, JSON)
4. Ask questions in the chat — answers cite specific source passages

## Architecture

```
User Question
     │
     ▼
 [Embedder]  all-mpnet-base-v2
     │
     ▼
 [ChromaDB]  EphemeralClient — pure in-memory, per-session
     │  top-k cosine similarity search
     ▼
 [Retriever]  deduplication + threshold filtering
     │
     ▼
 [RAGPipeline]  prompt construction + citation template
     │
     ▼
 [Groq LLM]  openai/gpt-oss-120b (user's own key)
     │
     ▼
 Cited Answer  [SOURCE 1], [SOURCE 2], …
```

## Technical Stack

- **Embeddings**: `sentence-transformers/all-mpnet-base-v2` — runs on cpu
- **Vector DB**: ChromaDB `EphemeralClient` — zero disk writes
- **LLM**: Groq `openai/gpt-oss-120b` — fast inference, free tier available
- **Chunking**: LangChain `RecursiveCharacterTextSplitter` (800 chars, 100 overlap)
- **Re-ranking**: Disabled on free tier (too slow on CPU); enable in `config/settings.py`
- **UI**: Streamlit with custom dark theme

## Clear Session Data

Click **"🗑️ Clear Session Data"** in the sidebar at any time to instantly wipe:
- All uploaded document embeddings
- Chat history
- API key from session memory
- Temporary files

## Local Development

```bash
git clone <your-repo>
cd legal-rag-assistant
pip install -r requirements.txt
streamlit run app.py
```

Set your Groq key in the sidebar — no `.env` file needed.

## Project Structure

```
├── app.py                          # Streamlit UI (entry point)
├── requirements.txt
├── packages.txt                    # System deps for HF Spaces
├── config/
│   └── settings.py                 # Centralised config (CPU, ephemeral)
└── src/
    ├── preprocessing/
    │   └── preprocessor.py         # PDF/TXT/CSV/JSON extraction + chunking
    ├── embedding/
    │   ├── embedder.py             # BGE model + ephemeral ChromaDB wrapper
    │   └── indexer.py              # index_file() / index_directory()
    ├── retrieval/
    │   └── retriever.py            # Semantic search + deduplication
    ├── rag/
    │   └── rag_pipeline.py         # Prompt builder + Groq caller
    └── utils/
        └── logger.py               # Rotating file logger → /tmp/
```

## Limitations (Free Tier)

- No GPU — embedding is CPU-only (~1-3s per document page)
- Groq free tier has rate limits — for heavy use, consider a paid plan
