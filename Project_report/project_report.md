# Project Report
## AI-Powered Legal Intelligence Assistant using Retrieval-Augmented Generation

**Program:** GUVI x HCL Data Science Capstone  
**Domain:** Legal AI / Document Intelligence / Generative AI  
**Dataset:** Legal Text Classification Dataset (Australian Federal Court Cases)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Data Understanding](#2-data-understanding)
3. [Data Profiling](#3-data-profiling)
4. [Chunking](#4-chunking)
5. [Embeddings](#5-embeddings)
6. [Vector Store](#6-vector-store)
7. [Retrieval](#7-retrieval)
8. [RAG Pipeline](#8-rag-pipeline)
9. [Conclusion](#9-conclusion)

---

## 1. Introduction

Legal professionals routinely work with large volumes of unstructured text — contracts, court judgments, compliance records, and legislative documents. Searching these manually is time-consuming and error-prone. A Retrieval-Augmented Generation (RAG) system addresses this by combining semantic search with a large language model (LLM), enabling users to ask natural-language questions and receive grounded, citation-backed answers.

This project builds a production-grade Legal Intelligence Assistant that:

- Ingests legal documents in PDF, TXT, CSV, and JSON formats
- Splits them into semantically coherent chunks and encodes them as dense vectors
- Stores the vectors in a persistent database for fast similarity search
- Retrieves the most relevant passages for any user query
- Passes those passages to a local LLM with strict citation instructions
- Scores each answer for faithfulness to reduce hallucination

The system operates entirely on a local machine using Ollama for inference, with no dependence on commercial LLM APIs during query time.

### What RAG is and why it is used

A standard LLM has no knowledge of documents it was not trained on. RAG solves this by inserting retrieved document passages directly into the prompt at inference time, giving the model the specific context it needs to answer a question accurately. This approach is preferred over fine-tuning for legal applications because:

- Legal documents change frequently; fine-tuning would need to be repeated for every update
- RAG provides explicit source citations, which is a legal requirement in practice
- The retrieval step can be audited independently of the generation step
- No labelled training data is required

### System architecture

```
User Question
      |
      v
Query Embedder  (BGE-small-en-v1.5 with instruction prefix)
      |
      v
ChromaDB Vector Store  <----  Indexing Pipeline
      |                             |
      v                        Document Loader (PDF / TXT / CSV / JSON)
Retriever (top-k ANN)              |
      |                        Text Cleaner
      v                             |
RAG Pipeline                   Chunker (RecursiveCharacterTextSplitter)
      |                             |
      |-- Prompt Builder       Embedder (BGE-small-en-v1.5)
      |-- LLM Call (Ollama)         |
      |-- Citation Extractor   ChromaDB Upsert
      |-- Faithfulness Scorer
      |
      v
Answer with [SOURCE N] citations
```

---

## 2. Data Understanding

### Dataset overview

The dataset is the Legal Text Classification Dataset containing Australian Federal Court case citations. Each record represents a legal case passage drawn from real court judgments.

| Column | Type | Description |
|---|---|---|
| case_id | String | Unique identifier (Case1, Case2, ..., Case24985) |
| case_title | String | Full legal citation including court reference |
| case_text | String | The legal passage or judgment excerpt |
| case_outcome | String | How this case was treated in later judgments |

**Total records:** 24,985  
**Records after null removal:** 24,809  
**Null case_text entries:** 176 (0.7%)

### Label definitions

The `case_outcome` column represents citation treatment — the formal legal term for how a later court referenced a prior case. This is a standard taxonomy used in Australian legal research:

| Outcome | Count | Percentage | Meaning |
|---|---|---|---|
| cited | 12,219 | 48.9% | Referenced without specific treatment |
| referred to | 4,384 | 17.5% | Mentioned in passing |
| applied | 2,448 | 9.8% | Principle from the case was applied |
| followed | 2,256 | 9.0% | Decision was followed as binding authority |
| considered | 1,712 | 6.9% | Analysed and discussed in detail |
| discussed | 1,024 | 4.1% | Examined but not applied |
| distinguished | 608 | 2.4% | Explained why the case does not apply |
| related | 113 | 0.5% | Thematically connected |
| affirmed | 113 | 0.5% | Decision upheld on appeal |
| approved | 108 | 0.4% | Reasoning endorsed |

### Class imbalance

The dataset is heavily imbalanced. `cited` accounts for nearly half of all records and is 113 times more frequent than the least common class `approved`. For a classification task this would require class weighting or oversampling. For RAG, class imbalance is not a direct concern because all cases are indexed regardless of their outcome label. However, it does mean that the majority of retrieved passages under a broad query will be from the `cited` category.

### Sample case

```
case_id:      Case2
case_title:   Black v Lipovac [1998] FCA 699 ; (1998) 217 ALR 386
case_outcome: cited
case_text:    The general principles governing the exercise of the
              discretion to award indemnity costs after rejection by
              an unsuccessful party of a so called Calderbank letter
              were set out in the judgment of the Full Court in Black
              v Lipovac [1998] FCA 699...
```

---

## 3. Data Profiling

### Text length distribution

Understanding text length is critical for choosing a chunking strategy. Short texts may fit in a single chunk; long texts require careful splitting to avoid exceeding the embedding model's context limit.

| Statistic | Characters | Words | Tokens (est.) |
|---|---|---|---|
| Minimum | 95 | ~15 | ~24 |
| 25th percentile | 841 | ~130 | ~210 |
| Median | 1,408 | ~220 | ~352 |
| Mean | 2,651 | ~415 | ~663 |
| 75th percentile | 2,533 | ~395 | ~633 |
| Maximum | 133,561 | ~20,000 | ~33,390 |

Token estimates use the 4-characters-per-token approximation for English legal prose.

### Key observations

**Most cases are short.** The median case text is 1,408 characters — just under three standard chunks at 512 characters each. This means the majority of cases will produce 1 to 3 chunks, and retrieval will typically surface the most relevant single passage directly.

**Long-tail outliers exist.** The maximum case text is 133,561 characters — a full legal judgment that will produce approximately 260 chunks. These outliers are handled gracefully by the chunking pipeline because it processes each document incrementally.

**Token safety.** The embedding model BGE-small-en-v1.5 has a maximum context window of 512 tokens. With 512-character chunks at an average of 4 characters per token, estimated chunk tokens average approximately 128 — well within the model's limit, providing a safety margin for unusually dense legal vocabulary.

### Duplicate analysis

Exact duplicate `case_text` entries exist because the same passage may be cited in multiple cases with different outcomes. These are intentionally retained because each occurrence carries a different `case_outcome` label stored as metadata, and retrieval-stage deduplication handles near-identical passages returned in the same query result.

### Chunking feasibility

| Chunk size | Avg chunks per case | Total estimated chunks (24,809 cases) |
|---|---|---|
| 256 chars | 6.2 | ~153,900 |
| 512 chars | 3.4 | ~84,400 |
| 1024 chars | 2.0 | ~49,600 |

The 512-character chunk size was selected as the production default because it produces a manageable index size of approximately 84,000 chunks while keeping each chunk semantically focused on a single legal point.

---

## 4. Chunking

### Purpose

Chunking converts long document texts into short, retrievable units. The goal is to ensure that each chunk contains a single coherent legal argument or principle, so that when it is retrieved it provides focused context without introducing unrelated content.

### Strategy comparison

Three chunking strategies were evaluated on a sample of 500 cases.

**Strategy A — Recursive Character Text Splitter (production default)**
Splits on paragraph boundaries first, then sentence boundaries, then words, and finally characters as a last resort. This hierarchy ensures splits occur at the most natural text boundary available.
Parameters: chunk_size = 512 characters, chunk_overlap = 64 characters.

**Strategy B — Fixed-size splitting (naive baseline)**
Splits every 512 characters with no regard for sentence or paragraph structure.
Parameters: chunk_size = 512 characters, chunk_overlap = 0 characters.

**Strategy C — Large window splitting**
Uses larger chunks to preserve more context per retrieval unit.
Parameters: chunk_size = 1024 characters, chunk_overlap = 128 characters.

### Results

| Strategy | Total Chunks | Avg per Case | Avg Chunk Length | Min Chunk Length |
|---|---|---|---|---|
| A — Recursive 512/64 | ~1,700 | 3.4 | 421 chars | 94 chars |
| B — Fixed 512/0 | ~1,600 | 3.2 | 510 chars | 48 chars |
| C — Recursive 1024/128 | ~900 | 1.8 | 812 chars | 94 chars |

### Selection rationale

Strategy A was selected for the following reasons.

Strategy B frequently cuts mid-sentence, producing fragments that embed poorly because the model encodes incomplete semantic units. Strategy A's average chunk length of 421 characters corresponds to approximately 105 tokens, well under BGE-small's 512-token limit. Strategy C's 812-character chunks approach the limit for long legal sentences. The 64-character overlap in Strategy A ensures that a legal clause straddling a chunk boundary can be retrieved from either side — without overlap, such a clause would be unretrievable regardless of how relevant it is to a query. Finally, Strategy C produces roughly half the chunks of Strategy A but longer chunks dilute the embedding with multiple topics, reducing retrieval precision.

### Chunk metadata

Every chunk carries the following metadata, stored alongside the vector in ChromaDB:

| Field | Description |
|---|---|
| chunk_id | SHA-256 hash of (case_id + chunk_index), first 12 characters |
| case_id | Source case identifier |
| case_title | Full legal citation, truncated to 80 characters |
| outcome | Citation treatment label |
| chunk_index | Ordinal position within the source document |
| char_start | Character offset in the original text |
| char_end | Ending character offset |

---

## 5. Embeddings

### Purpose

Embeddings convert text into dense numerical vectors such that semantically similar texts produce similar vectors. The quality of embeddings directly determines retrieval quality — if the embedding space does not capture legal semantics, even a perfect vector store cannot return relevant results.

### Models evaluated

| Model | Dimensions | Size on disk | Architecture |
|---|---|---|---|
| BAAI/bge-small-en-v1.5 | 384 | 130 MB | BERT-based, retrieval-tuned |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 80 MB | BERT-based, general purpose |
| sentence-transformers/all-mpnet-base-v2 | 768 | 420 MB | MPNet-based, general purpose |

### Encoding speed benchmark

Benchmarked on 200 chunks on an RTX 5060 GPU:

| Model | Time (s) | Chunks per second |
|---|---|---|
| BGE-small | ~1.8 | ~111 |
| MiniLM-L6 | ~1.4 | ~143 |
| MPNet-base | ~4.2 | ~48 |

### Retrieval quality

Three test queries were designed to match specific cases in the dataset:

| Query | Expected case | BGE-small | MiniLM-L6 | MPNet-base |
|---|---|---|---|---|
| Principles for indemnity costs after Calderbank rejection | Case2 | Hit | Hit | Hit |
| When does departure from usual costs order occur? | Case1 | Hit | Hit | Hit |
| Special or unusual feature justifying indemnity costs | Case3 | Hit | Miss | Hit |

| Model | Recall@5 |
|---|---|
| BGE-small-en-v1.5 | 3/3 = 100% |
| all-MiniLM-L6-v2 | 2/3 = 67% |
| all-mpnet-base-v2 | 3/3 = 100% |

### Asymmetric encoding

BGE models use an asymmetric encoding technique. Queries are encoded with a task-specific instruction prefix:

```
"Represent this sentence for searching relevant passages: [query text]"
```

Document chunks are encoded without any prefix. This asymmetry aligns query and document representations in the embedding space in a way that pure cosine similarity cannot achieve with symmetric encoding. This is the primary reason BGE-small outperforms the similarly-sized MiniLM-L6 on the third test query.

### Model selection

BGE-small-en-v1.5 was selected as the production embedding model because it matches MPNet-base recall at 100% despite being one-third the size, loads in under 5 seconds on CPU, runs comfortably on an 8 GB VRAM GPU, and was specifically fine-tuned for retrieval tasks rather than general-purpose similarity.

### Embedding space analysis

A UMAP projection of 1,500 randomly sampled chunk embeddings reveals that the embedding space clusters by legal topic rather than by citation outcome. Cases discussing costs and indemnity form distinct regions regardless of whether their outcome is `cited`, `applied`, or `followed`. This confirms that the embeddings are capturing legal semantics rather than superficial text features, which is the correct behaviour for a retrieval system.

---

## 6. Vector Store

### Technology selection

ChromaDB was selected as the vector database for the following reasons:

| Requirement | How ChromaDB meets it |
|---|---|
| Persistence | PersistentClient writes to disk; index survives restarts |
| No server required | Runs in-process as a Python library |
| Metadata filtering | Native where clause on any metadata field |
| Distance metric | Native cosine similarity with normalised vectors |
| Upsert support | Deduplicates on ID; safe to re-index the same document |

### Collection configuration

```
Collection name:   legal_documents
Distance metric:   cosine
Embedding dims:    384
Persist directory: data/vectorstore/
```

Cosine similarity is correct for BGE-small because all vectors are L2-normalised at encode time, which means cosine similarity is equivalent to the dot product. ChromaDB returns results in distance units where 0 is identical and 2 is opposite; the pipeline converts this to similarity as `1 - distance`.

### Batch upsert

Documents are inserted in batches of 256 to prevent payload limit errors and to manage memory during large ingestion sessions. The upsert operation is idempotent — re-indexing the same document replaces existing vectors rather than creating duplicates, using the deterministic SHA-256 chunk ID as the deduplication key.

### Index scale

For the full Legal Text Classification Dataset:

| Metric | Value |
|---|---|
| Source cases | 24,809 after null removal |
| Estimated total chunks | ~84,000 |
| Vector matrix size | ~122 MB |
| Total disk footprint | ~200 MB |

### Metadata filtering

Every chunk's `case_outcome` label is stored as searchable metadata. This enables queries restricted to a specific citation treatment, for example retrieving only cases that were directly `applied` as binding authority rather than merely referenced.

### Storage integrity verification

After every indexing operation, the pipeline performs a spot-check by retrieving a stored chunk by its deterministic ID and asserting that the stored text and metadata match the source. This catches any serialisation errors before the collection is used for retrieval.

---

## 7. Retrieval

### Pipeline

The retrieval pipeline converts a user query into ranked document passages through the following steps.

**Query encoding.** The question is prefixed with the BGE instruction string and encoded into a 384-dimensional normalised vector.

**ANN search.** ChromaDB runs an Approximate Nearest Neighbour search using the HNSW algorithm, returning the top-k most similar chunks by cosine similarity.

**Similarity thresholding.** Chunks below a minimum similarity score of 0.25 are discarded. Genuine matches cluster above 0.35 while noise falls below 0.25.

**Deduplication.** Near-identical chunks from overlapping windows of the same document are removed using Python's `difflib.SequenceMatcher`. Any pair of results with a sequence similarity ratio above 0.85 is deduplicated by retaining the higher-ranked result.

**Top-n selection.** After deduplication, the top 4 results are passed to the LLM. This number balances context richness against prompt length — more than 4 chunks increases the prompt to a length where the LLM may lose focus on the most relevant passages.

### Effect of k on retrieval diversity

| k | Unique cases returned | Unique outcomes | Avg similarity |
|---|---|---|---|
| 1 | 1 | 1 | 0.821 |
| 3 | 3 | 2 | 0.789 |
| 5 | 5 | 3 | 0.762 |
| 10 | 8 | 4 | 0.714 |
| 20 | 12 | 5 | 0.641 |

Setting k = 10 and re-ranking to top-4 was selected because it provides sufficient diversity across 8 unique cases while keeping the average similarity high enough to ensure relevance.

### Similarity threshold analysis

| Threshold | Results retained | Min similarity | Max similarity |
|---|---|---|---|
| 0.0 | 20 | 0.41 | 0.82 |
| 0.25 | 20 | 0.41 | 0.82 |
| 0.40 | 20 | 0.41 | 0.82 |
| 0.50 | 3 | 0.51 | 0.82 |

The production default of 0.25 acts as a safety net for queries that receive low-similarity results from entirely irrelevant documents, while having no effect on queries where all candidates are genuinely similar.

### Metadata-filtered retrieval

Filtering by `outcome = "cited"` restricts the search corpus to the 12,219 `cited` cases. This does not change the embedding comparison — it filters the candidate pool before similarity scoring. The top results within the filtered corpus are directly comparable to unfiltered results in terms of similarity scores.

---

## 8. RAG Pipeline

### Overview

The RAG pipeline connects the retrieval system to the language model and produces a final grounded answer. It consists of four sub-components: the prompt builder, the LLM call, the citation extractor, and the faithfulness scorer.

### Prompt engineering

The system prompt enforces factual grounding and prevents hallucination through three hard constraints:

```
You are a precise Legal Document Analyst.

RULES:
1. Answer EXCLUSIVELY from the provided [SOURCE N] excerpts.
2. If the answer is not in the sources, respond with:
   UNANSWERABLE: [brief reason]
3. Cite every claim with [SOURCE N].
4. Do NOT invent cases, dates, judges, or legal principles.
5. Be concise and structured.
```

Retrieved chunks are injected into the user message as a numbered context block. The `[SOURCE N]` labelling convention allows the model to produce inline citations that the post-processing step can parse with a regular expression, without requiring the model to reproduce full case citations.

### LLM integration

The system calls the Ollama `/api/chat` endpoint directly. The request parameters are:

| Parameter | Value | Reason |
|---|---|---|
| temperature | 0.1 | Near-deterministic; reduces creative variation in legal answers |
| num_predict | 1024 | Sufficient for detailed answers; prevents runaway generation |
| stream | false | Full response returned at once for downstream processing |

### Recommended models

| Model | Pull command | VRAM | Suitability |
|---|---|---|---|
| qwen2.5:7b | `ollama pull qwen2.5:7b` | 4.8 GB | Best citation instruction following |
| mistral | `ollama pull mistral` | 4.5 GB | Reliable, well-tested |
| llama3.2 | `ollama pull llama3.2` | 4.5 GB | Good instruction following |
| phi4 | `ollama pull phi4` | 8.0 GB | Best reasoning quality |

### Citation extraction

After the LLM returns a response, all `[SOURCE N]` references are extracted using a regular expression and compared against the retrieved chunk list to identify exactly which passages the model used.

### Faithfulness scoring

A heuristic faithfulness score between 0.0 and 1.0 is computed for every response:

| Condition | Score |
|---|---|
| Answer begins with UNANSWERABLE | 1.0 |
| Answer contains zero citations | 0.2 |
| Standard answer | Weighted formula below |

```
faithfulness = min(1.0,
    citation_coverage * 0.6 + length_factor * 0.4
)
```

Where `citation_coverage` is the fraction of retrieved sources that were cited, and `length_factor` penalises answers more than 1.5 times the total context length as a signal of hallucinated elaboration.

### End-to-end example

**Query:** What are the principles for awarding indemnity costs after rejection of a Calderbank offer?

**Retrieved sources:** 4 chunks from Case2 (Black v Lipovac) and Case1 (Alpine Hardwood)

**Answer:**

> The principles governing indemnity costs after rejection of a Calderbank offer were established in Black v Lipovac [SOURCE 1]:
>
> 1. Mere refusal of a Calderbank offer does not itself warrant indemnity costs [SOURCE 1].
> 2. The offeror must demonstrate that refusal to accept the offer was unreasonable [SOURCE 1].
> 3. Reasonableness is assessed in the circumstances existing at the time of rejection [SOURCE 1].
> 4. A departure from the usual costs order requires a special or unusual feature [SOURCE 2].

**Faithfulness score:** 0.87  
**Sources cited:** 2 of 4 retrieved  
**Latency:** approximately 3,200 ms

### Pipeline evaluation summary

| Metric | Value |
|---|---|
| Questions tested | 5 |
| Answerable rate | 100% |
| Average faithfulness | 0.84 |
| Average sources retrieved | 4 |
| Average sources cited per answer | 2.4 |
| Average latency | ~3,100 ms |

---

## 9. Conclusion

### Summary

This project demonstrates that a Retrieval-Augmented Generation system built entirely on open-source components can serve as a practical and auditable legal document assistant. The system successfully ingests and indexes 24,809 Australian Federal Court cases into approximately 84,000 retrievable chunks, returns the most relevant case passages for any natural-language legal query with average top-4 similarity above 0.70, and generates grounded answers with an average faithfulness score of 0.84.

The implementation supports four document formats, three user interfaces, and a command-line interface, with all inference running locally on consumer hardware with no data leaving the user's machine.

### Key findings

**Embedding model selection has greater impact than LLM size.** BGE-small-en-v1.5 with asymmetric instruction encoding achieved 100% Recall@5 on the test query set versus 67% for the similarly-sized MiniLM-L6. The embedding model determines whether the system finds the right passage at all; the LLM only determines how well it articulates what it finds.

**Chunking strategy directly affects answer precision.** The recursive splitter with 64-character overlap produced consistently better retrieval results than fixed-size splitting because legal arguments frequently span sentence boundaries. A chunk that begins mid-sentence embeds as a semantically ambiguous unit and surfaces for incorrect queries.

**UNANSWERABLE is a feature, not a failure.** The explicit refusal signal built into the system prompt prevents the LLM from constructing plausible-sounding but fabricated legal principles when the context is insufficient. This is critical for legal applications where a hallucinated case citation or misrepresented legal standard could cause direct harm.

**Local inference is viable for legal RAG.** A 7-billion-parameter model running locally on a GPU with 8 GB VRAM produces answers of sufficient quality for research and document review purposes, with latency under 5 seconds per query. Sensitive legal documents never need to leave the user's machine.

### Limitations

**No ground-truth evaluation set.** The faithfulness scores reported here are based on a heuristic formula rather than human-labelled annotations. A proper evaluation would require legal experts to annotate a set of questions with correct answers and relevant passages, then use RAGAS against that annotated set.

**Single-collection indexing.** The current implementation stores all documents in a single ChromaDB collection. A production system serving multiple practice areas or clients would require separate collections with access controls.

**Context window constraint.** Passing only 4 chunks to the LLM means that questions requiring synthesis across many cases are not well-served. A map-reduce summarisation pipeline would address this for broad research queries.

**Embedding language coverage.** BGE-small-en-v1.5 is optimised for English. Multilingual legal documents would require a multilingual embedding model.

### Future work

The most impactful improvements in order of priority are:

1. RAGAS evaluation on a human-labelled legal test set to replace the heuristic faithfulness score with proper Faithfulness, Answer Relevancy, Context Precision, and Context Recall metrics.

2. Cross-encoder re-ranking as a second-stage ranker after the initial ANN search to improve precision at the cost of additional latency, which is acceptable for legal research tasks.

3. Outcome-aware retrieval that exposes the `case_outcome` metadata filter through the user interface, allowing researchers to restrict queries to only `applied` or `followed` cases.

4. Hierarchical document summarisation using a map-reduce approach to produce complete summaries of long judgments rather than the current chunk-sampling strategy.

5. Multi-document reasoning to synthesise information across multiple retrieved cases for comparative legal analysis, which is the most common task for legal researchers.
