"""
rag_pipeline.py — Retrieval-Augmented Generation pipeline.

This module glues retrieval to generation.  Given a user query it:

  1. Retrieves the most relevant chunks (see retriever.py)
  2. Builds a prompt that instructs the LLM to answer ONLY from the context
  3. Calls the LLM
  4. Extracts citations (which chunks the answer references)
  5. Scores the answer for faithfulness (basic hallucination detection)

Prompt engineering notes
------------------------
* We use a SYSTEM prompt to set the LLM's "role" as a careful legal
  analyst, emphasising that it must not invent information.
* The context is formatted with clear [SOURCE N] labels so the model can
  cite specific passages and so the post-processing step can identify which
  chunks were referenced.
* Temperature is set very low (0.1) to reduce creative variation.
* We add an explicit "UNANSWERABLE" signal so the model can decline
  gracefully when the context doesn't contain an answer — this is critical
  for hallucination reduction.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

# from openai import OpenAI   # Groq uses the OpenAI-compatible API
import requests as _requests

from config.settings import cfg
from src.retrieval.retriever import Retriever, RetrievedChunk
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------

# DELETE the old function entirely and replace with this

def _call_ollama(prompt_messages: list[dict]) -> str:
    """
    Call Ollama's native /api/chat endpoint directly.
    This is more reliable than the OpenAI-compatibility wrapper
    for local Ollama installations.
    
    Args:
        prompt_messages: List of {"role": ..., "content": ...} dicts
    
    Returns:
        The model's response text.
    """
    payload = {
        "model":    cfg.llm.model_name,       # e.g. "mistral"
        "messages": prompt_messages,
        "stream":   False,                     # wait for full response, not streaming
        "options": {
            "temperature": cfg.llm.temperature,
            "num_predict": cfg.llm.max_tokens,
        }
    }
    response = _requests.post(
        f"{cfg.llm.base_url}/api/chat",        # http://localhost:11434/api/chat
        json=payload,
        timeout=cfg.llm.timeout,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise and careful Legal Document Analyst.

Your ONLY job is to answer the user's question using the provided legal document excerpts.

RULES you MUST follow:
1. Base your answer EXCLUSIVELY on the provided [SOURCE N] excerpts.
2. If the answer cannot be found in the sources, respond with exactly:
   UNANSWERABLE: [brief reason why the context is insufficient]
3. Cite your sources inline using [SOURCE N] notation after each claim.
4. Do NOT invent legal terms, dates, parties, or obligations.
5. Keep your answer concise and structured. Use bullet points for lists.
6. If sources conflict, note the conflict explicitly rather than choosing one.
"""

CONTEXT_TEMPLATE = "[SOURCE {n}] (From: {filename} | Section: {section})\n{text}"

USER_PROMPT_TEMPLATE = """LEGAL DOCUMENT EXCERPTS:
{context}

---
USER QUESTION: {question}

Provide a structured answer with inline citations [SOURCE N].
"""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    """The complete output of one RAG pipeline invocation."""
    question: str
    answer: str
    sources: list[RetrievedChunk]
    cited_source_indices: list[int]   # 1-based indices of sources the answer cites
    is_answerable: bool               # False if the model returned UNANSWERABLE
    faithfulness_score: float         # 0–1 heuristic (see _score_faithfulness)
    latency_ms: float
    prompt_tokens_estimate: int
    model_used: str


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Orchestrates retrieve → prompt → generate → validate.

    Thread-safety note: The LLM client is stateless (each call is
    independent HTTP), and the Retriever is read-only, so a single
    RAGPipeline instance can safely serve concurrent Streamlit users.
    """

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    def _build_context_block(self, chunks: list[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into a numbered context block for the prompt.

        Each chunk is labelled [SOURCE N] so the model can reference it.
        We include filename and section so the model can produce accurate
        citations like "per Clause 7 of contract_v3.pdf [SOURCE 2]".
        """
        parts = [
            CONTEXT_TEMPLATE.format(
                n=i + 1,
                filename=chunk.filename,
                section=chunk.section or "General",
                text=chunk.text.strip(),
            )
            for i, chunk in enumerate(chunks)
        ]
        return "\n\n".join(parts)

    def _extract_cited_indices(self, answer: str) -> list[int]:
        """
        Parse [SOURCE N] references from the generated answer.

        Returns a sorted, deduplicated list of 1-based source indices.
        """
        matches = re.findall(r"\[SOURCE\s+(\d+)\]", answer, re.IGNORECASE)
        return sorted(set(int(m) for m in matches))

    def _score_faithfulness(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> float:
        """
        Heuristic faithfulness score (0.0 – 1.0).

        A proper faithfulness check would use an NLI model to verify each
        claim, but that adds latency.  This lightweight version:
          • Returns 0.0 for UNANSWERABLE responses (no grounding needed)
          • Rewards high citation density
          • Penalises answers with zero citations
          • Penalises answers longer than 3x the total context length
            (a sign the model is hallucinating elaborations)

        This should be complemented with RAGAS evaluation in production.
        """
        if answer.startswith("UNANSWERABLE"):
            return 1.0  # Correctly refusing to answer is "faithful"

        cited = self._extract_cited_indices(answer)
        if not cited:
            # Answer with zero citations — model ignored the context
            log.warning("Answer contains no [SOURCE N] citations")
            return 0.2

        # What fraction of retrieved sources were actually cited?
        citation_coverage = len(cited) / max(len(chunks), 1)

        # Is the answer suspiciously longer than the context?
        context_len = sum(len(c.text) for c in chunks)
        answer_len = len(answer)
        length_penalty = min(answer_len / max(context_len * 3, 1), 1.0)
        length_factor = 1.0 - max(0.0, length_penalty - 0.5)  # only penalise >150% of context

        score = min(1.0, citation_coverage * 0.6 + length_factor * 0.4)
        return round(score, 3)

    def run(
        self,
        question: str,
        filename_filter: Optional[str] = None,
    ) -> RAGResponse:
        """
        Execute the full RAG pipeline for one user question.

        Args:
            question:        The user's natural-language legal question.
            filename_filter: Restrict retrieval to a specific document.

        Returns:
            A RAGResponse with the answer, sources, and quality metrics.
        """
        t_start = time.perf_counter()

        # ── 1. Retrieve ────────────────────────────────────────────────────
        chunks = self._retriever.retrieve(question, filename_filter=filename_filter)

        if not chunks:
            log.warning("No relevant chunks found for: '%s'", question[:80])
            return RAGResponse(
                question=question,
                answer="UNANSWERABLE: No relevant passages were found in the indexed documents. Please ensure the relevant document has been uploaded.",
                sources=[],
                cited_source_indices=[],
                is_answerable=False,
                faithfulness_score=1.0,
                latency_ms=(time.perf_counter() - t_start) * 1000,
                prompt_tokens_estimate=0,
                model_used=cfg.llm.model_name,
            )

        # ── 2. Build prompt ────────────────────────────────────────────────
        context_block = self._build_context_block(chunks)
        user_message = USER_PROMPT_TEMPLATE.format(
            context=context_block,
            question=question,
        )

        # Rough token estimate: 1 token ≈ 4 characters for English legal text
        prompt_tokens_estimate = (len(SYSTEM_PROMPT) + len(user_message)) // 4

        # ── 3. Generate ────────────────────────────────────────────────────
        log.info(
            "Calling LLM '%s' with ~%d prompt tokens",
            cfg.llm.model_name, prompt_tokens_estimate
        )
        try:
            answer = _call_ollama([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ])
        except Exception as exc:
            log.error("LLM call failed: %s", exc, exc_info=True)
            answer = f"UNANSWERABLE: LLM error — {str(exc)[:200]}"

        # ── 4. Post-process ────────────────────────────────────────────────
        cited_indices = self._extract_cited_indices(answer)
        is_answerable = not answer.startswith("UNANSWERABLE")
        faithfulness = self._score_faithfulness(answer, chunks)
        latency_ms = (time.perf_counter() - t_start) * 1000

        log.info(
            "RAG complete: answerable=%s, faithfulness=%.3f, latency=%.0fms",
            is_answerable, faithfulness, latency_ms,
        )

        return RAGResponse(
            question=question,
            answer=answer,
            sources=chunks,
            cited_source_indices=cited_indices,
            is_answerable=is_answerable,
            faithfulness_score=faithfulness,
            latency_ms=latency_ms,
            prompt_tokens_estimate=prompt_tokens_estimate,
            model_used=cfg.llm.model_name,
        )

    def summarise_document(self, filename: str) -> str:
        """
        Generate a structured summary of an entire indexed document.

        Strategy: retrieve a broad sample of chunks (high top_k, no query
        bias) and ask the LLM to synthesise a structured summary.  This is
        less precise than query-driven retrieval but gives a good overview.
        """
        # Use a generic retrieval query that will hit a wide range of clauses
        generic_queries = [
            "main purpose parties involved effective date",
            "obligations rights termination liability",
            "definitions warranties governing law",
        ]

        all_chunks: list[RetrievedChunk] = []
        seen_ids: set[str] = set()

        for query in generic_queries:
            chunks = self._retriever.retrieve(
                query, top_k=5, filename_filter=filename
            )
            for c in chunks:
                if c.chunk_id not in seen_ids:
                    seen_ids.add(c.chunk_id)
                    all_chunks.append(c)

        if not all_chunks:
            return "Could not retrieve content from the document. Ensure it has been indexed."

        context = self._build_context_block(all_chunks[:12])  # cap to avoid token overflow

        summary_prompt = f"""Based ONLY on the following legal document excerpts, provide a structured summary covering:
1. Document type and parties involved
2. Key obligations of each party
3. Important dates and terms
4. Termination and liability clauses
5. Any notable risks or unusual clauses

DOCUMENT EXCERPTS:
{context}

Provide a clear, professional summary. Cite [SOURCE N] for important claims."""

        try:
            return _call_ollama([
                {"role": "system", "content": "You are a legal document analyst. Summarise legal documents accurately and concisely, citing your sources."},
                {"role": "user",   "content": summary_prompt},
            ])
        except Exception as exc:
            log.error("Summarisation failed: %s", exc)
            return f"Summarisation failed: {exc}"
