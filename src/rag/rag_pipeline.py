"""
rag_pipeline.py — RAG pipeline + RAGAS guard.

"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional
import difflib

from config.settings import cfg
from src.retrieval.retriever import Retriever, RetrievedChunk
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

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
QUESTION: {question}

Provide a structured answer with inline citations [SOURCE N].
"""


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    cited_source_indices: list[int]
    is_answerable: bool
    faithfulness_score: float           # from RAGAS guard (or heuristic)
    answer_relevancy_score: float       # RAGAS metric
    context_relevancy_score: float      # RAGAS metric (normalised rerank)
    ragas_overall: float                # harmonic mean of the three
    ragas_passed: bool
    latency_ms: float
    prompt_tokens_estimate: int
    model_used: str
    gate_failed: str = ""               # which gate triggered UNANSWERABLE


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Retrieve → re-rank gate → prompt → generate → citation gate → RAGAS guard.

    api_key is passed per-call so it is never stored on the instance beyond
    the duration of a single request, giving the user full control.
    """

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever
        self._ragas_guard = None   # injected via set_ragas_guard()

    def set_ragas_guard(self, guard) -> None:
        """Accept a RAGASGuard instance from outside (allows shared embedder)."""
        self._ragas_guard = guard

    # ── LLM caller ───────────────────────────────────────────────────────
    def _call_groq(self, messages: list[dict], api_key: str) -> str:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=cfg.llm.timeout,
        )
        resp = client.chat.completions.create(
            model=cfg.llm.model_name,
            messages=messages,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
        )
        return resp.choices[0].message.content.strip()

    # ── Helpers ───────────────────────────────────────────────────────────
    def _build_context_block(self, chunks: list[RetrievedChunk]) -> str:
        return "\n\n".join(
            CONTEXT_TEMPLATE.format(
                n=i + 1,
                filename=chunk.filename,
                section=chunk.section or "General",
                score=chunk.rerank_score,
                text=chunk.text.strip(),
            )
            for i, chunk in enumerate(chunks)
        )

    def _extract_cited_indices(self, answer: str) -> list[int]:
        return sorted(set(
            int(m) for m in re.findall(r"\[SOURCE\s*(\d+)\]", answer, re.IGNORECASE)
        ))

    def _heuristic_faithfulness(self, answer: str, chunks: list[RetrievedChunk]) -> float:
        """
        Cheap pre-RAGAS faithfulness estimate (used when RAGAS guard is absent).
        """
        if answer.startswith("UNANSWERABLE"):
            return 1.0
        cited = self._extract_cited_indices(answer)
        if not cited:
            return 0.0
        citation_coverage = len(cited) / max(len(chunks), 1)
        context_len = sum(len(c.text) for c in chunks)
        length_ratio = len(answer) / max(context_len * 3, 1)
        length_factor = 1.0 - max(0.0, min(length_ratio, 1.0) - 0.5)
        return round(min(1.0, citation_coverage * 0.6 + length_factor * 0.4), 3)

    def _unanswerable_response(
        self,
        question: str,
        reason: str,
        gate: str,
        chunks: list[RetrievedChunk],
        t_start: float,
        model: str = "none",
    ) -> RAGResponse:
        return RAGResponse(
            question=question,
            answer=f"UNANSWERABLE: {reason}",
            sources=chunks,
            cited_source_indices=[],
            is_answerable=False,
            faithfulness_score=1.0,
            answer_relevancy_score=0.0,
            context_relevancy_score=0.0,
            ragas_overall=0.0,
            ragas_passed=False,
            latency_ms=(time.perf_counter() - t_start) * 1000,
            prompt_tokens_estimate=0,
            model_used=model,
            gate_failed=gate,
        )

    # ── Main run ──────────────────────────────────────────────────────────
    def run(
        self,
        question: str,
        api_key: str,
        filename_filter: Optional[str] = None,
    ) -> RAGResponse:
        t_start = time.perf_counter()

        # ── Gate 0: API key present ───────────────────────────────────────
        if not api_key or not api_key.strip():
            return self._unanswerable_response(
                question,
                "No API key provided. Enter your Groq API key in the sidebar.",
                gate="gate_0_no_api_key",
                chunks=[], t_start=t_start,
            )

        # ── Gate 1 + 2: Retrieval (re-rank within retrieval) ─────────────────────
        chunks = self._retriever.retrieve(question, filename_filter=filename_filter)
        if not chunks:
            return self._unanswerable_response(
                question,
                "No relevant passages were found in the indexed documents. "
                "The question may be outside the scope of the uploaded material, "
                "or no documents have been indexed yet.",
                gate="gate_1_2_no_retrieval",
                chunks=[], t_start=t_start,
            )

        # deduplicate chunks based on text similarity (avoid near-duplicates)
        filtered = []
        for c in chunks:
            if any(difflib.SequenceMatcher(None, c.text, s.text).ratio()>0.95 for s in filtered):
                continue
            filtered.append(c)
        
        chunks = filtered

        # ── Build prompt ──────────────────────────────────────────────────
        context_block = self._build_context_block(chunks)
        user_msg = USER_PROMPT_TEMPLATE.format(context=context_block, question=question)
        prompt_tokens_estimate = (len(SYSTEM_PROMPT) + len(user_msg)) // 4
        model_label = f"groq/{cfg.llm.model_name}"

        log.info("LLM call | model=%s tokens~%d chunks=%d", model_label, prompt_tokens_estimate, len(chunks))

        # ── LLM generation ────────────────────────────────────────────────
        try:
            answer = self._call_groq(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                api_key=api_key,
            )
        except Exception as exc:
            log.error("LLM call failed: %s", exc, exc_info=True)
            return self._unanswerable_response(
                question, f"LLM error — {str(exc)[:200]}",
                gate="gate_llm_error", chunks=chunks, t_start=t_start, model=model_label,
            )

        # ── Gate 3: LLM self-declared unanswerable ────────────────────────
        if answer.upper().startswith("UNANSWERABLE"):
            log.info("LLM declared UNANSWERABLE")
            return self._unanswerable_response(
                question,
                answer.split(":", 1)[-1].strip() if ":" in answer else "Insufficient context.",
                gate="gate_3_llm_unanswerable",
                chunks=chunks, t_start=t_start, model=model_label,
            )

        # ── Gate 4: No citations in answer ───────────────────────────────
        cited = self._extract_cited_indices(answer)
        if not cited:
            log.warning("Answer contains zero [SOURCE N] citations — forcing UNANSWERABLE")
            return self._unanswerable_response(
                question,
                "The generated answer contained no source citations, indicating it could not "
                "be grounded in the retrieved passages.",
                gate="gate_4_no_citations",
                chunks=chunks, t_start=t_start, model=model_label,
            )

        # ── Gate 5–7: RAGAS quality guard ────────────────────────────────
        faith = self._heuristic_faithfulness(answer, chunks)   # fallback
        ans_rel = 1.0
        ctx_rel = 1.0
        ragas_passed = True
        ragas_overall = faith
        ragas_failure = ""

        if self._ragas_guard is not None and cfg.ragas.enabled:
            ragas_result = self._ragas_guard.evaluate(question, answer, chunks, api_key)
            faith       = ragas_result.faithfulness
            ans_rel     = ragas_result.answer_relevancy
            ctx_rel     = ragas_result.context_relevancy
            ragas_passed = ragas_result.passed
            ragas_overall = ragas_result.overall_score

            if not ragas_passed:
                ragas_failure = ragas_result.failure_reason
                log.warning("RAGAS gate failed: %s", ragas_failure)
                return self._unanswerable_response(
                    question,
                    f"The answer did not meet quality standards and may contain "
                    f"hallucinated or off-topic content. ({ragas_failure})",
                    gate=f"gate_5_7_ragas",
                    chunks=chunks, t_start=t_start, model=model_label,
                )

        latency_ms = (time.perf_counter() - t_start) * 1000
        log.info(
            "RAG complete | answerable=True | faith=%.2f ans_rel=%.2f ctx_rel=%.2f | lat=%.0fms",
            faith, ans_rel, ctx_rel, latency_ms,
        )

        return RAGResponse(
            question=question,
            answer=answer,
            sources=chunks,
            cited_source_indices=cited,
            is_answerable=True,
            faithfulness_score=faith,
            answer_relevancy_score=ans_rel,
            context_relevancy_score=ctx_rel,
            ragas_overall=ragas_overall,
            ragas_passed=ragas_passed,
            latency_ms=latency_ms,
            prompt_tokens_estimate=prompt_tokens_estimate,
            model_used=model_label,
            gate_failed="",
        )

    # ── Summarise ─────────────────────────────────────────────────────────
    def summarise_document(self, filename: str, api_key: str) -> str:
        if not api_key or not api_key.strip():
            return "No API key provided. Enter your Groq key in the sidebar."

        generic_queries = [
            "main purpose parties involved effective date",
            "obligations rights termination liability",
            "definitions warranties governing law indemnification",
        ]
        all_chunks: list[RetrievedChunk] = []
        seen: set[str] = set()
        for q in generic_queries:
            for c in self._retriever.retrieve(q, top_k=6, filename_filter=filename):
                if c.chunk_id not in seen:
                    seen.add(c.chunk_id)
                    all_chunks.append(c)

        if not all_chunks:
            return "Could not retrieve content. Ensure the document has been indexed."

        context = self._build_context_block(all_chunks[:12])
        summary_prompt = f"""Based ONLY on the following legal document excerpts, provide a structured summary covering:
1. Document type and parties involved
2. Key obligations of each party
3. Important dates and financial terms
4. Termination and liability clauses
5. Governing law and dispute resolution
6. Notable risks or unusual clauses

DOCUMENT EXCERPTS:
{context}

Cite [SOURCE N] for every important claim. If a section cannot be determined from the excerpts, write "Not specified in available excerpts"."""

        try:
            return self._call_groq(
                [
                    {"role": "system", "content": "You are a legal document analyst. Summarise accurately, citing [SOURCE N] for every claim. Never fabricate information."},
                    {"role": "user",   "content": summary_prompt},
                ],
                api_key=api_key,
            )
        except Exception as exc:
            log.error("Summarisation failed: %s", exc)
            return f"Summarisation failed: {exc}"
