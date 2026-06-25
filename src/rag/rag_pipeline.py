from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

from config.settings import cfg
from src.retrieval.retriever import Retriever, RetrievedChunk
from src.utils.logger import get_logger

log = get_logger(__name__)


SYSTEM_PROMPT = """
You are a precise Legal Document Analyst.

RULES:
1. Use ONLY provided sources.
2. Every factual statement must include [SOURCE N].
3. Do NOT use outside knowledge.
4. If insufficient context exists, respond exactly:
   UNANSWERABLE: insufficient information in sources
"""

CONTEXT_TEMPLATE = "[SOURCE {n}] {text}"

USER_PROMPT_TEMPLATE = """
SOURCES:
{context}

QUESTION:
{question}

Answer clearly with citations [SOURCE N].
"""


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    cited_source_indices: list[int]
    is_answerable: bool
    faithfulness_score: float
    answer_relevancy_score: float
    context_relevancy_score: float
    ragas_overall: float
    ragas_passed: bool
    latency_ms: float
    model_used: str
    gate_failed: str = ""


class RAGPipeline:
    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever
        self._ragas_guard = None

    def set_ragas_guard(self, guard):
        self._ragas_guard = guard

    def _call_llm(self, messages, api_key):
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

    def _build_context(self, chunks):
        return "\n\n".join(
            CONTEXT_TEMPLATE.format(n=i + 1, text=c.text.strip())
            for i, c in enumerate(chunks)
        )

    def _extract_citations(self, text: str):
        return sorted(set(int(x) for x in re.findall(r"\[SOURCE\s*(\d+)\]", text)))

    # ---------------- MAIN ----------------
    def run(self, question: str, api_key: str, filename_filter: Optional[str] = None):

        t0 = time.perf_counter()

        if not api_key:
            return self._fail(question, "Missing API key", "gate_0", t0)

        # 1. Retrieve
        chunks = self._retriever.retrieve(question, filename_filter=filename_filter)

        if not chunks:
            return self._fail(question, "No relevant chunks found", "gate_1", t0)

        # 2. Build prompt
        context = self._build_context(chunks)

        prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

        # 3. Generate
        answer = self._call_llm(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            api_key,
        )

        citations = self._extract_citations(answer)

        # 4. RUN RAGAS FIRST (IMPORTANT FIX)
        if self._ragas_guard:
            ragas = self._ragas_guard.evaluate(question, answer, chunks, api_key)
        else:
            ragas = None

        # 5. Detect unanswerable
        is_unanswerable = answer.upper().startswith("UNANSWERABLE")

        # 6. FINAL DECISION (balanced gating)
        if is_unanswerable:
            return self._fail(question, answer, "gate_llm_unanswerable", t0, chunks)

        # soft citation enforcement (NOT hard kill)
        if len(citations) == 0:
            log.warning("No citations found but keeping answer for RAGAS review")

        # 7. Build response
        latency = (time.perf_counter() - t0) * 1000

        return RAGResponse(
            question=question,
            answer=answer,
            sources=chunks,
            cited_source_indices=citations,
            is_answerable=True,
            faithfulness_score=ragas.faithfulness if ragas else 0.0,
            answer_relevancy_score=ragas.answer_relevancy if ragas else 0.0,
            context_relevancy_score=ragas.context_relevancy if ragas else 0.0,
            ragas_overall=ragas.overall_score if ragas else 0.0,
            ragas_passed=ragas.passed if ragas else True,
            latency_ms=latency,
            model_used=cfg.llm.model_name,
        )

    def _fail(self, question, reason, gate, t0, chunks=None):
        return RAGResponse(
            question=question,
            answer=f"UNANSWERABLE: {reason}",
            sources=chunks or [],
            cited_source_indices=[],
            is_answerable=False,
            faithfulness_score=0.0,
            answer_relevancy_score=0.0,
            context_relevancy_score=0.0,
            ragas_overall=0.0,
            ragas_passed=False,
            latency_ms=(time.perf_counter() - t0) * 1000,
            model_used="none",
            gate_failed=gate,
        )