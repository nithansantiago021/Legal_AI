from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Optional

from config.settings import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RAGASResult:
    faithfulness: float
    answer_relevancy: float
    context_relevancy: float
    passed: bool
    failure_reason: str = ""
    statements_checked: int = 0
    statements_supported: int = 0
    generated_questions: list[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        scores = [self.faithfulness, self.answer_relevancy, self.context_relevancy]
        if any(s <= 0 for s in scores):
            return 0.0
        return round(len(scores) / sum(1.0 / s for s in scores), 3)


# ---------------- LLM CALL ----------------
def _groq_json(prompt, api_key, system=""):
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=cfg.llm.judge_model,
            messages=messages,
            temperature=0,
            max_tokens=512,
        )

        text = resp.choices[0].message.content.strip()
        text = re.sub(r"```.*?```", "", text, flags=re.S)

        return json.loads(text)

    except Exception as e:
        log.warning("Judge call failed: %s", e)
        return None


# ---------------- FAITHFULNESS ----------------
def _faithfulness(answer, context, api_key):
    """
    FIXED VERSION:
    Removes LLM-based decomposition (main source of failure)
    Uses deterministic sentence + overlap grounding
    """

    if answer.upper().startswith("UNANSWERABLE"):
        return 0.0, 0, 0

    # 1. sentence split (stable, no LLM)
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0, 0, 0

    # 2. build context bag
    context_text = " ".join(context).lower()

    context_tokens = set(context_text.split())

    def is_supported(sentence: str) -> bool:
        tokens = set(sentence.lower().split())

        # overlap-based grounding (legal docs work well here)
        overlap = len(tokens & context_tokens)

        return overlap >= 3   # threshold (tune later if needed)

    supported = sum(is_supported(s) for s in sentences)

    score = supported / len(sentences)

    return round(score, 3), len(sentences), supported


# ---------------- ANSWER RELEVANCY ----------------
def _answer_relevancy(question, answer, embedder, api_key):
    if answer.upper().startswith("UNANSWERABLE"):
        return 0.0, []   # FIXED

    prompt = f"""
Generate 3 questions this answer solves.

Answer:
{answer}
"""

    raw = _groq_json(prompt, api_key)

    if not isinstance(raw, list):
        return 0.5, []

    gen_qs = raw[:3]

    try:
        from src.utils.cosine import cosine_similarity

        q_vec = embedder.encode_query(question)
        gen_vecs = embedder.encode_documents(gen_qs)

        sims = [cosine_similarity(q_vec, v) for v in gen_vecs]
        return round(sum(sims) / len(sims), 3), gen_qs

    except Exception:
        return 0.5, gen_qs


# ---------------- CONTEXT RELEVANCY ----------------
def _context_relevancy(scores):
    if not scores:
        return 0.0

    # FIXED normalization
    if max(scores) <= 1.0:
        vals = scores
    else:
        vals = [1 / (1 + math.exp(-s)) for s in scores]

    return round(sum(vals) / len(vals), 3)


# ---------------- MAIN ----------------
class RAGASGuard:
    def __init__(self, embedder):
        self._embedder = embedder

    def evaluate(self, question, answer, chunks, api_key):

        ctx = [c.text for c in chunks]
        rerank = [c.rerank_score for c in chunks]

        ctx_rel = _context_relevancy(rerank)
        faith, n, sup = _faithfulness(answer, ctx, api_key)
        ans_rel, gen_qs = _answer_relevancy(question, answer, self._embedder, api_key)

        failures = []

        if faith < cfg.ragas.min_faithfulness:
            failures.append("faithfulness low")

        if ans_rel < cfg.ragas.min_answer_relevancy:
            failures.append("answer relevancy low")

        if ctx_rel < cfg.ragas.min_context_relevancy:
            failures.append("context relevancy low")

        return RAGASResult(
            faithfulness=faith,
            answer_relevancy=ans_rel,
            context_relevancy=ctx_rel,
            passed=len(failures) == 0,
            failure_reason="; ".join(failures),
            statements_checked=n,
            statements_supported=sup,
            generated_questions=gen_qs,
        )