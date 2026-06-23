"""
ragas_guard.py — Lightweight RAGAS-style quality guard for the RAG pipeline.

Why not the official RAGAS library?
------------------------------------
The official `ragas` package requires a separate LangChain LLM wrapper and
pulls in heavy async machinery that doesn't play well with Streamlit's
synchronous execution model on HF free tier.  This module reimplements the
three most important RAGAS metrics from first principles using the same Groq
API key the user already provided:

  1. Faithfulness
     ─────────────
     Decompose the answer into atomic statements via the judge LLM.
     For each statement, ask the judge: "Is this claim fully supported by the
     provided context passages?"  Faithfulness = |supported| / |total|.
     This is the primary anti-hallucination metric.

  2. Answer Relevancy
     ─────────────────
     Ask the judge LLM to generate N hypothetical questions that the given
     answer would best respond to.  Embed all generated questions AND the
     original question with the BGE model, then compute mean cosine similarity.
     High similarity → the answer is on-topic.  Low similarity → the answer
     drifted from the question.

  3. Context Relevancy
     ──────────────────
     Proxy metric: normalise the cross-encoder re-rank scores of retrieved
     chunks to [0,1] and take their mean.  This avoids an extra LLM call
     and is accurate because the cross-encoder score directly measures how
     well a passage answers the query.

Gating logic
────────────
After generation, RAGASGuard.evaluate() returns a RAGASResult.  The pipeline
checks:
  - faithfulness   < cfg.ragas.min_faithfulness       → UNANSWERABLE
  - answer_relevancy < cfg.ragas.min_answer_relevancy → UNANSWERABLE
  - context_relevancy < cfg.ragas.min_context_relevancy → UNANSWERABLE

All three gates must pass for the answer to be returned to the user.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Optional

from config.settings import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RAGASResult:
    faithfulness: float             # 0–1, fraction of statements grounded in context
    answer_relevancy: float         # 0–1, semantic alignment of answer with question
    context_relevancy: float        # 0–1, normalised re-rank score of retrieved chunks
    passed: bool                    # True if all gates pass
    failure_reason: str = ""        # populated when passed=False
    statements_checked: int = 0     # how many atomic statements were NLI-evaluated
    statements_supported: int = 0
    generated_questions: list[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Harmonic mean of the three metrics — same as RAGAS composite score."""
        scores = [self.faithfulness, self.answer_relevancy, self.context_relevancy]
        if any(s == 0 for s in scores):
            return 0.0
        return round(len(scores) / sum(1.0 / s for s in scores), 3)


# ── Groq caller (shared helper) ───────────────────────────────────────────────

def _groq_json(prompt: str, api_key: str, system: str = "") -> dict | list | None:
    """
    Call the fast judge model and parse JSON from the response.
    Returns None on any failure so callers can degrade gracefully.
    """
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=cfg.llm.timeout,
        )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=cfg.llm.judge_model,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )
        text = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        return json.loads(text)
    except Exception as exc:
        log.warning("RAGAS judge call failed: %s", exc)
        return None


# ── Metric 1: Faithfulness ────────────────────────────────────────────────────

def _compute_faithfulness(
    answer: str,
    context_texts: list[str],
    api_key: str,
) -> tuple[float, int, int]:
    """
    Returns (faithfulness_score, statements_checked, statements_supported).

    Step 1 — decompose answer into atomic statements.
    Step 2 — for each statement, ask judge if it is supported by context.
    """
    if answer.startswith("UNANSWERABLE"):
        return 1.0, 0, 0   # UNANSWERABLE is intrinsically faithful

    context_block = "\n\n".join(f"[C{i+1}] {t}" for i, t in enumerate(context_texts))

    # ── Step 1: decompose ──
    decompose_prompt = f"""Break the following answer into a list of self-contained atomic statements.
Each statement must make a single factual claim. Ignore hedges like "based on the sources".

ANSWER:
{answer}

Respond with ONLY a JSON array of strings, e.g.:
["Statement 1.", "Statement 2.", "Statement 3."]
Do NOT include any preamble or markdown."""

    raw = _groq_json(decompose_prompt, api_key)
    if not isinstance(raw, list) or not raw:
        log.warning("Faithfulness decomposition failed or returned no statements")
        # Fall back to simple citation-coverage heuristic
        cited = len(re.findall(r"\[SOURCE\s*\d+\]", answer, re.IGNORECASE))
        return min(1.0, cited / 3), 0, cited

    statements: list[str] = [str(s) for s in raw if str(s).strip()]
    statements = statements[: cfg.ragas.max_statements_to_check]

    # ── Step 2: NLI check ──
    nli_prompt = f"""You are a strict factual verifier.

CONTEXT PASSAGES:
{context_block}

For each statement below, determine whether it is FULLY supported by the context passages above.
A statement is supported only if the context explicitly or clearly implies it.
Do NOT use any outside knowledge.

STATEMENTS:
{json.dumps(statements)}

Respond with ONLY a JSON object mapping each statement (as a key) to true or false:
{{"Statement 1 text": true, "Statement 2 text": false, ...}}
Do NOT include any preamble or markdown."""

    verdicts = _groq_json(nli_prompt, api_key)
    if not isinstance(verdicts, dict):
        log.warning("NLI verdict parsing failed — using heuristic fallback")
        # Heuristic: credit any statement that overlaps with context vocab
        context_words = set(" ".join(context_texts).lower().split())
        supported = sum(
            len(set(s.lower().split()) & context_words) / max(len(s.split()), 1) > 0.4
            for s in statements
        )
        return round(supported / max(len(statements), 1), 3), len(statements), supported

    n_supported = sum(1 for v in verdicts.values() if v is True)
    n_total = len(verdicts)
    score = round(n_supported / max(n_total, 1), 3)
    return score, n_total, n_supported


# ── Metric 2: Answer Relevancy ────────────────────────────────────────────────

def _compute_answer_relevancy(
    question: str,
    answer: str,
    embedder,        # the shared Embedder instance
    api_key: str,
    n_questions: int = 3,
) -> tuple[float, list[str]]:
    """
    Returns (answer_relevancy_score, generated_questions).

    Asks the judge to reverse-engineer N questions that this answer would
    satisfy, then measures mean cosine similarity between those questions
    and the original question in embedding space.
    """
    if answer.startswith("UNANSWERABLE"):
        return 1.0, []   # not applicable — don't penalise

    gen_prompt = f"""Given the following answer, generate {n_questions} different questions that this answer would directly respond to.
The questions should be specific, not generic.

ANSWER:
{answer}

Respond with ONLY a JSON array of {n_questions} question strings.
Do NOT include any preamble or markdown."""

    raw = _groq_json(gen_prompt, api_key)
    if not isinstance(raw, list) or not raw:
        log.warning("Answer relevancy question generation failed — returning neutral score")
        return 0.5, []

    gen_questions = [str(q).strip() for q in raw if str(q).strip()][:n_questions]
    if not gen_questions:
        return 0.5, []

    # Embed original question and all generated questions
    try:
        from src.utils.cosine import cosine_similarity
        q_vec = embedder.encode_query(question)
        gen_vecs = embedder.encode_documents(gen_questions)
        sims = [cosine_similarity(q_vec, gv) for gv in gen_vecs]
        score = round(sum(sims) / len(sims), 3)
    except Exception as exc:
        log.warning("Answer relevancy embedding failed: %s", exc)
        score = 0.5

    return score, gen_questions


# ── Metric 3: Context Relevancy ───────────────────────────────────────────────

def _compute_context_relevancy(rerank_scores: list[float]) -> float:
    """
    Normalise cross-encoder logits to [0,1] using a sigmoid and return
    the mean across all retrieved chunks.

    ms-marco logit interpretation:
      logit >  2  → highly relevant  → sigmoid ≈ 0.88
      logit =  0  → moderate         → sigmoid = 0.50
      logit = -2  → weak             → sigmoid ≈ 0.12
      logit < -3  → already filtered by retriever
    """
    if not rerank_scores:
        return 0.0
    sigmoid_scores = [1.0 / (1.0 + math.exp(-s)) for s in rerank_scores]
    return round(sum(sigmoid_scores) / len(sigmoid_scores), 3)


# ── Main guard ────────────────────────────────────────────────────────────────

class RAGASGuard:
    """
    Post-generation quality evaluator.

    Usage
    -----
    guard = RAGASGuard(embedder)
    result = guard.evaluate(question, answer, chunks, api_key)
    if not result.passed:
        # override answer with UNANSWERABLE + reason
    """

    def __init__(self, embedder) -> None:
        self._embedder = embedder

    def evaluate(
        self,
        question: str,
        answer: str,
        chunks,                          # list[RetrievedChunk]
        api_key: str,
    ) -> RAGASResult:
        """
        Run all three metrics and return a RAGASResult.
        Never raises — degrades gracefully on any LLM/network failure.
        """
        if not cfg.ragas.enabled:
            return RAGASResult(
                faithfulness=1.0, answer_relevancy=1.0, context_relevancy=1.0,
                passed=True,
            )

        context_texts = [c.text for c in chunks]
        rerank_scores = [c.rerank_score for c in chunks]

        log.info("Running RAGAS evaluation for: '%s'", question[:60])

        # ── Metric 3 first (no LLM call needed) ──────────────────────────
        ctx_rel = _compute_context_relevancy(rerank_scores)

        # ── Metric 1: Faithfulness ────────────────────────────────────────
        faith, n_checked, n_supported = _compute_faithfulness(answer, context_texts, api_key)

        # ── Metric 2: Answer Relevancy ────────────────────────────────────
        ans_rel, gen_qs = _compute_answer_relevancy(
            question, answer, self._embedder, api_key
        )

        # ── Gate checks ───────────────────────────────────────────────────
        failures = []
        if faith < cfg.ragas.min_faithfulness:
            failures.append(
                f"faithfulness {faith:.2f} < {cfg.ragas.min_faithfulness} "
                f"({n_supported}/{n_checked} statements grounded)"
            )
        if ans_rel < cfg.ragas.min_answer_relevancy:
            failures.append(
                f"answer relevancy {ans_rel:.2f} < {cfg.ragas.min_answer_relevancy} "
                f"(answer may not address the question)"
            )
        if ctx_rel < cfg.ragas.min_context_relevancy:
            failures.append(
                f"context relevancy {ctx_rel:.2f} < {cfg.ragas.min_context_relevancy} "
                f"(retrieved passages may not be relevant)"
            )

        passed = len(failures) == 0
        failure_reason = "; ".join(failures) if failures else ""

        result = RAGASResult(
            faithfulness=faith,
            answer_relevancy=ans_rel,
            context_relevancy=ctx_rel,
            passed=passed,
            failure_reason=failure_reason,
            statements_checked=n_checked,
            statements_supported=n_supported,
            generated_questions=gen_qs,
        )

        log.info(
            "RAGAS | faith=%.2f ans_rel=%.2f ctx_rel=%.2f overall=%.2f passed=%s",
            faith, ans_rel, ctx_rel, result.overall_score, passed,
        )
        if not passed:
            log.warning("RAGAS gate FAILED: %s", failure_reason)

        return result
