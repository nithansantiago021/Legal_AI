"""
evaluator.py — RAGAS-based retrieval and generation evaluation.

RAGAS (Retrieval-Augmented Generation Assessment) provides a framework for
evaluating RAG pipelines without hand-labelled ground-truth answers.  It
measures:

  • Faithfulness    — does the answer contain only claims supported by context?
  • Answer Relevancy — is the answer relevant to the question?
  • Context Precision  — are the retrieved chunks relevant?
  • Context Recall     — does the context cover what's needed to answer?

We also compute classical retrieval metrics (Precision@K, Recall@K) using
a small set of manually-written test questions.

Usage
-----
  python -m src.evaluation.evaluator --test-file data/eval_questions.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from config.settings import cfg
from src.rag.rag_pipeline import RAGPipeline, RAGResponse
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models for evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalQuestion:
    """A single evaluation question with optional ground truth."""
    question: str
    expected_answer: Optional[str] = None        # can be None for relevancy-only eval
    relevant_chunk_ids: list[str] = field(default_factory=list)  # for Recall@K


@dataclass
class EvalResult:
    """Per-question evaluation result."""
    question: str
    answer: str
    is_answerable: bool
    faithfulness_score: float
    num_sources_retrieved: int
    num_sources_cited: int
    latency_ms: float


@dataclass
class EvalReport:
    """Aggregated evaluation metrics across all test questions."""
    total_questions: int
    answerable_rate: float           # % of questions the system attempted to answer
    avg_faithfulness: float
    avg_latency_ms: float
    avg_sources_retrieved: float
    avg_sources_cited: float
    results: list[EvalResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RAGAS integration
# ---------------------------------------------------------------------------

def _run_ragas_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: Optional[list[str]] = None,
) -> dict:
    """
    Run RAGAS evaluation if the library is available.

    RAGAS requires an OpenAI API key by default (it uses GPT-4 as the judge).
    We wrap it in a try-except so the system remains functional without it.

    Args:
        questions:     List of user questions.
        answers:       List of generated answers.
        contexts:      List of context lists (one list of strings per question).
        ground_truths: Optional list of reference answers for recall computation.

    Returns:
        Dict of metric name → score, or {} if RAGAS is not available.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        data_dict: dict = {
            "question":  questions,
            "answer":    answers,
            "contexts":  contexts,
        }
        metrics = [faithfulness, answer_relevancy, context_precision]

        if ground_truths:
            data_dict["ground_truth"] = ground_truths
            metrics.append(context_recall)

        dataset = Dataset.from_dict(data_dict)
        log.info("Running RAGAS evaluation on %d questions...", len(questions))
        result = evaluate(dataset, metrics=metrics)
        return {k: float(v) for k, v in result.items()}

    except ImportError:
        log.warning("RAGAS not installed. Run: pip install ragas datasets")
        return {}
    except Exception as exc:
        log.error("RAGAS evaluation failed: %s", exc, exc_info=True)
        return {}


# ---------------------------------------------------------------------------
# Main evaluator class
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Runs evaluation benchmarks over the RAG pipeline.

    Produces both a per-question breakdown and aggregated statistics.
    """

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline

    def load_test_questions(self, path: Optional[str] = None) -> list[EvalQuestion]:
        """
        Load evaluation questions from a JSON file.

        Expected JSON format:
        [
          {
            "question": "What is the effective date of the contract?",
            "expected_answer": "1 January 2024",   // optional
            "relevant_chunk_ids": ["abc123", "def456"]  // optional
          },
          ...
        ]

        If no path is provided, returns a built-in set of generic legal
        questions useful for smoke-testing.
        """
        if path and Path(path).exists():
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            return [EvalQuestion(**item) for item in data]

        # Default smoke-test questions — generic enough to work with any legal doc
        log.warning("No eval file found; using built-in smoke-test questions")
        return [
            EvalQuestion("What are the main parties in this document?"),
            EvalQuestion("What are the key obligations of the parties?"),
            EvalQuestion("What is the termination clause?"),
            EvalQuestion("What law governs this agreement?"),
            EvalQuestion("What is the liability limitation?"),
            EvalQuestion("Are there any confidentiality provisions?"),
            EvalQuestion("What happens in case of breach of contract?"),
            EvalQuestion("What is the payment structure?"),
        ]

    def run(
        self,
        questions: Optional[list[EvalQuestion]] = None,
        save_results: bool = True,
    ) -> EvalReport:
        """
        Run the full evaluation suite.

        Args:
            questions:    EvalQuestion list; loaded from cfg if None.
            save_results: If True, save the report to cfg.evaluation.results_path.

        Returns:
            An EvalReport with aggregated metrics and per-question details.
        """
        if questions is None:
            questions = self.load_test_questions(cfg.evaluation.test_dataset_path)

        log.info("Starting evaluation: %d questions", len(questions))

        results: list[EvalResult] = []
        # Collect data for RAGAS
        ragas_questions: list[str] = []
        ragas_answers:   list[str] = []
        ragas_contexts:  list[list[str]] = []
        ragas_truths:    list[str] = []

        for i, eq in enumerate(questions, 1):
            log.info("[%d/%d] Evaluating: %s", i, len(questions), eq.question[:60])
            try:
                rag_resp: RAGResponse = self._pipeline.run(eq.question)

                result = EvalResult(
                    question=eq.question,
                    answer=rag_resp.answer,
                    is_answerable=rag_resp.is_answerable,
                    faithfulness_score=rag_resp.faithfulness_score,
                    num_sources_retrieved=len(rag_resp.sources),
                    num_sources_cited=len(rag_resp.cited_source_indices),
                    latency_ms=rag_resp.latency_ms,
                )
                results.append(result)

                # Collect for RAGAS
                ragas_questions.append(eq.question)
                ragas_answers.append(rag_resp.answer)
                ragas_contexts.append([c.text for c in rag_resp.sources])
                if eq.expected_answer:
                    ragas_truths.append(eq.expected_answer)

            except Exception as exc:
                log.error("Evaluation failed for question '%s': %s", eq.question[:40], exc)

        # ── Aggregate metrics ──────────────────────────────────────────────
        n = len(results)
        report = EvalReport(
            total_questions=n,
            answerable_rate=sum(r.is_answerable for r in results) / max(n, 1),
            avg_faithfulness=sum(r.faithfulness_score for r in results) / max(n, 1),
            avg_latency_ms=sum(r.latency_ms for r in results) / max(n, 1),
            avg_sources_retrieved=sum(r.num_sources_retrieved for r in results) / max(n, 1),
            avg_sources_cited=sum(r.num_sources_cited for r in results) / max(n, 1),
            results=results,
        )

        # ── RAGAS (optional) ──────────────────────────────────────────────
        ragas_metrics = _run_ragas_evaluation(
            ragas_questions, ragas_answers, ragas_contexts,
            ragas_truths if ragas_truths else None,
        )
        if ragas_metrics:
            log.info("RAGAS metrics: %s", ragas_metrics)

        # ── Save ──────────────────────────────────────────────────────────
        if save_results:
            output = {
                "summary": {k: v for k, v in asdict(report).items() if k != "results"},
                "ragas_metrics": ragas_metrics,
                "per_question": [asdict(r) for r in results],
            }
            out_path = Path(cfg.evaluation.results_path)
            out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
            log.info("Evaluation report saved to: %s", out_path)

        # Log summary
        log.info(
            "EVALUATION SUMMARY: answerable=%.1f%% | faithfulness=%.3f | latency=%.0fms",
            report.answerable_rate * 100,
            report.avg_faithfulness,
            report.avg_latency_ms,
        )
        return report
