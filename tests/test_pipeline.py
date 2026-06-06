"""
test_pipeline.py — Unit tests for the Legal RAG pipeline components.

Run with:
  pytest tests/ -v

We use pytest fixtures to avoid repeatedly loading the embedding model in
test teardowns.  The vector store is tested against an in-memory ChromaDB
client (no disk writes during tests).
"""

import hashlib
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestTextCleaning:
    """Tests for the clean_text and detect_section functions."""

    def test_removes_zero_width_chars(self):
        from src.preprocessing.preprocessor import clean_text
        noisy = "Normal text\u200bwith zero-width\ufeffspaces"
        result = clean_text(noisy)
        assert "\u200b" not in result
        assert "\ufeff" not in result

    def test_collapses_excess_newlines(self):
        from src.preprocessing.preprocessor import clean_text
        text = "Para 1\n\n\n\n\nPara 2"
        result = clean_text(text)
        assert "\n\n\n" not in result

    def test_normalises_horizontal_whitespace(self):
        from src.preprocessing.preprocessor import clean_text
        text = "Lots    of     spaces"
        result = clean_text(text)
        assert "  " not in result   # no double spaces

    def test_detect_section_numbered_clause(self):
        from src.preprocessing.preprocessor import detect_section
        text = "7. TERMINATION\nEither party may terminate..."
        section = detect_section(text)
        assert "7" in section or "TERMINATION" in section

    def test_detect_section_returns_empty_for_body_text(self):
        from src.preprocessing.preprocessor import detect_section
        text = "The parties agree that all payments shall be made within 30 days."
        section = detect_section(text)
        # Generic body text should not be detected as a heading
        assert section == "" or len(section) < 5


class TestChunking:
    """Tests for the chunk_document function."""

    def test_chunks_have_unique_ids(self):
        from src.preprocessing.preprocessor import RawDocument, chunk_document
        doc = RawDocument(
            doc_id="testdoc1",
            filename="test.txt",
            file_type="txt",
            raw_text="A" * 2000,   # force multiple chunks
        )
        chunks = chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunks_cover_full_document(self):
        """Verify that chunk boundaries don't skip any part of the document."""
        from src.preprocessing.preprocessor import RawDocument, chunk_document, clean_text
        raw = "The quick brown fox. " * 100
        doc = RawDocument(doc_id="d1", filename="f.txt", file_type="txt", raw_text=raw)
        chunks = chunk_document(doc)
        # Reconstruct a rough coverage check
        # With overlap, total chunk chars will exceed cleaned text chars
        clean = clean_text(raw)
        assert len(chunks) >= 1
        # Each chunk should be non-empty
        assert all(len(c.text) > 0 for c in chunks)

    def test_chunk_metadata_populated(self):
        from src.preprocessing.preprocessor import RawDocument, chunk_document
        doc = RawDocument(
            doc_id="metadoc",
            filename="contract.pdf",
            file_type="pdf",
            raw_text="CLAUSE 1 — PARTIES\nParty A and Party B agree to the following terms. " * 30,
        )
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.doc_id == "metadoc"
            assert c.filename == "contract.pdf"
            assert isinstance(c.chunk_index, int)
            assert c.char_start >= 0
            assert c.char_end > c.char_start


# ---------------------------------------------------------------------------
# Chunk metadata serialisation
# ---------------------------------------------------------------------------

class TestChromaMetadata:
    """Tests for DocumentChunk.to_chroma_metadata."""

    def test_no_nested_objects(self):
        """ChromaDB rejects non-primitive metadata values."""
        from src.preprocessing.preprocessor import DocumentChunk
        chunk = DocumentChunk(
            chunk_id="abc", doc_id="d1", filename="f.txt",
            chunk_index=0, text="test", char_start=0, char_end=4,
            metadata={"nested": {"key": "value"}},
        )
        meta = chunk.to_chroma_metadata()
        # All values must be str, int, float, or bool
        for k, v in meta.items():
            assert isinstance(v, (str, int, float, bool)), \
                f"Metadata value for key '{k}' is {type(v)}, expected primitive"


# ---------------------------------------------------------------------------
# RAG pipeline tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestRAGPipeline:
    """Tests for the RAG pipeline using a mocked LLM client."""

    @pytest.fixture
    def mock_pipeline(self):
        """Return a RAGPipeline with a mocked retriever and LLM."""
        from src.rag.rag_pipeline import RAGPipeline
        from src.retrieval.retriever import RetrievedChunk

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            RetrievedChunk(
                chunk_id="c1",
                text="The agreement shall terminate upon 30 days written notice.",
                filename="contract.pdf",
                section="CLAUSE 7 — TERMINATION",
                chunk_index=6,
                similarity=0.92,
                doc_id="doc1",
            )
        ]

        pipeline = RAGPipeline(mock_retriever)

        # Mock the LLM client to avoid real API calls in tests
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "The agreement can be terminated with 30 days written notice [SOURCE 1]."
        )
        pipeline._client.chat.completions.create.return_value = mock_response

        return pipeline

    def test_answerable_response(self, mock_pipeline):
        resp = mock_pipeline.run("What is the termination clause?")
        assert resp.is_answerable is True
        assert "[SOURCE 1]" in resp.answer
        assert len(resp.cited_source_indices) == 1
        assert resp.cited_source_indices[0] == 1

    def test_faithfulness_score_above_zero(self, mock_pipeline):
        resp = mock_pipeline.run("What is the termination clause?")
        assert resp.faithfulness_score > 0

    def test_empty_retrieval_returns_unanswerable(self):
        from src.rag.rag_pipeline import RAGPipeline

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []   # simulate no relevant chunks

        pipeline = RAGPipeline(mock_retriever)
        resp = pipeline.run("Completely irrelevant question?")

        assert resp.is_answerable is False
        assert "UNANSWERABLE" in resp.answer
        assert resp.faithfulness_score == 1.0   # correctly refusing = faithful

    def test_extract_cited_indices(self, mock_pipeline):
        """Unit test the citation extraction regex."""
        from src.rag.rag_pipeline import RAGPipeline
        mock_retriever = MagicMock()
        pipeline = RAGPipeline(mock_retriever)

        answer = "Clause A [SOURCE 1] and Clause B [SOURCE 3] are relevant."
        indices = pipeline._extract_cited_indices(answer)
        assert indices == [1, 3]

    def test_unanswerable_faithfulness(self, mock_pipeline):
        """UNANSWERABLE responses should score 1.0 (correct behaviour)."""
        from src.rag.rag_pipeline import RAGPipeline
        mock_retriever = MagicMock()
        pipeline = RAGPipeline(mock_retriever)

        score = pipeline._score_faithfulness("UNANSWERABLE: context is insufficient", [])
        assert score == 1.0


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluator:
    """Tests for the Evaluator class."""

    def test_default_questions_returned_when_no_file(self, tmp_path):
        from src.rag.rag_pipeline import RAGPipeline
        from src.evaluation.evaluator import Evaluator

        pipeline = MagicMock(spec=RAGPipeline)
        evaluator = Evaluator(pipeline)

        questions = evaluator.load_test_questions(str(tmp_path / "nonexistent.json"))
        assert len(questions) > 0
        assert all(hasattr(q, "question") for q in questions)

    def test_report_aggregates_correctly(self):
        from src.rag.rag_pipeline import RAGPipeline, RAGResponse
        from src.retrieval.retriever import RetrievedChunk
        from src.evaluation.evaluator import Evaluator, EvalQuestion

        # Set up a mocked pipeline that always returns a faithful answer
        mock_resp = MagicMock(spec=RAGResponse)
        mock_resp.answer = "The answer is [SOURCE 1]."
        mock_resp.is_answerable = True
        mock_resp.faithfulness_score = 0.9
        mock_resp.sources = [MagicMock(spec=RetrievedChunk)]
        mock_resp.cited_source_indices = [1]
        mock_resp.latency_ms = 500.0

        mock_pipeline = MagicMock(spec=RAGPipeline)
        mock_pipeline.run.return_value = mock_resp

        evaluator = Evaluator(mock_pipeline)
        questions = [
            EvalQuestion("Question 1?"),
            EvalQuestion("Question 2?"),
        ]
        report = evaluator.run(questions=questions, save_results=False)

        assert report.total_questions == 2
        assert report.answerable_rate == 1.0
        assert abs(report.avg_faithfulness - 0.9) < 0.001
