"""
cli.py — Command-line interface for the Legal RAG Assistant.

Useful for batch indexing and testing without launching the Streamlit UI.

Usage:
  python cli.py index --dir data/raw/
  python cli.py index --file my_contract.pdf
  python cli.py query "What is the termination clause?"
  python cli.py evaluate
  python cli.py list-docs
"""

import argparse
import sys
from pathlib import Path


def cmd_index(args: argparse.Namespace) -> None:
    from src.embedding.embedder import Embedder, VectorStore
    from src.embedding.indexer import index_file, index_directory

    embedder     = Embedder()
    vector_store = VectorStore()

    def progress(msg: str) -> None:
        print(msg)

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"❌ File not found: {path}", file=sys.stderr)
            sys.exit(1)
        chunks = index_file(path, embedder, vector_store, progress)
        print(f"\n✅ Indexed {len(chunks)} chunks from '{path.name}'")
    elif args.dir:
        path = Path(args.dir)
        if not path.is_dir():
            print(f"❌ Directory not found: {path}", file=sys.stderr)
            sys.exit(1)
        total = index_directory(path, embedder, vector_store, progress)
        print(f"\n✅ Total chunks indexed: {total}")
    else:
        print("Specify --file or --dir", file=sys.stderr)
        sys.exit(1)


def cmd_query(args: argparse.Namespace) -> None:
    from src.embedding.embedder import Embedder, VectorStore
    from src.retrieval.retriever import Retriever
    from src.rag.rag_pipeline import RAGPipeline

    embedder     = Embedder()
    vector_store = VectorStore()
    retriever    = Retriever(embedder, vector_store)
    pipeline     = RAGPipeline(retriever)

    resp = pipeline.run(args.question)
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(resp.answer)
    print("\n" + "-" * 60)
    print(f"Answerable: {resp.is_answerable}")
    print(f"Faithfulness: {resp.faithfulness_score:.3f}")
    print(f"Sources retrieved: {len(resp.sources)}")
    print(f"Latency: {resp.latency_ms:.0f} ms")


def cmd_list_docs(args: argparse.Namespace) -> None:
    from src.embedding.embedder import VectorStore
    vector_store = VectorStore()
    docs = vector_store.list_documents()
    if not docs:
        print("No documents indexed.")
        return
    print(f"\n{len(docs)} document(s) indexed:")
    for d in docs:
        print(f"  • {d['filename']} (id={d['doc_id']})")
    print(f"\nTotal chunks: {vector_store.count}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from src.embedding.embedder import Embedder, VectorStore
    from src.retrieval.retriever import Retriever
    from src.rag.rag_pipeline import RAGPipeline
    from src.evaluation.evaluator import Evaluator

    embedder     = Embedder()
    vector_store = VectorStore()
    retriever    = Retriever(embedder, vector_store)
    pipeline     = RAGPipeline(retriever)
    evaluator    = Evaluator(pipeline)

    report = evaluator.run(save_results=True)
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Total questions:  {report.total_questions}")
    print(f"Answerable rate:  {report.answerable_rate:.1%}")
    print(f"Avg faithfulness: {report.avg_faithfulness:.3f}")
    print(f"Avg latency:      {report.avg_latency_ms:.0f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="legal-rag",
        description="Legal RAG Assistant CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index
    idx_parser = subparsers.add_parser("index", help="Index documents")
    idx_parser.add_argument("--file", help="Path to a single file")
    idx_parser.add_argument("--dir",  help="Path to a directory of files")
    idx_parser.set_defaults(func=cmd_index)

    # query
    q_parser = subparsers.add_parser("query", help="Ask a question")
    q_parser.add_argument("question", help="The legal question to answer")
    q_parser.set_defaults(func=cmd_query)

    # list-docs
    ld_parser = subparsers.add_parser("list-docs", help="List indexed documents")
    ld_parser.set_defaults(func=cmd_list_docs)

    # evaluate
    ev_parser = subparsers.add_parser("evaluate", help="Run evaluation benchmark")
    ev_parser.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
