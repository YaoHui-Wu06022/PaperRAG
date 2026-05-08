from __future__ import annotations

import argparse
import json

from paper_rag.config import Settings
from paper_rag.retrieval.dense.service import run_index, run_search
from paper_rag.retrieval.plan import run_plan


def add_retrieval_parsers(subparsers: argparse._SubParsersAction) -> None:
    index = subparsers.add_parser("index", help="Build the Milvus vector index from paper_data chunks")
    index.set_defaults(handler=handle_index)

    search = subparsers.add_parser("search", help="Search indexed paper chunks")
    search.add_argument("query", help="Search query")
    search.add_argument("--top-k", type=int, default=5, help="Number of chunks to return")
    search.set_defaults(handler=handle_search)

    plan = subparsers.add_parser("plan", help="Plan a paper question through the retrieval routers")
    plan.add_argument("query", nargs="+", help="Question to plan")
    plan.add_argument("--debug", action="store_true", help="Include parser and retrieval debug details")
    plan.set_defaults(handler=handle_plan)


def handle_index(args: argparse.Namespace) -> int:
    settings = Settings.load(args.project_root)
    summary = run_index(settings, reporter=print)
    print(f"Indexed {summary.chunk_count} chunk(s) into {summary.collection_name}.")
    return 0


def handle_search(args: argparse.Namespace) -> int:
    settings = Settings.load(args.project_root)
    results = run_search(settings, args.query, top_k=args.top_k)
    if not results:
        print("No results.")
        return 0
    for index, result in enumerate(results, start=1):
        print(f"[{index}] score={result.score:.4f} {result.title}")
        print(f"    section: {result.section_path_text or '-'}")
        print(f"    pages: {result.pages_text or '-'}")
        print(f"    chunk: {result.chunk_id}")
        print(f"    {result.snippet}")
    return 0


def handle_plan(args: argparse.Namespace) -> int:
    settings = Settings.load(args.project_root)
    query = " ".join(args.query).strip()
    payload = run_plan(settings, query, debug=args.debug)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0
