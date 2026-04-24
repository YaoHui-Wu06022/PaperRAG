from __future__ import annotations

import argparse
import json

from ..config import Settings
from ..retrieval.dense.service import run_index, run_search
from ..retrieval.planner import run_plan


def add_retrieval_parsers(subparsers: argparse._SubParsersAction) -> None:
    index = subparsers.add_parser("index", help="Build the Milvus vector index from paper_data chunks")
    index.add_argument("--quiet", action="store_true", help="Only print the final summary")
    index.set_defaults(handler=handle_index)

    search = subparsers.add_parser("search", help="Search indexed paper chunks")
    search.add_argument("query", help="Search query")
    search.add_argument("--top-k", type=int, default=5, help="Number of chunks to return")
    search.set_defaults(handler=handle_search)

    plan = subparsers.add_parser("plan", help="Build a JSON evidence pack for a question")
    plan.add_argument("query", help="Question to plan evidence for")
    plan.set_defaults(handler=handle_plan)


def handle_index(args: argparse.Namespace) -> int:
    settings = Settings.load(args.project_root)
    reporter = (lambda _: None) if args.quiet else print
    summary = run_index(settings, reporter=reporter)
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
    evidence_pack = run_plan(settings, args.query)
    print(json.dumps(evidence_pack, ensure_ascii=False, indent=2))
    return 0
