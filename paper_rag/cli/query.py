"""论文元数据搜索和唯一正文 RAG CLI。"""

from __future__ import annotations

import argparse
import json

from paper_rag.config import Settings


def add_query_parser(subparsers: argparse._SubParsersAction) -> None:
    """注册元数据搜索和正文检索命令。"""

    search = subparsers.add_parser("search", help="搜索本地论文元数据")
    search.add_argument("query")
    search.add_argument("--author")
    search.add_argument("--category")
    search.add_argument("--year")
    search.add_argument("--year-from")
    search.add_argument("--year-to")
    search.add_argument("--state")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--json", action="store_true")
    search.set_defaults(handler=handle_search)

    retrieve = subparsers.add_parser("retrieve", help="检索论文正文证据")
    retrieve.add_argument("query")
    retrieve.add_argument("--paper-id", action="append", default=[])
    retrieve.add_argument("--author")
    retrieve.add_argument("--category")
    retrieve.add_argument("--year")
    retrieve.add_argument("--year-from")
    retrieve.add_argument("--year-to")
    retrieve.add_argument("--state")
    retrieve.add_argument("--task", choices=("auto", "fact", "reason", "summary", "comparison"), default="auto")
    retrieve.add_argument("--limit", type=int, default=8)
    retrieve.add_argument("--max-chars", type=int, default=24000)
    retrieve.add_argument("--mode", choices=("lexical", "semantic", "hybrid"), default="hybrid")
    retrieve.add_argument("--json", action="store_true")
    retrieve.set_defaults(handler=handle_retrieve)


def handle_search(args: argparse.Namespace) -> int:
    from paper_rag.llamaindex.service import search

    filters = {key: value for key, value in {"author": args.author, "category": args.category, "year": args.year, "year_from": args.year_from, "year_to": args.year_to, "state": args.state}.items() if value}
    payload = search(Settings.load(args.project_root), args.query, filters=filters or None, limit=args.limit)
    _print(payload, args.json)
    return 0 if payload.get("status") == "ok" else 1


def handle_retrieve(args: argparse.Namespace) -> int:
    from paper_rag.llamaindex.service import retrieve

    filters = {key: value for key, value in {"author": args.author, "category": args.category, "year": args.year, "year_from": args.year_from, "year_to": args.year_to, "state": args.state}.items() if value}
    payload = retrieve(Settings.load(args.project_root), args.query, args.paper_id or None, filters=filters or None, task=args.task, limit=args.limit, max_chars=args.max_chars, mode=args.mode)
    _print(payload, args.json)
    return 0 if payload.get("status") in {"ok", "embedding_unavailable"} else 1


def _print(payload: dict, as_json: bool) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


__all__ = ["add_query_parser", "handle_retrieve", "handle_search"]
