from __future__ import annotations

import argparse
import json

from paper_rag.config import Settings


def add_read_parser(subparsers: argparse._SubParsersAction) -> None:
    chunks = subparsers.add_parser("search-chunks", help="检索 MinerU 正文 Chunk")
    chunks.add_argument("query")
    chunks.add_argument("--paper-id", action="append", default=[])
    chunks.add_argument("--limit", type=int, default=8)
    chunks.add_argument("--mode", choices=("lexical", "semantic", "hybrid"), default="hybrid")
    chunks.add_argument("--json", action="store_true")
    chunks.set_defaults(handler=handle_search_chunks)

    read = subparsers.add_parser("read", help="读取论文正文或 Chunk")
    commands = read.add_subparsers(dest="read_command", required=True)
    chunk = commands.add_parser("chunk")
    chunk.add_argument("chunk_id")
    chunk.add_argument("--json", action="store_true")
    chunk.set_defaults(handler=handle_read_chunk)
    full = commands.add_parser("fulltext")
    full.add_argument("paper_id")
    full.add_argument("--offset", type=int, default=0)
    full.add_argument("--limit", type=int, default=12000)
    full.add_argument("--json", action="store_true")
    full.set_defaults(handler=handle_read_fulltext)


def handle_search_chunks(args: argparse.Namespace) -> int:
    from paper_rag.semantic.service import hybrid_search
    payload = hybrid_search(Settings.load(args.project_root), args.query, args.paper_id or None, args.limit, args.mode)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else json.dumps(payload, ensure_ascii=False))
    return 0 if payload.get("status") == "ok" else 1


def handle_read_chunk(args: argparse.Namespace) -> int:
    from paper_rag.reading.service import read_chunk
    payload = read_chunk(Settings.load(args.project_root), args.chunk_id)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else json.dumps(payload, ensure_ascii=False))
    return 0 if payload.get("status") == "ok" else 1


def handle_read_fulltext(args: argparse.Namespace) -> int:
    from paper_rag.reading.service import read_fulltext
    payload = read_fulltext(Settings.load(args.project_root), args.paper_id, args.offset, args.limit)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else json.dumps(payload, ensure_ascii=False))
    return 0 if payload.get("status") == "ok" else 1

__all__ = ["add_read_parser"]
