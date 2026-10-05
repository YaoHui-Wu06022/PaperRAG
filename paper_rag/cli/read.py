"""论文全文和 Chunk 读取 CLI。"""

from __future__ import annotations

import argparse
import json

from paper_rag.config import Settings


def add_read_parser(subparsers: argparse._SubParsersAction) -> None:
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


def handle_read_chunk(args: argparse.Namespace) -> int:
    from paper_rag.catalog.service import CatalogIndexNotReady, get_chunk

    try:
        item = get_chunk(Settings.load(args.project_root), args.chunk_id)
    except CatalogIndexNotReady:
        payload = {"status": "catalog_not_ready", "data": {"chunk": None}}
    else:
        payload = {"status": "ok" if item else "not_found", "data": {"chunk": item}}
    _print(payload, args.json)
    return 0 if payload["status"] == "ok" else 1


def handle_read_fulltext(args: argparse.Namespace) -> int:
    from paper_rag.reading.service import read_fulltext

    payload = read_fulltext(Settings.load(args.project_root), args.paper_id, args.offset, args.limit)
    _print(payload, args.json)
    return 0 if payload.get("status") == "ok" else 1


def _print(payload: dict, as_json: bool) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2) if as_json else json.dumps(payload, ensure_ascii=False))


__all__ = ["add_read_parser", "handle_read_chunk", "handle_read_fulltext"]
