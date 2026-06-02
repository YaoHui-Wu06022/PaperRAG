from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paper_rag.cli.ask import add_ask_parser, add_chat_parser
from paper_rag.cli.ingest import add_ingest_parser
from paper_rag.cli.retrieval import add_retrieval_parsers
from paper_rag.retrieval.probe import add_probe_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper-rag")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="项目根目录，默认当前目录")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_ingest_parser(subparsers)
    add_retrieval_parsers(subparsers)
    add_ask_parser(subparsers)
    add_chat_parser(subparsers)
    add_probe_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler:
        return handler(args)
    parser.error(f"未知命令：{args.command}")
    return 2
