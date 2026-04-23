from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .ask import add_ask_parser
from .ingest import add_ingest_parser
from .retrieval import add_retrieval_parsers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper-rag")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_ingest_parser(subparsers)
    add_ask_parser(subparsers)
    add_retrieval_parsers(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler:
        return handler(args)
    parser.error(f"Unknown command {args.command}")
    return 2
