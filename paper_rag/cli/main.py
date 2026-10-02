from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paper_rag.cli.acquire import add_acquire_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper-rag")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="项目根目录")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_acquire_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    return args.handler(args)

