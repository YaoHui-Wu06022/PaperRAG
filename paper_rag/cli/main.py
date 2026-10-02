from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper-rag")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="项目根目录")
    subparsers = parser.add_subparsers(dest="command", required=True)
    # CLI 解析阶段不加载 MCP 和网络组件，保持启动轻量。
    from paper_rag.cli.acquire import add_acquire_parser
    from paper_rag.cli.ingest import add_ingest_parser

    add_acquire_parser(subparsers)
    add_ingest_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    return args.handler(args)

