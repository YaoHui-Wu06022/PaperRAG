from __future__ import annotations

import argparse
import json

from paper_rag.answer import run_ask
from paper_rag.config import Settings


def add_ask_parser(subparsers: argparse._SubParsersAction) -> None:
    ask = subparsers.add_parser("ask", help="基于检索证据回答论文问题")
    ask.add_argument("query", nargs="+", help="要回答的问题")
    ask.add_argument("--debug", action="store_true", help="在 evidence 中包含 planner 调试信息")
    ask.add_argument("--json", action="store_true", help="以 JSON 格式输出回答 payload")
    ask.set_defaults(handler=handle_ask)


def handle_ask(args: argparse.Namespace) -> int:
    settings = Settings.load(args.project_root)
    query = " ".join(args.query).strip()
    payload = run_ask(settings, query, debug=args.debug)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload["answer"])
    return 0
