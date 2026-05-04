from __future__ import annotations

import argparse
import json

from ..answer import run_ask
from ..config import Settings


def add_ask_parser(subparsers: argparse._SubParsersAction) -> None:
    ask = subparsers.add_parser("ask", help="Answer a paper question with retrieval evidence")
    ask.add_argument("query", nargs="+", help="Question to answer")
    ask.add_argument("--debug", action="store_true", help="Include planner debug details in evidence")
    ask.add_argument("--json", action="store_true", help="Print answer payload as JSON")
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
