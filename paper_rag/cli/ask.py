from __future__ import annotations

import argparse
import json

from ..config import Settings
from ..answer import run_ask


def add_ask_parser(subparsers: argparse._SubParsersAction) -> None:
    ask = subparsers.add_parser("ask", help="Answer a question from plan evidence")
    ask.add_argument("query", help="Question to answer")
    ask.add_argument("--debug", action="store_true", help="Print raw plan JSON for debugging")
    ask.set_defaults(handler=handle_ask)


def handle_ask(args: argparse.Namespace) -> int:
    settings = Settings.load(args.project_root)
    result = run_ask(settings, args.query)
    print(result.answer)
    if result.provenance:
        print()
        print("证据:")
        for item in result.provenance:
            print(f"- {item}")
    for warning in result.warnings:
        print(f"提示: {warning}")
    if args.debug:
        print()
        print(json.dumps(result.plan, ensure_ascii=False, indent=2))
    return 0
