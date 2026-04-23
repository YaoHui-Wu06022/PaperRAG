from __future__ import annotations

import argparse

from ..config import Settings
from ..retrieval.answer import run_ask


def add_ask_parser(subparsers: argparse._SubParsersAction) -> None:
    ask = subparsers.add_parser("ask", help="Answer a question from plan evidence")
    ask.add_argument("query", help="Question to answer")
    ask.set_defaults(handler=handle_ask)


def handle_ask(args: argparse.Namespace) -> int:
    settings = Settings.load(args.project_root)
    result = run_ask(settings, args.query)
    print(result.answer)
    if result.provenance:
        print()
        print(f"来源: {'; '.join(result.provenance)}")
    for warning in result.warnings:
        print(f"提示: {warning}")
    return 0
