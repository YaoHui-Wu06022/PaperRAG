from __future__ import annotations

import argparse
import sys
import json

from paper_rag.cli.library import main as library_main


def build_parser() -> argparse.ArgumentParser:
    from paper_rag.cli.library import parser

    return parser()


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    # Global options may precede commands. Normalize only actual command tokens.
    position = 0
    if effective_argv[:1] == ["--project-root"]:
        position = 2
    if position < len(effective_argv):
        command = effective_argv[position]
        if command in {"ask", "chat", "plan"}:
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "error",
                        "data": None,
                        "warnings": [
                            {
                                "code": "host_agent_required",
                                "message": "Use paper-research with search/read; answer generation belongs to the host Agent.",
                            }
                        ],
                    }
                )
            )
            return 2
        if command == "status":
            effective_argv[position] = "doctor"
        if command == "index" and position + 1 < len(effective_argv) and effective_argv[position + 1] in {"inspect", "archive", "restore", "retire"}:
            effective_argv[position : position + 2] = ["index-" + effective_argv[position + 1]]
        if command in {"papers", "wiki"} and position + 1 < len(effective_argv):
            effective_argv[position : position + 2] = [
                command + "-" + effective_argv[position + 1]
            ]
    return library_main(effective_argv)
