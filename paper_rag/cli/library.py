from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paper_rag.library.common import LibraryError
from paper_rag.library.contracts import (
    CitationRequest,
    DoctorRequest,
    EvalRequest,
    IndexRequest,
    IngestRequest,
    PapersRequest,
    ReadRequest,
    SearchRequest,
    WikiApply,
    WikiRequest,
)
from paper_rag.library.service import run
from paper_rag.library.settings import LibrarySettings

CONTRACTS = {
    "doctor": DoctorRequest,
    "ingest": IngestRequest,
    "index": IndexRequest,
    "index-inspect": IndexRequest,
    "index-archive": IndexRequest,
    "index-restore": IndexRequest,
    "index-retire": IndexRequest,
    "eval": EvalRequest,
    "papers-find": PapersRequest,
    "papers-count": PapersRequest,
    "papers-get": PapersRequest,
    "search": SearchRequest,
    "read": ReadRequest,
    "citations": CitationRequest,
    "wiki-search": WikiRequest,
    "wiki-read": WikiRequest,
    "wiki-prepare": WikiRequest,
    "wiki-lint": WikiRequest,
    "wiki-recover": WikiRequest,
    "wiki-apply": WikiApply,
}
COMMANDS = {
    "index-inspect": "index_inspect",
    "index-archive": "index",
    "index-restore": "index",
    "index-retire": "index",
    "papers-find": "papers_find",
    "papers-count": "papers_count",
    "papers-get": "papers_get",
    "wiki-search": "wiki_search",
    "wiki-read": "wiki_read",
    "wiki-prepare": "wiki_prepare",
    "wiki-lint": "wiki_lint",
    "wiki-recover": "wiki_recover",
    "wiki-apply": "wiki_apply",
}


def parser():
    p = argparse.ArgumentParser(prog="paper-rag")
    p.add_argument("--project-root", default=".", type=Path)
    p.add_argument("command", choices=sorted(CONTRACTS))
    p.add_argument(
        "--request", default="-", help="UTF-8 JSON request path; '-' reads stdin"
    )
    p.add_argument("--json", action="store_true", dest="machine")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        payload = json.loads(
            sys.stdin.read()
            if args.request == "-"
            else Path(args.request).read_text(encoding="utf-8-sig")
        )
        request = CONTRACTS[args.command].model_validate(payload)
        result = run(
            LibrarySettings.load(args.project_root),
            COMMANDS.get(args.command, args.command),
            request,
        )
        print(
            json.dumps(result, ensure_ascii=False, indent=None if args.machine else 2)
        )
        return 0
    except (OSError, json.JSONDecodeError, LibraryError, ValueError) as exc:
        data = {
            "schema_version": 1,
            "status": "error",
            "data": None,
            "warnings": [
                {"code": getattr(exc, "code", "invalid_request"), "message": str(exc)}
            ],
        }
        print(json.dumps(data, ensure_ascii=False))
        return 2
    except Exception as exc:
        # Third-party exception messages can contain endpoints or credentials.
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "error",
                    "data": None,
                    "warnings": [
                        {"code": "dependency_failure", "message": type(exc).__name__}
                    ],
                }
            )
        )
        return 2
