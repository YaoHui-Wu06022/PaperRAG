from __future__ import annotations

import argparse

from ..config import Settings
from ..dataprocess.ingest import IngestSummary, run_ingest


def add_ingest_parser(subparsers: argparse._SubParsersAction) -> None:
    ingest = subparsers.add_parser("ingest", help="Sync data/pdf into data/paper_data")
    ingest.add_argument("--refresh", action="store_true", help="Refresh metadata for all active PDFs")
    ingest.add_argument("--quiet", action="store_true", help="Only print the final summary")
    ingest.set_defaults(handler=handle_ingest)


def handle_ingest(args: argparse.Namespace) -> int:
    settings = Settings.load(args.project_root)
    reporter = None if args.quiet else print
    summary = run_ingest(
        settings,
        refresh_metadata=args.refresh,
        reporter=reporter,
    )
    print_summary(summary)
    return 1 if summary.has_errors() else 0


def print_summary(summary: IngestSummary) -> None:
    def section(title: str, rows: list[str]) -> None:
        if not rows:
            return
        print(f"\n{title}:")
        for row in rows:
            print(f"  - {row}")

    section("processed", summary.processed)
    section("reused mineru output", summary.reused)
    section("restored mineru output", summary.restored)
    section("renamed mineru output", summary.renamed_mineru_output)
    section("deleted", summary.deleted)
    section("duplicates", summary.duplicates)
    section("metadata unresolved", summary.unresolved)
    section("errors", summary.errors)
    if not any(vars(summary).values()):
        print("No changes.")
