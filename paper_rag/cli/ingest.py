from __future__ import annotations

import argparse

from paper_rag.config import Settings
from paper_rag.ingest.pipeline import IngestSummary, run_ingest


def add_ingest_parser(subparsers: argparse._SubParsersAction) -> None:
    ingest = subparsers.add_parser("ingest", help="将 data/pdf 同步入 data/paper_data")
    ingest.add_argument("--refresh", action="store_true", help="刷新所有 active PDF 的元数据")
    ingest.set_defaults(handler=handle_ingest)


def handle_ingest(args: argparse.Namespace) -> int:
    settings = Settings.load(args.project_root)
    summary = run_ingest(
        settings,
        refresh_metadata=args.refresh,
        reporter=print,
    )
    print_summary(summary)
    return 1 if summary.errors else 0


def print_summary(summary: IngestSummary) -> None:
    def section(title: str, rows: list[str]) -> None:
        if not rows:
            return
        print(f"\n{title}:")
        for row in rows:
            print(f"  - {row}")

    section("已处理", summary.processed)
    section("复用的 MinerU 输出", summary.reused)
    section("从归档恢复的 MinerU 输出", summary.restored)
    section("已重命名的 MinerU 输出", summary.renamed_mineru_output)
    section("已删除", summary.deleted)
    section("重复文件", summary.duplicates)
    section("元数据未解析", summary.unresolved)
    section("错误", summary.errors)
    if not any(vars(summary).values()):
        print("没有变化。")
