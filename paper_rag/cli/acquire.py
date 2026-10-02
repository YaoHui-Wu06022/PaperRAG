from __future__ import annotations

import argparse
import json

from paper_rag.acquisition.arxiv import download_arxiv_inputs, preview_arxiv_inputs
from paper_rag.config import Settings


def add_acquire_parser(subparsers: argparse._SubParsersAction) -> None:
    acquire = subparsers.add_parser("acquire", help="获取原始论文资料")
    acquire_subparsers = acquire.add_subparsers(dest="acquire_source", required=True)
    arxiv = acquire_subparsers.add_parser("arxiv", help="从 ArXiv 获取最新论文 PDF")
    arxiv.add_argument("inputs", nargs="+", help="ArXiv ID 或 abs/pdf URL")
    arxiv.add_argument("--dry-run", action="store_true", help="只预览，不下载")
    arxiv.add_argument("--json", action="store_true", help="输出 JSON")
    arxiv.set_defaults(handler=handle_arxiv)


def handle_arxiv(args: argparse.Namespace) -> int:
    settings = Settings.load(args.project_root)
    if args.dry_run:
        payload = {"status": "preview", "items": preview_arxiv_inputs(settings, args.inputs)}
        print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else format_preview(payload))
        return 0 if not any(item.get("status") == "failed" for item in payload["items"]) else 1
    result = download_arxiv_inputs(settings, args.inputs, reporter=None)
    payload = result.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else format_result(payload))
    return 1 if result.failed else 0


def format_preview(payload: dict) -> str:
    lines = ["ArXiv 获取预览："]
    for item in payload["items"]:
        suffix = f"：{item['error']}" if item.get("error") else ""
        lines.append(f"- {item.get('input')}: {item.get('status')}{suffix}")
    return "\n".join(lines)


def format_result(payload: dict) -> str:
    lines = [
        f"已下载 {payload['downloaded']}，跳过 {payload['skipped']}，失败 {payload['failed']}。"
    ]
    for item in payload["items"]:
        suffix = f"：{item['error']}" if item.get("error") else ""
        lines.append(f"- {item.get('input')}: {item.get('status')}{suffix}")
    return "\n".join(lines)

