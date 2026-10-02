"""MinerU ArXiv 入库 CLI。"""

from __future__ import annotations

import argparse
import json

from paper_rag.config import Settings


def add_ingest_parser(subparsers: argparse._SubParsersAction) -> None:
    ingest = subparsers.add_parser("ingest", help="使用 MinerU 解析已下载论文")
    ingest_subparsers = ingest.add_subparsers(dest="ingest_source", required=True)
    arxiv = ingest_subparsers.add_parser("arxiv", help="解析 ArXiv 本地 PDF")
    arxiv.add_argument("inputs", nargs="+", help="ArXiv ID 或 abs/pdf URL")
    arxiv.add_argument("--dry-run", action="store_true", help="只预览，不提交 MinerU")
    arxiv.add_argument("--json", action="store_true", help="输出 JSON")
    arxiv.set_defaults(handler=handle_arxiv_ingest)


def handle_arxiv_ingest(args: argparse.Namespace) -> int:
    # 延迟导入网络客户端，让 CLI 的其他命令保持轻量。
    from paper_rag.ingest.mineru import ingest_arxiv_inputs, preview_arxiv_ingest

    settings = Settings.load(args.project_root)
    if args.dry_run:
        payload = {"status": "preview", "items": preview_arxiv_ingest(settings, args.inputs)}
        print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else format_preview(payload))
        return 0 if not any(item.get("status") in {"failed", "missing_source"} for item in payload["items"]) else 1
    result = ingest_arxiv_inputs(settings, args.inputs, reporter=None)
    payload = result.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else format_result(payload))
    return 1 if result.failed else 0


def format_preview(payload: dict) -> str:
    lines = ["MinerU ArXiv 入库预览："]
    for item in payload["items"]:
        suffix = f"：{item['error']}" if item.get("error") else ""
        lines.append(f"- {item.get('input')}: {item.get('status')}{suffix}")
    return "\n".join(lines)


def format_result(payload: dict) -> str:
    lines = [
        f"已解析 {payload['succeeded']}，跳过 {payload['skipped']}，失败 {payload['failed']}。"
    ]
    for item in payload["items"]:
        suffix = f"：{item['error']}" if item.get("error") else ""
        lines.append(f"- {item.get('input')}: {item.get('status')}{suffix}")
    return "\n".join(lines)

