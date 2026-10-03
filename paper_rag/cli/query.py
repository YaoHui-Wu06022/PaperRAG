"""Paper RAG 统一查询 CLI。"""

from __future__ import annotations

import argparse
import json

from paper_rag.config import Settings


def add_query_parser(subparsers: argparse._SubParsersAction) -> None:
    """注册只读自然语言查询命令。"""

    query = subparsers.add_parser("query", help="查询本地论文知识库")
    query.add_argument("query", help="自然语言查询")
    query.add_argument("--paper-id", action="append", default=[], help="限定论文 ID，可重复")
    query.add_argument("--json", action="store_true", help="输出 JSON")
    query.set_defaults(handler=handle_query)


def handle_query(args: argparse.Namespace) -> int:
    """执行只读查询；网络组件在真正执行时延迟导入。"""

    from paper_rag.query.service import query_papers

    result = query_papers(Settings.load(args.project_root), args.query, args.paper_id)
    payload = result.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else format_result(payload))
    return 0


def format_result(payload: dict) -> str:
    intent = payload["intent"]
    lines = [f"查询意图：{intent['intent']}（来源：{intent['provider']}）"]
    if intent.get("needs_clarification"):
        lines.append("需要补充信息后才能继续查询。")
    if payload.get("message"):
        lines.append(str(payload["message"]))
    lines.append(f"匹配论文：{len(payload.get('items', []))}")
    return "\n".join(lines)


__all__ = ["add_query_parser", "handle_query"]
