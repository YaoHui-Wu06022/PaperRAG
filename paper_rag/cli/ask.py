from __future__ import annotations

import argparse
import json
from typing import Any

from paper_rag.answer import run_ask
from paper_rag.config import Settings
from paper_rag.corpus.context import CorpusContext
from paper_rag.retrieval.plan import run_plan


EVIDENCE_SOURCE_LIMIT = 5
SNIPPET_LIMIT = 180
EXIT_COMMANDS = {"exit", "quit", "退出"}


def add_ask_parser(subparsers: argparse._SubParsersAction) -> None:
    ask = subparsers.add_parser("ask", help="基于检索证据回答论文问题")
    ask.add_argument("query", nargs="+", help="要回答的问题")
    ask.add_argument("--debug", action="store_true", help="输出完整 payload，用于排查 planner 和 retrieval")
    ask.add_argument("--evidence", action="store_true", help="在答案后附带最多 5 条证据来源")
    ask.set_defaults(handler=handle_ask)


def add_chat_parser(subparsers: argparse._SubParsersAction) -> None:
    chat = subparsers.add_parser("chat", help="在同一进程中连续执行论文问答或检索规划")
    chat.add_argument("--mode", choices=["ask", "plan"], default="ask", help="连续执行 ask 或 plan，默认 ask")
    chat.add_argument("--debug", action="store_true", help="输出完整 payload，用于排查 planner 和 retrieval")
    chat.add_argument("--evidence", action="store_true", help="在 ask 答案后附带最多 5 条证据来源")
    chat.set_defaults(handler=handle_chat, chat_parser=chat)


def handle_ask(args: argparse.Namespace) -> int:
    settings = Settings.load(args.project_root)
    query = " ".join(args.query).strip()
    payload = run_ask(settings, query, debug=args.debug)
    print_ask_payload(payload, debug=args.debug, evidence=args.evidence)
    return 0


def handle_chat(args: argparse.Namespace) -> int:
    """连续执行相互独立的问题，并复用一次会话中的本地语料。"""
    if args.mode == "plan" and args.evidence:
        args.chat_parser.error("--evidence 仅适用于 ask 模式")
    settings = Settings.load(args.project_root)
    corpus = CorpusContext(settings)
    print("已进入连续模式。输入 exit、quit 或 退出结束。")
    while True:
        try:
            query = input("问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not query:
            continue
        if query.casefold() in EXIT_COMMANDS:
            return 0
        try:
            if args.mode == "plan":
                payload = run_plan(settings, query, debug=args.debug, corpus=corpus)
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                payload = run_ask(settings, query, debug=args.debug, corpus=corpus)
                print_ask_payload(payload, debug=args.debug, evidence=args.evidence)
        except KeyboardInterrupt:
            print()
            return 0
        except Exception as exc:
            print(f"本轮执行失败：{exc}")
    return 0


def print_ask_payload(payload: dict[str, Any], *, debug: bool, evidence: bool) -> None:
    """按 ask CLI 参数输出答案、来源或完整排查 payload。"""
    if debug:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(payload["answer"])
    if evidence:
        print_evidence_sources(payload.get("evidence"))


def print_evidence_sources(evidence: Any) -> None:
    """打印适合人工快速核对的证据来源摘要。"""
    sources = list(dict.fromkeys(evidence_sources(evidence)))[:EVIDENCE_SOURCE_LIMIT]
    print("\n证据来源：")
    if not sources:
        print("没有可展示的证据来源。")
        return
    for index, source in enumerate(sources, start=1):
        print(f"[{index}] {source}")


def evidence_sources(evidence: Any) -> list[str]:
    """按 route 提取最多用于展示的来源，不改变内部 evidence。"""
    if not isinstance(evidence, dict):
        return []
    results = evidence.get("results")
    if not isinstance(results, dict):
        return []
    route = evidence.get("route")
    if route == "content":
        return [format_content_source(context) for context in results.get("contexts") or []]
    if route == "reference":
        return reference_sources(results)
    if route == "metadata":
        return metadata_sources(results)
    return []


def format_content_source(context: dict[str, Any]) -> str:
    """格式化正文 chunk 来源，并附短摘录便于判断召回质量。"""
    title = str(context.get("title") or "未知论文")
    section = join_values(context.get("section_path")) or "-"
    pages = join_values(context.get("pages")) or "-"
    chunk_id = str(context.get("chunk_id") or "-")
    snippet = shorten_text(context.get("text"))
    source = f"{title} | 章节: {section} | 页码: {pages} | chunk: {chunk_id}"
    return f"{source}\n    摘录: {snippet}" if snippet else source


def reference_sources(results: dict[str, Any]) -> list[str]:
    """优先展示引用边；缺少边时退回命中论文标题。"""
    sources = []
    for edge in results.get("edges") or []:
        source = str(edge.get("source") or "未知论文")
        obj = str(edge.get("object") or "未知论文")
        location = format_location(edge)
        sources.append(f"{source} -> {obj}{location}")
    if sources:
        return sources
    return [str(paper) for paper in results.get("papers") or []]


def metadata_sources(results: dict[str, Any]) -> list[str]:
    """展示 metadata 命中的论文及本地字段。"""
    items = list(results.get("items") or results.get("actual") or [])
    for group in results.get("groups") or []:
        items.extend(group.get("items") or [])
    return [format_metadata_source(item) for item in items]


def format_metadata_source(item: dict[str, Any]) -> str:
    title = str(item.get("title") or "未知论文")
    values = item.get("values")
    if not values:
        return title
    return f"{title} | {json.dumps(values, ensure_ascii=False, separators=(',', ':'))}"


def format_location(edge: dict[str, Any]) -> str:
    parts = []
    if edge.get("page"):
        parts.append(f"页码: {edge['page']}")
    if edge.get("block"):
        parts.append(f"block: {edge['block']}")
    return f" | {' | '.join(parts)}" if parts else ""


def join_values(value: Any) -> str:
    if isinstance(value, list):
        return " > ".join(str(item) for item in value)
    return str(value or "")


def shorten_text(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= SNIPPET_LIMIT:
        return text
    return text[:SNIPPET_LIMIT].rstrip() + "..."
