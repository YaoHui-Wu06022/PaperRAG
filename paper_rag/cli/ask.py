from __future__ import annotations

import argparse
import json
from typing import Any

from paper_rag.answer import run_ask
from paper_rag.config import Settings
from paper_rag.corpus.context import CorpusContext
from paper_rag.retrieval.plan import run_plan


EXIT_COMMANDS = {"exit", "quit", "退出"}


def add_ask_parser(subparsers: argparse._SubParsersAction) -> None:
    ask = subparsers.add_parser("ask", help="检索论文证据（最终回答由 Agent 生成）")
    ask.add_argument("query", nargs="+", help="查询问题")
    ask.add_argument("--debug", action="store_true", help="输出 planner/retrieval payload")
    ask.add_argument("--evidence", action="store_true", help="以 JSON 输出证据")
    ask.set_defaults(handler=handle_ask)


def add_chat_parser(subparsers: argparse._SubParsersAction) -> None:
    chat = subparsers.add_parser("chat", help="连续执行论文检索问题")
    chat.add_argument("--mode", choices=["ask", "plan"], default="ask")
    chat.add_argument("--debug", action="store_true", help="输出 planner/retrieval payload")
    chat.add_argument("--evidence", action="store_true", help="以 JSON 输出证据")
    chat.set_defaults(handler=handle_chat, chat_parser=chat)


def handle_ask(args: argparse.Namespace) -> int:
    settings = Settings.load(args.project_root)
    query = " ".join(args.query).strip()
    payload = run_ask(settings, query, debug=args.debug)
    print_ask_payload(payload, debug=args.debug, evidence=args.evidence)
    return 0


def handle_chat(args: argparse.Namespace) -> int:
    if args.mode == "plan" and args.evidence:
        args.chat_parser.error("--evidence 仅适用于 ask 模式")
    settings = Settings.load(args.project_root)
    corpus = CorpusContext(settings)
    print("已进入连续检索模式。输入 exit 或 quit 退出。")
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


def print_ask_payload(payload: dict[str, Any], *, debug: bool, evidence: bool) -> None:
    """输出检索 evidence，不生成最终自然语言答案。"""
    if debug or evidence:
        print(json.dumps({
            "answer_mode": payload.get("answer_mode", "evidence"),
            "evidence": payload.get("evidence"),
        }, ensure_ascii=False, indent=2))
        return
    print(json.dumps(payload.get("evidence") or {}, ensure_ascii=False, indent=2))


def print_evidence_sources(evidence: Any) -> None:
    sources = list(dict.fromkeys(evidence_sources(evidence)))
    print("\n证据来源：")
    if not sources:
        print("没有可展示的证据来源。")
        return
    for index, source in enumerate(sources, start=1):
        print(f"[{index}] {source}")


def evidence_sources(evidence: Any) -> list[str]:
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
    section = join_values(context.get("section_path")) or "-"
    pages = join_values(context.get("pages")) or "-"
    chunk_id = str(context.get("chunk_id") or "-")
    return f"章节: {section} | 页码: {pages} | chunk: {chunk_id}"


def reference_sources(results: dict[str, Any]) -> list[str]:
    sources = []
    for edge in results.get("edges") or []:
        source = str(edge.get("source") or "未知论文")
        obj = str(edge.get("object") or "未知论文")
        location = format_location(edge)
        sources.append(f"{source} -> {obj}{location}")
    return sources or [str(paper) for paper in results.get("papers") or []]


def metadata_sources(results: dict[str, Any]) -> list[str]:
    items = list(results.get("items") or results.get("actual") or [])
    for group in results.get("groups") or []:
        items.extend(group.get("items") or [])
    return [format_metadata_source(item) for item in items]


def format_metadata_source(item: dict[str, Any]) -> str:
    title = str(item.get("title") or "未知论文")
    values = item.get("values")
    return title if not values else f"{title} | {json.dumps(values, ensure_ascii=False, separators=(',', ':'))}"


def format_location(edge: dict[str, Any]) -> str:
    parts = []
    if edge.get("page"):
        parts.append(f"页码: {edge['page']}")
    if edge.get("block"):
        parts.append(f"block: {edge['block']}")
    return f" | {' | '.join(parts)}" if parts else ""


def join_values(value: Any) -> str:
    return " > ".join(str(item) for item in value) if isinstance(value, list) else str(value or "")
