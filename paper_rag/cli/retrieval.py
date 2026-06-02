from __future__ import annotations

import argparse
import json

from paper_rag.config import Settings
from paper_rag.retrieval.dense.service import run_index, run_search
from paper_rag.retrieval.plan import run_plan


def add_retrieval_parsers(subparsers: argparse._SubParsersAction) -> None:
    index = subparsers.add_parser("index", help="基于 paper_data chunks 构建 Milvus 向量索引")
    index.set_defaults(handler=handle_index)

    search = subparsers.add_parser("search", help="检索已索引的论文 chunks")
    search.add_argument("query", help="检索问题")
    search.add_argument("--top-k", type=int, default=5, help="返回的 chunk 数量")
    search.set_defaults(handler=handle_search)

    plan = subparsers.add_parser("plan", help="通过 retrieval routers 规划一个论文问题")
    plan.add_argument("query", nargs="+", help="要规划的问题")
    plan.add_argument("--debug", action="store_true", help="包含 parser 和 retrieval 调试信息")
    plan.set_defaults(handler=handle_plan)


def handle_index(args: argparse.Namespace) -> int:
    settings = Settings.load(args.project_root)
    summary = run_index(settings, reporter=print)
    print(f"已将 {summary.chunk_count} 个 chunk 写入 {summary.collection_name}。")
    return 0


def handle_search(args: argparse.Namespace) -> int:
    settings = Settings.load(args.project_root)
    results = run_search(settings, args.query, top_k=args.top_k)
    if not results:
        print("没有检索结果。")
        return 0
    for index, result in enumerate(results, start=1):
        print(f"[{index}] 得分={result.score:.4f} {result.title}")
        print(f"    章节: {result.section_path_text or '-'}")
        print(f"    页码: {result.pages_text or '-'}")
        print(f"    chunk：{result.chunk_id}")
        print(f"    {result.snippet}")
    return 0


def handle_plan(args: argparse.Namespace) -> int:
    settings = Settings.load(args.project_root)
    query = " ".join(args.query).strip()
    payload = run_plan(settings, query, debug=args.debug)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0
