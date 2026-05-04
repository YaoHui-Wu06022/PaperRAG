"""metadata planner 的手动调试入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_project_root() -> Path:
    """向上寻找项目根目录，支持直接运行本文件。"""
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".env").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from paper_rag.config import Settings
from paper_rag.retrieval.domains.metadata.planner import plan_metadata
from paper_rag.retrieval.domains.metadata.router import build_metadata_decision
from paper_rag.retrieval.route import RouteDecision


DEFAULT_QUERIES = [
    "ResNet 和 Transformer 分别是哪一年发表的？",
]


def main() -> int:
    """解析命令行参数并输出 metadata planner evidence。"""
    parser = argparse.ArgumentParser(description="Probe metadata planner evidence outputs.")
    parser.add_argument("query", nargs="*", help="Optional single query. Defaults to the first DEFAULT_QUERIES item.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Project root containing .env")
    parser.add_argument("--debug", action="store_true", help="Include planner debug details")
    parser.add_argument("--show-route", action="store_true", help="Include the parsed RouteDecision summary")
    args = parser.parse_args()

    settings = Settings.load(args.project_root)
    query = " ".join(args.query).strip() if args.query else (DEFAULT_QUERIES[0] if DEFAULT_QUERIES else "")
    if not query:
        print(json.dumps({"error": "No query provided. Add an item to DEFAULT_QUERIES or pass one on the command line."}, ensure_ascii=False, indent=2))
        return 1

    warnings: list[str] = []
    try:
        route = build_metadata_decision(
            settings,
            RouteDecision(route="metadata", query=query, parse_status="ok"),
            warnings,
        )
        payload: dict[str, object] = plan_metadata(settings, route, warnings, debug=args.debug)
        if args.show_route:
            payload["route_summary"] = route_summary(route)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(json.dumps({"query": query, "error": str(exc), "warnings": warnings}, ensure_ascii=False, indent=2))
        return 1
    return 0


def route_summary(route: RouteDecision) -> dict[str, object]:
    """输出 metadata RouteDecision 的关键字段。"""
    return {
        "route": route.route,
        "intent": route.intent,
        "return_fields": route.return_fields,
        "paper_semantic": route.paper_semantic,
        "filters": route.filters,
        "paper_groups": route.paper_groups,
        "group_mode": route.group_mode,
        "parse_status": route.parse_status,
        "parser_error": route.parser_error,
        "parser_result": route.parser_result,
    }


if __name__ == "__main__":
    sys.exit(main())
