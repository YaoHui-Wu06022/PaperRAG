"""reference planner 的手动调试入口。"""

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
from paper_rag.retrieval.routes.reference.planner import plan_reference
from paper_rag.retrieval.routes.reference.router import build_reference_decision
from paper_rag.retrieval.route import RouteDecision


DEFAULT_QUERIES = [
    "哪些论文引用了 ResNet？",
]


def main() -> int:
    """解析命令行参数并输出 reference planner evidence。"""
    parser = argparse.ArgumentParser(description="Probe reference planner evidence outputs.")
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
        route = build_reference_decision(
            settings,
            RouteDecision(route="reference", query=query, parse_status="ok"),
            warnings,
        )
        payload: dict[str, object] = plan_reference(settings, route, warnings, debug=args.debug)
        if args.show_route:
            payload["route_summary"] = route_summary(route)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(json.dumps({"query": query, "error": str(exc), "warnings": warnings}, ensure_ascii=False, indent=2))
        return 1
    return 0


def route_summary(route: RouteDecision) -> dict[str, object]:
    """输出 reference RouteDecision 的关键字段。"""
    return {
        "route": route.route,
        "intent": route.intent,
        "return_side": route.return_side,
        "source_semantic": route.source_semantic,
        "source_filters": route.source_filters,
        "source_groups": route.source_groups,
        "source_mode": route.source_mode,
        "object_semantic": route.object_semantic,
        "object_filters": route.object_filters,
        "object_groups": route.object_groups,
        "object_mode": route.object_mode,
        "parse_status": route.parse_status,
        "parser_error": route.parser_error,
        "parser_result": route.parser_result,
    }


if __name__ == "__main__":
    sys.exit(main())
