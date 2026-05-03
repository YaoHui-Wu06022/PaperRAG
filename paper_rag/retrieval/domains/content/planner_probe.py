from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_project_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".env").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_rag.config import Settings
from paper_rag.retrieval.domains.content.planner import plan_body
from paper_rag.retrieval.domains.content.router import build_content_decision
from paper_rag.retrieval.route import RouteDecision


DEFAULT_QUERIES = [
    "ResNet 的模型结构是什么？",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe content planner evidence outputs.")
    parser.add_argument("query", nargs="*", help="Optional single query. Defaults to the first DEFAULT_QUERIES item.")
    parser.add_argument("--project-root", type=Path, default=find_project_root(), help="Project root containing .env")
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
        route = build_content_decision(
            settings,
            RouteDecision(route="content", query=query, parse_status="ok"),
            warnings,
        )
        payload: dict[str, object] = plan_body(settings, route, warnings, debug=args.debug)
        if args.show_route:
            payload["route_summary"] = route_summary(route)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(json.dumps({"query": query, "error": str(exc), "warnings": warnings}, ensure_ascii=False, indent=2))
        return 1
    return 0


def route_summary(route: RouteDecision) -> dict[str, object]:
    return {
        "route": route.route,
        "intent": route.intent,
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
