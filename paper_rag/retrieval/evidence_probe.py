from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


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
from paper_rag.retrieval.domains.metadata.planner import plan_metadata
from paper_rag.retrieval.domains.metadata.router import build_metadata_decision
from paper_rag.retrieval.domains.reference.planner import plan_reference
from paper_rag.retrieval.domains.reference.router import build_reference_decision
from paper_rag.retrieval.plan import run_plan
from paper_rag.retrieval.route import RouteDecision


DEFAULT_QUERIES = {
    "metadata": "BERT 是谁写的？",
    "reference": "哪些论文引用了 ResNet？",
    "content": "ResNet 的模型结构是什么？",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe unified planner evidence outputs.")
    parser.add_argument("query", nargs="*", help="Optional single query. Defaults depend on --route.")
    parser.add_argument("--project-root", type=Path, default=find_project_root(), help="Project root containing .env")
    parser.add_argument(
        "--route",
        choices=["auto", "metadata", "reference", "content"],
        default="auto",
        help="auto uses top route; a domain route bypasses top parser and tests that evidence directly",
    )
    parser.add_argument("--all", action="store_true", help="Run one default query for each domain route")
    parser.add_argument("--debug", action="store_true", help="Include planner debug details")
    parser.add_argument("--show-route", action="store_true", help="Include RouteDecision summary for domain routes")
    args = parser.parse_args()

    settings = Settings.load(args.project_root)
    if args.all:
        payload = [
            run_domain_probe(settings, route, DEFAULT_QUERIES[route], debug=args.debug, show_route=args.show_route)
            for route in ("metadata", "reference", "content")
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    query = " ".join(args.query).strip()
    if not query:
        query = DEFAULT_QUERIES.get(args.route) if args.route != "auto" else DEFAULT_QUERIES["content"]
    if not query:
        print(json.dumps({"error": "No query provided."}, ensure_ascii=False, indent=2))
        return 1

    if args.route == "auto":
        payload = run_plan(settings, query, debug=args.debug)
    else:
        payload = run_domain_probe(settings, args.route, query, debug=args.debug, show_route=args.show_route)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_domain_probe(
    settings: Settings,
    route_name: str,
    query: str,
    *,
    debug: bool,
    show_route: bool,
) -> dict[str, Any]:
    warnings: list[str] = []
    try:
        route = build_domain_decision(settings, route_name, query, warnings)
        payload = plan_domain(settings, route, warnings, debug=debug)
        if show_route:
            payload["route_summary"] = route_summary(route)
        return payload
    except Exception as exc:
        return {
            "query": query,
            "route": route_name,
            "status": "probe_failed",
            "error": str(exc),
            "warnings": warnings,
        }


def build_domain_decision(settings: Settings, route_name: str, query: str, warnings: list[str]) -> RouteDecision:
    base = RouteDecision(route=route_name, query=query, parse_status="ok")
    if route_name == "metadata":
        return build_metadata_decision(settings, base, warnings)
    if route_name == "reference":
        return build_reference_decision(settings, base, warnings)
    if route_name == "content":
        return build_content_decision(settings, base, warnings)
    raise ValueError(f"Unsupported route: {route_name}")


def plan_domain(settings: Settings, route: RouteDecision, warnings: list[str], *, debug: bool) -> dict[str, Any]:
    if route.route == "metadata":
        return plan_metadata(settings, route, warnings, debug=debug)
    if route.route == "reference":
        return plan_reference(settings, route, warnings, debug=debug)
    if route.route == "content":
        return plan_body(settings, route, warnings, debug=debug)
    raise ValueError(f"Unsupported route: {route.route}")


def route_summary(route: RouteDecision) -> dict[str, Any]:
    return {
        "route": route.route,
        "intent": route.intent,
        "return_fields": route.return_fields,
        "paper_semantic": route.paper_semantic,
        "filters": route.filters,
        "paper_groups": route.paper_groups,
        "group_mode": route.group_mode,
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
