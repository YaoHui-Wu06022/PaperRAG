"""检索 plan 的顶层薄编排：只分路由，再交给各 domain 执行。"""

from __future__ import annotations

from typing import Any

from ..config import Settings
from .routes.common.errors import PlanParseError
from .routes.content.planner import plan_body
from .routes.content.router import build_content_decision
from .routes.metadata.planner import plan_metadata
from .routes.metadata.router import build_metadata_decision
from .routes.reference.planner import plan_reference
from .routes.reference.router import build_reference_decision
from .routes.top.parser import TopParserClient
from .route import RouteDecision


def run_plan(
    settings: Settings,
    query: str,
    *,
    debug: bool = False,
    top_parser=None,
) -> dict[str, Any]:
    """执行一次完整 plan：top router -> domain router -> domain planner。"""
    warnings: list[str] = []
    route = build_plan_route(settings, query, warnings, top_parser=top_parser)
    if route.route == "metadata":
        decision = build_metadata_decision(settings, route, warnings)
        return plan_metadata(settings, decision, warnings, debug=debug)
    if route.route == "reference":
        decision = build_reference_decision(settings, route, warnings)
        return plan_reference(settings, decision, warnings, debug=debug)
    if route.route == "content":
        decision = build_content_decision(settings, route, warnings)
        return plan_body(settings, decision, warnings, debug=debug)
    return unclear_plan(query, route, warnings, debug=debug)


def build_plan_route(
    settings: Settings,
    query: str,
    warnings: list[str],
    *,
    top_parser=None,
) -> RouteDecision:
    """调用 top parser，把 query 分类成 metadata/reference/content/unclear。"""
    try:
        parser = top_parser or TopParserClient.from_settings(settings)
        if not hasattr(parser, "parse_top"):
            raise PlanParseError("top_parser must provide parse_top(query)")
        parser_result = parser.parse_top(query)
    except (PlanParseError, OSError, ValueError) as exc:
        warnings.append(f"top_parse_failed: {exc}")
        return RouteDecision(
            route="unclear",
            query=query,
            parser_result=None,
            parse_status="parse_failed",
            parser_error=str(exc),
        )

    route_name = parser_result["router"]
    return RouteDecision(
        route=route_name,
        query=query,
        parser_result=parser_result,
        parse_status="ok",
    )


def unclear_plan(
    query: str,
    route: RouteDecision,
    warnings: list[str],
    *,
    debug: bool = False,
) -> dict[str, Any]:
    """把 top 层失败或 unclear 结果包装成统一 evidence 骨架。"""
    evidence: dict[str, Any] = {
        "query": query,
        "route": "unclear",
        "status": "parse_failed" if route.parse_status == "parse_failed" else "unclear",
        "results": {},
        "warnings": warnings or ["top parser returned unclear route"],
    }
    if route.parser_error:
        evidence["parser_error"] = route.parser_error
    if debug:
        evidence["debug"] = {
            "parser_result": route.parser_result,
            "parse_status": route.parse_status,
            "parser_error": route.parser_error,
        }
    return evidence
