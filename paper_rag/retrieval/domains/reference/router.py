from __future__ import annotations

from typing import Any

from ....config import Settings
from ..common.aliases import alias_matches_for_unresolved_anchors, dedupe_alias_matches, resolve_anchor_papers
from ..common.errors import PlanParseError
from ...top_router import RouteDecision
from .parser import ReferenceParserClient


REFERENCE_CHINESE_TERMS = {
    "引用",
    "引用了",
    "引用过",
    "引用关系",
    "被引用",
    "被引",
    "参考",
    "参考了",
    "参考文献",
    "引文",
    "文献引用",
    "列进参考文献",
    "作为参考文献",
}


def reference_route(query: str, tokens: list[str] | None = None) -> RouteDecision | None:
    _ = tokens
    chinese_term = first_chinese_reference_term(query)
    if chinese_term:
        return RouteDecision(
            route="reference",
            reason=f"匹配到关键词: {chinese_term}",
            intent=None,
            target_query=query,
        )
    return None


def has_reference_term(query: str) -> bool:
    return first_chinese_reference_term(query) is not None


def first_chinese_reference_term(query: str) -> str | None:
    for term in sorted(REFERENCE_CHINESE_TERMS, key=len, reverse=True):
        if term in query:
            return term
    return None


def build_reference_decision(
    settings: Settings,
    decision: RouteDecision,
    query: str,
    warnings: list[str],
    *,
    plan_parser=None,
) -> RouteDecision:
    try:
        parser_result = parse_reference_query(settings, query, plan_parser)
    except (PlanParseError, OSError, ValueError) as exc:
        warnings.append(f"reference_parse_failed: {exc}")
        return RouteDecision(
            route=decision.route,
            reason=decision.reason,
            intent=None,
            target_query=query,
            parse_status="parse_failed",
            parser_error=str(exc),
        )
    anchor_queries = parser_result["anchors"]
    target_papers, alias_matches = resolve_anchor_papers(settings, anchor_queries)
    alias_matches.extend(alias_matches_for_unresolved_anchors(settings, anchor_queries))
    parse_status = "ok" if parser_result["direction"] else "unknown_direction"
    if parse_status == "unknown_direction":
        warnings.append("reference parser returned direction=null")
    return RouteDecision(
        route=decision.route,
        reason=decision.reason,
        intent=parser_result["intent"],
        target_query=parser_result["raw_query"],
        target_queries=anchor_queries,
        target_papers=target_papers,
        alias_matches=dedupe_alias_matches(alias_matches),
        parser_result=parser_result,
        parse_status=parse_status,
        filters=parser_result["filters"],
        direction=parser_result["direction"],
        anchors=parser_result["anchors"],
        anchor_mode=parser_result["anchor_mode"],
    )


def parse_reference_query(settings: Settings, query: str, plan_parser=None) -> dict[str, Any]:
    parser = plan_parser or ReferenceParserClient.from_settings(settings)
    if hasattr(parser, "parse_reference"):
        return parser.parse_reference(query)
    raise PlanParseError("plan_parser must provide parse_reference(query)")
