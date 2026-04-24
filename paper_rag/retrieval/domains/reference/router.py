from __future__ import annotations

from copy import deepcopy
from typing import Any

from ....config import Settings
from ..common.paper_resolved import alias_matches_for_unresolved_anchors, dedupe_alias_matches, resolve_paper_mentions
from ..common.errors import PlanParseError
from ..common.filters import resolve_anchor_year_filters
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
            query=query,
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
        parser = plan_parser or ReferenceParserClient.from_settings(settings)
        if not hasattr(parser, "parse_reference"):
            raise PlanParseError("plan_parser must provide parse_reference(query)")
        parser_result = parser.parse_reference(query)
    except (PlanParseError, OSError, ValueError) as exc:
        warnings.append(f"reference_parse_failed: {exc}")
        return RouteDecision(
            route=decision.route,
            reason=decision.reason,
            intent=None,
            query=query,
            parse_status="parse_failed",
            parser_error=str(exc),
        )
    paper_mentions = parser_result["anchors"]
    resolved_papers, alias_matches = resolve_paper_mentions(settings, paper_mentions)
    alias_matches.extend(alias_matches_for_unresolved_anchors(settings, paper_mentions))
    parse_status = "ok" if parser_result["direction"] else "unknown_direction"
    if parse_status == "unknown_direction":
        warnings.append("reference parser returned direction=null")
    decision = RouteDecision(
        route=decision.route,
        reason=decision.reason,
        intent=parser_result["intent"],
        query=query,
        paper_mentions=paper_mentions,
        resolved_papers=resolved_papers,
        alias_matches=dedupe_alias_matches(alias_matches),
        parser_result=parser_result,
        parse_status=parse_status,
        filters=parser_result["filters"],
        direction=parser_result["direction"],
        anchors=parser_result["anchors"],
        anchor_mode=parser_result["anchor_mode"],
    )
    return apply_anchor_year_filters(decision, warnings)


def apply_anchor_year_filters(decision: RouteDecision, warnings: list[str]) -> RouteDecision:
    filters = list(decision.filters)
    resolved_filters = resolve_anchor_year_filters(filters, decision.resolved_papers, warnings)
    if resolved_filters == filters:
        return decision
    parser_result = deepcopy(decision.parser_result) if decision.parser_result is not None else None
    if parser_result is not None:
        parser_result["filters"] = resolved_filters
    return RouteDecision(
        route=decision.route,
        reason=decision.reason,
        intent=decision.intent,
        query=decision.query,
        paper_mentions=decision.paper_mentions,
        resolved_papers=decision.resolved_papers,
        alias_matches=decision.alias_matches,
        parser_result=parser_result,
        parse_status=decision.parse_status,
        parser_error=decision.parser_error,
        filters=resolved_filters,
        direction=decision.direction,
        anchors=decision.anchors,
        anchor_mode=decision.anchor_mode,
    )
