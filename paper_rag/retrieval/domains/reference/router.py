from __future__ import annotations

from copy import deepcopy

from ....config import Settings
from ..common.errors import PlanParseError
from ..common.filters import resolve_paper_year_filters
from ..common.paper_resolver import resolve_parser_papers
from ...top_router import RouteDecision
from .parser import ReferenceParserClient


REFERENCE_ENTRY_TERMS = {
    "引用",
    "被引",
    "参考",
    "引文",
}


def reference_route(query: str, tokens: list[str] | None = None) -> RouteDecision | None:
    _ = tokens
    term = first_reference_entry_term(query)
    if term:
        return RouteDecision(
            route="reference",
            reason=f"匹配到路由词: {term}",
            intent=None,
            query=query,
        )
    return None

def first_reference_entry_term(query: str) -> str | None:
    for term in sorted(REFERENCE_ENTRY_TERMS, key=len, reverse=True):
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
    resolved = resolve_parser_papers(settings, parser_result)
    parser_result = {**parser_result, "filters": resolved["filters"]}
    parse_status = "ok" if parser_result["direction"] else "unknown_direction"
    if parse_status == "unknown_direction":
        warnings.append("reference parser returned direction=null")
    decision = RouteDecision(
        route=decision.route,
        reason=decision.reason,
        intent=parser_result["intent"],
        query=query,
        resolved_papers=resolved["resolved_papers"],
        resolved_anchor_papers=resolved["resolved_anchor_papers"],
        alias_matches=resolved["alias_matches"],
        parser_result=parser_result,
        parse_status=parse_status,
        filters=parser_result["filters"],
        direction=parser_result["direction"],
        anchors=parser_result["anchors"],
        anchor_mode=parser_result["anchor_mode"],
    )
    return apply_anchor_year_filters(settings, decision, warnings)


def apply_anchor_year_filters(settings: Settings, decision: RouteDecision, warnings: list[str]) -> RouteDecision:
    filters = list(decision.filters)
    resolved_filters = resolve_paper_year_filters(settings, filters, warnings)
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
        resolved_papers=decision.resolved_papers,
        resolved_anchor_papers=decision.resolved_anchor_papers,
        alias_matches=decision.alias_matches,
        parser_result=parser_result,
        parse_status=decision.parse_status,
        parser_error=decision.parser_error,
        filters=resolved_filters,
        direction=decision.direction,
        anchors=decision.anchors,
        anchor_mode=decision.anchor_mode,
    )
