from __future__ import annotations

from copy import deepcopy

from ....config import Settings
from ..common.errors import PlanParseError
from ..common.filters import resolve_paper_year_filters
from ..common.paper_resolver import resolve_parser_papers
from ...top_router import RouteDecision
from .parser import MetadataParserClient


METADATA_ENTRY_TERMS = {
    # author
    "作者",
    "写",
    "谁",
    "发布",
    "提出",
    # title
    "题",
    "题目",
    "标题",
    "名字",
    # year
    "年",
    "时",

    # venue
    "期刊",
    "会议",
    "发表",
    "出版",
    "来源",
    "出处",
    "在哪",
    "哪里",
    # 计数词
    "多少",
    "几",
    "数量",
    "哪些",
}


def metadata_route(query: str, tokens: list[str]) -> RouteDecision | None:
    reason = metadata_entry_reason(query, tokens)
    if not reason:
        return None
    return RouteDecision(
        route="metadata",
        reason=reason,
        intent=None,
        query=query,
    )


def build_metadata_decision(
    settings: Settings,
    decision: RouteDecision,
    query: str,
    warnings: list[str],
    *,
    plan_parser=None,
) -> RouteDecision:
    try:
        parser = plan_parser or MetadataParserClient.from_settings(settings)
        if not hasattr(parser, "parse_metadata"):
            raise PlanParseError("plan_parser must provide parse_metadata(query)")
        parser_result = parser.parse_metadata(query)
    except (PlanParseError, OSError, ValueError) as exc:
        warnings.append(f"metadata_parse_failed: {exc}")
        return RouteDecision(
            route=decision.route,
            reason=decision.reason,
            intent=None,
            query=query,
            parse_status="parse_failed",
            parser_error=str(exc),
            return_field=None,
        )
    resolved = resolve_parser_papers(
        settings,
        parser_result,
        fallback_query=query if parser_result["intent"] == "lookup" else None,
    )
    parser_result = {**parser_result, "filters": resolved["filters"]}
    enriched = RouteDecision(
        route=decision.route,
        reason=decision.reason,
        intent=parser_result["intent"],
        query=query,
        resolved_papers=resolved["resolved_papers"],
        alias_matches=resolved["alias_matches"],
        parser_result=parser_result,
        parse_status="ok",
        return_field=parser_result["return_field"],
        filters=parser_result["filters"],
        anchors=parser_result["anchors"],
    )
    return apply_anchor_year_filters(settings, enriched, warnings)


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
        return_field=decision.return_field,
        filters=resolved_filters,
        anchors=decision.anchors,
    )


def metadata_entry_reason(query: str, tokens: list[str]) -> str:
    """Only decide whether to enter metadata parsing."""
    _ = tokens
    term = first_matching_term(query, METADATA_ENTRY_TERMS)
    if term:
        return f"matched metadata entry clue: {term}"
    return ""


def first_matching_term(query: str, terms: set[str]) -> str | None:
    for term in sorted(terms, key=len, reverse=True):
        if term in query:
            return term
    return None
