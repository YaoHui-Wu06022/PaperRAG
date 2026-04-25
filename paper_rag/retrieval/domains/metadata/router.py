from __future__ import annotations

from copy import deepcopy

from ....config import Settings
from ..common.errors import PlanParseError
from ..common.filters import resolve_paper_year_filters
from ..common.paper_resolver import resolve_parser_papers
from ...top_router import RouteDecision
from .parser import MetadataParserClient


METADATA_ENTRY_TERMS = {
    "作者",
    "谁",
    "题目",
    "标题",
    "名字",
    "年",
    "发表",
    "发布",
    "期刊",
    "会议",
    "venue",
}

PAPER_TERMS = {
    "论文",
    "文章",
    "篇",
}

METADATA_LIST_TERMS = {
    "哪些",
    "有",
    "多少",
    "几",
    "数量",
    "一共",
    "最早",
    "最新",
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


def metadata_entry_reason(query: str, tokens: list[str]) -> str:
    _ = tokens
    term = first_metadata_entry_term(query)
    if term:
        return f"匹配到路由词: {term}"
    paper_term = first_paper_term(query)
    list_term = first_metadata_list_term(query)
    if paper_term and list_term:
        return f"匹配到论文集合和列表词: {paper_term}/{list_term}"
    return ""


def first_metadata_entry_term(query: str) -> str | None:
    for term in sorted(METADATA_ENTRY_TERMS, key=len, reverse=True):
        if term in query:
            return term
    return None


def first_paper_term(query: str) -> str | None:
    for term in sorted(PAPER_TERMS, key=len, reverse=True):
        if term in query:
            return term
    return None


def first_metadata_list_term(query: str) -> str | None:
    for term in sorted(METADATA_LIST_TERMS, key=len, reverse=True):
        if term in query:
            return term
    return None


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
