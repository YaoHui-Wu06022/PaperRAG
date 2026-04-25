from __future__ import annotations

from copy import deepcopy

from ....config import Settings
from ..common.filters import resolve_paper_year_filters
from ..common.paper_resolver import paper_mentions_from_anchors_and_title_filters, resolve_paper_mentions
from ..common.errors import PlanParseError
from ...top_router import RouteDecision
from .parser import MetadataParserClient


METADATA_ENTRY_TERMS = {
    "作者",
    "标题",
    "题目",
    "会议",
    "期刊",
    "年份",
    "发表",
    "哪一年",
    "谁写",
    "写的",
    "谁提出",
    "多少篇",
    "几篇",
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
    paper_mentions = paper_mentions_from_anchors_and_title_filters(parser_result)
    if parser_result["intent"] == "lookup" and not paper_mentions:
        paper_mentions = [query]
    resolved_papers, alias_matches = resolve_paper_mentions(settings, paper_mentions)
    enriched = RouteDecision(
        route=decision.route,
        reason=decision.reason,
        intent=parser_result["intent"],
        query=query,
        paper_mentions=paper_mentions,
        resolved_papers=resolved_papers,
        alias_matches=alias_matches,
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
        paper_mentions=decision.paper_mentions,
        resolved_papers=decision.resolved_papers,
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
    term = first_metadata_entry_term(query)
    if term:
        return f"匹配到关键词: {term}"
    return ""


def first_metadata_entry_term(query: str) -> str | None:
    for term in sorted(METADATA_ENTRY_TERMS, key=len, reverse=True):
        if term in query:
            return term
    return None
