from __future__ import annotations

from ....config import Settings
from ..common.aliases import resolve_anchor_papers, target_queries_from_anchors_and_title_filters
from ..common.errors import PlanParseError
from ..common.filters import resolve_anchor_year_filters
from ...top_router import RouteDecision
from .parser import MetadataParserClient


METADATA_ENTRY_TERMS = {
    "作者",
    "标题",
    "会议",
    "期刊",
    "题目",
    "发布",
    "发表",
    "年份",
    "哪一年",
    "几年",
    "谁写",
    "谁提出",
    "哪些论文",
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
        target_query=query,
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
        parser_result = parse_metadata_query(settings, query, plan_parser)
    except (PlanParseError, OSError, ValueError) as exc:
        warnings.append(f"metadata_parse_failed: {exc}")
        return RouteDecision(
            route=decision.route,
            reason=decision.reason,
            intent=None,
            target_query=query,
            parse_status="parse_failed",
            parser_error=str(exc),
            return_field=None,
        )
    target_queries = target_queries_from_anchors_and_title_filters(parser_result)
    if parser_result["intent"] == "lookup" and not target_queries:
        target_queries = [query]
    enriched = RouteDecision(
        route=decision.route,
        reason=decision.reason,
        intent=parser_result["intent"],
        target_query=query,
        target_queries=target_queries,
        parser_result=parser_result,
        parse_status="ok",
        return_field=parser_result["return_field"],
        filters=parser_result["filters"],
        anchors=parser_result["anchors"],
    )
    resolved = resolve_decision_targets(settings, enriched, target_queries)
    return apply_anchor_year_filters(resolved, warnings)


def parse_metadata_query(settings: Settings, query: str, plan_parser=None) -> dict[str, Any]:
    parser = plan_parser or MetadataParserClient.from_settings(settings)
    if not hasattr(parser, "parse_metadata"):
        raise PlanParseError("plan_parser must provide parse_metadata(query)")
    return parser.parse_metadata(query)


def resolve_decision_targets(settings: Settings, decision: RouteDecision, target_queries: list[str]) -> RouteDecision:
    target_papers, alias_matches = resolve_anchor_papers(settings, target_queries)
    return RouteDecision(
        route=decision.route,
        reason=decision.reason,
        intent=decision.intent,
        target_query=decision.target_query,
        target_queries=target_queries,
        target_papers=target_papers,
        alias_matches=alias_matches,
        parser_result=decision.parser_result,
        parse_status=decision.parse_status,
        parser_error=decision.parser_error,
        return_field=decision.return_field,
        filters=decision.filters,
        anchors=decision.anchors,
    )


def apply_anchor_year_filters(decision: RouteDecision, warnings: list[str]) -> RouteDecision:
    filters = list(decision.filters)
    resolved_filters = resolve_anchor_year_filters(filters, decision.target_papers, warnings)
    if resolved_filters == filters:
        return decision
    return RouteDecision(
        route=decision.route,
        reason=decision.reason,
        intent=decision.intent,
        target_query=decision.target_query,
        target_queries=decision.target_queries,
        target_papers=decision.target_papers,
        alias_matches=decision.alias_matches,
        parser_result=decision.parser_result,
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
