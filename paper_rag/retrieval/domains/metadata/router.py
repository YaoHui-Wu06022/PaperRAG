from __future__ import annotations

from copy import deepcopy

from ....config import Settings
from ..common.errors import PlanParseError
from ..common.filters import resolve_paper_year_filters
from ..common.paper_resolver import dedupe_alias_matches, merge_papers, resolve_parser_papers
from ...top_router import RouteDecision
from .parser import MetadataParserClient


def build_metadata_decision(
    settings: Settings,
    decision: RouteDecision,
    warnings: list[str],
    *,
    plan_parser=None,
) -> RouteDecision:
    query = decision.extract_query
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
            extract_query=query,
            resolved_papers=decision.resolved_papers,
            resolved_anchor_papers=decision.resolved_anchor_papers,
            alias_matches=decision.alias_matches,
            parse_status="parse_failed",
            parser_error=str(exc),
            return_field=None,
            filters=decision.filters,
        )

    combined_filters = [*decision.filters, *parser_result["filters"]]
    resolved = resolve_parser_papers(settings, {**parser_result, "filters": combined_filters})
    parser_result = {**parser_result, "filters": resolved["filters"]}
    enriched = RouteDecision(
        route=decision.route,
        reason=decision.reason,
        intent=parser_result["intent"],
        extract_query=query,
        resolved_papers=merge_papers(decision.resolved_papers, resolved["resolved_papers"]),
        resolved_anchor_papers=resolved["resolved_anchor_papers"],
        alias_matches=dedupe_alias_matches([*decision.alias_matches, *resolved["alias_matches"]]),
        parser_result=parser_result,
        parse_status="ok",
        return_field=parser_result["return_field"],
        filters=parser_result["filters"],
        anchors=parser_result.get("anchors") or [],
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
        extract_query=decision.extract_query,
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
