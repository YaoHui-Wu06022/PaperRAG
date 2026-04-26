from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from .data.aliases import AliasMatch
from .domains.common.errors import PlanParseError
from .domains.common.filters import resolve_paper_year_filters
from .domains.top.parser import TopParserClient


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str
    intent: str | None = None
    extract_query: str = ""
    resolved_papers: list[dict[str, Any]] = field(default_factory=list)
    resolved_anchor_papers: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    alias_matches: list[AliasMatch] = field(default_factory=list)
    parser_result: dict[str, Any] | None = None
    parse_status: str = "not_parsed"
    parser_error: str | None = None
    return_field: str | None = None
    filters: list[dict[str, Any]] = field(default_factory=list)
    direction: str | None = None
    anchors: list[str] = field(default_factory=list)
    anchor_mode: str | None = None


def build_route_decision(
    settings: Settings,
    query: str,
    *,
    warnings: list[str],
    plan_parser=None,
) -> RouteDecision:
    try:
        parser = plan_parser or TopParserClient.from_settings(settings)
        if not hasattr(parser, "parse_top"):
            raise PlanParseError("plan_parser must provide parse_top(query)")
        top_result = parser.parse_top(query)
    except (PlanParseError, OSError, ValueError) as exc:
        warnings.append(f"top_parse_failed: {exc}")
        return RouteDecision(
            route="unclear",
            reason=f"top_parse_failed: {exc}",
            intent=None,
            extract_query=query,
            parse_status="parse_failed",
            parser_error=str(exc),
        )
    decision = build_top_decision(settings, query, top_result, warnings)
    if decision.route == "metadata":
        from .domains.metadata.router import build_metadata_decision

        return build_metadata_decision(settings, decision, warnings, plan_parser=plan_parser)
    if decision.route == "reference":
        from .domains.reference.router import build_reference_decision

        return build_reference_decision(settings, decision, warnings, plan_parser=plan_parser)
    if decision.route == "content":
        from .domains.content.router import build_content_decision

        return build_content_decision(settings, decision, warnings, plan_parser=plan_parser)
    return decision


def build_top_decision(
    settings: Settings,
    original_query: str,
    top_result: dict[str, Any],
    warnings: list[str],
) -> RouteDecision:
    filters = resolve_paper_year_filters(settings, top_result["filters"], warnings)
    parser_result = {**top_result, "filters": filters}
    router = top_result["router"]
    extract_query = top_result["extract_query"] or original_query
    if router == "unclear":
        return RouteDecision(
            route="unclear",
            reason="top parser selected unclear",
            intent=None,
            extract_query=extract_query,
            parser_result=parser_result,
            parse_status="ok",
            filters=filters,
        )
    return RouteDecision(
        route=router,
        reason=f"top parser selected route: {router}",
        intent=None,
        extract_query=extract_query,
        parser_result=parser_result,
        parse_status="ok",
        filters=filters,
    )
