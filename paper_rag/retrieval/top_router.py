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
    filter_groups: list[dict[str, Any]] = field(default_factory=list)
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
    normalized_top = normalize_single_filter_group(top_result, warnings)
    filters = resolve_paper_year_filters(settings, normalized_top["filters"], warnings)
    filter_groups = resolve_filter_groups(settings, normalized_top.get("filter_groups") or [], warnings)
    parser_result = {
        **normalized_top,
        "filters": filters,
        "filter_groups": filter_groups,
    }
    router = normalized_top["router"]
    extract_query = normalized_top["extract_query"] or original_query
    if router == "unclear":
        return RouteDecision(
            route="unclear",
            reason="top parser selected unclear",
            intent=None,
            extract_query=extract_query,
            parser_result=parser_result,
            parse_status="ok",
            filters=filters,
            filter_groups=filter_groups,
        )
    return RouteDecision(
        route=router,
        reason=f"top parser selected route: {router}",
        intent=None,
        extract_query=extract_query,
        parser_result=parser_result,
        parse_status="ok",
        filters=filters,
        filter_groups=filter_groups,
    )


def normalize_single_filter_group(top_result: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    filters = top_result.get("filters") or []
    filter_groups = top_result.get("filter_groups") or []
    if filters or len(filter_groups) != 1:
        return top_result
    group = filter_groups[0]
    subject = str(group.get("subject") or "").strip()
    extract_query = top_result.get("extract_query") or ""
    rewritten_query = replace_subject_placeholder(extract_query, subject)
    if rewritten_query == extract_query:
        warnings.append("top single filter_group normalized without subject placeholder replacement")
    return {
        **top_result,
        "extract_query": rewritten_query,
        "filters": list(group.get("filters") or []),
        "filter_groups": [],
    }


def replace_subject_placeholder(extract_query: str, subject: str) -> str:
    if not subject:
        return extract_query
    return extract_query.replace("{subject}", subject).replace("{subject_1}", subject)


def resolve_filter_groups(
    settings: Settings,
    filter_groups: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    resolved_groups: list[dict[str, Any]] = []
    for group in filter_groups:
        resolved_groups.append({
            "subject": group.get("subject") or "",
            "filters": resolve_paper_year_filters(settings, group.get("filters") or [], warnings),
        })
    return resolved_groups
