from __future__ import annotations

from copy import deepcopy
from typing import Any

from ....config import Settings
from ..common.errors import PlanParseError
from ...data.manifest_records import merge_paper_records
from ...data.parser_scope_resolver import resolve_parser_scope, resolve_year_filter_values
from ...data.utils import dedupe_alias_matches
from ...route import RouteDecision
from .parser import MetadataParserClient


def build_metadata_decision(
    settings: Settings,
    decision: RouteDecision,
    warnings: list[str],
    *,
    plan_parser=None,
) -> RouteDecision:
    original_query = decision.original_query
    try:
        parser = plan_parser or MetadataParserClient.from_settings(settings)
        if not hasattr(parser, "parse_metadata"):
            raise PlanParseError("plan_parser must provide parse_metadata(query)")
        parser_result = parser.parse_metadata(original_query)
    except (PlanParseError, OSError, ValueError) as exc:
        warnings.append(f"metadata_parse_failed: {exc}")
        return RouteDecision(
            route=decision.route,
            intent=None,
            original_query=original_query,
            resolved_papers=decision.resolved_papers,
            alias_matches=decision.alias_matches,
            parser_result=decision.parser_result,
            parse_status="parse_failed",
            parser_error=str(exc),
            return_fields=[],
            paper_semantic=decision.paper_semantic,
            filters=decision.filters,
            paper_groups=decision.paper_groups,
            group_mode=decision.group_mode,
        )

    parser_result = {
        **parser_result,
        "filters": [*decision.filters, *parser_result["filters"]],
    }
    resolved = resolve_parser_scope(settings, parser_result)
    parser_result = {
        **parser_result,
        "filters": resolved["filters"],
        "paper_groups": resolved["paper_groups"],
    }
    enriched = RouteDecision(
        route=decision.route,
        intent=parser_result["intent"],
        original_query=original_query,
        resolved_papers=merge_paper_records(decision.resolved_papers, resolved["resolved_papers"]),
        alias_matches=dedupe_alias_matches([*decision.alias_matches, *resolved["alias_matches"]]),
        parser_result=parser_result,
        parse_status="ok",
        return_fields=parser_result["return_fields"],
        paper_semantic=parser_result["paper_semantic"],
        filters=parser_result["filters"],
        paper_groups=parser_result["paper_groups"],
        group_mode=parser_result["group_mode"],
    )
    return apply_paper_year_filters(settings, enriched, warnings)


def apply_paper_year_filters(settings: Settings, decision: RouteDecision, warnings: list[str]) -> RouteDecision:
    filters = resolve_year_filter_values(settings, list(decision.filters), warnings)
    paper_groups = [
        {**group, "filters": resolve_year_filter_values(settings, list(group.get("filters") or []), warnings)}
        for group in decision.paper_groups
    ]
    if filters == decision.filters and paper_groups == decision.paper_groups:
        return decision

    parser_result = deepcopy(decision.parser_result) if decision.parser_result is not None else None
    if parser_result is not None:
        parser_result["filters"] = filters
        parser_result["paper_groups"] = paper_groups
    return RouteDecision(
        route=decision.route,
        intent=decision.intent,
        original_query=decision.original_query,
        resolved_papers=decision.resolved_papers,
        alias_matches=decision.alias_matches,
        parser_result=parser_result,
        parse_status=decision.parse_status,
        parser_error=decision.parser_error,
        return_fields=decision.return_fields,
        paper_semantic=decision.paper_semantic,
        filters=filters,
        paper_groups=paper_groups,
        group_mode=decision.group_mode,
    )


def effective_group_filters(route: RouteDecision, group: dict[str, Any]) -> list[dict[str, Any]]:
    return [*route.filters, *(group.get("filters") or [])]
