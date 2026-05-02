from __future__ import annotations

from copy import deepcopy

from ....config import Settings
from ..common.errors import PlanParseError
from ..common.filter_normalizer import resolve_paper_year_filters
from ..common.paper_resolver import dedupe_alias_matches, merge_papers, resolve_parser_paper_scope
from ...route import RouteDecision
from .parser import ReferenceParserClient


def build_reference_decision(
    settings: Settings,
    decision: RouteDecision,
    warnings: list[str],
    *,
    plan_parser=None,
) -> RouteDecision:
    original_query = decision.original_query
    try:
        parser = plan_parser or ReferenceParserClient.from_settings(settings)
        if not hasattr(parser, "parse_reference"):
            raise PlanParseError("plan_parser must provide parse_reference(query)")
        parser_result = parser.parse_reference(original_query)
    except (PlanParseError, OSError, ValueError) as exc:
        warnings.append(f"reference_parse_failed: {exc}")
        return RouteDecision(
            route=decision.route,
            intent=None,
            original_query=original_query,
            resolved_papers=decision.resolved_papers,
            alias_matches=decision.alias_matches,
            parser_result=decision.parser_result,
            parse_status="parse_failed",
            parser_error=str(exc),
            return_side=decision.return_side,
            source_semantic=decision.source_semantic,
            source_filters=decision.source_filters,
            source_groups=decision.source_groups,
            source_mode=decision.source_mode,
            object_semantic=decision.object_semantic,
            object_filters=decision.object_filters,
            object_groups=decision.object_groups,
            object_mode=decision.object_mode,
        )

    source_resolved = resolve_parser_paper_scope(settings, {
        "filters": parser_result["source_filters"],
        "paper_groups": parser_result["source_groups"],
    })
    object_resolved = resolve_parser_paper_scope(settings, {
        "filters": parser_result["object_filters"],
        "paper_groups": parser_result["object_groups"],
    })
    parser_result = {
        **parser_result,
        "source_filters": source_resolved["filters"],
        "source_groups": source_resolved["paper_groups"],
        "object_filters": object_resolved["filters"],
        "object_groups": object_resolved["paper_groups"],
        "source_resolved_papers": source_resolved["resolved_papers"],
        "object_resolved_papers": object_resolved["resolved_papers"],
    }
    enriched = RouteDecision(
        route=decision.route,
        intent=parser_result["intent"],
        original_query=original_query,
        resolved_papers=merge_papers(
            decision.resolved_papers,
            source_resolved["resolved_papers"],
            object_resolved["resolved_papers"],
        ),
        alias_matches=dedupe_alias_matches([
            *decision.alias_matches,
            *source_resolved["alias_matches"],
            *object_resolved["alias_matches"],
        ]),
        parser_result=parser_result,
        parse_status="ok",
        return_side=parser_result["return_side"],
        source_semantic=parser_result["source_semantic"],
        source_filters=parser_result["source_filters"],
        source_groups=parser_result["source_groups"],
        source_mode=parser_result["source_mode"],
        object_semantic=parser_result["object_semantic"],
        object_filters=parser_result["object_filters"],
        object_groups=parser_result["object_groups"],
        object_mode=parser_result["object_mode"],
    )
    return apply_reference_year_filters(settings, enriched, warnings)


def apply_reference_year_filters(settings: Settings, decision: RouteDecision, warnings: list[str]) -> RouteDecision:
    source_filters = resolve_paper_year_filters(settings, list(decision.source_filters), warnings)
    source_groups = [
        {**group, "filters": resolve_paper_year_filters(settings, list(group.get("filters") or []), warnings)}
        for group in decision.source_groups
    ]
    object_filters = resolve_paper_year_filters(settings, list(decision.object_filters), warnings)
    object_groups = [
        {**group, "filters": resolve_paper_year_filters(settings, list(group.get("filters") or []), warnings)}
        for group in decision.object_groups
    ]
    if (
        source_filters == decision.source_filters
        and source_groups == decision.source_groups
        and object_filters == decision.object_filters
        and object_groups == decision.object_groups
    ):
        return decision

    parser_result = deepcopy(decision.parser_result) if decision.parser_result is not None else None
    if parser_result is not None:
        parser_result["source_filters"] = source_filters
        parser_result["source_groups"] = source_groups
        parser_result["object_filters"] = object_filters
        parser_result["object_groups"] = object_groups
    return RouteDecision(
        route=decision.route,
        intent=decision.intent,
        original_query=decision.original_query,
        resolved_papers=decision.resolved_papers,
        alias_matches=decision.alias_matches,
        parser_result=parser_result,
        parse_status=decision.parse_status,
        parser_error=decision.parser_error,
        return_side=decision.return_side,
        source_semantic=decision.source_semantic,
        source_filters=source_filters,
        source_groups=source_groups,
        source_mode=decision.source_mode,
        object_semantic=decision.object_semantic,
        object_filters=object_filters,
        object_groups=object_groups,
        object_mode=decision.object_mode,
    )
