from __future__ import annotations

from typing import Any

from .....config import Settings
from ....data.aliases import AliasMatch, expand_query_with_aliases, resolve_target_papers
from ...top_router import RouteDecision, first_matching_term, route_tokens
from ..metadata.schema import PlanParseError
from .parser import ReferenceParserClient, validate_reference_parse


REFERENCE_TERMS = {
    "bibliography",
    "bibliographies",
    "citation",
    "citations",
    "cite",
    "cited",
    "cites",
    "citing",
    "reference",
    "referenced",
    "references",
    "referencing",
}


def reference_route(query: str, tokens: list[str] | None = None) -> RouteDecision | None:
    tokens = tokens or route_tokens(query)
    reference_term = first_matching_term(tokens, REFERENCE_TERMS)
    if not reference_term:
        return None
    return RouteDecision(
        route="reference",
        reason=f"matched reference term: {reference_term}",
        intent=None,
        target_query=query,
    )


def has_reference_term(query: str) -> bool:
    return first_matching_term(route_tokens(query), REFERENCE_TERMS) is not None


def build_reference_decision(
    settings: Settings,
    decision: RouteDecision,
    query: str,
    warnings: list[str],
    *,
    plan_parser=None,
) -> RouteDecision:
    try:
        parser_result = parse_reference_query(settings, query, plan_parser)
    except (PlanParseError, OSError, ValueError) as exc:
        warnings.append(f"reference_parse_failed: {exc}")
        return RouteDecision(
            route=decision.route,
            reason=decision.reason,
            intent=None,
            target_query=query,
            parse_status="parse_failed",
            parser_error=str(exc),
        )
    anchor_queries = [item["value"] for item in parser_result["anchor"]]
    target_papers, alias_matches = resolve_target_papers(settings, anchor_queries)
    alias_matches.extend(alias_matches_for_unresolved_anchors(settings, anchor_queries))
    parse_status = "ok" if parser_result["direction"] else "unknown_direction"
    if parse_status == "unknown_direction":
        warnings.append("reference parser returned direction=null")
    return RouteDecision(
        route=decision.route,
        reason=decision.reason,
        intent=parser_result["intent"],
        target_query=parser_result["raw_query"],
        target_queries=anchor_queries,
        target_papers=target_papers,
        alias_matches=dedupe_alias_matches(alias_matches),
        parser_result=parser_result,
        parse_status=parse_status,
        filters=parser_result["filters"],
        direction=parser_result["direction"],
        anchor=parser_result["anchor"],
        anchor_mode=parser_result["anchor_mode"],
    )


def parse_reference_query(settings: Settings, query: str, plan_parser=None) -> dict[str, Any]:
    parser = plan_parser or ReferenceParserClient.from_settings(settings)
    if hasattr(parser, "parse_reference"):
        result = parser.parse_reference(query)
        return validate_reference_parse(result, query)
    raise PlanParseError("plan_parser must provide parse_reference(query)")


def alias_matches_for_unresolved_anchors(settings: Settings, anchor_queries: list[str]) -> list[AliasMatch]:
    matches: list[AliasMatch] = []
    for query in anchor_queries:
        _, query_matches = expand_query_with_aliases(settings, query)
        matches.extend(query_matches)
    return matches


def dedupe_alias_matches(matches: list[AliasMatch]) -> list[AliasMatch]:
    seen: set[tuple[str, str]] = set()
    result: list[AliasMatch] = []
    for match in matches:
        key = (match.alias, match.canonical)
        if key in seen:
            continue
        seen.add(key)
        result.append(match)
    return result
