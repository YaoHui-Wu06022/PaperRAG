from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from .data.aliases import AliasMatch
from .domains.common.text import route_tokens


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str
    intent: str | None = None
    query: str = ""
    paper_mentions: list[str] = field(default_factory=list)
    resolved_papers: list[dict[str, Any]] = field(default_factory=list)
    alias_matches: list[AliasMatch] = field(default_factory=list)
    parser_result: dict[str, Any] | None = None
    parse_status: str = "not_parsed"
    parser_error: str | None = None
    return_field: str | None = None
    filters: list[dict[str, Any]] = field(default_factory=list)
    direction: str | None = None
    anchors: list[str] = field(default_factory=list)
    anchor_mode: str | None = None


def route_query(query: str) -> RouteDecision:
    tokens = route_tokens(query)
    from .domains.reference.router import reference_route  # local import avoids circular import

    reference = reference_route(query, tokens)
    if reference:
        return reference
    from .domains.metadata.router import metadata_route  # local import avoids circular import

    metadata = metadata_route(query, tokens)
    if metadata:
        return metadata
    return RouteDecision(
        route="content",
        reason="default content route for Chinese query",
        intent=None,
        query=query,
    )


def build_route_decision(
    settings: Settings,
    query: str,
    *,
    warnings: list[str],
    plan_parser=None,
) -> RouteDecision:
    decision = route_query(query)
    if decision.route == "metadata":
        from .domains.metadata.router import build_metadata_decision

        return build_metadata_decision(settings, decision, query, warnings, plan_parser=plan_parser)
    if decision.route == "reference":
        from .domains.reference.router import build_reference_decision

        return build_reference_decision(settings, decision, query, warnings, plan_parser=plan_parser)
    return decision


def has_reference_term(query: str) -> bool:
    from .domains.reference.router import has_reference_term as reference_has_term

    return reference_has_term(query)
