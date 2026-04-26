from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Settings
from .domains.content.planner import plan_body
from .domains.metadata.planner import plan_metadata
from .domains.reference.planner import plan_reference
from .top_router import build_route_decision


@dataclass(frozen=True)
class PreparedQuery:
    original_query: str
    warnings: list[str]
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


def run_plan(
    settings: Settings,
    query: str,
    *,
    plan_parser=None,
    embedder=None,
    store=None,
) -> dict[str, Any]:
    prepared = prepare_query(settings, query)
    warnings = list(prepared.warnings)
    if prepared.failed:
        return {
            "original_query": prepared.original_query,
            "route": "error",
            "router_reason": prepared.error,
            "evidence": {},
            "warnings": warnings,
        }
    route = build_route_decision(settings, prepared.original_query, warnings=warnings, plan_parser=plan_parser)
    evidence: dict[str, Any]
    if route.route == "metadata":
        evidence = plan_metadata(settings, route, warnings)
    elif route.route == "reference":
        evidence = plan_reference(settings, route, warnings)
    elif route.route == "unclear":
        warnings.append(route.reason)
        evidence = {
            "parse_status": "unclear",
            "message": route.reason,
        }
    else:
        evidence = plan_body(settings, route, warnings, embedder=embedder, store=store)
    result: dict[str, Any] = {
        "original_query": prepared.original_query,
        "extract_query": route.extract_query,
        "route": route.route,
        "intent": route.intent,
        "router_reason": route.reason,
    }
    if route.route == "metadata":
        result["return_field"] = route.return_field
    elif route.route == "reference":
        result["direction"] = route.direction
        result["anchors"] = route.anchors
        result["anchor_mode"] = route.anchor_mode
    result["filters"] = route.filters
    result["evidence"] = evidence
    result["warnings"] = warnings
    return result


def prepare_query(
    settings: Settings,
    query: str,
) -> PreparedQuery:
    _ = settings
    return PreparedQuery(query, [])
