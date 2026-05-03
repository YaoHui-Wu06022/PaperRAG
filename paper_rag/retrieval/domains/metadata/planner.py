from __future__ import annotations

from typing import Any

from ....config import Settings
from ....dataprocess.venues import display_venue
from ...data.aliases import alias_match_to_dict
from ...data.manifest_records import dedupe_paper_records
from ...data.paper_scope import combined_semantic, records_for_scope
from ...evidence import to_evidence_metadata_record
from ...route import RouteDecision


def plan_metadata(
    settings: Settings,
    route: RouteDecision,
    warnings: list[str],
) -> dict[str, Any]:
    if route.parse_status == "parse_failed":
        return {
            **build_metadata_evidence_base(route),
            "parse_status": "parse_failed",
            "parser_error": route.parser_error,
            "records": [],
        }

    if route.group_mode == "per":
        group_results = metadata_per_group_results(settings, route)
        records = dedupe_paper_records([record for group in group_results for record in group["records"]])
        evidence = build_metadata_evidence(settings, route, records, group_results=group_results)
    elif route.group_mode == "and":
        group_results = metadata_per_group_results(settings, route)
        records = dedupe_paper_records([record for group in group_results for record in group["records"]])
        evidence = build_metadata_evidence(settings, route, records, group_results=group_results)
        evidence["exists"] = all(bool(group["records"]) for group in group_results)
    else:
        records = metadata_records_for_route(settings, route)
        evidence = build_metadata_evidence(settings, route, records)
        if route.intent == "exists":
            evidence["exists"] = bool(records)

    if route.intent == "count":
        evidence["count"] = len(evidence["records"])
    if not evidence["records"]:
        warnings.append("metadata route found no matching manifest records")
    return evidence


def build_metadata_evidence_base(route: RouteDecision) -> dict[str, Any]:
    return {
        "intent": route.intent,
        "return_fields": route.return_fields,
        "paper_semantic": route.paper_semantic,
        "filters": route.filters,
        "paper_groups": route.paper_groups,
        "group_mode": route.group_mode,
        "alias_matches": [alias_match_to_dict(match) for match in route.alias_matches],
    }


def build_metadata_evidence(
    settings: Settings,
    route: RouteDecision,
    records: list[dict[str, Any]],
    *,
    group_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        **build_metadata_evidence_base(route),
        "parse_status": "ok",
        "records": [metadata_record_with_values(settings, record, route.return_fields) for record in records],
    }
    if group_results is not None:
        evidence["group_results"] = [
            {
                **group,
                "records": [metadata_record_with_values(settings, record, route.return_fields) for record in group["records"]],
                "count": len(group["records"]),
                "exists": bool(group["records"]),
            }
            for group in group_results
        ]
    return evidence


def metadata_per_group_results(settings: Settings, route: RouteDecision) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for group in route.paper_groups:
        semantic = combined_semantic(route.paper_semantic, group.get("semantic") or "")
        filters = [*route.filters, *(group.get("filters") or [])]
        records = metadata_records_for_scope(settings, semantic, filters, route.group_mode)
        results.append({
            "semantic": group.get("semantic") or "",
            "filters": group.get("filters") or [],
            "records": records,
        })
    return results


def metadata_records_for_route(settings: Settings, route: RouteDecision) -> list[dict[str, Any]]:
    if route.group_mode == "or":
        records = [
            record
            for group in route.paper_groups
            for record in metadata_records_for_scope(
                settings,
                combined_semantic(route.paper_semantic, group.get("semantic") or ""),
                [*route.filters, *(group.get("filters") or [])],
                route.group_mode,
            )
        ]
        return dedupe_paper_records(records)
    return metadata_records_for_scope(settings, route.paper_semantic, route.filters, route.group_mode)


def metadata_records_for_scope(
    settings: Settings,
    paper_semantic: str,
    filters: list[dict[str, Any]],
    group_mode: str,
) -> list[dict[str, Any]]:
    return records_for_scope(settings, paper_semantic, filters, group_mode)


def metadata_record_with_values(settings: Settings, record: dict[str, Any], return_fields: list[str]) -> dict[str, Any]:
    public_record = to_evidence_metadata_record(record)
    public_record["venue"] = display_venue(settings, public_record.get("venue"))
    if return_fields:
        public_record["values"] = {field: public_record.get(field) for field in return_fields}
    return public_record
