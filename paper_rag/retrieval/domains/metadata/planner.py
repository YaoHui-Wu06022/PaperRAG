from __future__ import annotations

from typing import Any

from ....config import Settings
from ...data.filters import match_record_filter
from ...data.manifest_lookup import load_active_manifest_records, match_manifest_records, to_evidence_manifest_record
from ...evidence import to_evidence_metadata_record, to_evidence_metadata_records
from ...top_router import RouteDecision
from ...data.aliases import alias_match_to_dict


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
    records: list[dict[str, Any]]
    if route.intent == "lookup":
        records = metadata_lookup_records(settings, route, warnings)
    else:
        records = metadata_records_by_parser_filters(settings, route.filters)
    if not records:
        warnings.append("metadata route found no matching manifest records")
    evidence = {
        **build_metadata_evidence_base(route),
        "parse_status": "ok",
        "records": records,
    }
    if route.intent == "count":
        evidence["count"] = len(records)
    return evidence


def build_metadata_evidence_base(route: RouteDecision) -> dict[str, Any]:
    return {
        "intent": route.intent,
        "return_field": route.return_field,
        "filters": route.filters,
        "alias_matches": [alias_match_to_dict(match) for match in route.alias_matches],
    }


def metadata_lookup_records(
    settings: Settings,
    route: RouteDecision,
    warnings: list[str],
) -> list[dict[str, Any]]:
    return_field = route.return_field
    if return_field is None:
        warnings.append("metadata lookup missing return_field")
    records = to_evidence_metadata_records(route.resolved_papers)
    if not records and route.extract_query:
        records = match_manifest_records(settings, route.extract_query)
    if return_field is None:
        return records
    return [
        {**record, "return_field": return_field, "value": record.get(str(return_field))}
        for record in records
    ]


def metadata_records_by_parser_filters(settings: Settings, filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in load_active_manifest_records(settings):
        if all(match_record_filter(settings, record, filter_item) for filter_item in filters):
            records.append(to_evidence_metadata_record(to_evidence_manifest_record(record)))
    return records
