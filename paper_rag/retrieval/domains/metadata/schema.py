from __future__ import annotations

from typing import Any

from ..common.errors import PlanParseError
from ..common.schema import normalize_string_list, parse_json_object, validate_paper_filter


METADATA_INTENTS = {"lookup", "list", "count"}
METADATA_RETURN_FIELDS = {"author", "year", "venue", "title", None}


def validate_metadata_parse(content: str | dict[str, Any], fallback_query: str = "") -> dict[str, Any]:
    _ = fallback_query
    payload = parse_json_object(content, "Metadata")
    intent = payload.get("intent")
    if intent not in METADATA_INTENTS:
        raise PlanParseError(f"Invalid metadata intent: {intent}")
    if "return_field" not in payload:
        raise PlanParseError("Metadata parser missing return_field")
    return_field = payload.get("return_field")
    if return_field == "null":
        return_field = None
    if return_field not in METADATA_RETURN_FIELDS:
        raise PlanParseError(f"Invalid metadata return_field: {return_field}")
    filters = payload.get("filters", [])
    if filters is None:
        filters = []
    if not isinstance(filters, list):
        raise PlanParseError("Metadata filters must be a list")
    return {
        "intent": intent,
        "return_field": return_field,
        "anchors": normalize_string_list(payload.get("anchors") or [], "Metadata anchors"),
        "filters": [validate_paper_filter(filter_item) for filter_item in filters],
    }
