from __future__ import annotations

from typing import Any

from ..common.errors import PlanParseError
from ..common.schema import load_payload, norm_string_list, validate_paper_filters


METADATA_INTENTS = {"lookup", "list", "count"}
METADATA_RETURN_FIELDS = {"author", "year", "venue", "title", None}
PARSER_NAME = "Metadata"


def validate_metadata_parse(content: str | dict[str, Any], fallback_query: str = "") -> dict[str, Any]:
    _ = fallback_query
    payload = load_payload(content, PARSER_NAME)
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
    return {
        "intent": intent,
        "return_field": return_field,
        "anchors": norm_string_list(payload.get("anchors") or [], f"{PARSER_NAME} anchors"),
        "filters": validate_paper_filters(payload.get("filters", []), PARSER_NAME),
    }
