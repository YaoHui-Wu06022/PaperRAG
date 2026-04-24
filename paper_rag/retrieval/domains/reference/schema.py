from __future__ import annotations

from typing import Any

from ..common.errors import PlanParseError
from ..common.schema import normalize_string_list, parse_json_object, validate_paper_filter


REFERENCE_INTENTS = {"list", "count"}
REFERENCE_DIRECTIONS = {"cites", "cited_by", None}
REFERENCE_ANCHOR_MODES = {"per", "or", "and"}


def validate_reference_parse(content: str | dict[str, Any], fallback_query: str = "") -> dict[str, Any]:
    _ = fallback_query
    payload = parse_json_object(content, "Reference")
    intent = payload.get("intent")
    if intent not in REFERENCE_INTENTS:
        raise PlanParseError(f"Invalid reference intent: {intent}")
    direction = normalize_nullable_enum(payload.get("direction"))
    if direction not in REFERENCE_DIRECTIONS:
        raise PlanParseError(f"Invalid reference direction: {direction}")
    anchor_mode = normalize_nullable_enum(payload.get("anchor_mode")) or "per"
    if anchor_mode not in REFERENCE_ANCHOR_MODES:
        raise PlanParseError(f"Invalid reference anchor_mode: {anchor_mode}")
    filters = payload.get("filters", [])
    if filters is None:
        filters = []
    if not isinstance(filters, list):
        raise PlanParseError("Reference filters must be a list")
    return {
        "intent": intent,
        "direction": direction,
        "anchors": normalize_string_list(payload.get("anchors") or [], "Reference anchors"),
        "anchor_mode": anchor_mode,
        "filters": [validate_paper_filter(filter_item) for filter_item in filters],
    }


def normalize_nullable_enum(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "null":
        return None
    return value
