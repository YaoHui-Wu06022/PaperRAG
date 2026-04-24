from __future__ import annotations

from typing import Any
import json

from ..metadata.schema import PlanParseError, validate_metadata_filter


REFERENCE_INTENTS = {"list", "count", None}
REFERENCE_DIRECTIONS = {"outgoing", "incoming", None}
REFERENCE_ANCHOR_FIELDS = {"title"}
REFERENCE_ANCHOR_MODES = {"per", "or", "and"}


def validate_reference_parse(content: str | dict[str, Any], fallback_query: str = "") -> dict[str, Any]:
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"Reference parser returned invalid JSON: {exc}") from exc
    else:
        payload = dict(content)
    if not isinstance(payload, dict):
        raise PlanParseError("Reference parser JSON root must be an object")
    if payload.get("router") != "reference":
        raise PlanParseError("Reference parser router must be reference")
    intent = normalize_nullable_enum(payload.get("intent"))
    if intent not in REFERENCE_INTENTS:
        raise PlanParseError(f"Invalid reference intent: {intent}")
    direction = normalize_nullable_enum(payload.get("direction"))
    if direction not in REFERENCE_DIRECTIONS:
        raise PlanParseError(f"Invalid reference direction: {direction}")
    anchors = payload.get("anchors")
    if not isinstance(anchors, list):
        raise PlanParseError("Reference anchors must be a list")
    normalized_anchors = [validate_reference_anchor(item) for item in anchors]
    anchor_mode = normalize_nullable_enum(payload.get("anchor_mode")) or "per"
    if anchor_mode not in REFERENCE_ANCHOR_MODES:
        raise PlanParseError(f"Invalid reference anchor_mode: {anchor_mode}")
    filters = payload.get("filters", [])
    if filters is None:
        filters = []
    if not isinstance(filters, list):
        raise PlanParseError("Reference filters must be a list")
    normalized_filters = [validate_metadata_filter(filter_item) for filter_item in filters]
    raw_query = payload.get("raw_query")
    if not isinstance(raw_query, str) or not raw_query.strip():
        raw_query = fallback_query
    return {
        "router": "reference",
        "intent": intent,
        "direction": direction,
        "anchors": normalized_anchors,
        "anchor_mode": anchor_mode,
        "filters": normalized_filters,
        "raw_query": raw_query,
    }


def validate_reference_anchor(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise PlanParseError("Reference anchor must be an object")
    field = value.get("field")
    if field not in REFERENCE_ANCHOR_FIELDS:
        raise PlanParseError(f"Invalid reference anchor field: {field}")
    anchor_value = str(value.get("value") or "").strip()
    if not anchor_value:
        raise PlanParseError("Reference anchor value is required")
    return {"field": field, "value": anchor_value}


def normalize_nullable_enum(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "null":
        return None
    return value
