from __future__ import annotations

from typing import Any
import json

from ..common.errors import PlanParseError
from ..common.schema import validate_plan_filter


REFERENCE_INTENTS = {"list", "count", None}
REFERENCE_DIRECTIONS = {"cites", "cited_by", None}
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
    if "router" in payload:
        raise PlanParseError("Reference parser payload must not include router")
    intent = normalize_nullable_enum(payload.get("intent"))
    if intent not in REFERENCE_INTENTS:
        raise PlanParseError(f"Invalid reference intent: {intent}")
    direction = normalize_nullable_enum(payload.get("direction"))
    if direction not in REFERENCE_DIRECTIONS:
        raise PlanParseError(f"Invalid reference direction: {direction}")
    anchors = payload.get("anchors")
    if not isinstance(anchors, list):
        raise PlanParseError("Reference anchors must be a list")
    anchor_mode = normalize_nullable_enum(payload.get("anchor_mode")) or "per"
    if anchor_mode not in REFERENCE_ANCHOR_MODES:
        raise PlanParseError(f"Invalid reference anchor_mode: {anchor_mode}")
    filters = payload.get("filters", [])
    if filters is None:
        filters = []
    if not isinstance(filters, list):
        raise PlanParseError("Reference filters must be a list")
    raw_query = payload.get("raw_query")
    if not isinstance(raw_query, str) or not raw_query.strip():
        raw_query = fallback_query
    return {
        "intent": intent,
        "direction": direction,
        "anchors": normalize_reference_anchors(anchors),
        "anchor_mode": anchor_mode,
        "filters": [validate_plan_filter(filter_item) for filter_item in filters],
        "raw_query": raw_query,
    }


def normalize_reference_anchors(anchors: list[Any]) -> list[str]:
    normalized: list[str] = []
    for anchor in anchors:
        if not isinstance(anchor, str):
            raise PlanParseError("Reference anchor must be a string title")
        text = anchor.strip()
        if text:
            normalized.append(text)
    return normalized


def normalize_nullable_enum(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "null":
        return None
    return value
