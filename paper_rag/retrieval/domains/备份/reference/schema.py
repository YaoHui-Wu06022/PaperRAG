from __future__ import annotations

from typing import Any

from ..common.errors import PlanParseError
from ..common.schema import load_payload, norm_string_list, validate_paper_filters


REFERENCE_INTENTS = {"list", "count"}
REFERENCE_DIRECTIONS = {"cites", "cited_by", None}
REFERENCE_ANCHOR_MODES = {"per", "or", "and"}
PARSER_NAME = "Reference"


def validate_reference_parse(content: str | dict[str, Any], fallback_query: str = "") -> dict[str, Any]:
    _ = fallback_query
    payload = load_payload(content, PARSER_NAME)
    intent = payload.get("intent")
    if intent not in REFERENCE_INTENTS:
        raise PlanParseError(f"Invalid reference intent: {intent}")
    direction = normalize_nullable_enum(payload.get("direction"))
    if direction not in REFERENCE_DIRECTIONS:
        raise PlanParseError(f"Invalid reference direction: {direction}")
    anchor_mode = normalize_nullable_enum(payload.get("anchor_mode")) or "per"
    if anchor_mode not in REFERENCE_ANCHOR_MODES:
        raise PlanParseError(f"Invalid reference anchor_mode: {anchor_mode}")
    return {
        "intent": intent,
        "direction": direction,
        "anchors": norm_string_list(payload.get("anchors") or [], f"{PARSER_NAME} anchors"),
        "anchor_mode": anchor_mode,
        "filters": validate_paper_filters(payload.get("filters", []), PARSER_NAME),
    }


def normalize_nullable_enum(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "null":
        return None
    return value
