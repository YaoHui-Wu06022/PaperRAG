from __future__ import annotations

from typing import Any

from ..common.errors import PlanParseError
from ..common.schema import load_payload, norm_string_list, validate_paper_filters


CONTENT_INTENTS = {"fact", "method", "reason", "compare", "summary", "list"}
PARSER_NAME = "Content"


def validate_content_parse(content: str | dict[str, Any], fallback_query: str = "") -> dict[str, Any]:
    _ = fallback_query
    payload = load_payload(content, PARSER_NAME)
    intent = payload.get("intent")
    if intent not in CONTENT_INTENTS:
        raise PlanParseError(f"Invalid content intent: {intent}")
    return {
        "intent": intent,
        "anchors": norm_string_list(payload.get("anchors") or [], f"{PARSER_NAME} anchors"),
        "compare_objects": norm_string_list(payload.get("compare_objects") or [], f"{PARSER_NAME} compare_objects"),
        "objects": norm_string_list(payload.get("objects") or [], f"{PARSER_NAME} objects"),
        "filters": validate_paper_filters(payload.get("filters", []), PARSER_NAME),
    }
