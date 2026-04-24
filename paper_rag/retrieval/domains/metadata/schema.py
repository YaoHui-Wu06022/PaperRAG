from __future__ import annotations

from typing import Any
import json

from ..common.errors import PlanParseError
from ..common.schema import validate_plan_filter


METADATA_INTENTS = {"lookup", "list", "count"}
METADATA_RETURN_FIELDS = {"author", "year", "venue", "title", None}


def validate_metadata_parse(content: str | dict[str, Any], fallback_query: str = "") -> dict[str, Any]:
    _ = fallback_query
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"Metadata parser returned invalid JSON: {exc}") from exc
    else:
        payload = dict(content)
    if not isinstance(payload, dict):
        raise PlanParseError("Metadata parser JSON root must be an object")
    if "router" in payload:
        raise PlanParseError("Metadata parser payload must not include router")
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
    anchors = payload.get("anchors") or []
    if not isinstance(anchors, list):
        raise PlanParseError("Metadata anchors must be a list")
    return {
        "intent": intent,
        "return_field": return_field,
        "anchors": normalize_metadata_anchors(anchors),
        "filters": [validate_plan_filter(filter_item) for filter_item in filters],
    }


def normalize_metadata_anchors(anchors: list[Any]) -> list[str]:
    normalized: list[str] = []
    for anchor in anchors:
        if not isinstance(anchor, str):
            raise PlanParseError("Metadata anchor must be a string title")
        text = anchor.strip()
        if text:
            normalized.append(text)
    return normalized
