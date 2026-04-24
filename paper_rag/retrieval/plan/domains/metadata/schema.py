from __future__ import annotations

from typing import Any
import json


class PlanParseError(RuntimeError):
    pass


METADATA_INTENTS = {"lookup", "list", "count", "unknown"}
METADATA_RETURN_FIELDS = {"author", "year", "venue", "title", None}
METADATA_FILTER_FIELDS = {"author", "year", "venue", "title"}
METADATA_FILTER_OPS = {"=", "in", "contains", "interval"}


def validate_metadata_parse(content: str | dict[str, Any], fallback_query: str = "") -> dict[str, Any]:
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"Plan parser returned invalid JSON: {exc}") from exc
    else:
        payload = dict(content)
    if not isinstance(payload, dict):
        raise PlanParseError("Plan parser JSON root must be an object")
    if payload.get("router") != "metadata":
        raise PlanParseError("Metadata parser router must be metadata")
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
    normalized_filters = [validate_metadata_filter(filter_item) for filter_item in filters]
    anchors = payload.get("anchors") or []
    if not isinstance(anchors, list):
        raise PlanParseError("Metadata anchors must be a list")
    normalized_anchors = []
    for anchor in anchors:
        normalized_anchor = validate_metadata_anchor(anchor)
        if normalized_anchor is not None:
            normalized_anchors.append(normalized_anchor)
    return {
        "router": "metadata",
        "intent": intent,
        "return_field": return_field,
        "anchors": normalized_anchors,
        "filters": normalized_filters,
    }


def validate_metadata_anchor(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        raise PlanParseError("Metadata anchor must be an object")
    field = value.get("field")
    if field != "title":
        raise PlanParseError(f"Invalid metadata anchor field: {field}")
    anchor_value = str(value.get("value") or "").strip()
    if not anchor_value:
        return None
    return {
        "field": "title",
        "value": anchor_value,
    }


def validate_metadata_filter(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanParseError("Metadata filter must be an object")
    field = value.get("field")
    if field not in METADATA_FILTER_FIELDS:
        raise PlanParseError(f"Invalid metadata filter field: {field}")
    op = value.get("op")
    if op not in METADATA_FILTER_OPS:
        raise PlanParseError(f"Invalid metadata filter op: {op}")
    if "value" not in value:
        raise PlanParseError("Metadata filter missing value")
    negated = value.get("negated")
    if not isinstance(negated, bool):
        raise PlanParseError("Metadata filter negated must be true or false")
    normalized_value = normalize_metadata_filter_value(op, value.get("value"))
    return {
        "field": field,
        "op": op,
        "value": normalized_value,
        "negated": negated,
    }


def normalize_metadata_filter_value(op: str, value: Any) -> Any:
    if op != "interval":
        return value
    if isinstance(value, list):
        if len(value) != 2:
            raise PlanParseError("Metadata interval filter requires two bounds")
        return [_normalize_interval_bound(value[0]), _normalize_interval_bound(value[1])]
    raise PlanParseError("Metadata interval filter requires a two-item range")


def _normalize_interval_bound(value: Any) -> int | str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"-inf", "-infinity"}:
            return "-inf"
        if normalized in {"inf", "+inf", "infinity", "+infinity"}:
            return "inf"
        if normalized == "anchor":
            return "anchor"
        try:
            return int(normalized)
        except ValueError as exc:
            raise PlanParseError("Metadata interval filter bounds must be numeric or inf sentinels") from exc
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PlanParseError("Metadata interval filter bounds must be numeric or inf sentinels") from exc
