from __future__ import annotations

import json
from typing import Any

from .errors import PlanParseError


PAPER_FILTER_FIELDS = {"author", "year", "venue", "title"}
PAPER_FILTER_OPS = {"=", "in", "contains", "interval"}
NEGATIVE_INFINITY = {"-inf", "-infinity"}
POSITIVE_INFINITY = {"inf", "+inf", "infinity", "+infinity"}


def load_payload(content: str | dict[str, Any], name: str) -> dict[str, Any]:
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"{name} parser returned invalid JSON: {exc}") from exc
    else:
        payload = dict(content)
    if not isinstance(payload, dict):
        raise PlanParseError(f"{name} parser JSON root must be an object")
    return payload


def norm_string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list):
        raise PlanParseError(f"{name} must be a list")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise PlanParseError(f"{name} items must be strings")
        text = item.strip()
        if text:
            items.append(text)
    return items


def validate_paper_filters(value: Any, name: str) -> list[dict[str, Any]]:
    if value is None:
        value = []
    if not isinstance(value, list):
        raise PlanParseError(f"{name} filters must be a list")
    return [validate_paper_filter(item) for item in value]


def validate_paper_filter(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanParseError("Paper filter must be an object")
    field = value.get("field")
    if field not in PAPER_FILTER_FIELDS:
        raise PlanParseError(f"Invalid paper filter field: {field}")
    op = value.get("op")
    if op not in PAPER_FILTER_OPS:
        raise PlanParseError(f"Invalid paper filter op: {op}")
    if "value" not in value:
        raise PlanParseError("Paper filter missing value")
    negated = value.get("negated")
    if not isinstance(negated, bool):
        raise PlanParseError("Paper filter negated must be true or false")
    return {
        "field": field,
        "op": op,
        "value": norm_filter_value(op, value.get("value")),
        "negated": negated,
    }


def norm_filter_value(op: str, value: Any) -> Any:
    if op != "interval":
        return value
    if isinstance(value, list) and len(value) == 2:
        return [norm_interval_bound(value[0]), norm_interval_bound(value[1])]
    raise PlanParseError("Plan interval filter requires a two-item range")


def norm_interval_bound(value: Any) -> int | str:
    if isinstance(value, str):
        text = value.strip()
        normalized = text.lower()
        if normalized in NEGATIVE_INFINITY:
            return "-inf"
        if normalized in POSITIVE_INFINITY:
            return "inf"
        try:
            return int(normalized)
        except ValueError:
            if text:
                return text
            raise PlanParseError("Plan interval filter bounds must be numeric, paper mentions, or inf sentinels")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PlanParseError("Plan interval filter bounds must be numeric or inf sentinels") from exc
