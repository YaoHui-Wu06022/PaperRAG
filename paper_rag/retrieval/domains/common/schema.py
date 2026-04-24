from __future__ import annotations

import json
from typing import Any

from .errors import PlanParseError


PAPER_FILTER_FIELDS = {"author", "year", "venue", "title"}
PAPER_FILTER_OPS = {"=", "in", "contains", "interval"}


def parse_json_object(content: str | dict[str, Any], parser_name: str) -> dict[str, Any]:
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"{parser_name} parser returned invalid JSON: {exc}") from exc
    else:
        payload = dict(content)
    if not isinstance(payload, dict):
        raise PlanParseError(f"{parser_name} parser JSON root must be an object")
    return payload


def normalize_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise PlanParseError(f"{field_name} must be a list")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise PlanParseError(f"{field_name} items must be strings")
        text = item.strip()
        if text:
            normalized.append(text)
    return normalized

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
        "value": normalize_plan_filter_value(op, value.get("value")),
        "negated": negated,
    }


def normalize_plan_filter_value(op: str, value: Any) -> Any:
    if op != "interval":
        return value
    if isinstance(value, list):
        if len(value) != 2:
            raise PlanParseError("Plan interval filter requires two bounds")
        return [_normalize_interval_bound(value[0]), _normalize_interval_bound(value[1])]
    raise PlanParseError("Plan interval filter requires a two-item range")


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
            raise PlanParseError("Plan interval filter bounds must be numeric or inf sentinels") from exc
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PlanParseError("Plan interval filter bounds must be numeric or inf sentinels") from exc
