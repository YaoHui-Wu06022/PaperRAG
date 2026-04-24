from __future__ import annotations

from typing import Any

from .errors import PlanParseError


PLAN_FILTER_FIELDS = {"author", "year", "venue", "title"}
PLAN_FILTER_OPS = {"=", "in", "contains", "interval"}


def validate_plan_filter(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanParseError("Plan filter must be an object")
    field = value.get("field")
    if field not in PLAN_FILTER_FIELDS:
        raise PlanParseError(f"Invalid plan filter field: {field}")
    op = value.get("op")
    if op not in PLAN_FILTER_OPS:
        raise PlanParseError(f"Invalid plan filter op: {op}")
    if "value" not in value:
        raise PlanParseError("Plan filter missing value")
    negated = value.get("negated")
    if not isinstance(negated, bool):
        raise PlanParseError("Plan filter negated must be true or false")
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

