"""三条 domain 共用的 parser payload 与 paper filter 校验工具。"""

from __future__ import annotations

import json
from typing import Any

from .errors import PlanParseError


PAPER_FILTER_FIELDS = {"author", "year", "venue", "title", "paper"}
PAPER_FILTER_OPS = {"=", "in", "contains", "interval", "follow", "prior"}
PAPER_FILTER_FIELD_OPS = {
    "paper": {"=", "follow", "prior"},
    "year": {"=", "interval"},
    "venue": {"=", "in"},
    "author": {"contains"},
    "title": {"contains"},
}
PAPER_GROUP_MODES = {"single", "per", "or", "and"}
NEGATIVE_INFINITY = {"-inf"}
POSITIVE_INFINITY = {"inf", "+inf"}


def load_payload(content: str | dict[str, Any], name: str) -> dict[str, Any]:
    """把 parser 返回的 JSON 字符串或 dict 规整成对象根。"""
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


def normalize_string_list(value: Any, name: str) -> list[str]:
    """校验字符串列表，并过滤空字符串。"""
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
    """校验一组 paper scope filters。"""
    if value is None:
        value = []
    if not isinstance(value, list):
        raise PlanParseError(f"{name} filters must be a list")
    return [validate_paper_filter(item) for item in value]


def validate_semantic(value: Any, name: str) -> str:
    """校验 semantic 文本字段。"""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise PlanParseError(f"{name} must be a string")
    return value.strip()


def normalize_nullable_enum(value: Any) -> str | None:
    """把 JSON null 或字符串 'null' 统一成 None。"""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "null":
        return None
    return str(value)


def validate_paper_groups(value: Any, parser_name: str, field_name: str) -> list[dict[str, Any]]:
    """校验 paper_groups/source_groups/object_groups 的统一结构。"""
    if value is None:
        value = []
    if not isinstance(value, list):
        raise PlanParseError(f"{parser_name} {field_name} must be a list")
    groups: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise PlanParseError(f"{parser_name} {field_name} items must be objects")
        extra_fields = set(item) - {"semantic", "filters"}
        if extra_fields:
            fields = ", ".join(sorted(extra_fields))
            raise PlanParseError(f"{parser_name} {field_name}[{index}] returned unsupported fields: {fields}")
        groups.append({
            "semantic": validate_semantic(item.get("semantic", ""), f"{parser_name} {field_name}[{index}].semantic"),
            "filters": validate_paper_filters(item.get("filters", []), f"{parser_name} {field_name}[{index}]"),
        })
    return groups


def validate_group_mode(mode: Any, groups: list[dict[str, Any]], parser_name: str, field_name: str) -> str:
    """校验 group_mode 和 groups 是否成对出现。"""
    if mode is None:
        mode = "single"
    if mode not in PAPER_GROUP_MODES:
        raise PlanParseError(f"Invalid {parser_name} {field_name}: {mode}")
    if mode == "single" and groups:
        raise PlanParseError(f"{parser_name} {field_name}=single requires empty groups")
    if mode != "single" and not groups:
        raise PlanParseError(f"{parser_name} grouped modes require non-empty groups")
    return str(mode)


def validate_paper_filter(value: Any) -> dict[str, Any]:
    """校验单个 filter，并按 op 规范化 value。"""
    if not isinstance(value, dict):
        raise PlanParseError("Paper filter must be an object")
    field = value.get("field")
    if field not in PAPER_FILTER_FIELDS:
        raise PlanParseError(f"Invalid paper filter field: {field}")
    op = value.get("op")
    if op not in PAPER_FILTER_OPS:
        raise PlanParseError(f"Invalid paper filter op: {op}")
    validate_filter_field_op(field, op)
    if "value" not in value:
        raise PlanParseError("Paper filter missing value")
    negated = value.get("negated")
    if not isinstance(negated, bool):
        raise PlanParseError("Paper filter negated must be true or false")
    return {
        "field": field,
        "op": op,
        "value": normalize_filter_value(op, value.get("value")),
        "negated": negated,
    }


def validate_filter_field_op(field: str, op: str) -> None:
    """拒绝 field/op 非法组合，避免执行层出现隐式宽容语义。"""
    allowed_ops = PAPER_FILTER_FIELD_OPS.get(field, set())
    if op not in allowed_ops:
        raise PlanParseError(f"Invalid paper filter op for {field} field: {op}")


def normalize_filter_value(op: str, value: Any) -> Any:
    """只对 interval value 做边界规范化，其它 value 原样保留。"""
    if op != "interval":
        return value
    if isinstance(value, list) and len(value) == 2:
        return [normalize_interval_bound(value[0]), normalize_interval_bound(value[1])]
    raise PlanParseError("Plan interval filter requires a two-item range")


def normalize_interval_bound(value: Any) -> int | str:
    """把 interval 边界转成 int、inf 标记或论文 mention 文本。"""
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
