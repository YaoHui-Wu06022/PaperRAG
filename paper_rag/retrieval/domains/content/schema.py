"""content parser schema 校验。"""

from __future__ import annotations

from typing import Any

from ..common.errors import PlanParseError
from ..common.schema import (
    load_payload,
    norm_string_list,
    validate_group_mode,
    validate_paper_filters,
    validate_paper_groups,
    validate_semantic,
)


CONTENT_INTENTS = {"lookup", "reason", "compare", "summary", "list", "count", "exists", None}
CONTENT_FIELDS = {
    "intent",
    "paper_semantic",
    "filters",
    "paper_groups",
    "group_mode",
    "content_objects",
    "compare_objects",
}
PARSER_NAME = "Content"


def validate_content_parse(content: str | dict[str, Any], fallback_query: str = "") -> dict[str, Any]:
    """校验 content 输出，并约束 compare/count/exists 的对象字段。"""
    _ = fallback_query
    payload = load_payload(content, PARSER_NAME)
    extra_fields = set(payload) - CONTENT_FIELDS
    if extra_fields:
        fields = ", ".join(sorted(extra_fields))
        raise PlanParseError(f"Content parser returned unsupported fields: {fields}")

    intent = normalize_nullable_enum(payload.get("intent"))
    if intent not in CONTENT_INTENTS:
        raise PlanParseError(f"Invalid content intent: {intent}")

    paper_semantic = validate_semantic(payload.get("paper_semantic", ""), "Content paper_semantic")
    filters = validate_paper_filters(payload.get("filters", []), PARSER_NAME)
    paper_groups = validate_paper_groups(payload.get("paper_groups", []), PARSER_NAME, "paper_groups")
    group_mode = validate_group_mode(payload.get("group_mode", "single"), paper_groups, PARSER_NAME, "group_mode")
    if group_mode == "and" and intent != "exists":
        raise PlanParseError('Content group_mode="and" requires intent="exists"')

    content_objects = norm_string_list(payload.get("content_objects") or [], f"{PARSER_NAME} content_objects")
    compare_objects = norm_string_list(payload.get("compare_objects") or [], f"{PARSER_NAME} compare_objects")
    if intent == "compare":
        if len(compare_objects) < 2:
            raise PlanParseError("Content compare requires at least two compare_objects")
    elif compare_objects:
        raise PlanParseError("Content non-compare intents require compare_objects=[]")
    if intent in {"count", "exists"} and not content_objects:
        raise PlanParseError(f"Content {intent} requires content_objects")

    return {
        "intent": intent,
        "paper_semantic": paper_semantic,
        "filters": filters,
        "paper_groups": paper_groups,
        "group_mode": group_mode,
        "content_objects": content_objects,
        "compare_objects": compare_objects,
    }


def normalize_nullable_enum(value: Any) -> str | None:
    """把 JSON null 或字符串 'null' 统一成 None。"""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "null":
        return None
    return str(value)
