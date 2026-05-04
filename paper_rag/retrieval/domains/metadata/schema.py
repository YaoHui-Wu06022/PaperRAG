"""metadata parser schema 校验。"""

from __future__ import annotations

from typing import Any

from ..common.errors import PlanParseError
from ..common.schema import (
    load_payload,
    normalize_nullable_enum,
    validate_group_mode,
    validate_paper_filters,
    validate_paper_groups,
    validate_semantic,
)


METADATA_INTENTS = {"lookup", "list", "count", "exists", None}
METADATA_RETURN_FIELDS = {"author", "year", "venue", "title"}
METADATA_GROUP_MODES = {"single", "per", "or", "and"}
METADATA_FIELDS = {"intent", "return_fields", "paper_semantic", "filters", "paper_groups", "group_mode"}
PARSER_NAME = "Metadata"


def validate_metadata_parse(content: str | dict[str, Any], fallback_query: str = "") -> dict[str, Any]:
    """校验 metadata 输出，并补齐 list 默认返回 title 的约定。"""
    _ = fallback_query
    payload = load_payload(content, PARSER_NAME)
    extra_fields = set(payload) - METADATA_FIELDS
    if extra_fields:
        fields = ", ".join(sorted(extra_fields))
        raise PlanParseError(f"Metadata parser returned unsupported fields: {fields}")

    intent = normalize_nullable_enum(payload.get("intent"))
    if intent not in METADATA_INTENTS:
        raise PlanParseError(f"Invalid metadata intent: {intent}")

    return_fields = validate_return_fields(payload.get("return_fields", []))
    if intent == "list" and not return_fields:
        return_fields = ["title"]
    if intent == "lookup" and not return_fields:
        raise PlanParseError("Metadata lookup requires return_fields")
    if intent in {"count", "exists"} or intent is None:
        if return_fields:
            raise PlanParseError("Metadata count/exists/null requires return_fields=[]")

    paper_semantic = validate_semantic(payload.get("paper_semantic", ""), "paper_semantic")
    filters = validate_paper_filters(payload.get("filters", []), PARSER_NAME)
    paper_groups = validate_paper_groups(payload.get("paper_groups", []), PARSER_NAME, "paper_groups")
    group_mode = payload.get("group_mode", "single")
    group_mode = validate_group_mode(group_mode, paper_groups, PARSER_NAME, "group_mode")
    if group_mode == "and" and intent != "exists":
        raise PlanParseError('Metadata group_mode="and" requires intent="exists"')

    return {
        "intent": intent,
        "return_fields": return_fields,
        "paper_semantic": paper_semantic,
        "filters": filters,
        "paper_groups": paper_groups,
        "group_mode": group_mode,
    }


def validate_return_fields(value: Any) -> list[str]:
    """校验 metadata lookup/list 允许返回的字段列表。"""
    if value is None:
        value = []
    if not isinstance(value, list):
        raise PlanParseError("Metadata return_fields must be a list")
    fields: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise PlanParseError("Metadata return_fields items must be strings")
        field = item.strip()
        if field not in METADATA_RETURN_FIELDS:
            raise PlanParseError(f"Invalid metadata return field: {field}")
        if field not in seen:
            seen.add(field)
            fields.append(field)
    return fields
