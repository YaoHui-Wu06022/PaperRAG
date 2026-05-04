"""content parser schema 校验。"""

from __future__ import annotations

from typing import Any

from ..common.errors import PlanParseError
from ..common.schema import (
    load_payload,
    normalize_nullable_enum,
    normalize_string_list,
    validate_group_mode,
    validate_paper_filters,
    validate_paper_groups,
    validate_semantic,
)


CONTENT_INTENTS = {"lookup", "reason", "compare", "summary", "list", "count", "exists", None}

# content parser 只允许这些字段
# 否则会被视为 prompt/schema 未对齐而直接报错
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
    """校验 content 输出，并返回 planner 可直接消费的规范化 dict"""
    _ = fallback_query

    # load_payload 会处理原始 JSON 字符串 / dict，并统一抛 PlanParseError。
    payload = load_payload(content, PARSER_NAME)

    # 不忽略未知字段：content prompt 经常调，字段漂移要尽早暴露
    extra_fields = set(payload) - CONTENT_FIELDS
    if extra_fields:
        fields = ", ".join(sorted(extra_fields))
        raise PlanParseError(f"Content parser returned unsupported fields: {fields}")

    # null 会被 normalize 成 None，便于后续用 Python 集合判断
    intent = normalize_nullable_enum(payload.get("intent"))
    if intent not in CONTENT_INTENTS:
        raise PlanParseError(f"Invalid content intent: {intent}")

    # paper scope 决定“先看哪些论文”
    # 不应该混进后续 dense/BM25 的正文检索 query
    paper_semantic = validate_semantic(payload.get("paper_semantic", ""), "Content paper_semantic")
    filters = validate_paper_filters(payload.get("filters", []), PARSER_NAME)
    paper_groups = validate_paper_groups(payload.get("paper_groups", []), PARSER_NAME, "paper_groups")
    group_mode = validate_group_mode(payload.get("group_mode", "single"), paper_groups, PARSER_NAME, "group_mode")

    # content 的 and 只表达“多个论文范围是否都满足同一正文判断”
    # 因此只能服务 exists，不能用于 lookup/list/summary 这类回答型意图
    if group_mode == "and" and intent != "exists":
        raise PlanParseError('Content group_mode="and" requires intent="exists"')

    # content_objects 是真正要查正文的对象；compare_objects 只描述被比较对象
    content_objects = normalize_string_list(payload.get("content_objects") or [], f"{PARSER_NAME} content_objects")
    compare_objects = normalize_string_list(payload.get("compare_objects") or [], f"{PARSER_NAME} compare_objects")

    if intent == "compare":
        if len(compare_objects) < 2:
            raise PlanParseError("Content compare requires at least two compare_objects")
        # LLM 偶尔会把 A/B 同时放进 compare_objects 和 content_objects
        # 这里做轻量兜底：A/B 只保留为被比较对象，content_objects 只留下比较维度
        compare_keys = {value.casefold() for value in compare_objects}
        content_objects = [value for value in content_objects if value.casefold() not in compare_keys]
    elif compare_objects:
        # 非 compare 意图不允许残留 compare_objects，避免 planner 误拼检索词
        raise PlanParseError("Content non-compare intents require compare_objects=[]")

    # count/exists 必须知道数什么、判断什么；否则执行层只能盲检
    if intent in {"count", "exists"} and not content_objects:
        raise PlanParseError(f"Content {intent} requires content_objects")

    # 返回值保持与 schema 同构，但所有字段已经过类型/枚举/列表归一化
    return {
        "intent": intent,
        "paper_semantic": paper_semantic,
        "filters": filters,
        "paper_groups": paper_groups,
        "group_mode": group_mode,
        "content_objects": content_objects,
        "compare_objects": compare_objects,
    }
