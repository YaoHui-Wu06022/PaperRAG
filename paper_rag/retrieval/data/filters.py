"""manifest record 的布尔 filter evaluator。"""

from __future__ import annotations

from typing import Any

from ...config import Settings
from ...dataprocess.manifest import normalize_year
from ...dataprocess.venues import expand_venue_query_terms, expand_venue_record_terms, venue_key, venue_keys_match
from .utils import filter_value_to_list, normalized_text


ARXIV_VALUES = {"arxiv", "arxiv preprint"}


def match_record_filters(settings: Settings, record, filters: list[dict[str, Any]]) -> bool:
    """判断单条 manifest record 是否满足全部 metadata filters。"""
    # ArXiv 是特殊范围：同组出现 venue=ArXiv 时，year 改用 preprint_year。
    year_source = "preprint" if has_arxiv_filter(filters) else "publish"
    return all(match_record_filter(settings, record, filter_item, year_source=year_source) for filter_item in filters)


def normalized_bound_text(value: Any) -> str:
    """规范化 year interval 的无穷边界文本。"""
    return value.strip().lower() if isinstance(value, str) else ""


def is_negative_infinity(value: Any) -> bool:
    """判断 interval 下界是否为 -inf。"""
    return normalized_bound_text(value) == "-inf"


def is_positive_infinity(value: Any) -> bool:
    """判断 interval 上界是否为 inf/+inf。"""
    return normalized_bound_text(value) in {"inf", "+inf"}


def bound_as_int(value: Any) -> int | None:
    """把区间边界转成 int，失败时返回 None。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def match_record_filter(
    settings: Settings,
    record,
    filter_item: dict[str, Any],
    *,
    year_source: str = "publish",
) -> bool:
    """处理单个 filter 的正向/否定匹配。"""
    matched = match_record_positive_filter(settings, record, filter_item, year_source=year_source)
    return not matched if filter_item.get("negated") else matched


def match_record_positive_filter(
    settings: Settings,
    record,
    filter_item: dict[str, Any],
    *,
    year_source: str = "publish",
) -> bool:
    """按 field 分派到具体的正向匹配函数。"""
    field = filter_item.get("field")
    op = filter_item.get("op")
    value = filter_item.get("value")
    if field == "year":
        if op not in {"=", "interval"}:
            return False
        return compare_number(record.year, op, value, year_source=year_source)
    if field == "author":
        if op != "contains":
            return False
        return compare_authors(record.author, op, value)
    if field == "venue":
        if op not in {"=", "in"}:
            return False
        # venue=ArXiv 不要求 manifest.venue 写成 ArXiv，只看是否有 preprint_year。
        if matches_arxiv_preprint(record, value, op):
            return True
        return compare_venue(settings, record.venue, op, value)
    if field == "title":
        if op != "contains":
            return False
        return compare_text(record.title, op, value)
    return False


def compare_number(actual: Any, op: str, expected: Any, *, year_source: str = "publish") -> bool:
    """比较 year 字段，支持 = 和 interval。"""
    actual_year = filter_year(actual, year_source)
    if actual_year is None:
        return False
    actual_number = int(actual_year)
    if op == "interval":
        bounds = list(expected) if isinstance(expected, list) else []
        if len(bounds) != 2:
            return False
        lower_bound, upper_bound = bounds
        if not is_negative_infinity(lower_bound):
            lower_number = bound_as_int(lower_bound)
            if lower_number is None:
                return False
            if actual_number < lower_number:
                return False
        if not is_positive_infinity(upper_bound):
            upper_number = bound_as_int(upper_bound)
            if upper_number is None:
                return False
            if actual_number > upper_number:
                return False
        return True
    expected_number = int(expected)
    if op == "=":
        return actual_number == expected_number
    return False


def compare_authors(authors: list[str], op: str, expected: Any) -> bool:
    """比较 author 字段，目前只支持 contains。"""
    values = filter_value_to_list(expected)
    if not values:
        return False
    if op == "contains":
        return any(matching_author(authors, value) for value in values)
    return False


def compare_text(actual: Any, op: str, expected: Any) -> bool:
    """比较 title/paper 等文本字段。"""
    actual_text = str(actual or "")
    actual_key = normalized_text(actual_text)
    values = filter_value_to_list(expected)
    if not values:
        return False
    value_keys = [normalized_text(value) for value in values]
    if op == "=":
        return actual_key == value_keys[0]
    if op == "in":
        return actual_key in set(value_keys)
    if op == "contains":
        return any(value_key and value_key in actual_key for value_key in value_keys)
    return False


def compare_venue(settings: Settings, actual: Any, op: str, expected: Any) -> bool:
    """比较 venue 字段，带 venue aliases 规范化。"""
    actual_keys = [venue_key(value) for value in expand_venue_record_terms(settings, actual)]
    actual_keys = [key for key in actual_keys if key]
    expected_keys = [venue_key(value) for value in expand_venue_query_terms(settings, filter_value_to_list(expected))]
    expected_keys = [key for key in expected_keys if key]
    if not actual_keys or not expected_keys:
        return False
    if op in {"=", "in"}:
        return any(venue_keys_match(actual_key, expected_key) for actual_key in actual_keys for expected_key in expected_keys)
    return False


def matches_arxiv_preprint(record, expected: Any, op: str) -> bool:
    """判断 venue=ArXiv 是否应按 preprint_year 命中。"""
    if op not in {"=", "in"}:
        return False
    actual_year = normalize_year(record.year)
    return actual_year.get("preprint_year") is not None and value_is_arxiv(expected)


def has_arxiv_filter(filters: list[dict[str, Any]]) -> bool:
    """判断同组 filters 中是否存在非否定 ArXiv 约束。"""
    return any(
        filter_item.get("field") == "venue"
        and not filter_item.get("negated")
        and value_is_arxiv(filter_item.get("value"))
        for filter_item in filters
    )


def value_is_arxiv(value: Any) -> bool:
    """纯判断：filter value 是否表达 ArXiv。"""
    values = [str(item).strip().lower() for item in filter_value_to_list(value) if str(item).strip()]
    return any(item in ARXIV_VALUES for item in values)


def filter_year(value: Any, year_source: str) -> int | None:
    """按当前过滤上下文选择 preprint_year 或 publish_year。"""
    year = normalize_year(value)
    if year_source == "preprint":
        return year.get("preprint_year")
    return year.get("publish_year")


def matching_author(authors: list[str], author_query: str) -> str | None:
    """在作者列表中查找与查询文本规范化后相同的作者。"""
    query_text = normalized_text(author_query)
    if not query_text:
        return None
    for author in authors:
        if normalized_text(author) == query_text:
            return author
    return None
