"""解析 parser 输出里的 paper/venue/year scope value。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ...config import Settings
from ...dataprocess.manifest import normalize_year
from ...dataprocess.venues import normalize_venue_for_storage
from .aliases_match import AliasMatch, dedupe_alias_matches, resolve_paper_queries
from .manifest_records import merge_paper_records
from .utils import is_negative_infinity, is_positive_infinity, normalize_interval_bound_text, value_to_text_list


def resolve_parser_scope(
    settings: Settings,
    parser_result: dict[str, Any],
    *,
    fallback_query: str | None = None,
) -> dict[str, Any]:
    """解析完整 parser scope，必要时用 fallback query 补论文范围。"""
    resolved = resolve_parser_paper_scope(settings, parser_result)
    resolved_papers = resolved["resolved_papers"]
    alias_matches = list(resolved["alias_matches"])
    if fallback_query and not resolved_papers:
        fallback_papers, fallback_matches = resolve_paper_queries(settings, [fallback_query])
        resolved_papers = merge_paper_records(resolved_papers, fallback_papers)
        alias_matches.extend(fallback_matches)
    return {
        **resolved,
        "resolved_papers": resolved_papers,
        "alias_matches": dedupe_alias_matches(alias_matches),
    }


def resolve_parser_paper_scope(settings: Settings, parser_result: dict[str, Any]) -> dict[str, Any]:
    """解析 parser 输出中的 filters 和 paper_groups。"""
    filters, filter_matches, filter_papers = resolve_filter_values(settings, parser_result.get("filters") or [])
    paper_groups: list[dict[str, Any]] = []
    group_matches: list[AliasMatch] = []
    group_papers: list[dict[str, Any]] = []
    for group in parser_result.get("paper_groups") or []:
        if not isinstance(group, dict):
            continue
        group_filters, matches, papers = resolve_filter_values(settings, group.get("filters") or [])
        group_matches.extend(matches)
        group_papers = merge_paper_records(group_papers, papers)
        paper_groups.append({
            **group,
            "semantic": str(group.get("semantic") or "").strip(),
            "filters": group_filters,
        })

    resolved_papers = merge_paper_records(filter_papers, group_papers)
    return {
        "filters": filters,
        "paper_groups": paper_groups,
        "resolved_papers": resolved_papers,
        "alias_matches": dedupe_alias_matches([*filter_matches, *group_matches]),
    }


def resolve_filter_values(
    settings: Settings,
    filters: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[AliasMatch], list[dict[str, Any]]]:
    """规范化 parser filters 中的 paper、venue 和 year interval value。"""
    resolved_filters: list[dict[str, Any]] = []
    alias_matches: list[AliasMatch] = []
    resolved_papers: list[dict[str, Any]] = []
    for filter_item in filters:
        item = deepcopy(filter_item)
        field = item.get("field")
        if field == "paper":
            # paper = / follow / prior 都先把 value 解析成本地 canonical title。
            value, matches, papers = resolve_paper_filter_value(settings, item.get("value"))
            item["value"] = value
            alias_matches.extend(matches)
            if not item.get("negated"):
                resolved_papers = merge_paper_records(resolved_papers, papers)
        elif field == "year" and item.get("op") == "interval":
            # year interval 允许边界写论文名，例如 ["ResNet", "inf"]。
            value, matches, papers = resolve_interval_paper_bounds(settings, item.get("value"))
            item["value"] = value
            alias_matches.extend(matches)
            resolved_papers = merge_paper_records(resolved_papers, papers)
        elif field == "venue":
            item["value"] = resolve_venue_filter_value(settings, item.get("value"))
        resolved_filters.append(item)
    return resolved_filters, alias_matches, resolved_papers


def resolve_paper_filter_value(settings: Settings, value: Any) -> tuple[Any, list[AliasMatch], list[dict[str, Any]]]:
    """把 paper filter value 解析为本地规范论文标题。"""
    values = value_to_text_list(value)
    titles, matches, papers = resolve_paper_mentions_to_titles(settings, values)
    if not isinstance(value, list):
        return (titles[0] if titles else str(value or "").strip()), matches, papers
    return titles, matches, papers


def resolve_interval_paper_bounds(settings: Settings, value: Any) -> tuple[Any, list[AliasMatch], list[dict[str, Any]]]:
    """解析 year interval 两侧可能出现的论文名边界。"""
    if not isinstance(value, list) or len(value) != 2:
        return value, [], []
    left, left_matches, left_papers = resolve_interval_bound_paper(settings, value[0])
    right, right_matches, right_papers = resolve_interval_bound_paper(settings, value[1])
    return [left, right], [*left_matches, *right_matches], merge_paper_records(left_papers, right_papers)


def resolve_interval_bound_paper(settings: Settings, value: Any) -> tuple[Any, list[AliasMatch], list[dict[str, Any]]]:
    """解析单个 interval 边界中的论文 mention。"""
    if not isinstance(value, str) or not value.strip():
        return value, [], []
    if is_negative_infinity(value) or is_positive_infinity(value):
        return value, [], []
    titles, matches, papers = resolve_paper_mentions_to_titles(settings, [value])
    return (titles[0] if titles else value), matches, papers


def resolve_venue_filter_value(settings: Settings, value: Any) -> Any:
    """按 venue aliases 规范化 venue filter value。"""
    if isinstance(value, list):
        return [normalize_venue_for_storage(settings, item) or str(item or "").strip() for item in value]
    return normalize_venue_for_storage(settings, value) or str(value or "").strip()


def resolve_paper_mentions_to_titles(
    settings: Settings,
    mentions: list[str],
) -> tuple[list[str], list[AliasMatch], list[dict[str, Any]]]:
    """把多个论文 mention 解析为标题、alias matches 和 records。"""
    titles: list[str] = []
    alias_matches: list[AliasMatch] = []
    resolved_papers: list[dict[str, Any]] = []
    for mention in mentions:
        papers, matches = resolve_paper_queries(settings, [mention])
        alias_matches.extend(matches)
        resolved_papers = merge_paper_records(resolved_papers, papers)
        if papers:
            titles.extend(str(paper.get("title") or "").strip() for paper in papers if paper.get("title"))
        elif matches:
            titles.extend(match.canonical for match in matches if match.canonical)
        else:
            titles.append(mention)
    return [title for title in titles if title], alias_matches, resolved_papers


def resolve_year_filter_values(
    settings: Settings,
    filters: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """解析并合并一组 year interval filters。"""
    resolved_filters = [resolve_year_interval_filter(settings, filter_item, warnings) for filter_item in filters]
    return merge_year_interval_filters(resolved_filters)


def resolve_year_interval_filter(
    settings: Settings,
    filter_item: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """解析单个 year interval filter 的论文边界。"""
    if filter_item.get("field") != "year" or filter_item.get("op") != "interval":
        return filter_item
    value = filter_item.get("value")
    if not isinstance(value, list) or len(value) != 2:
        return filter_item

    left, right = [resolve_year_boundary(settings, boundary, warnings) for boundary in value]
    if left == value[0] and right == value[1]:
        return filter_item
    if not has_resolved_interval_bounds(left, right):
        return {**filter_item, "value": [left, right]}
    return {**filter_item, "value": normalize_interval_filter_bounds(left, right)}


def resolve_year_boundary(settings: Settings, boundary: Any, warnings: list[str]) -> Any:
    """把 year interval 边界中的论文 mention 转成年份。"""
    if isinstance(boundary, int):
        return boundary
    if isinstance(boundary, str):
        text = boundary.strip()
        if is_negative_infinity(text):
            return "-inf"
        if is_positive_infinity(text):
            return "inf"
        if not text:
            return boundary
    else:
        return boundary

    papers, _ = resolve_paper_queries(settings, [text])
    years = [publish_or_preprint_year(paper.get("year")) for paper in papers]
    years = [year for year in years if year is not None]
    if years:
        return min(years)
    warnings.append(f"paper interval could not resolve boundary year: {text}")
    return boundary


def publish_or_preprint_year(value: Any) -> int | None:
    """从 year 字段中取 publish_year，缺失时退到 preprint_year。"""
    year = normalize_year(value)
    return year.get("publish_year") or year.get("preprint_year")


def normalize_interval_filter_bounds(left: Any, right: Any) -> list[Any]:
    """规范化区间边界和方向。"""
    if isinstance(left, int) and is_positive_infinity(right):
        # “X 之后”转成开区间，因此下界 +1。
        return [left + 1, "inf"]
    if is_negative_infinity(left) and isinstance(right, int):
        # “X 之前”同样转成开区间，因此上界 -1。
        return ["-inf", right - 1]
    if isinstance(left, int) and isinstance(right, int) and left > right:
        return [right, left]
    return [left, "inf" if normalize_interval_bound_text(right) == "+inf" else right]


def merge_year_interval_filters(filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并同组可合并的 year interval filters。"""
    merged: dict[str, Any] | None = None
    output: list[dict[str, Any]] = []
    for filter_item in filters:
        if is_mergeable_year_interval(filter_item):
            merged = merge_year_interval(merged, filter_item)
        else:
            output.append(filter_item)
    if merged is not None:
        output.append(merged)
    return output


def is_mergeable_year_interval(filter_item: dict[str, Any]) -> bool:
    """判断 year interval filter 是否可以并入公共区间。"""
    value = filter_item.get("value")
    if not (
        filter_item.get("field") == "year"
        and filter_item.get("op") == "interval"
        and not filter_item.get("negated")
        and isinstance(value, list)
        and len(value) == 2
    ):
        return False
    return has_resolved_interval_bounds(value[0], value[1])


def merge_year_interval(current: dict[str, Any] | None, next_filter: dict[str, Any]) -> dict[str, Any]:
    """把两个 year interval filter 合成更窄的区间。"""
    if current is None:
        return dict(next_filter)
    current_lower, current_upper = current["value"]
    next_lower, next_upper = next_filter["value"]
    return {
        **current,
        "value": [
            max_lower_bound(current_lower, next_lower),
            min_upper_bound(current_upper, next_upper),
        ],
    }


def max_lower_bound(left: Any, right: Any) -> Any:
    """取两个 interval 下界中更严格的一个。"""
    if is_negative_infinity(left):
        return right
    if is_negative_infinity(right):
        return left
    return max(left, right)


def min_upper_bound(left: Any, right: Any) -> Any:
    """取两个 interval 上界中更严格的一个。"""
    if is_positive_infinity(left):
        return "inf" if normalize_interval_bound_text(right) == "+inf" else right
    if is_positive_infinity(right):
        return "inf" if normalize_interval_bound_text(left) == "+inf" else left
    return min(left, right)


def has_resolved_interval_bounds(left: Any, right: Any) -> bool:
    """判断 interval 两侧是否已经解析成可比较边界。"""
    return (isinstance(left, int) or is_negative_infinity(left)) and (
        isinstance(right, int) or is_positive_infinity(right)
    )
