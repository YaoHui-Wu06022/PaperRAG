"""解析 parser 输出里的 paper/venue/year scope value。"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from paper_rag.config import Settings
from paper_rag.ingest.manifest import normalize_year
from paper_rag.corpus.venues import normalize_venue_for_storage
from paper_rag.corpus.aliases import AliasMatch, dedupe_alias_matches, resolve_paper_queries
from paper_rag.corpus.records import merge_paper_records
from paper_rag.corpus.utils import is_negative_infinity, is_positive_infinity, normalize_interval_bound_text, value_to_text_list

if TYPE_CHECKING:
    from paper_rag.corpus.context import CorpusContext


def resolve_parser_scope(
    settings: Settings,
    parser_result: dict[str, Any],
    *,
    fallback_query: str | None = None,
    corpus: "CorpusContext | None" = None,
) -> dict[str, Any]:
    """解析完整 parser scope，必要时用 fallback query 补论文范围。"""
    resolved = resolve_parser_paper_scope(settings, parser_result, corpus=corpus)
    resolved_papers = resolved["resolved_papers"]
    alias_matches = list(resolved["alias_matches"])
    if fallback_query and not resolved_papers:
        fallback_papers, fallback_matches = resolve_paper_queries(settings, [fallback_query], corpus=corpus)
        resolved_papers = merge_paper_records(resolved_papers, fallback_papers)
        alias_matches.extend(fallback_matches)
    return {
        **resolved,
        "resolved_papers": resolved_papers,
        "alias_matches": dedupe_alias_matches(alias_matches),
    }


def resolve_parser_paper_scope(
    settings: Settings,
    parser_result: dict[str, Any],
    *,
    corpus: "CorpusContext | None" = None,
) -> dict[str, Any]:
    """解析 parser 输出中的 filters 和 paper_groups。"""
    filters, filter_matches, filter_papers = resolve_filter_values(settings, parser_result.get("filters") or [], corpus=corpus)
    paper_groups: list[dict[str, Any]] = []
    group_matches: list[AliasMatch] = []
    group_papers: list[dict[str, Any]] = []
    for group in parser_result.get("paper_groups") or []:
        if not isinstance(group, dict):
            continue
        group_filters, matches, papers = resolve_filter_values(settings, group.get("filters") or [], corpus=corpus)
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
    *,
    corpus: "CorpusContext | None" = None,
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
            value, matches, papers = resolve_paper_filter_value(settings, item.get("value"), corpus=corpus)
            item["value"] = value
            alias_matches.extend(matches)
            if not item.get("negated"):
                resolved_papers = merge_paper_records(resolved_papers, papers)
        elif field == "year" and item.get("op") == "interval":
            # year interval 允许边界写论文名，例如 ["ResNet", "inf"]。
            value, matches, papers = resolve_interval_paper_bounds(settings, item.get("value"), corpus=corpus)
            item["value"] = value
            alias_matches.extend(matches)
            resolved_papers = merge_paper_records(resolved_papers, papers)
        elif field == "venue":
            item["value"] = resolve_venue_filter_value(settings, item.get("value"))
        resolved_filters.append(item)
    return resolved_filters, alias_matches, resolved_papers


def resolve_paper_filter_value(
    settings: Settings,
    value: Any,
    *,
    corpus: "CorpusContext | None" = None,
) -> tuple[Any, list[AliasMatch], list[dict[str, Any]]]:
    """把 paper filter value 解析为本地规范论文标题。"""
    values = value_to_text_list(value)
    titles, matches, papers = resolve_paper_mentions_to_titles(settings, values, corpus=corpus)
    if not isinstance(value, list):
        return (titles[0] if titles else str(value or "").strip()), matches, papers
    return titles, matches, papers


def resolve_interval_paper_bounds(
    settings: Settings,
    value: Any,
    *,
    corpus: "CorpusContext | None" = None,
) -> tuple[Any, list[AliasMatch], list[dict[str, Any]]]:
    """解析 year interval 两侧可能出现的论文名边界。"""
    if not isinstance(value, list) or len(value) != 2:
        return value, [], []
    resolved_values: list[Any] = []
    alias_matches: list[AliasMatch] = []
    resolved_papers: list[dict[str, Any]] = []
    for boundary in value:
        if (
            isinstance(boundary, str)
            and boundary.strip()
            and not is_negative_infinity(boundary)
            and not is_positive_infinity(boundary)
        ):
            titles, matches, papers = resolve_paper_mentions_to_titles(settings, [boundary], corpus=corpus)
            boundary = titles[0] if titles else boundary
            alias_matches.extend(matches)
            resolved_papers = merge_paper_records(resolved_papers, papers)
        resolved_values.append(boundary)
    return resolved_values, alias_matches, resolved_papers


def resolve_venue_filter_value(settings: Settings, value: Any) -> Any:
    """按 venue aliases 规范化 venue filter value。"""
    if isinstance(value, list):
        return [normalize_venue_for_storage(settings, item) or str(item or "").strip() for item in value]
    return normalize_venue_for_storage(settings, value) or str(value or "").strip()


def resolve_paper_mentions_to_titles(
    settings: Settings,
    mentions: list[str],
    *,
    corpus: "CorpusContext | None" = None,
) -> tuple[list[str], list[AliasMatch], list[dict[str, Any]]]:
    """把多个论文 mention 解析为标题、alias matches 和 records。"""
    titles: list[str] = []
    alias_matches: list[AliasMatch] = []
    resolved_papers: list[dict[str, Any]] = []
    for mention in mentions:
        papers, matches = resolve_paper_queries(settings, [mention], corpus=corpus)
        alias_matches.extend(matches)
        resolved_papers = merge_paper_records(resolved_papers, papers)
        if papers:
            titles.extend(str(paper.get("title") or "").strip() for paper in papers if paper.get("title"))
        elif matches:
            titles.extend(match.canonical for match in matches if match.canonical)
        else:
            titles.append(mention)
    return [title for title in titles if title], alias_matches, resolved_papers


def resolve_scope_year_filters(
    settings: Settings,
    filters: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    warnings: list[str],
    *,
    corpus: "CorpusContext | None" = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """解析一组 scope filters 和 paper groups 内的 year interval 边界。"""
    resolved_filters = resolve_year_filter_values(settings, list(filters), warnings, corpus=corpus)
    resolved_groups = [
        {
            **group,
            "filters": resolve_year_filter_values(settings, list(group.get("filters") or []), warnings, corpus=corpus),
        }
        for group in groups
    ]
    return resolved_filters, resolved_groups


def resolve_year_filter_values(
    settings: Settings,
    filters: list[dict[str, Any]],
    warnings: list[str],
    *,
    corpus: "CorpusContext | None" = None,
) -> list[dict[str, Any]]:
    """解析并合并一组 year interval filters。"""
    resolved_filters: list[dict[str, Any]] = []
    for filter_item in filters:
        if filter_item.get("field") != "year" or filter_item.get("op") != "interval":
            resolved_filters.append(filter_item)
            continue
        value = filter_item.get("value")
        if not isinstance(value, list) or len(value) != 2:
            resolved_filters.append(filter_item)
            continue

        left, right = [
            resolve_year_boundary(settings, boundary, warnings, corpus=corpus)
            for boundary in value
        ]
        if left == value[0] and right == value[1]:
            resolved_filters.append(filter_item)
        elif has_resolved_interval_bounds(left, right):
            resolved_filters.append({**filter_item, "value": normalize_interval_filter_bounds(left, right)})
        else:
            resolved_filters.append({**filter_item, "value": [left, right]})
    return merge_year_interval_filters(resolved_filters)


def resolve_year_boundary(
    settings: Settings,
    boundary: Any,
    warnings: list[str],
    *,
    corpus: "CorpusContext | None" = None,
) -> Any:
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

    papers, _ = resolve_paper_queries(settings, [text], corpus=corpus)
    years: list[int] = []
    for paper in papers:
        year = normalize_year(paper.get("year"))
        candidate = year.get("publish_year") or year.get("preprint_year")
        if candidate is not None:
            years.append(candidate)
    if years:
        return min(years)
    warnings.append(f"paper interval 无法解析边界年份：{text}")
    return boundary


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
    if is_negative_infinity(current_lower):
        lower = next_lower
    elif is_negative_infinity(next_lower):
        lower = current_lower
    else:
        lower = max(current_lower, next_lower)
    if is_positive_infinity(current_upper):
        upper = "inf" if normalize_interval_bound_text(next_upper) == "+inf" else next_upper
    elif is_positive_infinity(next_upper):
        upper = "inf" if normalize_interval_bound_text(current_upper) == "+inf" else current_upper
    else:
        upper = min(current_upper, next_upper)
    return {
        **current,
        "value": [lower, upper],
    }


def has_resolved_interval_bounds(left: Any, right: Any) -> bool:
    """判断 interval 两侧是否已经解析成可比较边界。"""
    return (isinstance(left, int) or is_negative_infinity(left)) and (
        isinstance(right, int) or is_positive_infinity(right)
    )
