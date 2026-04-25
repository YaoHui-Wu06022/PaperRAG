from __future__ import annotations

from typing import Any

from ....config import Settings
from ....dataprocess.manifest import effective_year


def resolve_paper_year_filters(
    settings: Settings,
    filters: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    resolved_filters = [resolve_year_interval_filter(settings, filter_item, warnings) for filter_item in filters]
    return merge_year_interval_filters(resolved_filters)


def resolve_year_interval_filter(
    settings: Settings,
    filter_item: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
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
    return {**filter_item, "value": norm_interval_filter_bounds(left, right)}


def resolve_year_boundary(settings: Settings, boundary: Any, warnings: list[str]) -> Any:
    if isinstance(boundary, int) or is_infinity(boundary):
        return boundary
    if not isinstance(boundary, str) or not boundary.strip():
        return boundary

    from .paper_resolver import resolve_paper_queries

    mention = boundary.strip()
    papers, _ = resolve_paper_queries(settings, [mention])
    years = [effective_year(paper.get("year")) for paper in papers]
    years = [year for year in years if year is not None]
    if years:
        return min(years)
    warnings.append(f"paper interval could not resolve boundary year: {mention}")
    return boundary


def norm_interval_filter_bounds(left: Any, right: Any) -> list[Any]:
    if isinstance(left, int) and is_positive_infinity(right):
        return [left + 1, right]
    if is_negative_infinity(left) and isinstance(right, int):
        return [left, right - 1]
    if isinstance(left, int) and isinstance(right, int) and left > right:
        return [right, left]
    return [left, right]


def merge_year_interval_filters(filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    value = filter_item.get("value")
    return (
        filter_item.get("field") == "year"
        and filter_item.get("op") == "interval"
        and not filter_item.get("negated")
        and isinstance(value, list)
        and len(value) == 2
        and has_resolved_interval_bounds(value[0], value[1])
    )


def merge_year_interval(current: dict[str, Any] | None, next_filter: dict[str, Any]) -> dict[str, Any]:
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
    if is_negative_infinity(left):
        return right
    if is_negative_infinity(right):
        return left
    return max(left, right)


def min_upper_bound(left: Any, right: Any) -> Any:
    if is_positive_infinity(left):
        return right
    if is_positive_infinity(right):
        return left
    return min(left, right)


def has_resolved_interval_bounds(left: Any, right: Any) -> bool:
    return (isinstance(left, int) or is_infinity(left)) and (isinstance(right, int) or is_infinity(right))


def is_infinity(value: Any) -> bool:
    return is_negative_infinity(value) or is_positive_infinity(value)


def is_negative_infinity(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"-inf", "-infinity"}


def is_positive_infinity(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"inf", "+inf", "infinity", "+infinity"}
