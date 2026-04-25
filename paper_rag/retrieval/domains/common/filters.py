from __future__ import annotations

from typing import Any

from ....config import Settings
from ....dataprocess.manifest import effective_year


def resolve_paper_year_filters(
    settings: Settings,
    filters: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    resolved_filters = [
        resolve_paper_interval_filter(settings, filter_item, warnings)
        for filter_item in filters
    ]
    return merge_year_interval_filters(resolved_filters)


def resolve_paper_interval_filter(
    settings: Settings,
    filter_item: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    if filter_item.get("field") != "year" or filter_item.get("op") != "interval":
        return filter_item
    value = filter_item.get("value")
    if not isinstance(value, list) or len(value) != 2:
        return filter_item

    left, right = value
    resolved_left = resolve_year_boundary(settings, left, warnings)
    resolved_right = resolve_year_boundary(settings, right, warnings)
    if resolved_left == left and resolved_right == right:
        return filter_item
    if not interval_bounds_are_resolved(resolved_left, resolved_right):
        return {**filter_item, "value": [resolved_left, resolved_right]}
    return {**filter_item, "value": normalize_interval_filter_bounds(resolved_left, resolved_right)}


def resolve_year_boundary(settings: Settings, boundary: Any, warnings: list[str]) -> Any:
    if isinstance(boundary, int) or is_infinity(boundary):
        return boundary
    if not isinstance(boundary, str):
        return boundary
    mention = boundary.strip()
    if not mention:
        return boundary
    from .paper_resolver import resolve_paper_mentions

    papers, _ = resolve_paper_mentions(settings, [mention])
    years = [effective_year(paper.get("year")) for paper in papers]
    years = [year for year in years if year is not None]
    if not years:
        warnings.append(f"paper interval could not resolve boundary year: {mention}")
        return boundary
    return min(years)


def normalize_interval_filter_bounds(left: Any, right: Any) -> list[Any]:
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
        if (
            filter_item.get("field") == "year"
            and filter_item.get("op") == "interval"
            and not filter_item.get("negated")
            and isinstance(filter_item.get("value"), list)
            and len(filter_item["value"]) == 2
            and interval_bounds_are_resolved(filter_item["value"][0], filter_item["value"][1])
        ):
            merged = merge_interval_filter(merged, filter_item)
        else:
            output.append(filter_item)
    if merged is not None:
        output.append(merged)
    return output


def merge_interval_filter(current: dict[str, Any] | None, next_filter: dict[str, Any]) -> dict[str, Any]:
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


def interval_bounds_are_resolved(left: Any, right: Any) -> bool:
    return (isinstance(left, int) or is_infinity(left)) and (isinstance(right, int) or is_infinity(right))


def is_infinity(value: Any) -> bool:
    return is_negative_infinity(value) or is_positive_infinity(value)


def is_negative_infinity(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"-inf", "-infinity"}


def is_positive_infinity(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"inf", "+inf", "infinity", "+infinity"}
