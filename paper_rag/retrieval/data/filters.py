from __future__ import annotations

from typing import Any

from ...config import Settings
from ...dataprocess.manifest import effective_year
from ..domains.common.text import flatten_filter_value, normalized_text_key, route_tokens
from .venues import canonicalize_venue, expand_venue_query_terms


def match_record_filter(settings: Settings, record, filter_item: dict[str, Any]) -> bool:
    matched = match_record_positive_filter(settings, record, filter_item)
    return not matched if filter_item.get("negated") else matched


def match_record_positive_filter(settings: Settings, record, filter_item: dict[str, Any]) -> bool:
    field = filter_item.get("field")
    op = filter_item.get("op")
    value = filter_item.get("value")
    if field == "year":
        return compare_number(record.year, op, value)
    if field == "author":
        return compare_authors(record.author, op, value)
    if field == "venue":
        return compare_venue(settings, record.venue, op, value)
    if field == "title":
        return compare_text(record.title, op, value)
    return False


def compare_number(actual: Any, op: str, expected: Any) -> bool:
    actual_effective_year = effective_year(actual)
    if actual_effective_year is None:
        return False
    actual_number = int(actual_effective_year)
    if op == "interval":
        bounds = list(expected) if isinstance(expected, list) else []
        if len(bounds) != 2:
            return False
        lower_bound, upper_bound = bounds
        if not _is_negative_infinity(lower_bound):
            try:
                lower_number = int(lower_bound)
            except (TypeError, ValueError):
                return False
            if actual_number < lower_number:
                return False
        if not _is_positive_infinity(upper_bound):
            try:
                upper_number = int(upper_bound)
            except (TypeError, ValueError):
                return False
            if actual_number > upper_number:
                return False
        return True
    if op == "in":
        return isinstance(expected, list) and actual_number in {int(item) for item in expected}
    expected_number = int(expected)
    if op == "=":
        return actual_number == expected_number
    if op == "contains":
        return str(expected_number) in str(actual_number)
    return False


def _is_negative_infinity(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"-inf", "-infinity"}


def _is_positive_infinity(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"inf", "+inf", "infinity", "+infinity"}


def compare_authors(authors: list[str], op: str, expected: Any) -> bool:
    values = flatten_filter_value(expected)
    if not values:
        return False
    if op in {"=", "contains", "in"}:
        return any(matching_author(authors, value) for value in values)
    return False


def compare_text(actual: Any, op: str, expected: Any) -> bool:
    actual_text = str(actual or "")
    actual_key = normalized_text_key(actual_text)
    values = flatten_filter_value(expected)
    if not values:
        return False
    value_keys = [normalized_text_key(value) for value in values]
    if op == "=":
        return actual_key == value_keys[0]
    if op == "in":
        return actual_key in set(value_keys)
    if op == "contains":
        return any(value_key and value_key in actual_key for value_key in value_keys)
    return False


def compare_venue(settings: Settings, actual: Any, op: str, expected: Any) -> bool:
    actual_text = canonicalize_venue(settings, actual)
    values = expand_venue_query_terms(settings, flatten_filter_value(expected))
    return compare_text(actual_text, op, values) or compare_text(actual, op, values)


def matching_author(authors: list[str], author_query: str) -> str | None:
    query_tokens = route_tokens(author_query)
    if not query_tokens:
        return None
    for author in authors:
        if route_tokens(author) == query_tokens:
            return author
    return None
