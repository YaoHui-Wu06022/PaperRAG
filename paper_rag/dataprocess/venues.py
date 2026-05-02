from __future__ import annotations

import json
import re
from typing import Any

from ..config import Settings


def load_venue_aliases(settings: Settings) -> list[dict[str, Any]]:
    path = settings.data_dir / "venue_aliases.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def normalize_venue_for_storage(settings: Settings, venue: Any) -> str | None:
    text = clean_venue_text(venue)
    if not text:
        return None
    return display_venue(settings, text)


def display_venue(settings: Settings, venue: Any) -> str:
    text = clean_venue_text(venue)
    if not text:
        return ""
    text_key = venue_key(text)
    for entry in load_venue_aliases(settings):
        display = venue_entry_display(entry)
        for candidate in venue_entry_terms(entry):
            if venue_keys_match(text_key, venue_key(candidate)):
                return display
    return text


def expand_venue_query_terms(settings: Settings, values: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        value_key = venue_key(value)
        matched = False
        for entry in load_venue_aliases(settings):
            term_keys = [venue_key(term) for term in venue_entry_terms(entry)]
            if value_key and value_key in term_keys:
                expanded.extend(venue_entry_terms(entry))
                matched = True
                break
        if not matched:
            expanded.append(clean_venue_text(value))
    return unique_terms(expanded)


def expand_venue_record_terms(settings: Settings, venue: Any) -> list[str]:
    text = clean_venue_text(venue)
    if not text:
        return []
    text_key = venue_key(text)
    expanded = [text]
    for entry in load_venue_aliases(settings):
        for candidate in venue_entry_terms(entry):
            if venue_keys_match(text_key, venue_key(candidate)):
                expanded.extend(venue_entry_terms(entry))
                expanded.append(venue_entry_display(entry))
                return unique_terms(expanded)
    return unique_terms(expanded)


def venue_entry_display(entry: dict[str, Any]) -> str:
    canonical = str(entry.get("canonical") or "").strip()
    display = str(entry.get("display") or "").strip()
    return display or canonical


def venue_entry_terms(entry: dict[str, Any]) -> list[str]:
    canonical = str(entry.get("canonical") or "").strip()
    display = str(entry.get("display") or "").strip()
    aliases = [str(alias).strip() for alias in entry.get("aliases") or [] if str(alias).strip()]
    return [term for term in [canonical, display, *aliases] if term]


def unique_terms(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = clean_venue_text(value)
        key = venue_key(text)
        if key and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def venue_key(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", clean_venue_text(value).lower()))


def venue_keys_match(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return left == right or left in right or right in left


def clean_venue_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,.;:-")
