from __future__ import annotations

import json
from typing import Any

from ...config import Settings
from ..sparse.bm25 import tokenize


def load_venue_aliases(settings: Settings) -> list[dict[str, Any]]:
    path = settings.data_dir / "venue_aliases.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def canonicalize_venue(settings: Settings, venue: Any) -> str:
    text = str(venue or "").strip()
    if not text:
        return ""
    text_key = venue_key(text)
    for entry in load_venue_aliases(settings):
        canonical = str(entry.get("canonical") or "").strip()
        if not canonical:
            continue
        display = str(entry.get("display") or "").strip() or canonical
        for candidate in venue_entry_terms(entry):
            candidate_key = venue_key(candidate)
            if candidate_key and (text_key == candidate_key or candidate_key in text_key):
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
            expanded.append(value)
    return unique_terms(expanded)


def venue_entry_terms(entry: dict[str, Any]) -> list[str]:
    canonical = str(entry.get("canonical") or "").strip()
    aliases = [str(alias).strip() for alias in entry.get("aliases") or [] if str(alias).strip()]
    return [term for term in [canonical, *aliases] if term]


def unique_terms(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = venue_key(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def venue_key(value: str) -> str:
    return " ".join(tokenize(value))
