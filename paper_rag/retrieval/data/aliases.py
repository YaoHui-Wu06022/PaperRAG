from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...config import Settings
from .manifest_lookup import match_manifest_records
from ..sparse.bm25 import tokenize


@dataclass(frozen=True)
class AliasMatch:
    alias: str
    canonical: str
    expanded_terms: list[str]


def load_paper_annotation_aliases(settings: Settings) -> list[dict[str, Any]]:
    path = settings.data_dir / "paper_annotations.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    entries: list[dict[str, Any]] = []
    for annotation in payload.values():
        if not isinstance(annotation, dict):
            continue
        title = str(annotation.get("title") or "").strip()
        aliases = [str(alias).strip() for alias in annotation.get("aliases") or [] if str(alias).strip()]
        if title and aliases:
            entries.append({"canonical": title, "aliases": aliases})
    return entries


def expand_query_with_aliases(settings: Settings, query: str) -> tuple[str, list[AliasMatch]]:
    matches = find_alias_matches(load_paper_annotation_aliases(settings), query)
    if not matches:
        return query, []
    terms = tokenize(query)
    seen = set(terms)
    expanded = list(terms)
    for match in matches:
        for term in match.expanded_terms:
            for token in tokenize(term):
                if token not in seen:
                    seen.add(token)
                    expanded.append(token)
    return " ".join(expanded), matches


def find_alias_matches(entries: list[dict[str, Any]], query: str) -> list[AliasMatch]:
    query_tokens = set(tokenize(query))
    matches: list[AliasMatch] = []
    for entry in entries:
        canonical = str(entry.get("canonical") or "").strip()
        aliases = [str(alias).strip() for alias in entry.get("aliases") or [] if str(alias).strip()]
        for alias in aliases:
            alias_tokens = set(tokenize(alias))
            if alias_tokens and alias_tokens.issubset(query_tokens):
                matches.append(AliasMatch(alias, canonical, [alias, canonical, *aliases]))
                break
    return matches


def alias_match_to_dict(match: AliasMatch) -> dict[str, Any]:
    return {
        "alias": match.alias,
        "canonical": match.canonical,
        "expanded_terms": unique_terms(match.expanded_terms),
    }


def unique_terms(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = " ".join(tokenize(value))
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def resolve_paper_queries(settings: Settings, queries: list[str]) -> tuple[list[dict[str, Any]], list[AliasMatch]]:
    targets: list[dict[str, Any]] = []
    alias_matches: list[AliasMatch] = []
    seen: set[str] = set()
    for query in queries:
        expanded_query, matches = expand_query_with_aliases(settings, query)
        alias_matches.extend(matches)
        for record in match_manifest_records(settings, expanded_query):
            paper_id = path_name(record.get("paper_data_path"))
            key = paper_id or str(record.get("title") or "")
            if key in seen:
                continue
            seen.add(key)
            targets.append({
                "file_hash": record.get("file_hash"),
                "title": record.get("title"),
                "author": record.get("author"),
                "year": record.get("year"),
                "venue": record.get("venue"),
                "pdf_path": record.get("pdf_path"),
                "paper_id": paper_id,
                "paper_data_path": record.get("paper_data_path"),
                "matched_alias": matched_alias_for_record(record.get("title"), matches),
            })
    return targets, alias_matches


def matched_alias_for_record(title: Any, matches: list[AliasMatch]) -> str | None:
    normalized_title = " ".join(tokenize(str(title or "")))
    for match in matches:
        if " ".join(tokenize(match.canonical)) == normalized_title:
            return match.alias
    return None


def path_name(path: Any) -> str | None:
    if not path:
        return None
    return str(path).replace("\\", "/").rstrip("/").split("/")[-1]
