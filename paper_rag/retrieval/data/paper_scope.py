from __future__ import annotations

from pathlib import Path
from typing import Any

from ...config import Settings
from ...dataprocess.annotations import load_paper_annotations
from ...dataprocess.manifest import ManifestRecord
from .citation_scope import record_matches_citation_scope
from .filters import compare_text, match_record_filters
from .manifest_lookup import load_active_manifest_records, match_manifest_records, to_evidence_manifest_record
from .text import normalized_text_key


PaperTags = dict[str, list[str]]


def records_for_scope(
    settings: Settings,
    paper_semantic: str,
    filters: list[dict[str, Any]],
    group_mode: str = "single",
) -> list[dict[str, Any]]:
    semantic = paper_semantic.strip()
    semantic_keys = semantic_candidate_keys(settings, semantic)
    if semantic and not semantic_keys:
        return []
    records: list[dict[str, Any]] = []
    for record in load_active_manifest_records(settings):
        if semantic and record_key(record) not in semantic_keys:
            continue
        if match_scope_filters(settings, record, filters, group_mode):
            evidence = to_evidence_manifest_record(record)
            evidence["_record_key"] = record_key(record)
            records.append(evidence)
    return records


def match_scope_filters(
    settings: Settings,
    record: ManifestRecord,
    filters: list[dict[str, Any]],
    group_mode: str,
) -> bool:
    return all(
        match_scope_filter(settings, record, filter_item, group_mode)
        for filter_item in filters
        if filter_item.get("field") == "paper"
    ) and match_record_filters(settings, record, [filter_item for filter_item in filters if filter_item.get("field") != "paper"])


def match_scope_filter(
    settings: Settings,
    record: ManifestRecord,
    filter_item: dict[str, Any],
    group_mode: str,
) -> bool:
    if filter_item.get("field") == "paper":
        matched = match_paper_filter(settings, record, filter_item, group_mode)
        return not matched if filter_item.get("negated") else matched
    return match_record_filters(settings, record, [filter_item])


def match_paper_filter(
    settings: Settings,
    record: ManifestRecord,
    filter_item: dict[str, Any],
    group_mode: str,
) -> bool:
    _ = group_mode
    op = filter_item.get("op")
    values = [str(value).strip() for value in flatten_value(filter_item.get("value")) if str(value).strip()]
    if not values:
        return False
    if op == "=":
        return any(compare_text(record.title, "=", value) for value in values)
    if op in {"follow", "prior"}:
        return match_citation_graph_filter(settings, record, values, op)
    return False


def match_citation_graph_filter(
    settings: Settings,
    record: ManifestRecord,
    titles: list[str],
    op: str,
) -> bool:
    return record_matches_citation_scope(settings, record_key(record), titles, op)


def semantic_candidate_keys(settings: Settings, paper_semantic: str) -> set[str]:
    semantic = paper_semantic.strip()
    if not semantic:
        return set()
    keys: set[str] = set()
    matches = match_manifest_records(settings, semantic)
    keys.update(record_key_from_dict(record) for record in matches if record_key_from_dict(record))
    tag_title_keys = semantic_tag_title_keys(settings, semantic)
    if tag_title_keys:
        keys.update(
            record_key(record)
            for record in load_active_manifest_records(settings)
            if title_key(record.title) in tag_title_keys
        )
    return keys


def load_paper_annotation_tag_index(settings: Settings) -> dict[str, PaperTags]:
    index: dict[str, PaperTags] = {}
    for annotation in load_paper_annotations(settings).values():
        title = str(annotation.get("title") or "").strip()
        key = title_key(title)
        if not key:
            continue
        tags = normalize_tags(annotation.get("tags"))
        if tags["zh"] or tags["en"]:
            index[key] = tags
    return index


def semantic_tag_title_keys(settings: Settings, paper_semantic: str) -> set[str]:
    semantic = paper_semantic.strip()
    if not semantic:
        return set()
    return {
        title
        for title, tags in load_paper_annotation_tag_index(settings).items()
        if semantic_matches_tags(semantic, tags)
    }


def semantic_matches_tags(semantic: str, tags: PaperTags) -> bool:
    semantic_text = compact_text(semantic)
    semantic_key = normalized_text_key(semantic)
    for tag in [*tags.get("zh", []), *tags.get("en", [])]:
        if semantic_matches_tag(semantic_text, semantic_key, tag):
            return True
    return False


def semantic_matches_tag(semantic_text: str, semantic_key: str, tag: str) -> bool:
    tag_text = compact_text(tag)
    if tag_text and semantic_text and (tag_text in semantic_text or semantic_text in tag_text):
        return True
    tag_key = normalized_text_key(tag)
    if tag_key and semantic_key:
        return tag_key in semantic_key or semantic_key in tag_key
    return False


def normalize_tags(value: Any) -> PaperTags:
    if not isinstance(value, dict):
        return {"zh": [], "en": []}
    return {
        "zh": normalize_string_list(value.get("zh")),
        "en": normalize_string_list(value.get("en")),
    }


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := str(item).strip())]


def title_key(title: Any) -> str:
    text = str(title or "").strip()
    return normalized_text_key(text) or " ".join(text.lower().split())


def compact_text(value: Any) -> str:
    return "".join(str(value or "").lower().split())


def combined_semantic(shared: str, local: str) -> str:
    return " ".join(part for part in [shared.strip(), local.strip()] if part)


def flatten_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def unique_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for record in records:
        key = record_key_from_dict(record)
        if key and key not in seen:
            seen.add(key)
            result.append(record)
    return result


def record_key(record: ManifestRecord) -> str:
    if record.paper_data_path:
        return Path(str(record.paper_data_path)).name
    return str(record.title or "")


def record_key_from_dict(record: dict[str, Any]) -> str:
    if record.get("_record_key"):
        return str(record["_record_key"])
    if record.get("paper_data_path"):
        return Path(str(record["paper_data_path"])).name
    return str(record.get("paper_id") or record.get("title") or "")
