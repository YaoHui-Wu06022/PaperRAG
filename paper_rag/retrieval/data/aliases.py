from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...config import Settings
from .annotations_index import load_paper_annotation_entries
from .manifest_records import match_manifest_records, paper_record_key
from .utils import dedupe_alias_matches, dedupe_text_values_for_search, normalized_text, tokenize


@dataclass(frozen=True)
class AliasMatch:
    """记录一次别名命中及其对应的规范论文名。"""
    alias: str
    canonical: str


def load_paper_annotation_aliases(settings: Settings) -> list[dict[str, Any]]:
    """从人工 annotation 中读取论文别名表。"""
    entries: list[dict[str, Any]] = []
    for entry in load_paper_annotation_entries(settings):
        if entry.aliases:
            entries.append({"canonical": entry.title, "aliases": entry.aliases})
    return entries


def find_alias_matches(entries: list[dict[str, Any]], query: str) -> list[AliasMatch]:
    """用搜索 token 判断 query 是否命中某个论文别名。"""
    query_tokens = set(tokenize(query))
    matches: list[AliasMatch] = []
    for entry in entries:
        canonical = str(entry.get("canonical") or "").strip()
        aliases = [str(alias).strip() for alias in entry.get("aliases") or [] if str(alias).strip()]
        for alias in aliases:
            alias_tokens = set(tokenize(alias))
            if alias_tokens and alias_tokens.issubset(query_tokens):
                matches.append(AliasMatch(alias, canonical))
                break
    return matches


def alias_match_to_dict(match: AliasMatch) -> dict[str, Any]:
    """把 AliasMatch 裁剪成 evidence 可输出的 dict。"""
    return {
        "alias": match.alias,
        "canonical": match.canonical,
    }


def resolve_paper_queries(settings: Settings, queries: list[str]) -> tuple[list[dict[str, Any]], list[AliasMatch]]:
    """把 parser 里的论文 mention 解析成本地 manifest 论文。"""
    targets: list[dict[str, Any]] = []
    alias_matches: list[AliasMatch] = []
    seen: set[str] = set()
    alias_entries = load_paper_annotation_aliases(settings)
    for query in queries:
        query_text = str(query or "").strip()
        if not query_text:
            continue
        matches = find_alias_matches(alias_entries, query_text)
        alias_matches.extend(matches)
        candidate_queries = dedupe_text_values_for_search([
            query_text,
            *[match.canonical for match in matches if match.canonical],
        ])
        for candidate_query in candidate_queries:
            for record in match_manifest_records(settings, candidate_query):
                key = paper_record_key(record)
                if key in seen:
                    continue
                seen.add(key)
                targets.append(resolved_paper_record(record, matches))
    return targets, dedupe_alias_matches(alias_matches)


def resolved_paper_record(record: dict[str, Any], matches: list[AliasMatch]) -> dict[str, Any]:
    """在标准 manifest record 上补充解析链路需要的字段。"""
    key = paper_record_key(record)
    return {
        **record,
        "_record_key": key,
        "paper_id": key,
        "matched_alias": matched_alias_for_record(record.get("title"), matches),
    }

def matched_alias_for_record(title: Any, matches: list[AliasMatch]) -> str | None:
    """如果 record 标题等于 canonical，返回触发它的 alias。"""
    title_text = normalized_text(str(title or ""))
    for match in matches:
        if normalized_text(match.canonical) == title_text:
            return match.alias
    return None
