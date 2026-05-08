"""论文别名解析：把 parser 中的 paper mention 映射到本地 canonical title。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from paper_rag.config import Settings
from paper_rag.corpus.annotations import load_paper_annotation_entries
from paper_rag.corpus.records import match_manifest_records, paper_record_key
from paper_rag.corpus.utils import dedupe_bm25_text, dedupe_by, normalize_bm25_token, normalize_token


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
    query_tokens = set(normalize_bm25_token(query))
    matches: list[AliasMatch] = []
    for entry in entries:
        canonical = str(entry.get("canonical") or "").strip()
        aliases = [str(alias).strip() for alias in entry.get("aliases") or [] if str(alias).strip()]
        for alias in aliases:
            alias_tokens = set(normalize_bm25_token(alias))
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


def dedupe_alias_matches(matches: list[AliasMatch]) -> list[AliasMatch]:
    """按 alias/canonical 对 alias match 保序去重。"""
    return dedupe_by(matches, lambda match: (match.alias, match.canonical))


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
        # 先用用户原始 mention 直接搜 manifest，再用 alias 映射出的 canonical title 搜。
        candidate_queries = dedupe_bm25_text([
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
    title_text = normalize_token(str(title or ""))
    for match in matches:
        if normalize_token(match.canonical) == title_text:
            return match.alias
    return None
