"""把 paper scope 转成候选论文 records。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from paper_rag.config import Settings
from paper_rag.ingest.manifest import ManifestRecord
from paper_rag.corpus.aliases import resolve_paper_queries
from paper_rag.corpus.annotation_index import PaperTags, load_paper_annotation_entries, paper_title_key
from paper_rag.corpus.citation_index import record_matches_citation_scope
from paper_rag.corpus.filters import compare_text, match_record_filters
from paper_rag.corpus.records import (
    dedupe_paper_records,
    load_active_manifest_records,
    match_manifest_records,
    paper_record_key,
    to_evidence_manifest_record,
)
from paper_rag.corpus.utils import normalize_token, value_to_text_list

if TYPE_CHECKING:
    from paper_rag.corpus.context import CorpusContext


def records_for_scope(
    settings: Settings,
    paper_semantic: str,
    filters: list[dict[str, Any]],
    group_mode: str = "single",
    *,
    corpus: "CorpusContext | None" = None,
) -> list[dict[str, Any]]:
    """根据 semantic 和已解析 filters 找到候选论文 records。"""
    semantic = paper_semantic.strip()
    semantic_keys = semantic_candidate_keys(settings, semantic, corpus=corpus)
    if semantic and not semantic_keys:
        # 有 semantic 但没有任何标题/tag 召回时，直接返回空候选集。
        return []
    records: list[dict[str, Any]] = []
    active_records = corpus.active_manifest_records if corpus else load_active_manifest_records(settings)
    for record in active_records:
        if semantic and paper_record_key(record) not in semantic_keys:
            continue
        if match_scope_filters(settings, record, filters, corpus=corpus):
            records.append(to_evidence_manifest_record(record))
    return records


def resolve_scope_records(
    settings: Settings,
    semantic: str,
    filters: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    mode: str,
    *,
    corpus: "CorpusContext | None" = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """统一展开 single/per/or/and paper scope，返回扁平 records 和逐组结果。"""
    if mode not in {"per", "or", "and"}:
        return records_for_scope(settings, semantic, filters, mode, corpus=corpus), []

    group_results = [
        {
            "semantic": group.get("semantic") or "",
            "filters": group.get("filters") or [],
            "records": records_for_scope(
                settings,
                combined_semantic(semantic, group.get("semantic") or ""),
                [*filters, *(group.get("filters") or [])],
                mode,
                corpus=corpus,
            ),
        }
        for group in groups
    ]
    records = dedupe_paper_records([
        record
        for group in group_results
        for record in group["records"]
    ])
    return records, group_results


def match_scope_filters(
    settings: Settings,
    record: ManifestRecord,
    filters: list[dict[str, Any]],
    *,
    corpus: "CorpusContext | None" = None,
) -> bool:
    """判断 record 是否满足 scope 中的 paper filters 和 metadata filters。"""
    paper_filters = [filter_item for filter_item in filters if filter_item.get("field") == "paper"]
    metadata_filters = [filter_item for filter_item in filters if filter_item.get("field") != "paper"]
    for filter_item in paper_filters:
        matched = match_paper_filter(settings, record, filter_item, corpus=corpus)
        if filter_item.get("negated"):
            matched = not matched
        if not matched:
            return False
    return match_record_filters(settings, record, metadata_filters)


def match_paper_filter(
    settings: Settings,
    record: ManifestRecord,
    filter_item: dict[str, Any],
    *,
    corpus: "CorpusContext | None" = None,
) -> bool:
    """匹配 paper = / follow / prior 三类论文范围条件。"""
    op = filter_item.get("op")
    values = value_to_text_list(filter_item.get("value"))
    if not values:
        return False
    if op == "=":
        return any(compare_text(record.title, "=", value) for value in values)
    if op in {"follow", "prior"}:
        graph = corpus.citation_graph if corpus else None
        if corpus and graph is None:
            return False
        return record_matches_citation_scope(settings, paper_record_key(record), values, op, graph=graph)
    return False


def semantic_candidate_keys(
    settings: Settings,
    paper_semantic: str,
    *,
    corpus: "CorpusContext | None" = None,
) -> set[str]:
    """把 paper_semantic 转成候选论文 key 集合。"""
    semantic = paper_semantic.strip()
    if not semantic:
        return set()
    keys: set[str] = set()
    active_records = corpus.active_manifest_records if corpus else load_active_manifest_records(settings)
    semantics = semantic_query_variants(semantic)
    for value in semantics:
        matches = match_manifest_records(settings, value, records=active_records)
        keys.update(paper_record_key(record) for record in matches if paper_record_key(record))
        tag_title_keys = semantic_tag_title_keys(settings, value, corpus=corpus)
        if tag_title_keys:
            # tags 先按 title key 对齐 annotation 和 manifest，再转成统一 paper_record_key。
            keys.update(
                paper_record_key(record)
                for record in active_records
                if paper_title_key(record.title) in tag_title_keys
            )
    alias_papers, _ = resolve_paper_queries(settings, semantics, corpus=corpus)
    keys.update(paper_record_key(record) for record in alias_papers if paper_record_key(record))
    return keys


def semantic_query_variants(semantic: str) -> list[str]:
    """生成 paper_semantic 的匹配变体，去掉常见中文论文描述后缀。"""
    stripped = strip_paper_semantic_suffix(semantic)
    values = [semantic, stripped]
    return [value for index, value in enumerate(values) if value and value not in values[:index]]


def strip_paper_semantic_suffix(semantic: str) -> str:
    """清理 parser 常把论文别名带上的中文描述后缀。"""
    value = semantic.strip()
    for suffix in ["这篇论文", "原论文", "论文"]:
        if value.endswith(suffix):
            return value[:-len(suffix)].strip(" \t\r\n，。！？?；;：:、")
    return value


def load_paper_annotation_tag_index(
    settings: Settings,
    *,
    corpus: "CorpusContext | None" = None,
) -> dict[str, PaperTags]:
    """从 annotation entries 构建标题 key 到 tags 的索引。"""
    if corpus:
        return corpus.annotation_tag_index
    index: dict[str, PaperTags] = {}
    for entry in load_paper_annotation_entries(settings):
        if entry.tags["zh"] or entry.tags["en"]:
            index[entry.paper_title_key] = entry.tags
    return index


def semantic_tag_title_keys(
    settings: Settings,
    paper_semantic: str,
    *,
    corpus: "CorpusContext | None" = None,
) -> set[str]:
    """找出 tags 能匹配 paper_semantic 的论文标题 key。"""
    semantic = paper_semantic.strip()
    if not semantic:
        return set()
    return {
        title
        for title, tags in load_paper_annotation_tag_index(settings, corpus=corpus).items()
        if semantic_matches_tags(semantic, tags)
    }


def semantic_matches_tags(semantic: str, tags: PaperTags) -> bool:
    """判断 semantic 是否命中一篇论文的中英文标签。"""
    semantic_text = compact_text(semantic)
    semantic_key = normalize_token(semantic)
    for tag in [*tags.get("zh", []), *tags.get("en", [])]:
        if semantic_matches_tag(semantic_text, semantic_key, tag):
            return True
    return False


def semantic_matches_tag(semantic_text: str, semantic_key: str, tag: str) -> bool:
    """判断 semantic 与单个 tag 是否互相包含。"""
    tag_text = compact_text(tag)
    if tag_text and semantic_text and (tag_text in semantic_text or semantic_text in tag_text):
        return True
    tag_key = normalize_token(tag)
    if tag_key and semantic_key:
        return tag_key in semantic_key or semantic_key in tag_key
    return False


def compact_text(value: Any) -> str:
    """生成中英文标签包含判断用的紧凑文本。"""
    return "".join(str(value or "").lower().split())


def combined_semantic(shared: str, local: str) -> str:
    """合并全局 semantic 和 group 内 semantic。"""
    return " ".join(part for part in [shared.strip(), local.strip()] if part)
