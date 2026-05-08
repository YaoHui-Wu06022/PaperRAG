"""content 专用检索 query 生成：dense 自然语言，BM25 关键词候选。"""

from __future__ import annotations

import re
from typing import Any

from paper_rag.config import Settings
from paper_rag.corpus.utils import value_to_text_list
from paper_rag.corpus.utils import dedupe_text
from paper_rag.retrieval.route import RouteDecision
from paper_rag.retrieval.routes.content.translation import KeywordTranslatorProtocol, translate_bm25_terms


QUERY_STOP_PHRASES = (
    "是什么",
    "是多少",
    "有哪些",
    "哪些",
    "分别",
    "各自",
    "是否",
    "有没有",
    "这篇论文",
    "论文",
    "使用了",
    "用了",
    "的",
    "吗",
)


def build_content_retrieval_query(
    settings: Settings,
    route: RouteDecision,
    warnings: list[str],
    *,
    translator: KeywordTranslatorProtocol | None = None,
) -> dict[str, Any]:
    """根据 content parser 结果组装 dense/BM25 两套检索 query。"""
    parser_result = route.parser_result or {}
    compare_objects = list(parser_result.get("compare_objects") or [])
    content_objects = list(parser_result.get("content_objects") or [])
    excluded_scope_terms = scope_query_exclusion_terms(route)
    compare_terms = non_scope_compare_objects(compare_objects, excluded_scope_terms)
    cleaned_query = remove_scope_terms_from_query(route.query, excluded_scope_terms)
    query_terms = query_keyword_terms(cleaned_query) if not content_objects and not compare_terms else []
    source_terms = {
        # source_terms 只用于 debug，帮助检查哪些词被纳入或排除。
        "content_objects": content_objects,
        "compare_objects": compare_objects,
        "compare_terms": compare_terms,
        "query_terms": query_terms,
        "excluded_scope_terms": excluded_scope_terms,
    }
    base_bm25_terms = dedupe_text([*content_objects, *compare_terms, *query_terms])
    translated_terms = translate_bm25_terms(settings, base_bm25_terms, translator=translator, warnings=warnings)
    bm25_queries = dedupe_text([*base_bm25_terms, *translated_terms]) or [route.query]
    return {
        "dense_query": build_dense_query(route.query, route.intent, content_objects, compare_objects),
        "bm25_queries": bm25_queries,
        "source_terms": source_terms,
        "intent": route.intent,
        "content_objects": content_objects,
        "compare_objects": compare_objects,
    }


def build_dense_query(
    query: str,
    intent: str | None,
    content_objects: list[str],
    compare_objects: list[str],
) -> str:
    """把 content 语义结构改写成适合 embedding 的中文自然语言句子。"""
    object_text = "、".join(content_objects)
    compare_text = "、".join(compare_objects)
    if intent == "compare" and compare_text and object_text:
        return f"比较{compare_text}在{object_text}方面的差异和相关描述"
    if intent == "compare" and compare_text:
        return f"比较{compare_text}之间的差异和相关描述"
    if intent == "exists" and object_text:
        return f"判断论文中是否提到、使用或包含{object_text}"
    if intent == "count" and object_text:
        return f"查找与{object_text}数量和统计相关的信息"
    if object_text:
        return f"查找论文中关于{object_text}的相关内容"
    if compare_text:
        return f"查找论文中关于{compare_text}的相关内容"
    return query


def query_keyword_terms(query: str) -> list[str]:
    """从剩余原问题中抽取 BM25 fallback 关键词。"""
    terms: list[str] = []
    # 英文缩写/术语通常直接能命中论文正文，因此先保留连续 latin phrase。
    latin_phrase = " ".join(re.findall(r"[A-Za-z][A-Za-z0-9-]*", query))
    if latin_phrase:
        terms.append(latin_phrase)
    cleaned = query
    for phrase in QUERY_STOP_PHRASES:
        cleaned = cleaned.replace(phrase, " ")
    cleaned = re.sub(r"[，。！？?；;：:、（）()【】\\[\\]\"'“”‘’]", " ", cleaned)
    terms.extend(part.strip() for part in cleaned.split() if part.strip())
    return dedupe_text(terms)


def non_scope_compare_objects(compare_objects: list[str], scope_terms: list[str]) -> list[str]:
    """保留正文比较对象，过滤掉已作为论文 scope 的 compare_objects。"""
    scope_keys = {scope_term_key(term) for term in scope_terms if scope_term_key(term)}
    return [
        compare_object
        for compare_object in dedupe_text(compare_objects)
        if scope_term_key(compare_object) not in scope_keys
    ]


def scope_term_key(value: str) -> str:
    """用于判断 compare_object 是否已经被论文 scope 吸收。"""
    return str(value or "").strip().casefold()


def scope_query_exclusion_terms(route: RouteDecision) -> list[str]:
    """收集已结构化为 scope 的值，避免它们再次进入 BM25 query。"""
    terms: list[str] = []
    terms.append(route.paper_semantic)
    terms.extend(scope_filter_values(route.filters))
    for group in route.paper_groups:
        terms.append(str(group.get("semantic") or ""))
        terms.extend(scope_filter_values(group.get("filters") or []))
    for paper in route.resolved_papers:
        terms.extend([
            str(paper.get("title") or ""),
            str(paper.get("matched_alias") or ""),
        ])
    for match in route.alias_matches:
        terms.extend([match.alias, match.canonical])
    return dedupe_text(terms)


def scope_filter_values(filters: list[dict[str, Any]]) -> list[str]:
    """把 filters 中的结构化范围值展开成可从 query 中扣掉的文本。"""
    values: list[str] = []
    for filter_item in filters:
        values.extend(value_to_text_list(filter_item.get("value")))
    return values


def remove_scope_terms_from_query(query: str, scope_terms: list[str]) -> str:
    """从原问题中删除已进入 scope 的短语，剩余部分再抽 BM25 关键词。"""
    cleaned = query
    # 长词优先删除，避免先删短 alias 破坏完整标题/术语。
    for term in sorted(scope_terms, key=len, reverse=True):
        text = str(term or "").strip()
        if len(text) <= 1:
            continue
        cleaned = re.sub(re.escape(text), " ", cleaned, flags=re.IGNORECASE)
    return cleaned
