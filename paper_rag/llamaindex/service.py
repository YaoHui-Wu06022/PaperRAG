"""论文库唯一正文 RAG 服务和 LlamaIndex 索引生命周期。"""

from __future__ import annotations

from functools import lru_cache
from collections import Counter
import json
import re
import unicodedata
from typing import Any, Iterable

from llama_index.core.schema import QueryBundle

from paper_rag.catalog.service import CatalogIndexNotReady, get_metadata, list_chunks, search_catalog
from paper_rag.config import Settings
from paper_rag.lexical import build_fts_query, normalize_lexical_text
from paper_rag.llamaindex.index import IndexService, LlamaIndexError
from paper_rag.llamaindex.nodes import node_metadata
from paper_rag.llamaindex.retrievers import HybridRetriever
from paper_rag.llamaindex.translation import LexicalQuery, prepare_lexical_query
from paper_rag.presentation import attach_presentation, retrieve_presentation, search_presentation
from paper_rag.routing import RetrieveDecision, RetrieveTask, RouteIntent, classify_retrieve


_BODY_REGIONS = ("abstract", "content", "appendix")
_FILTER_KEYS = {"author", "category", "year", "year_from", "year_to", "state"}
_DEFAULT_REGIONS = ("abstract", "content")
_APPENDIX_QUERY_RE = re.compile(r"(?:\bappendi(?:x|ces)\b|\bsupplement(?:ary|al)?\b|附录)", re.IGNORECASE)
_TECH_ENTITY_RE = re.compile(r"\b(?:[A-Za-z]+[A-Z][A-Za-z0-9.-]*|[A-Z]{2,}[A-Za-z0-9.-]*|[A-Za-z]+[0-9][A-Za-z0-9.-]*)\b")
_TABLE_REF_RE = re.compile(r"(?:\btable\s*|\btab\.?\s*|表\s*)(\d+)", re.IGNORECASE)
_FIGURE_REF_RE = re.compile(r"(?:\bfigure\s*|\bfig\.?\s*|图\s*)(\d+)", re.IGNORECASE)
_GENERIC_ENTITIES = {"attention", "transformer", "transformers", "model", "models", "image", "images", "network", "networks", "method", "methods"}


def search(settings: Settings, query: str, filters: dict[str, Any] | None = None, limit: int = 20) -> dict[str, Any]:
    """只搜索论文元数据，不读取正文。"""
    try:
        normalized_filters = _normalize_filters(filters)
    except ValueError as exc:
        return _search_response("invalid_input", {"query": query, "items": [], "count": 0}, [str(exc)])
    prepared = prepare_lexical_query(query, settings, purpose="metadata", filters=normalized_filters)
    try:
        records = search_catalog(
            settings,
            prepared.query,
            normalized_filters,
            _limit(limit, 100),
            fts_query=prepared.fts_query,
        )
    except ValueError as exc:
        return _search_response("invalid_input", {"query": query, "items": [], "count": 0, "query_debug": prepared.debug()}, [*prepared.warnings, str(exc)])
    except CatalogIndexNotReady:
        return _search_response("catalog_not_ready", {"query": query, "items": [], "count": 0, "query_debug": prepared.debug()}, list(prepared.warnings))
    return _search_response("ok", {"query": query, "items": [record.to_dict() for record in records], "count": len(records), "query_debug": prepared.debug()}, list(prepared.warnings))


def retrieve(settings: Settings, query: str, paper_ids: list[str] | None = None, *, filters: dict[str, Any] | None = None, task: str = "auto", limit: int = 8, mode: str = "hybrid", regions: list[str] | None = None, max_chars: int = 24000) -> dict[str, Any]:
    """唯一正文入口：先约束范围，再召回，最后执行任务策略。"""
    mode = str(mode).casefold()
    try:
        if mode not in {"lexical", "semantic", "hybrid"} or not str(query).strip():
            raise ValueError("query must be nonempty; mode must be lexical, semantic or hybrid")
        regions = _normalize_regions(regions, query)
        filters = _normalize_filters(filters)
        limit = _limit(limit, 50)
        scope = _resolve_candidates(settings, paper_ids, filters)
        if scope == []:
            return _retrieve_response("not_found", _empty_payload(query, mode=mode), ["no papers match the supplied constraints"], mode)
    except CatalogIndexNotReady:
        return _retrieve_response("catalog_not_ready", _empty_payload(query, mode=mode), [], mode)
    except LookupError as exc:
        return _retrieve_response("not_found", _empty_payload(query, mode=mode), [str(exc)], mode)
    except (TypeError, ValueError) as exc:
        return _retrieve_response("invalid_input", _empty_payload(query, mode=mode), [str(exc)], mode)
    try:
        decision = classify_retrieve(settings, query, paper_ids) if str(task).casefold() == "auto" else _explicit_task(task)
    except ValueError as exc:
        return _retrieve_response("invalid_input", _empty_payload(query, mode=mode), [str(exc)], mode)
    task = decision.task.value
    prepared = prepare_lexical_query(query, settings, purpose="body", task=task, filters=filters)
    warnings = list(prepared.warnings) + ([decision.warning] if decision.warning else [])
    debug: dict[str, Any] = {"query_debug": prepared.debug(), "constraint_paper_ids": scope, "evidence_fallback_used": False, "recall": {}}
    state: dict[str, Any] = {"query_bundle": QueryBundle(query)}
    cache: dict[str, list[dict[str, Any]]] = {}

    def recall(ids: list[str] | None, lexical: LexicalQuery = prepared, fallback: LexicalQuery | None = None, recall_mode: str = mode):
        """一次请求共享索引和问题向量，各阶段只改变论文约束。"""
        items, messages, stats = _retrieve_items(settings, query, ids, regions, recall_mode, limit, task=task, lexical_query_override=lexical, lexical_fallback_query=fallback, request_state=state)
        warnings.extend(messages)
        return items, stats

    try:
        targets: list[str] = []
        ranked: list[dict[str, Any]] = []
        if task in {"summary", "comparison"}:
            if paper_ids:
                targets = list(scope or ())
            else:
                targets, matches = _resolve_targets(settings, prepared, scope)
                debug["entity_matches"] = matches
            required = 2 if task == "comparison" else 1
            if paper_ids and len(targets) < required:
                return _retrieve_response("invalid_input", _retrieve_data(query, decision, mode, [], debug), [*warnings, "comparison requires at least two unique papers"], mode)
            if len(targets) < required:
                ranked, stats = recall(scope)
                debug["recall"]["discovery"] = stats
                discovered = _unique_ids(item.get("paper_id") for item in ranked)
                targets = _unique_ids(targets + discovered)[:required]
            if task == "comparison" and len(targets) < 2:
                return _retrieve_response("invalid_input", _retrieve_data(query, decision, mode, [], debug), [*warnings, "comparison requires at least two unique papers"], mode)
            if len(targets) > limit:
                return _retrieve_response("invalid_input", _retrieve_data(query, decision, mode, [], debug), [*warnings, "limit must accommodate every target paper"], mode)
            debug["target_paper_ids"] = targets
            if task == "comparison":
                queues = []
                dimension_query = prepared.with_terms(prepared.core_terms) if prepared.core_terms else prepared
                entity_query = prepared.with_terms(prepared.translated_entities)
                for paper_id in targets:
                    paper_ranked, stats = recall([paper_id], dimension_query, entity_query if prepared.core_terms else None)
                    debug["recall"][paper_id] = stats
                    chunks = [item for item in _paper_chunks(settings, paper_id, cache) if item.get("region") in regions]
                    if not chunks:
                        warnings.append(f"no_evidence_for_paper:{paper_id}")
                    quota = limit // len(targets) + (targets.index(paper_id) < limit % len(targets))
                    queues.append(_comparison_sources(paper_ranked, chunks, quota))
                items = _balanced_sources(queues, limit)
            else:
                if targets and not ranked:
                    ranked, stats = recall(targets)
                    debug["recall"]["targets"] = stats
                items = _ordered_sources(settings, targets, limit, query, regions, task, ranked, cache)
                debug["chapter_supplements"] = [{"paper_id": item["paper_id"], "chunk_id": item["chunk_id"], "category": _section_category(item)} for item in items if item.get("chapter_supplement")]
        else:
            preferred, matches = _resolve_targets(settings, prepared, scope)
            state["preferred_paper_ids"] = preferred
            debug["entity_matches"] = matches
            debug["preferred_paper_ids"] = preferred
            ranked, stats = recall(scope)
            debug["recall"]["primary"] = stats
            if not ranked:
                fallback = _build_evidence_fallback_query(query, debug)
                if fallback:
                    debug.update({"evidence_fallback_used": True, "evidence_fallback_query": fallback.fts_query, "evidence_fallback_reason": "primary_retrieval_empty"})
                    ranked, stats = recall(scope, fallback, recall_mode="lexical")
                    debug["recall"]["fallback"] = stats
            items = _reason_items(settings, ranked, limit, cache, regions) if task == "reason" else ranked[:limit]
            if task == "reason":
                debug["reason_windows"] = {"core_count": len(ranked), "window_count": len(items), "source_chunk_count": len({value for item in items for value in item.get("source_chunk_ids", [])})}
        _assign_source_ids(items)
        debug["final_evidence_count"] = len(items)
        debug["per_paper_evidence_count"] = dict(Counter(item["paper_id"] for item in items))
        status = _warning_status(warnings)
        if not items:
            status = "insufficient_evidence"
            warnings.append("no_evidence_chunks")
        return _retrieve_response(status, _retrieve_data(query, decision, mode, items, debug), list(dict.fromkeys(warnings)), mode)
    except CatalogIndexNotReady:
        return _retrieve_response("catalog_not_ready", _retrieve_data(query, decision, mode, [], debug), warnings, mode)


def _retrieve_items(settings: Settings, query: str, candidates: list[str] | None, regions: tuple[str, ...], mode: str, limit: int, *, task: str = "fact", lexical_query_override: LexicalQuery | None = None, lexical_fallback_query: LexicalQuery | None = None, request_state: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """保留完整融合池，复用 QueryBundle 的原问题向量。"""
    state = request_state if request_state is not None else {"query_bundle": QueryBundle(query)}
    messages = []
    index = None
    semantic = None
    if mode != "lexical" and not state.get("semantic_failed"):
        try:
            if "index" not in state:
                state["index"] = _get_index_service(settings).load()
            index = state["index"]
            semantic = _build_semantic_retriever(index, candidates, regions, settings.llamaindex_semantic_top_k)
        except CatalogIndexNotReady:
            raise
        except Exception as exc:
            state["semantic_failed"] = True
            error_status = _index_error_status(exc) if isinstance(exc, LlamaIndexError) else "milvus_unavailable"
            state["failure_warning"] = f"semantic_unavailable:{error_status}:{type(exc).__name__}"
            index = None
    if mode != "lexical" and state.get("semantic_failed"):
        messages.extend([state.get("failure_warning", "semantic_unavailable:embedding_unavailable"), "lexical_fallback"])
    effective_mode = "lexical" if mode != "lexical" and semantic is None else mode
    retriever = HybridRetriever(settings, semantic, paper_ids=candidates, regions=regions, mode=effective_mode, task=task, lexical_top_k=settings.llamaindex_lexical_top_k, semantic_top_k=settings.llamaindex_semantic_top_k, rrf_k=settings.llamaindex_rrf_k, lexical_query_override=lexical_query_override, lexical_fallback_query=lexical_fallback_query, preferred_paper_ids=state.get("preferred_paper_ids"))
    nodes = retriever.retrieve(state["query_bundle"])
    if any(value.startswith("semantic_unavailable:") for value in retriever.warnings):
        state["semantic_failed"] = True
        state["failure_warning"] = next(value for value in retriever.warnings if value.startswith("semantic_unavailable:"))
        messages.append("lexical_fallback")
    items = []
    for result in nodes:
        item = node_metadata(result.node)
        item.update({"score": result.score, "semantic_score": result.node.metadata.get("semantic_score"), "lexical_rank": result.node.metadata.get("lexical_rank"), "semantic_rank": result.node.metadata.get("semantic_rank"), "rrf_score": result.node.metadata.get("rrf_score", result.score), "page_start_display": _display_page(item.get("page_start")), "page_end_display": _display_page(item.get("page_end")), "evidence_role": "direct"})
        items.append(item)
    stats = dict(retriever.recall_debug)
    # 在任务窗口合并前保留真实两路排名，评分调试不混入回答证据。
    stats["chunk_rankings"] = [
        {"retrieval_rank": rank, **{key: item.get(key) for key in (
            "chunk_id", "paper_id", "lexical_rank", "semantic_rank", "semantic_score", "rrf_score",
        )}}
        for rank, item in enumerate(items, start=1)
    ]
    return items, messages + list(retriever.warnings), stats


def _build_semantic_retriever(index: Any, paper_ids: list[str] | None, regions: tuple[str, ...], top_k: int) -> Any:
    """为 Milvus 语义检索构造与 lexical 相同的 metadata 约束。"""

    from llama_index.core.vector_stores import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters

    filters = [MetadataFilter(key="region", value=list(regions), operator=FilterOperator.IN)]
    if paper_ids:
        filters.append(MetadataFilter(key="paper_id", value=list(paper_ids), operator=FilterOperator.IN))
    return index.as_retriever(similarity_top_k=top_k, filters=MetadataFilters(filters=filters, condition=FilterCondition.AND))


def _explicit_task(task: str) -> RetrieveDecision:
    try:
        return RetrieveDecision(RouteIntent.RETRIEVE, RetrieveTask(str(task).casefold()), provider="explicit")
    except ValueError as exc:
        raise ValueError("task must be auto, fact, reason, summary or comparison") from exc


def _resolve_candidates(settings: Settings, paper_ids: list[str] | None, filters: dict[str, Any]) -> list[str] | None:
    """仅显式 ID 和 SQL 元数据条件构成硬约束，同时检查 Catalog。"""
    records = search_catalog(settings, "", filters, 100000)
    requested = _unique_ids(paper_ids or ())
    if not requested:
        return [record.base_id for record in records] if filters else None
    allowed = {record.base_id for record in records}
    by_id = {record.base_id.casefold(): record for record in records}
    by_canonical = {record.canonical_id.casefold(): record for record in records}
    result, missing = [], []
    for value in requested:
        record = by_id.get(value.casefold()) or by_canonical.get(value.casefold()) or get_metadata(settings, value)
        if record is None or record.base_id not in allowed:
            missing.append(value)
        elif record.base_id not in result:
            result.append(record.base_id)
    if missing:
        raise LookupError("unmatched_paper_ids:" + ",".join(missing))
    return result


def _identity_text(value: str) -> str:
    """标题身份匹配只规范 Unicode、大小写和标点，不做模糊猜测。"""
    return re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def _resolve_targets(settings: Settings, prepared: LexicalQuery, scope: list[str] | None) -> tuple[list[str], list[dict[str, Any]]]:
    """对象名称只有在标题完整或方法名唯一命中时才锁定论文。"""
    records = search_catalog(settings, "", {}, 100000)
    if scope is not None:
        records = [record for record in records if record.base_id in scope]
    names = list(prepared.entities)
    if not names:
        # 改写失败时只使用原文技术名称或明确的完整标题，不把回退普通词当实体。
        names = _query_entities(prepared.original_query)
        names.extend(record.title for record in records if len(record.title) > 10 and _identity_text(record.title) in _identity_text(prepared.original_query))
    result, matches = [], []
    for name in dict.fromkeys(names):
        if _identity_text(name) not in _identity_text(prepared.original_query):
            matches.append({"entity": name, "matches": [], "resolution": "not_in_question"})
            continue
        variants = [name]
        if name in prepared.entities:
            position = prepared.entities.index(name)
            if position < len(prepared.translated_entities):
                variants.append(prepared.translated_entities[position])
        # 保留原始抽取结果，仅去掉名称边界上的泛指“论文”后缀用于标题匹配。
        variants.extend(re.sub(r"\s*(?:论文|文章|\bpaper)\s*$", "", value, flags=re.IGNORECASE).strip() for value in list(variants))
        exact = [record.base_id for record in records if any(_identity_text(record.title) == _identity_text(value) for value in variants)]
        candidates = _unique_ids(exact or [record.base_id for record in records if any(_term_hit(record.title, value) for value in variants)])
        resolution = "unique_title" if len(candidates) == 1 else "ambiguous" if candidates else "unresolved"
        matches.append({"entity": name, "matches": candidates, "resolution": resolution})
        if len(candidates) == 1:
            result.extend(candidates)
    return _unique_ids(result), matches


def _paper_chunks(settings: Settings, paper_id: str, chunk_cache: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """一次请求只读取每篇论文的 Chunk 一次。"""
    if paper_id not in chunk_cache:
        chunk_cache[paper_id] = list_chunks(settings, paper_id, 10000)
    return chunk_cache[paper_id]


def _balanced_sources(queues: list[list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    """逐篇轮流取证，空队列不占名额，避免截断最后一篇。"""
    selected, used = [], set()
    offsets = [0] * len(queues)
    while len(selected) < limit:
        progressed = False
        for index, queue in enumerate(queues):
            while offsets[index] < len(queue):
                item = queue[offsets[index]]
                offsets[index] += 1
                if item["chunk_id"] in used:
                    continue
                used.add(item["chunk_id"])
                selected.append(item)
                progressed = True
                break
            if len(selected) == limit:
                break
        if not progressed:
            break
    return selected


def _section_category(item: dict[str, Any]) -> str:
    """按最具体的章节标题识别摘要覆盖类别，父章节仅作补充。"""
    if item.get("region") == "abstract":
        return "abstract"
    headings = [str(item.get("section_label") or ""), *reversed(item.get("section_path") or [])]
    categories = (
        ("acknowledgements", r"acknowledg|致谢"),
        ("method", r"\b(method|approach|architecture|algorithm|model|framework)\b|方法|架构|算法|模型"),
        ("results", r"\b(experiment|experiments|evaluation|results?|ablation|analysis|analyses|understanding)\b|实验|结果|评估|分析"),
        ("conclusion", r"\b(conclusions?|discussion|limitations?)\b|结论|讨论|局限"),
        ("introduction", r"\b(introduction|background|related works?|problem statement)\b|引言|背景|相关工作|问题定义"),
    )
    for heading in headings:
        for category, pattern in categories:
            if re.search(pattern, heading, re.IGNORECASE):
                return category
    return "other"


def _summary_sources(ranked: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """先覆盖关键章节，再轮流覆盖未知顶层章节，最后补相关来源。"""
    ranked = [item for item in ranked if _section_category(item) != "acknowledgements"]
    chunks = sorted((item for item in chunks if str(item.get("text") or "").strip() and _section_category(item) != "acknowledgements"), key=lambda item: item.get("ordinal", 0))
    selected, used = [], set()
    for category in ("abstract", "method", "results", "conclusion", "introduction"):
        item = next((item for item in ranked if _section_category(item) == category and item["chunk_id"] not in used), None)
        if item is None:
            source = next((item for item in chunks if _section_category(item) == category and item["chunk_id"] not in used), None)
            item = {**source, "chapter_supplement": True, "score": None} if source else None
        if item:
            selected.append(item)
            used.add(item["chunk_id"])
    # 未知章节先各取一条，其余条目以轮流方式补入。
    sections: dict[str, list[dict[str, Any]]] = {}
    for item in chunks:
        if _section_category(item) == "other" and item["chunk_id"] not in used:
            path = item.get("section_path") or []
            key = str(path[1] if len(path) > 1 else item.get("section_label") or "")
            sections.setdefault(key, []).append(item)
    for item in _balanced_sources([items[:1] for items in sections.values()], len(sections)):
        if item["chunk_id"] not in used:
            selected.append({**item, "chapter_supplement": True, "score": None})
            used.add(item["chunk_id"])
    for item in ranked:
        if item["chunk_id"] not in used:
            selected.append(item)
            used.add(item["chunk_id"])
    return selected


def _ordered_sources(settings: Settings, paper_ids: Iterable[str], limit: int, query: str, regions: tuple[str, ...], task: str, ranked: list[dict[str, Any]], chunk_cache: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """逐篇章节覆盖后，轮流分配最终证据名额。"""
    queues = []
    for paper_id in _unique_ids(paper_ids):
        chunks = [item for item in _paper_chunks(settings, paper_id, chunk_cache) if item.get("region") in regions]
        candidates = [item for item in ranked if item.get("paper_id") == paper_id]
        queues.append(_summary_sources(candidates, chunks))
    return _balanced_sources(queues, limit)


def _comparison_sources(ranked: list[dict[str, Any]], chunks: list[dict[str, Any]], quota: int) -> list[dict[str, Any]]:
    """正文优先；名额足够时为该论文保留一条摘要背景。"""
    content = [item for item in ranked if item.get("region") != "abstract"]
    abstract = next((item for item in ranked if item.get("region") == "abstract"), None)
    if abstract is None:
        source = next((item for item in chunks if item.get("region") == "abstract"), None)
        abstract = {**source, "score": None} if source else None
    if content and abstract and quota >= 2:
        return content[:1] + [abstract] + content[1:]
    return content + ([abstract] if abstract else [])


def _section_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (item.get("paper_id"), item.get("region"), tuple(item.get("section_path") or [item.get("section_label")]))


def _reason_items(settings: Settings, direct: list[dict[str, Any]], limit: int, chunk_cache: dict[str, list[dict[str, Any]]], regions: tuple[str, ...] = _BODY_REGIONS) -> list[dict[str, Any]]:
    """遍历完整候选池，合并同章节重叠窗口，不再固定砍半。"""
    result: list[dict[str, Any]] = []
    for item in direct:
        if len(result) >= limit:
            break
        chunks = _paper_chunks(settings, str(item["paper_id"]), chunk_cache)
        section = sorted((chunk for chunk in chunks if chunk.get("region") in regions and _section_key(chunk) == _section_key(item)), key=lambda chunk: chunk.get("ordinal", 0))
        ordinal = int(item.get("ordinal") or 0)
        nearby = [chunk for chunk in section if abs(int(chunk.get("ordinal") or 0) - ordinal) <= 1]
        window = _build_evidence_window(item, section, nearby or [item])
        ids = set(window["source_chunk_ids"])
        overlaps = [index for index, old in enumerate(result) if _section_key(old) == _section_key(item) and ids.intersection(old["source_chunk_ids"])]
        if overlaps:
            for index in overlaps:
                ids.update(result[index]["source_chunk_ids"])
            first = overlaps[0]
            merged = [chunk for chunk in section if chunk["chunk_id"] in ids]
            result[first] = _build_evidence_window(result[first], section, merged)
            for index in reversed(overlaps[1:]):
                result.pop(index)
        else:
            result.append(window)
    return result


def _query_entities(query: str) -> list[str]:
    """提取论文方法名、缩写和模型名，避免候选发现被问题措辞淹没。"""

    values = [str(query or "")]
    entities: list[str] = []
    seen: set[str] = set()
    for value in values:
        for match in _TECH_ENTITY_RE.findall(value):
            entity = match.strip(".,:;()[]{}")
            key = entity.casefold()
            if len(entity) < 3 or key in _GENERIC_ENTITIES or key in seen:
                continue
            seen.add(key)
            entities.append(entity)
    return entities


def _query_reference(query: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(str(query or ""))
    return match.group(1) if match else None


def _focus_terms(query: str, core_terms: Iterable[str], entities: Iterable[str]) -> list[str]:
    """提取正文零证据回退使用的非通用短语，不改变论文范围。"""

    values = [str(value).strip() for value in core_terms if str(value).strip()]
    values.extend(str(value).strip() for value in entities if str(value).strip())
    if not values:
        values.append(str(query or "").strip())
    result: list[str] = []
    for value in values:
        normalized = _normalize_match_text(value)
        if len(normalized) < 3 or normalized in _GENERIC_ENTITIES:
            continue
        if normalized not in result:
            result.append(normalized)
    return result


def _build_evidence_fallback_query(query: str, retrieval_debug: dict[str, Any]) -> LexicalQuery | None:
    """构造受候选范围约束的窄词法回退查询，不重新调用模型。"""

    query_debug = retrieval_debug.get("query_debug") or {}
    core_terms = query_debug.get("translated_core_terms") or []
    entities = list(query_debug.get("translated_entities") or []) + _query_entities(query)
    terms = _focus_terms(query, core_terms, entities) if (core_terms or entities) else []
    table_ref = _query_reference(query, _TABLE_REF_RE)
    figure_ref = _query_reference(query, _FIGURE_REF_RE)
    if table_ref:
        terms.append(f"table {table_ref}")
    if figure_ref:
        terms.append(f"figure {figure_ref}")
    normalized: list[str] = []
    seen: set[str] = set()
    for term in terms:
        value, _ = normalize_lexical_text(str(term))
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    if not normalized:
        return None
    return LexicalQuery(
        original_query=str(query),
        query=" ".join(normalized),
        fts_query=build_fts_query("", remove_stopwords=False, phrases=normalized),
        translation_used=False,
        translation_provider=None,
        translation_fallback=False,
        stopwords_removed=(),
        rewriter_used=False,
        rewriter_fallback=False,
        core_terms=tuple(normalized),
        warnings=(),
        rewriter_error=None,
    )


def _term_hit(text: str, term: str) -> bool:
    normalized_text = _normalize_match_text(text)
    normalized_term = _normalize_match_text(term)
    if not normalized_term:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])"
    plural_pattern = rf"(?<![a-z0-9]){re.escape(normalized_term)}s(?![a-z0-9])"
    return bool(re.search(pattern, normalized_text) or re.search(plural_pattern, normalized_text))


def _normalize_match_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-_/]+", " ", str(value or "").casefold())).strip()


def _candidate_regions(query: str) -> tuple[str, ...]:
    """候选发现默认忽略附录，只有问题明确提及时才加入。"""

    return _DEFAULT_REGIONS + (("appendix",) if _APPENDIX_QUERY_RE.search(str(query or "")) else ())


def _build_evidence_window(core: dict[str, Any], same_section: list[dict[str, Any]], nearby: list[dict[str, Any]]) -> dict[str, Any]:
    """把同章节邻接 Chunk 合并成可读证据窗口，不改写 Catalog 原文。"""

    ordered = sorted(nearby, key=lambda item: int(item.get("ordinal") or 0))
    positions = {str(item.get("chunk_id")): index for index, item in enumerate(same_section)}
    first = positions.get(str(ordered[0].get("chunk_id"))) if ordered else None
    last = positions.get(str(ordered[-1].get("chunk_id"))) if ordered else None
    while ordered and _starts_mid_token(str(ordered[0].get("text") or "")) and first is not None and first > 0:
        first -= 1
        ordered.insert(0, same_section[first])
    while ordered and _ends_mid_token(str(ordered[-1].get("text") or "")) and last is not None and last + 1 < len(same_section):
        last += 1
        ordered.append(same_section[last])
    text = _merge_chunk_texts(ordered)
    if _starts_mid_token(text):
        text = _trim_leading_fragment(text)
    source_chunk_ids = [str(item.get("chunk_id")) for item in ordered if item.get("chunk_id")]
    pages = [item.get("page_start") for item in ordered if isinstance(item.get("page_start"), int)]
    end_pages = [item.get("page_end") for item in ordered if isinstance(item.get("page_end"), int)]
    result = dict(core)
    result.update(
        {
            "text": text or core.get("text", ""),
            "type": "text",
            "evidence_role": "context",
            "window_id": f"{core.get('paper_id')}:{core.get('ordinal')}",
            "source_chunk_ids": source_chunk_ids or [str(core.get("chunk_id"))],
            "asset_refs": _window_refs(ordered, "asset_refs"),
            "source_blocks": _window_refs(ordered, "source_blocks"),
            "continuity_status": "complete" if text and not _starts_mid_token(text) else "trimmed_prefix",
            "page_start": min(pages) if pages else core.get("page_start"),
            "page_end": max(end_pages) if end_pages else core.get("page_end"),
            "page_start_display": _display_page(min(pages)) if pages else _display_page(core.get("page_start")),
            "page_end_display": _display_page(max(end_pages)) if end_pages else _display_page(core.get("page_end")),
            "score": None,
            "semantic_score": None,
            "lexical_rank": None,
            "semantic_rank": None,
            "rrf_score": None,
            "ranking_features": {},
        }
    )
    return result


def _window_refs(chunks: list[dict[str, Any]], field: str) -> list[Any]:
    """窗口资源及来源按原文顺序去重，支持字符串和结构化引用。"""
    values, seen = [], set()
    for chunk in chunks:
        for value in chunk.get(field) or []:
            key = json.dumps(value, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                values.append(value)
                seen.add(key)
    return values


def _merge_chunk_texts(chunks: list[dict[str, Any]]) -> str:
    """合并有重叠的相邻文本，避免窗口中重复整段内容。"""

    merged = ""
    for chunk in chunks:
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        if not merged:
            merged = text
            continue
        overlap = _text_overlap(merged, text)
        merged += text[overlap:] if overlap else f"\n\n{text}"
    return merged.strip()


def _text_overlap(left: str, right: str) -> int:
    """返回左右文本的最大后缀/前缀重叠长度。"""

    left_folded = left.casefold()
    right_folded = right.casefold()
    maximum = min(240, len(left), len(right))
    for size in range(maximum, 23, -1):
        if left_folded[-size:] == right_folded[:size]:
            return size
    return 0


def _starts_mid_token(text: str) -> bool:
    return bool(re.match(r"^[a-z]", text.strip()))


def _ends_mid_token(text: str) -> bool:
    value = text.rstrip()
    return bool(value and value[-1].isalnum())


def _trim_leading_fragment(text: str) -> str:
    match = re.search(r"(?<=[.!?])\s+(?=[A-Z0-9○])", text)
    if match:
        return text[match.end():].lstrip()
    return text


def _assign_source_ids(items: list[dict[str, Any]]) -> None:
    for number, item in enumerate(items, start=1):
        item["source_id"] = f"S{number}"


def _normalize_regions(regions: list[str] | None, query: str) -> tuple[str, ...]:
    selected = tuple(str(value) for value in (regions or _candidate_regions(query)) if str(value))
    if not selected or any(value not in _BODY_REGIONS for value in selected):
        raise ValueError("regions may only contain abstract, content or appendix")
    return tuple(dict.fromkeys(selected))


def _normalize_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    if filters is None:
        return {}
    if not isinstance(filters, dict):
        raise ValueError("filters must be an object")
    unknown = sorted(set(filters) - _FILTER_KEYS)
    if unknown:
        raise ValueError(f"unsupported filters: {', '.join(unknown)}")
    result = {key: value for key, value in filters.items() if value not in (None, "")}
    for key in ("year", "year_from", "year_to"):
        if key in result and (not str(result[key]).isdigit() or len(str(result[key])) != 4):
            raise ValueError(f"{key} must be a four digit year")
    if "year_from" in result and "year_to" in result and str(result["year_from"]) > str(result["year_to"]):
        raise ValueError("year_from must not be later than year_to")
    return result


def _unique_ids(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            result.append(text)
    return result


def _display_page(value: Any) -> int | None:
    return value + 1 if isinstance(value, int) else None


def _limit(value: Any, maximum: int) -> int:
    return max(1, min(int(value), maximum))


def _index_error_status(error: LlamaIndexError) -> str:
    if "not ready" in str(error).casefold() or "stale" in str(error).casefold():
        return "index_not_ready"
    return "milvus_unavailable" if "Milvus" in str(error) else "embedding_unavailable"


def _warning_status(warnings: Iterable[str]) -> str:
    """将语义层降级 warning 映射为稳定的 MCP 状态。"""

    for warning in warnings:
        if str(warning).startswith("semantic_unavailable:"):
            if "index_not_ready" in str(warning):
                return "index_not_ready"
            return "milvus_unavailable" if "milvus" in str(warning).casefold() else "embedding_unavailable"
    return "ok"


def _empty_payload(query: str, *, mode: str | None = None) -> dict[str, Any]:
    """构造统一的空正文响应，便于 Agent 区分失败阶段。"""

    data: dict[str, Any] = {"query": str(query or ""), "evidence": [], "count": 0}
    if mode:
        data["mode"] = mode
    return data


def _retrieve_data(
    query: str,
    decision: RetrieveDecision,
    mode: str,
    items: list[dict[str, Any]],
    retrieval_debug: dict[str, Any],
) -> dict[str, Any]:
    """统一生成正文检索的可观测数据，不让主流程重复拼接字段。"""

    return {
        "query": str(query or ""),
        "task": decision.task.value,
        "mode": mode,
        "routing": decision.to_dict(),
        "retrieval_debug": retrieval_debug,
        "evidence": [_agent_evidence(item) for item in items],
        "count": len(items),
    }


def _agent_evidence(item: dict[str, Any]) -> dict[str, Any]:
    """只向 Agent 暴露正文证据、来源定位和检索评分。"""

    fields = (
        "source_id", "paper_id", "chunk_id", "text", "type",
        "section_path", "page_start", "page_end", "source_chunk_ids",
        "score",
    )
    return {field: item[field] for field in fields if field in item}


def index_status(settings: Settings) -> dict[str, Any]:
    return _envelope("ok", _get_index_service(settings).status())


def rebuild_index(settings: Settings, mode: str = "auto") -> dict[str, Any]:
    try:
        result = _get_index_service(settings).rebuild(mode=mode)
        _get_index_service.cache_clear()
        return _envelope("completed", result)
    except CatalogIndexNotReady:
        return _envelope("catalog_not_ready", {})
    except LlamaIndexError as exc:
        status = str(exc.details.get("status") or "failed")
        return _envelope(status, exc.details, [str(exc)])


def _envelope(status: str, data: dict[str, Any], warnings: list[str] | None = None) -> dict[str, Any]:
    return {"status": status, "data": data, "warnings": [item for item in (warnings or []) if item], "read_only": status not in {"completed"}}


def _search_response(status: str, data: dict[str, Any], warnings: list[str] | None = None) -> dict[str, Any]:
    """为元数据查询附加必须原样输出的确定性答案。"""

    payload = _envelope(status, data, warnings)
    return attach_presentation(payload, search_presentation(payload["data"]))


def _retrieve_response(status: str, data: dict[str, Any], warnings: list[str] | None, mode: str) -> dict[str, Any]:
    """标记正文证据由客户端原样使用还是组织生成。"""

    payload = _envelope(status, data, warnings)
    return attach_presentation(payload, retrieve_presentation(payload["data"], mode, status=status))


@lru_cache(maxsize=4)
def _get_index_service(settings: Settings) -> IndexService:
    return IndexService(settings)


__all__ = ["index_status", "rebuild_index", "retrieve", "search"]
