from __future__ import annotations

from dataclasses import dataclass
import re

from langchain_core.documents import Document

from config import AppConfig
from retrieval.metadata_filter import apply_query_metadata_filter, build_doc_id_expr
from retrieval.query_router import QueryRoute
from retrieval.query_rewrite import build_query_variants, extract_canonical_title_hints
from retrieval.retriever import (
    attach_rerank_decision,
    decide_rerank,
    diversify_documents_by_source,
    rerank_documents,
    retrieve,
)
from services.local_cache_store import expand_to_parent_contexts


EVIDENCE_MAX_PER_PAGE_SECTION = 1
COMPARISON_ENTITY_MIN_EVIDENCE = 1


@dataclass
class RetrievalFlowResult:
    evidence_docs: list[Document]
    generation_docs: list[Document]
    paper_docs: list[Document]
    filtered_corpus_docs: list[Document]
    filtered_corpus_key: str | None
    query_variants: list[str]
    comparison_entities: list[str]
    route_type: str


def run_retrieval_flow(
    config: AppConfig,
    question: str,
    vector_store,
    *,
    chunk_corpus: list[Document],
    chunk_corpus_key: str | None,
    paper_vector_store=None,
    paper_corpus: list[Document] | None = None,
    paper_corpus_key: str | None = None,
    section_corpus: list[Document] | None = None,
    parent_map: dict[str, Document] | None = None,
    apply_metadata_filters: bool = True,
    retrieval_mode_override: str | None = None,
    use_parent_context: bool = True,
    route: QueryRoute | None = None,
) -> RetrievalFlowResult:
    """执行单个问题的路由化检索流程。

    顺序是刻意设计的：
    metadata 约束 -> query rewrite -> paper recall -> chunk recall ->
    rerank / diversify -> 构造生成上下文。
    """
    route = route or QueryRoute(
        route_type="factual",
        retrieval_scope="main",
        retrieval_mode=retrieval_mode_override or config.retrieval_mode,
        use_parent_context=use_parent_context,
        prefer_paper_context=False,
        paper_top_k=max(6, config.final_top_k + 2),
        chunk_top_k=config.retriever_top_k,
        final_top_k=config.final_top_k,
        prompt_mode="factual",
        reason="default",
    )
    retrieval_mode = retrieval_mode_override or route.retrieval_mode or config.retrieval_mode
    filtered_corpus = chunk_corpus
    filtered_corpus_key = chunk_corpus_key
    filtered_paper_corpus = paper_corpus or []
    filtered_paper_corpus_key = paper_corpus_key
    allowed_doc_ids = None
    metadata_milvus_expr = None
    if apply_metadata_filters:
        # Metadata filter 既会裁剪内存中的语料，
        # 也会生成可以下推到 Milvus 的 doc_id 表达式。
        metadata_seed_docs = paper_corpus or chunk_corpus
        metadata_filter_result = apply_query_metadata_filter(
            question,
            metadata_seed_docs,
            enabled=config.metadata_filter_enabled,
        )
        filtered_paper_corpus = metadata_filter_result.docs if paper_corpus else filtered_paper_corpus
        if metadata_filter_result.applied:
            filtered_corpus = _filter_docs_by_ids(
                chunk_corpus,
                metadata_filter_result.allowed_doc_ids,
            )
            filtered_corpus_key = None
            filtered_paper_corpus_key = None
        allowed_doc_ids = metadata_filter_result.allowed_doc_ids
        metadata_milvus_expr = metadata_filter_result.milvus_expr

    query_variants = build_query_variants(
        question,
        enabled=config.query_rewrite_enabled,
        max_variants=config.query_rewrite_max_variants,
    )
    title_hints = extract_canonical_title_hints(question)
    comparison_entities = (
        _extract_comparison_entities(question)
        if route.route_type == "comparison"
        else []
    )

    paper_docs: list[Document] = []
    top_paper_doc_ids: set[str] | None = None
    if filtered_paper_corpus:
        hinted_paper_corpus: list[Document] = []
        if route.route_type == "metadata" and title_hints:
            hinted_paper_corpus = _filter_docs_by_title_hints(
                filtered_paper_corpus,
                title_hints,
            )
            if hinted_paper_corpus:
                filtered_paper_corpus = hinted_paper_corpus
                filtered_paper_corpus_key = None
                # 如果 metadata 查询已经命中了某个 canonical paper alias
                #（例如 "ResNet" -> "Deep Residual Learning ..."），
                # 就直接使用命中的 paper docs，而不是再对单文档语料跑 BM25。
                # 否则单文档 BM25 的分数很容易退化到 0。
                paper_docs = _select_paper_docs_from_title_hints(
                    hinted_paper_corpus,
                    title_hints,
                    top_k=route.paper_top_k,
                )

        if not paper_docs:
            # Paper-level 检索是粗召回阶段。
            # 它先缩小搜索空间，再让 chunk 检索去找更精确的证据。
            paper_docs = retrieve(
                query=question,
                vector_store=paper_vector_store,
                top_k=route.paper_top_k,
                score_threshold=config.retrieval_score_threshold,
                score_threshold_mode=config.retrieval_score_threshold_mode,
                score_relative_ratio=config.retrieval_score_relative_ratio,
                score_quantile=config.retrieval_score_quantile,
                retrieval_mode=retrieval_mode,
                hybrid_corpus_docs=filtered_paper_corpus,
                hybrid_corpus_key=filtered_paper_corpus_key,
                hybrid_dense_top_k=max(route.paper_top_k, config.hybrid_dense_top_k),
                hybrid_bm25_top_k=max(route.paper_top_k, config.hybrid_bm25_top_k),
                hybrid_rrf_k=config.hybrid_rrf_k,
                query_variants=query_variants,
                metadata_allow_doc_ids=allowed_doc_ids,
                metadata_milvus_expr=metadata_milvus_expr,
            )
        paper_docs = _dedupe_docs_by_doc_id(paper_docs)
        top_paper_doc_ids = {
            str((doc.metadata or {}).get("doc_id", "")).strip()
            for doc in paper_docs
            if str((doc.metadata or {}).get("doc_id", "")).strip()
        }
        if top_paper_doc_ids:
            allowed_doc_ids = (
                allowed_doc_ids & top_paper_doc_ids
                if allowed_doc_ids is not None
                else top_paper_doc_ids
            )
            metadata_milvus_expr = build_doc_id_expr(allowed_doc_ids)
            filtered_corpus = _filter_docs_by_ids(filtered_corpus, allowed_doc_ids)
            filtered_corpus_key = None

    # Chunk 检索仍然是主要证据来源，
    # 但现在它只在上一步选中的 paper 集合里继续搜索，而不是全库扫一遍。
    docs = retrieve(
        query=question,
        vector_store=vector_store,
        top_k=route.chunk_top_k,
        score_threshold=config.retrieval_score_threshold,
        score_threshold_mode=config.retrieval_score_threshold_mode,
        score_relative_ratio=config.retrieval_score_relative_ratio,
        score_quantile=config.retrieval_score_quantile,
        retrieval_mode=retrieval_mode,
        hybrid_corpus_docs=filtered_corpus,
        hybrid_corpus_key=filtered_corpus_key,
        hybrid_dense_top_k=config.hybrid_dense_top_k,
        hybrid_bm25_top_k=config.hybrid_bm25_top_k,
        hybrid_rrf_k=config.hybrid_rrf_k,
        query_variants=query_variants,
        metadata_allow_doc_ids=allowed_doc_ids,
        metadata_milvus_expr=metadata_milvus_expr,
    )
    if comparison_entities:
        docs = _supplement_docs_for_comparison_entities(
            config=config,
            question=question,
            vector_store=vector_store,
            base_docs=docs,
            entities=comparison_entities,
            retrieval_mode=retrieval_mode,
            hybrid_corpus_docs=filtered_corpus,
            hybrid_corpus_key=filtered_corpus_key,
            metadata_allow_doc_ids=allowed_doc_ids,
            metadata_milvus_expr=metadata_milvus_expr,
        )

    if not docs and route.prefer_paper_context and paper_docs:
        docs = paper_docs[: route.final_top_k]

    candidate_k = max(route.final_top_k * 3, route.final_top_k + 4)
    if config.use_reranker:
        # 先对更宽的候选池做 rerank，
        # 再应用多样性或实体覆盖约束，最后截断到 final_top_k。
        decision = decide_rerank(
            docs,
            enabled=config.rerank_conditional_enabled,
            min_top1_score=config.rerank_skip_min_top1_score,
            min_score_gap=config.rerank_skip_min_score_gap,
            min_rel_gap=config.rerank_skip_min_rel_gap,
        )
        docs = attach_rerank_decision(docs, decision)
        if decision.should_rerank:
            pre_rerank_docs = docs[:candidate_k]
            reranked_docs = rerank_documents(
                query=question,
                docs=docs,
                provider=config.reranker_provider,
                reranker_model=config.reranker_model,
                api_key=config.reranker_api_key,
                base_url=config.reranker_base_url,
                top_k=candidate_k,
                score_threshold=config.reranker_score_threshold,
            )
            if not reranked_docs:
                docs = pre_rerank_docs
            else:
                docs = reranked_docs
                if comparison_entities:
                    docs = _fill_missing_entities_from_backup(
                        primary_docs=docs,
                        backup_docs=pre_rerank_docs,
                        entities=comparison_entities,
                        top_k=candidate_k,
                    )
        else:
            docs = docs[:candidate_k]
    else:
        docs = docs[:candidate_k]

    if comparison_entities:
        docs = _select_docs_with_entity_coverage(
            docs,
            entities=comparison_entities,
            top_k=route.final_top_k,
            diversify_by_source=config.diversify_by_source,
            max_per_source=config.max_chunks_per_source,
        )
    elif config.diversify_by_source:
        docs = diversify_documents_by_source(
            docs,
            max_per_source=config.max_chunks_per_source,
            top_k=route.final_top_k,
        )
    else:
        docs = docs[: route.final_top_k]
    docs = _dedupe_and_diversify_evidence_docs(docs, top_k=route.final_top_k)

    generation_docs = []
    if route.prefer_paper_context and paper_docs:
        generation_docs = _build_paper_generation_docs(
            question=question,
            route=route,
            paper_docs=paper_docs,
            section_corpus=section_corpus or [],
            allowed_doc_ids=allowed_doc_ids,
            query_variants=query_variants,
        )
    elif docs:
        if route.use_parent_context and use_parent_context and parent_map is not None:
            generation_docs = expand_to_parent_contexts(config, docs, parent_map)
        else:
            generation_docs = docs[: route.final_top_k]

    return RetrievalFlowResult(
        evidence_docs=docs,
        generation_docs=generation_docs,
        paper_docs=paper_docs,
        filtered_corpus_docs=filtered_corpus,
        filtered_corpus_key=filtered_corpus_key,
        query_variants=query_variants,
        comparison_entities=comparison_entities,
        route_type=route.route_type,
    )


def _filter_docs_by_ids(
    docs: list[Document],
    doc_ids: set[str] | None,
) -> list[Document]:
    if not doc_ids:
        return docs
    allowed = {str(item).strip() for item in doc_ids if str(item).strip()}
    if not allowed:
        return docs
    return [
        doc
        for doc in docs
        if str((doc.metadata or {}).get("doc_id", "")).strip() in allowed
    ]


def _dedupe_docs_by_doc_id(docs: list[Document]) -> list[Document]:
    deduped: list[Document] = []
    seen: set[str] = set()
    for doc in docs:
        metadata = dict(doc.metadata or {})
        doc_id = str(metadata.get("doc_id", "")).strip()
        key = doc_id or str(metadata.get("source", "")).strip() or str(id(doc))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def _filter_docs_by_title_hints(
    docs: list[Document],
    title_hints: list[str],
) -> list[Document]:
    normalized_hints = [str(item).strip().lower() for item in title_hints if str(item).strip()]
    if not normalized_hints:
        return docs

    matched: list[Document] = []
    for doc in docs:
        metadata = dict(doc.metadata or {})
        title = str(metadata.get("title", metadata.get("paper_title", ""))).strip().lower()
        if not title:
            continue
        if any(hint in title for hint in normalized_hints):
            matched.append(doc)
    return matched


def _select_paper_docs_from_title_hints(
    docs: list[Document],
    title_hints: list[str],
    *,
    top_k: int,
) -> list[Document]:
    normalized_hints = [str(item).strip().lower() for item in title_hints if str(item).strip()]
    if not normalized_hints:
        return docs[:top_k]

    def _score(doc: Document) -> tuple[int, int]:
        metadata = dict(doc.metadata or {})
        title = str(metadata.get("title", metadata.get("paper_title", ""))).strip().lower()
        exact_hits = sum(1 for hint in normalized_hints if hint == title)
        partial_hits = sum(1 for hint in normalized_hints if hint in title)
        return (exact_hits, partial_hits)

    ranked = sorted(
        docs,
        key=lambda doc: _score(doc),
        reverse=True,
    )
    return ranked[:top_k]


def _build_paper_generation_docs(
    *,
    question: str,
    route: QueryRoute,
    paper_docs: list[Document],
    section_corpus: list[Document],
    allowed_doc_ids: set[str] | None,
    query_variants: list[str],
) -> list[Document]:
    """为 paper-centric 路由构造 LLM 上下文。

    这里先放 paper summary，再补少量 section summary，
    让模型看到带结构的信息，而不是一口气塞进大量原始 chunk。
    """
    selected = list(paper_docs[: route.final_top_k])
    if not section_corpus:
        return selected
    scoped_sections = _filter_docs_by_ids(section_corpus, allowed_doc_ids)
    if not scoped_sections:
        return selected
    section_docs = retrieve(
        query=question,
        vector_store=None,
        top_k=max(2, route.final_top_k),
        retrieval_mode="bm25",
        hybrid_corpus_docs=scoped_sections,
        hybrid_corpus_key=None,
        query_variants=query_variants,
        metadata_allow_doc_ids=allowed_doc_ids,
    )
    seen = {
        (
            str((doc.metadata or {}).get("doc_id", "")).strip(),
            str((doc.metadata or {}).get("representation", "")).strip(),
            str((doc.metadata or {}).get("section_name", "")).strip(),
        )
        for doc in selected
    }
    for doc in section_docs:
        identity = (
            str((doc.metadata or {}).get("doc_id", "")).strip(),
            str((doc.metadata or {}).get("representation", "")).strip(),
            str((doc.metadata or {}).get("section_name", "")).strip(),
        )
        if identity in seen:
            continue
        selected.append(doc)
        seen.add(identity)
        if len(selected) >= max(route.final_top_k + 2, route.final_top_k):
            break
    return selected


def _is_comparison_query(query: str) -> bool:
    text = query.strip().lower()
    if not text:
        return False
    hints = (
        "比较",
        "对比",
        "区别",
        "差异",
        "异同",
        "共通",
        "共同点",
        "相同",
        "不同",
        "compare",
        "comparison",
        "difference",
        "different",
        "common",
        "similar",
        "vs",
    )
    has_hint = any(item in text for item in hints)
    has_connector = bool(
        re.search(
            r"(和|与|及|以及|跟|vs|/|、|&|\band\b)",
            text,
            flags=re.IGNORECASE,
        )
    )
    model_like = {
        item.lower()
        for item in re.findall(r"[A-Za-z][A-Za-z0-9._+-]{1,30}", query)
        if len(item) >= 3
    }
    has_two_entities = len(model_like) >= 2
    return (has_hint and (has_connector or has_two_entities)) or (
        has_connector and has_two_entities
    )


def _clean_entity_term(term: str) -> str:
    text = " ".join(term.strip().split())
    if not text:
        return ""
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(
        r"(的论文|模型|方法|架构|创新点|创新|共同点|共通点|区别|差异|异同|比较|对比|common points?|differences?|compare|comparison)$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip(" ,，。？?；;：:")


def _extract_comparison_entities(query: str) -> list[str]:
    if not _is_comparison_query(query):
        return []

    text = query.strip()
    cut_markers = (
        "有什么",
        "是什么",
        "区别",
        "差异",
        "异同",
        "比较",
        "对比",
        "共通",
        "共同点",
        "difference",
        "differences",
        "common",
        "similar",
        "compare",
    )
    cutoff = len(text)
    for marker in cut_markers:
        index = text.lower().find(marker.lower())
        if index >= 0:
            cutoff = min(cutoff, index)
    if cutoff < len(text):
        text = text[:cutoff]
    if "的" in text:
        text = text.split("的", 1)[0]

    parts = [
        _clean_entity_term(item)
        for item in re.split(
            r"(?:和|与|及|以及|跟|vs\.?|VS|/|、|&|\band\b)",
            text,
            flags=re.IGNORECASE,
        )
    ]
    parts = [item for item in parts if item]

    model_terms = [
        item
        for item in re.findall(r"[A-Za-z][A-Za-z0-9._+-]{1,30}", query)
        if len(item) >= 3
    ]
    if len(parts) < 2 and len(model_terms) >= 2:
        parts = model_terms

    entities: list[str] = []
    seen = set()
    for item in parts:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(item)
        if len(entities) >= 4:
            break
    return entities


def _doc_matches_entity(doc: Document, entity: str) -> bool:
    needle = entity.strip().lower()
    if not needle:
        return False
    metadata = dict(doc.metadata or {})
    fields = [
        str(metadata.get("source", "")),
        str(metadata.get("title", metadata.get("paper_title", ""))),
        str(metadata.get("section_path", "")),
        doc.page_content[:3000],
    ]
    return needle in "\n".join(fields).lower()


def _supplement_docs_for_comparison_entities(
    *,
    config: AppConfig,
    question: str,
    vector_store,
    base_docs: list[Document],
    entities: list[str],
    retrieval_mode: str,
    hybrid_corpus_docs: list[Document],
    hybrid_corpus_key: str | None,
    metadata_allow_doc_ids: set[str] | None,
    metadata_milvus_expr: str | None,
) -> list[Document]:
    if not entities:
        return base_docs

    merged = list(base_docs)
    seen_identity = {_evidence_identity(doc) for doc in merged}
    existing_hits = {entity: 0 for entity in entities}
    for doc in merged:
        for entity in entities:
            if _doc_matches_entity(doc, entity):
                existing_hits[entity] = existing_hits.get(entity, 0) + 1

    missing_entities = [
        entity
        for entity in entities
        if existing_hits.get(entity, 0) < COMPARISON_ENTITY_MIN_EVIDENCE
    ]
    if not missing_entities:
        return merged

    entity_top_k = max(config.retriever_top_k, config.final_top_k * 2, 6)
    for entity in missing_entities:
        entity_query = f"{entity} 论文 主要创新点"
        entity_variants = build_query_variants(
            entity_query,
            enabled=config.query_rewrite_enabled,
            max_variants=config.query_rewrite_max_variants,
        )
        candidates = retrieve(
            query=entity_query,
            vector_store=vector_store,
            top_k=entity_top_k,
            score_threshold=config.retrieval_score_threshold,
            score_threshold_mode=config.retrieval_score_threshold_mode,
            score_relative_ratio=config.retrieval_score_relative_ratio,
            score_quantile=config.retrieval_score_quantile,
            retrieval_mode=retrieval_mode,
            hybrid_corpus_docs=hybrid_corpus_docs,
            hybrid_corpus_key=hybrid_corpus_key,
            hybrid_dense_top_k=config.hybrid_dense_top_k,
            hybrid_bm25_top_k=config.hybrid_bm25_top_k,
            hybrid_rrf_k=config.hybrid_rrf_k,
            query_variants=entity_variants,
            metadata_allow_doc_ids=metadata_allow_doc_ids,
            metadata_milvus_expr=metadata_milvus_expr,
        )
        if not candidates and config.retrieval_score_threshold is not None:
            candidates = retrieve(
                query=entity_query,
                vector_store=vector_store,
                top_k=entity_top_k,
                score_threshold=None,
                score_threshold_mode=config.retrieval_score_threshold_mode,
                score_relative_ratio=config.retrieval_score_relative_ratio,
                score_quantile=config.retrieval_score_quantile,
                retrieval_mode=retrieval_mode,
                hybrid_corpus_docs=hybrid_corpus_docs,
                hybrid_corpus_key=hybrid_corpus_key,
                hybrid_dense_top_k=config.hybrid_dense_top_k,
                hybrid_bm25_top_k=config.hybrid_bm25_top_k,
                hybrid_rrf_k=config.hybrid_rrf_k,
                query_variants=entity_variants,
                metadata_allow_doc_ids=metadata_allow_doc_ids,
                metadata_milvus_expr=metadata_milvus_expr,
            )

        picked = None
        for doc in candidates:
            identity = _evidence_identity(doc)
            if identity in seen_identity:
                continue
            if _doc_matches_entity(doc, entity):
                picked = doc
                break
        if picked is None:
            for doc in candidates:
                identity = _evidence_identity(doc)
                if identity not in seen_identity:
                    picked = doc
                    break
        if picked is not None:
            merged.append(picked)
            seen_identity.add(_evidence_identity(picked))
    return merged


def _select_docs_with_entity_coverage(
    docs: list[Document],
    *,
    entities: list[str],
    top_k: int,
    diversify_by_source: bool,
    max_per_source: int,
) -> list[Document]:
    if not docs:
        return []

    limit = max(1, int(top_k))
    source_limit = max(1, int(max_per_source))
    ordered = _dedupe_and_diversify_evidence_docs(docs, top_k=max(len(docs), limit))
    selected: list[Document] = []
    selected_ids = set()
    source_counts: dict[str, int] = {}

    def can_add(doc: Document) -> bool:
        if not diversify_by_source:
            return True
        source = str((doc.metadata or {}).get("source", "unknown"))
        return source_counts.get(source, 0) < source_limit

    def add_doc(doc: Document) -> None:
        identity = _evidence_identity(doc)
        if identity in selected_ids:
            return
        selected.append(doc)
        selected_ids.add(identity)
        source = str((doc.metadata or {}).get("source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1

    for entity in entities:
        for doc in ordered:
            if len(selected) >= limit:
                break
            if not _doc_matches_entity(doc, entity):
                continue
            if not can_add(doc):
                continue
            add_doc(doc)
            break

    for doc in ordered:
        if len(selected) >= limit:
            break
        if not can_add(doc):
            continue
        add_doc(doc)

    if len(selected) < limit:
        for doc in ordered:
            if len(selected) >= limit:
                break
            add_doc(doc)

    return selected[:limit]


def _fill_missing_entities_from_backup(
    *,
    primary_docs: list[Document],
    backup_docs: list[Document],
    entities: list[str],
    top_k: int,
) -> list[Document]:
    if not entities:
        return primary_docs[:top_k]

    selected = list(primary_docs)
    seen_identity = {_evidence_identity(doc) for doc in selected}
    counts = {
        entity: sum(1 for doc in selected if _doc_matches_entity(doc, entity))
        for entity in entities
    }

    for entity in entities:
        if counts.get(entity, 0) >= COMPARISON_ENTITY_MIN_EVIDENCE:
            continue
        for doc in backup_docs:
            identity = _evidence_identity(doc)
            if identity in seen_identity:
                continue
            if not _doc_matches_entity(doc, entity):
                continue
            selected.append(doc)
            seen_identity.add(identity)
            counts[entity] = counts.get(entity, 0) + 1
            break

    if len(selected) < top_k:
        for doc in backup_docs:
            identity = _evidence_identity(doc)
            if identity in seen_identity:
                continue
            selected.append(doc)
            seen_identity.add(identity)
            if len(selected) >= top_k:
                break
    return selected[:top_k]


def _first_block_id(doc: Document) -> str:
    metadata = dict(doc.metadata or {})
    block_ids = metadata.get("mineru_block_ids")
    if isinstance(block_ids, list) and block_ids:
        return str(block_ids[0])
    if isinstance(block_ids, str) and block_ids.strip():
        return block_ids.strip()
    parent_id = str(metadata.get("parent_id", "")).strip()
    if parent_id:
        return parent_id
    return f"chunk_{metadata.get('chunk_id', '?')}"


def _evidence_identity(doc: Document) -> str:
    metadata = dict(doc.metadata or {})
    doc_id = str(metadata.get("doc_id", "")).strip()
    if not doc_id:
        doc_id = str(metadata.get("source", "unknown")).strip() or "unknown"
    return f"{doc_id}::{_first_block_id(doc)}"


def _page_label(doc: Document) -> str:
    value = (doc.metadata or {}).get("page")
    try:
        page = int(value)
        return str(page) if page > 0 else "?"
    except (TypeError, ValueError):
        return "?"


def _section_label(doc: Document) -> str:
    section = str((doc.metadata or {}).get("section_path", "")).strip()
    return section or "正文"


def _evidence_bucket(doc: Document) -> tuple[str, str, str]:
    metadata = dict(doc.metadata or {})
    doc_id = str(metadata.get("doc_id", "")).strip()
    if not doc_id:
        doc_id = str(metadata.get("source", "unknown")).strip() or "unknown"
    return (doc_id, _page_label(doc), _section_label(doc))


def _dedupe_and_diversify_evidence_docs(
    docs: list[Document],
    *,
    top_k: int,
    max_per_page_section: int = EVIDENCE_MAX_PER_PAGE_SECTION,
) -> list[Document]:
    if not docs:
        return []

    limit = max(1, int(top_k))
    bucket_limit = max(1, int(max_per_page_section))
    selected: list[Document] = []
    seen_identity: set[str] = set()
    bucket_counts: dict[tuple[str, str, str], int] = {}

    for doc in docs:
        identity = _evidence_identity(doc)
        if identity in seen_identity:
            continue
        bucket = _evidence_bucket(doc)
        if bucket_counts.get(bucket, 0) >= bucket_limit:
            continue
        selected.append(doc)
        seen_identity.add(identity)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if len(selected) >= limit:
            return selected

    if len(selected) < limit:
        for doc in docs:
            identity = _evidence_identity(doc)
            if identity in seen_identity:
                continue
            selected.append(doc)
            seen_identity.add(identity)
            if len(selected) >= limit:
                break
    return selected
