from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
import re
from typing import Any

from langchain_core.documents import Document


_RERANKER_CACHE: dict[str, Any] = {}
_BM25_CACHE: dict[str, tuple[Any, list[Document]]] = {}


@dataclass(frozen=True)
class RerankDecision:
    should_rerank: bool
    reason: str
    score_key: str | None = None
    top1_score: float | None = None
    top2_score: float | None = None
    score_gap: float | None = None
    score_rel_gap: float | None = None


def build_retriever(vector_store, top_k: int):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def _copy_doc(doc: Document, **extra_metadata: Any) -> Document:
    metadata = dict(doc.metadata or {})
    metadata.update(extra_metadata)
    return Document(page_content=doc.page_content, metadata=metadata)


def _attach_score(doc: Document, key: str, score: float) -> Document:
    return _copy_doc(doc, **{key: float(score)})


def _looks_like_similarity_scores(scores: list[float]) -> bool:
    if not scores:
        return False
    return min(scores) >= 0.0 and max(scores) <= 1.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _score_threshold_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized in {"absolute", "relative", "quantile"}:
        return normalized
    return "absolute"


def _quantile_value(values: list[float], q: float) -> float:
    if not values:
        return float("-inf")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    q = _clamp(q, 0.0, 1.0)
    pos = q * (len(ordered) - 1)
    lower_idx = int(pos)
    upper_idx = min(lower_idx + 1, len(ordered) - 1)
    lower = ordered[lower_idx]
    upper = ordered[upper_idx]
    ratio = pos - lower_idx
    return lower + (upper - lower) * ratio


def _compute_dynamic_threshold(
    scores: list[float],
    *,
    mode: str,
    absolute_threshold: float | None,
    relative_ratio: float,
    quantile: float,
) -> float | None:
    if not scores:
        return absolute_threshold
    mode = _score_threshold_mode(mode)
    if mode == "absolute":
        return absolute_threshold

    if mode == "relative":
        top = max(scores)
        bottom = min(scores)
        ratio = _clamp(relative_ratio, 0.0, 1.0)
        dynamic = top - (top - bottom) * (1.0 - ratio)
    else:
        dynamic = _quantile_value(scores, quantile)

    if absolute_threshold is None:
        return dynamic
    return max(dynamic, absolute_threshold)


def _filter_docs_by_score(
    docs: list[Document],
    score_key: str,
    threshold: float | None,
    *,
    threshold_mode: str = "absolute",
    relative_ratio: float = 0.7,
    quantile: float = 0.6,
) -> list[Document]:
    if threshold is None and _score_threshold_mode(threshold_mode) == "absolute":
        return docs
    if not any(score_key in (doc.metadata or {}) for doc in docs):
        return docs

    scored_docs: list[tuple[Document, float]] = []
    for doc in docs:
        try:
            score = float(doc.metadata.get(score_key, float("-inf")))
        except (TypeError, ValueError):
            continue
        scored_docs.append((doc, score))
    if not scored_docs:
        return docs

    dynamic_threshold = _compute_dynamic_threshold(
        [score for _, score in scored_docs],
        mode=threshold_mode,
        absolute_threshold=threshold,
        relative_ratio=relative_ratio,
        quantile=quantile,
    )
    if dynamic_threshold is None:
        return docs

    return [
        doc for doc, score in scored_docs if score >= dynamic_threshold
    ]


def _filter_docs_by_doc_ids(
    docs: list[Document],
    allowed_doc_ids: set[str] | None,
) -> list[Document]:
    if not allowed_doc_ids:
        return docs
    allowed = {str(item).strip() for item in allowed_doc_ids if str(item).strip()}
    if not allowed:
        return docs
    return [
        doc
        for doc in docs
        if str((doc.metadata or {}).get("doc_id", "")).strip() in allowed
    ]


def _tokenize_for_bm25(text: str) -> list[str]:
    normalized = text.strip().lower()
    if not normalized:
        return []
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9_]+", normalized)


def _bm25_cache_key(corpus_docs: list[Document], corpus_key: str | None) -> str:
    if corpus_key:
        return f"key:{corpus_key}"
    return f"obj:{id(corpus_docs)}:{len(corpus_docs)}"


def _get_bm25_index(corpus_docs: list[Document], cache_key: str):
    if cache_key in _BM25_CACHE:
        return _BM25_CACHE[cache_key]

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return None

    tokenized = []
    for doc in corpus_docs:
        tokens = _tokenize_for_bm25(doc.page_content)
        tokenized.append(tokens if tokens else ["_"])

    if not tokenized:
        return None

    index = BM25Okapi(tokenized)
    _BM25_CACHE[cache_key] = (index, corpus_docs)
    return _BM25_CACHE[cache_key]


def _retrieve_bm25(
    query: str,
    *,
    corpus_docs: list[Document] | None,
    top_k: int,
    corpus_key: str | None,
) -> list[Document]:
    if not corpus_docs or top_k <= 0:
        return []

    cache_key = _bm25_cache_key(corpus_docs, corpus_key)
    cached = _get_bm25_index(corpus_docs, cache_key)
    if cached is None:
        return []
    bm25, docs = cached

    query_tokens = _tokenize_for_bm25(query)
    if not query_tokens:
        return []

    try:
        scores = bm25.get_scores(query_tokens)
    except Exception:
        return []

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: float(scores[i]),
        reverse=True,
    )[:top_k]

    results: list[Document] = []
    for rank, idx in enumerate(ranked_indices, start=1):
        score = float(scores[idx])
        if score <= 0:
            continue
        results.append(
            _copy_doc(
                docs[idx],
                bm25_score=score,
                bm25_rank=rank,
            )
        )
    return results


def _doc_fingerprint(doc: Document) -> str:
    metadata = dict(doc.metadata or {})
    transient_keys = {
        "retrieval_score",
        "retrieval_raw_score",
        "retrieval_rrf_score",
        "retrieval_multi_rrf_score",
        "dense_rank",
        "bm25_rank",
        "bm25_score",
        "reranker_score",
    }
    stable_meta = {
        key: metadata[key]
        for key in sorted(metadata.keys())
        if key not in transient_keys
    }
    payload = {
        "meta": stable_meta,
        "text": doc.page_content[:1200],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _fuse_with_rrf(
    dense_docs: list[Document],
    bm25_docs: list[Document],
    *,
    top_k: int,
    rrf_k: int,
) -> list[Document]:
    return _fuse_ranked_lists(
        [dense_docs, bm25_docs],
        top_k=top_k,
        rrf_k=rrf_k,
        score_key="retrieval_rrf_score",
        source_rank_keys=("dense_rank", "bm25_rank"),
    )


def _fuse_ranked_lists(
    ranked_lists: list[list[Document]],
    *,
    top_k: int,
    rrf_k: int,
    score_key: str,
    source_rank_keys: tuple[str, ...] | None = None,
) -> list[Document]:
    if not ranked_lists:
        return []
    if rrf_k <= 0:
        rrf_k = 60

    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    per_list_ranks: list[dict[str, int]] = [dict() for _ in ranked_lists]

    for list_idx, docs_list in enumerate(ranked_lists):
        for rank, doc in enumerate(docs_list, start=1):
            key = _doc_fingerprint(doc)
            if key not in docs:
                docs[key] = doc
            per_list_ranks[list_idx][key] = rank
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]
    fused_docs: list[Document] = []
    for key in ranked_keys:
        metadata = dict((docs[key].metadata or {}))
        metadata[score_key] = float(scores[key])
        if source_rank_keys:
            for idx, rank_key in enumerate(source_rank_keys):
                if idx < len(per_list_ranks) and key in per_list_ranks[idx]:
                    metadata[rank_key] = per_list_ranks[idx][key]
        fused_docs.append(Document(page_content=docs[key].page_content, metadata=metadata))
    return fused_docs


def _retrieve_with_relevance_scores(
    query: str,
    vector_store,
    top_k: int,
    *,
    metadata_expr: str | None = None,
) -> list[Document] | None:
    method = getattr(vector_store, "similarity_search_with_relevance_scores", None)
    if not callable(method):
        return None
    kwargs: dict[str, Any] = {"k": top_k}
    if metadata_expr:
        try:
            if "expr" in inspect.signature(method).parameters:
                kwargs["expr"] = metadata_expr
        except (TypeError, ValueError):
            pass
    try:
        pairs = method(query, **kwargs)
    except Exception:
        return None

    docs: list[Document] = []
    for doc, score in pairs:
        docs.append(_attach_score(doc, "retrieval_score", float(score)))
    return docs


def _retrieve_with_raw_scores(
    query: str,
    vector_store,
    top_k: int,
    *,
    metadata_expr: str | None = None,
) -> list[Document] | None:
    method = getattr(vector_store, "similarity_search_with_score", None)
    if not callable(method):
        return None
    kwargs: dict[str, Any] = {"k": top_k}
    if metadata_expr:
        try:
            if "expr" in inspect.signature(method).parameters:
                kwargs["expr"] = metadata_expr
        except (TypeError, ValueError):
            pass
    try:
        pairs = method(query, **kwargs)
    except Exception:
        return None

    docs: list[Document] = []
    raw_scores: list[float] = []
    for doc, raw_score in pairs:
        numeric_score = float(raw_score)
        raw_scores.append(numeric_score)
        docs.append(_attach_score(doc, "retrieval_raw_score", numeric_score))

    # 只有当分数像 0-1 相似度时，才映射到统一阈值字段。
    if _looks_like_similarity_scores(raw_scores):
        docs = [
            _attach_score(doc, "retrieval_score", doc.metadata["retrieval_raw_score"])
            for doc in docs
        ]
    return docs


def _retrieve_dense(
    query: str,
    vector_store,
    top_k: int,
    *,
    metadata_expr: str | None = None,
) -> list[Document]:
    docs = _retrieve_with_relevance_scores(
        query,
        vector_store,
        top_k,
        metadata_expr=metadata_expr,
    )
    if docs is not None:
        return docs

    docs = _retrieve_with_raw_scores(
        query,
        vector_store,
        top_k,
        metadata_expr=metadata_expr,
    )
    if docs is not None:
        return docs

    retriever = build_retriever(vector_store, top_k)
    return list(retriever.invoke(query))


def _retrieve_dense_with_doc_id_allowlist(
    query: str,
    vector_store,
    *,
    top_k: int,
    allowed_doc_ids: set[str] | None,
    metadata_expr: str | None,
) -> list[Document]:
    if not allowed_doc_ids and not metadata_expr:
        return _retrieve_dense(query, vector_store, top_k)
    if not allowed_doc_ids:
        return _retrieve_dense(
            query,
            vector_store,
            top_k,
            metadata_expr=metadata_expr,
        )

    search_k = max(top_k, 32)
    max_search_k = max(128, top_k * 8)
    best: list[Document] = []
    while True:
        docs = _retrieve_dense(
            query,
            vector_store,
            search_k,
            metadata_expr=metadata_expr,
        )
        filtered = _filter_docs_by_doc_ids(docs, allowed_doc_ids)
        if filtered:
            best = filtered
        if len(filtered) >= top_k or search_k >= max_search_k:
            break
        search_k = min(max_search_k, search_k * 2)
    return best[:top_k]


def _normalize_query_variants(query: str, query_variants: list[str] | None) -> list[str]:
    if not query_variants:
        return [query]
    variants: list[str] = []
    seen = set()
    for item in query_variants:
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        variants.append(text)
    return variants or [query]


def retrieve(
    query: str,
    vector_store,
    top_k: int,
    score_threshold: float | None = None,
    score_threshold_mode: str = "absolute",
    score_relative_ratio: float = 0.7,
    score_quantile: float = 0.6,
    *,
    retrieval_mode: str = "dense",
    hybrid_corpus_docs: list[Document] | None = None,
    hybrid_corpus_key: str | None = None,
    hybrid_dense_top_k: int | None = None,
    hybrid_bm25_top_k: int | None = None,
    hybrid_rrf_k: int = 60,
    query_variants: list[str] | None = None,
    metadata_allow_doc_ids: set[str] | None = None,
    metadata_milvus_expr: str | None = None,
) -> list[Document]:
    mode = retrieval_mode.strip().lower()
    if mode not in {"dense", "hybrid", "bm25"}:
        mode = "dense"

    variants = _normalize_query_variants(query, query_variants)

    if mode == "bm25":
        variant_lists: list[list[Document]] = []
        for variant in variants:
            bm25_docs = _retrieve_bm25(
                variant,
                corpus_docs=hybrid_corpus_docs,
                top_k=max(top_k, int(hybrid_bm25_top_k or top_k)),
                corpus_key=hybrid_corpus_key,
            )
            bm25_docs = _filter_docs_by_doc_ids(bm25_docs, metadata_allow_doc_ids)
            variant_lists.append(bm25_docs[:top_k])

        if len(variant_lists) == 1:
            return variant_lists[0][:top_k]
        return _fuse_ranked_lists(
            variant_lists,
            top_k=top_k,
            rrf_k=hybrid_rrf_k,
            score_key="retrieval_multi_rrf_score",
        )

    if mode == "dense":
        if vector_store is None:
            return []
        dense_lists: list[list[Document]] = []
        for variant in variants:
            docs = _retrieve_dense_with_doc_id_allowlist(
                variant,
                vector_store,
                top_k=max(top_k, int(hybrid_dense_top_k or top_k)),
                allowed_doc_ids=metadata_allow_doc_ids,
                metadata_expr=metadata_milvus_expr,
            )
            docs = _filter_docs_by_score(
                docs,
                "retrieval_score",
                score_threshold,
                threshold_mode=score_threshold_mode,
                relative_ratio=score_relative_ratio,
                quantile=score_quantile,
            )
            dense_lists.append(docs[:top_k])

        if len(dense_lists) == 1:
            return dense_lists[0][:top_k]
        return _fuse_ranked_lists(
            dense_lists,
            top_k=top_k,
            rrf_k=hybrid_rrf_k,
            score_key="retrieval_multi_rrf_score",
        )

    dense_k = max(top_k, int(hybrid_dense_top_k or top_k))
    bm25_k = max(top_k, int(hybrid_bm25_top_k or top_k))

    variant_lists: list[list[Document]] = []
    for variant in variants:
        dense_docs = []
        if vector_store is not None:
            dense_docs = _retrieve_dense_with_doc_id_allowlist(
                variant,
                vector_store,
                top_k=dense_k,
                allowed_doc_ids=metadata_allow_doc_ids,
                metadata_expr=metadata_milvus_expr,
            )
        dense_docs = _filter_docs_by_score(
            dense_docs,
            "retrieval_score",
            score_threshold,
            threshold_mode=score_threshold_mode,
            relative_ratio=score_relative_ratio,
            quantile=score_quantile,
        )
        bm25_docs = _retrieve_bm25(
            variant,
            corpus_docs=hybrid_corpus_docs,
            top_k=bm25_k,
            corpus_key=hybrid_corpus_key,
        )
        bm25_docs = _filter_docs_by_doc_ids(bm25_docs, metadata_allow_doc_ids)
        if bm25_docs:
            if dense_docs:
                fused_docs = _fuse_with_rrf(
                    dense_docs,
                    bm25_docs,
                    top_k=dense_k,
                    rrf_k=hybrid_rrf_k,
                )
                variant_lists.append(fused_docs)
            else:
                variant_lists.append(bm25_docs[:dense_k])
        else:
            variant_lists.append(dense_docs)

    if len(variant_lists) == 1:
        return variant_lists[0][:top_k]
    return _fuse_ranked_lists(
        variant_lists,
        top_k=top_k,
        rrf_k=hybrid_rrf_k,
        score_key="retrieval_multi_rrf_score",
    )


def _get_reranker_model(reranker_model: str):
    if reranker_model not in _RERANKER_CACHE:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            return None

        # 关闭进度条和多余输出，避免终端刷屏。
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                _RERANKER_CACHE[reranker_model] = CrossEncoder(reranker_model)

    return _RERANKER_CACHE[reranker_model]


def clear_retrieval_cache() -> None:
    _BM25_CACHE.clear()
    _RERANKER_CACHE.clear()


def rerank_documents(
    query: str,
    docs: list[Document],
    reranker_model: str | None = None,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[Document]:
    """
    可选重排阶段：使用 cross-encoder 对候选文档重排并写入 `reranker_score`。
    当设置 `score_threshold` 时，会过滤低于阈值的候选文档。
    """
    if not docs:
        return []

    if not reranker_model:
        ranked = docs
        return ranked[:top_k] if top_k else ranked

    model = _get_reranker_model(reranker_model)
    if model is None:
        ranked = docs
        return ranked[:top_k] if top_k else ranked

    try:
        pairs = [(query, d.page_content) for d in docs]
        scores = model.predict(pairs)
    except Exception:
        ranked = docs
        return ranked[:top_k] if top_k else ranked

    ranked_pairs = sorted(zip(scores, docs), key=lambda x: float(x[0]), reverse=True)
    ranked_docs: list[Document] = []
    for score, doc in ranked_pairs:
        ranked_docs.append(_attach_score(doc, "reranker_score", float(score)))

    ranked_docs = _filter_docs_by_score(
        ranked_docs,
        "reranker_score",
        score_threshold,
    )
    return ranked_docs[:top_k] if top_k else ranked_docs


def _extract_numeric(metadata: dict[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_score_key(docs: list[Document]) -> str | None:
    candidates = (
        "retrieval_score",
        "retrieval_multi_rrf_score",
        "retrieval_rrf_score",
        "bm25_score",
    )
    if not docs:
        return None
    top1_meta = dict(docs[0].metadata or {})
    top2_meta = dict(docs[1].metadata or {}) if len(docs) > 1 else {}
    for key in candidates:
        top1_score = _extract_numeric(top1_meta, key)
        top2_score = _extract_numeric(top2_meta, key)
        if top1_score is not None and top2_score is not None:
            return key
    return None


def _rank_consensus_strength(doc: Document) -> int:
    metadata = dict(doc.metadata or {})
    score = 0
    dense_rank = _extract_numeric(metadata, "dense_rank")
    bm25_rank = _extract_numeric(metadata, "bm25_rank")
    if dense_rank is not None and dense_rank <= 2:
        score += 1
    if bm25_rank is not None and bm25_rank <= 2:
        score += 1
    return score


def decide_rerank(
    docs: list[Document],
    *,
    enabled: bool,
    min_top1_score: float,
    min_score_gap: float,
    min_rel_gap: float,
    min_top1_score_small_scale: float = 0.015,
    min_score_gap_small_scale: float = 0.001,
) -> RerankDecision:
    if not enabled:
        return RerankDecision(False, "conditional_disabled")
    if len(docs) <= 1:
        return RerankDecision(False, "single_candidate")

    score_key = _pick_score_key(docs)
    if not score_key:
        return RerankDecision(True, "missing_scores")

    top1 = _extract_numeric(dict(docs[0].metadata or {}), score_key)
    top2 = _extract_numeric(dict(docs[1].metadata or {}), score_key)
    if top1 is None or top2 is None:
        return RerankDecision(True, "invalid_top_scores", score_key=score_key)

    gap = top1 - top2
    rel_gap = gap / max(abs(top1), 1e-8)

    # 对 hybrid/rrf 这类小尺度分数使用单独阈值，避免误判。
    if abs(top1) <= 0.1:
        top1_threshold = min_top1_score_small_scale
        gap_threshold = min_score_gap_small_scale
    else:
        top1_threshold = min_top1_score
        gap_threshold = min_score_gap

    top1_consensus = _rank_consensus_strength(docs[0])
    top2_consensus = _rank_consensus_strength(docs[1])
    top1_dense_rank = _extract_numeric(dict(docs[0].metadata or {}), "dense_rank")
    top2_dense_rank = _extract_numeric(dict(docs[1].metadata or {}), "dense_rank")
    top1_bm25_rank = _extract_numeric(dict(docs[0].metadata or {}), "bm25_rank")
    top2_bm25_rank = _extract_numeric(dict(docs[1].metadata or {}), "bm25_rank")
    high_consensus = top1_consensus >= 2 and top2_consensus == 0
    high_single_rank_gap = (
        (top1_dense_rank == 1 and top2_dense_rank is not None and top2_dense_rank >= 4)
        or (top1_bm25_rank == 1 and top2_bm25_rank is not None and top2_bm25_rank >= 4)
    )
    high_margin = top1 >= top1_threshold and (gap >= gap_threshold or rel_gap >= min_rel_gap)

    if high_consensus or high_single_rank_gap or high_margin:
        return RerankDecision(
            should_rerank=False,
            reason="high_confidence_skip",
            score_key=score_key,
            top1_score=top1,
            top2_score=top2,
            score_gap=gap,
            score_rel_gap=rel_gap,
        )
    return RerankDecision(
        should_rerank=True,
        reason="low_confidence_need_rerank",
        score_key=score_key,
        top1_score=top1,
        top2_score=top2,
        score_gap=gap,
        score_rel_gap=rel_gap,
    )


def attach_rerank_decision(docs: list[Document], decision: RerankDecision) -> list[Document]:
    if not docs:
        return docs
    annotated: list[Document] = []
    for doc in docs:
        metadata = dict(doc.metadata or {})
        metadata["rerank_decision"] = decision.reason
        metadata["rerank_should_run"] = bool(decision.should_rerank)
        if decision.score_key:
            metadata["rerank_decision_score_key"] = decision.score_key
        if decision.top1_score is not None:
            metadata["rerank_decision_top1_score"] = float(decision.top1_score)
        if decision.top2_score is not None:
            metadata["rerank_decision_top2_score"] = float(decision.top2_score)
        if decision.score_gap is not None:
            metadata["rerank_decision_score_gap"] = float(decision.score_gap)
        if decision.score_rel_gap is not None:
            metadata["rerank_decision_rel_gap"] = float(decision.score_rel_gap)
        annotated.append(Document(page_content=doc.page_content, metadata=metadata))
    return annotated


def diversify_documents_by_source(
    docs: list[Document],
    *,
    max_per_source: int = 2,
    top_k: int | None = None,
) -> list[Document]:
    """
    按 source 限制同一论文的块数，减少结果同质化。
    若过滤过严导致数量不足，会回填原序文档以保证 top_k。
    """
    if not docs:
        return []
    limit = max(1, max_per_source)
    source_counts: dict[str, int] = {}
    selected: list[Document] = []
    selected_keys = set()

    for doc in docs:
        source = str((doc.metadata or {}).get("source", "unknown"))
        if source_counts.get(source, 0) >= limit:
            continue
        source_counts[source] = source_counts.get(source, 0) + 1
        selected.append(doc)
        selected_keys.add(_doc_fingerprint(doc))
        if top_k and len(selected) >= top_k:
            return selected

    if top_k and len(selected) < top_k:
        for doc in docs:
            key = _doc_fingerprint(doc)
            if key in selected_keys:
                continue
            selected.append(doc)
            selected_keys.add(key)
            if len(selected) >= top_k:
                break

    return selected
