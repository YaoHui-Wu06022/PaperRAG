from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...config import Settings
from ..data.aliases import alias_match_to_dict
from ..data.chunks import ChunkDocument, load_chunk_documents
from ..data.manifest_lookup import load_active_manifest_records, manifest_record_to_evidence, match_manifest_records
from ..dense.milvus_store import SearchResult
from ..dense.service import build_embedder, build_store
from ..sparse.bm25 import BM25Document, BM25Index
from .context import context_unit
from .router import RouteDecision, build_route_decision, flatten_filter_value, route_tokens
from .translation import BaiduTranslator, TranslationError, contains_chinese


RRF_K = 60


@dataclass(frozen=True)
class PreparedQuery:
    original_query: str
    retrieval_query: str
    language: str
    warnings: list[str]
    translation_provider: str | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass(frozen=True)
class FusedChunk:
    document: ChunkDocument
    score: float
    sources: dict[str, dict[str, float | int]]
    dense_result: SearchResult | None = None


def run_plan(
    settings: Settings,
    query: str,
    *,
    translator: BaiduTranslator | None = None,
    plan_parser=None,
    embedder=None,
    store=None,
) -> dict[str, Any]:
    prepared = prepare_query(settings, query, translator=translator)
    warnings = list(prepared.warnings)
    if prepared.failed:
        return {
            "original_query": prepared.original_query,
            "retrieval_query": prepared.retrieval_query,
            "route": "error",
            "router_reason": prepared.error,
            "translation_provider": prepared.translation_provider,
            "evidence": {},
            "warnings": warnings,
        }
    route = build_route_decision(settings, prepared.retrieval_query, warnings=warnings, plan_parser=plan_parser)
    evidence: dict[str, Any]
    if route.route == "metadata":
        evidence = plan_metadata(settings, route, warnings)
    elif route.route == "reference":
        evidence = plan_reference(route, warnings)
    else:
        evidence = plan_body(settings, prepared, route, warnings, embedder=embedder, store=store)
    return {
        "original_query": prepared.original_query,
        "retrieval_query": prepared.retrieval_query,
        "route": route.route,
        "intent": route.intent,
        "return_field": route.return_field,
        "router_reason": route.reason,
        "translation_provider": prepared.translation_provider,
        "evidence": evidence,
        "warnings": warnings,
    }


def prepare_query(
    settings: Settings,
    query: str,
    *,
    translator: BaiduTranslator | None = None,
) -> PreparedQuery:
    language = "zh" if contains_chinese(query) else "en"
    warnings: list[str] = []
    if language == "en":
        return PreparedQuery(query, query, language, warnings)
    translator = translator or BaiduTranslator(
        app_id=settings.baidu_translate_app_id,
        secret_key=settings.baidu_translate_secret_key,
        endpoint=settings.baidu_translate_endpoint,
        domain=settings.baidu_translate_domain,
    )
    try:
        result = translator.translate_to_english(query)
    except (TranslationError, OSError, ValueError) as exc:
        warnings.append(f"translation_failed: {exc}")
        return PreparedQuery(query, "", language, warnings, error="translation_failed")
    return PreparedQuery(query, result.text, language, warnings, result.provider)


def base_evidence(route: RouteDecision) -> dict[str, Any]:
    return {
        "top_route": route.route,
        "intent": route.intent,
        "return_field": route.return_field,
        "target_papers": route.target_papers,
        "query": route.target_query,
        "filters": route.filters,
        "alias_matches": [alias_match_to_dict(match) for match in route.alias_matches],
    }


def plan_metadata(
    settings: Settings,
    route: RouteDecision,
    warnings: list[str],
) -> dict[str, Any]:
    if route.parse_status == "parse_failed":
        return {
            **base_evidence(route),
            "parse_status": "parse_failed",
            "parser_error": route.parser_error,
            "parser_result": None,
            "records": [],
        }
    if route.intent == "unknown":
        return {
            **base_evidence(route),
            "parse_status": "unknown",
            "parser_result": route.parser_result,
            "records": [],
        }
    records: list[dict[str, Any]]
    if route.intent == "lookup":
        records = metadata_lookup_records(settings, route, warnings)
    else:
        records = metadata_records_by_parser_filters(settings, route.filters)
    if not records:
        warnings.append("metadata route found no matching manifest records")
    evidence = {
        **base_evidence(route),
        "query": route.target_query,
        "parse_status": "ok",
        "parser_result": route.parser_result,
        "records": records,
    }
    if route.intent == "count":
        evidence["count"] = len(records)
    return evidence


def plan_reference(route: RouteDecision, warnings: list[str]) -> dict[str, Any]:
    warnings.append("reference route is recognized but reference evidence is not implemented yet")
    return {
        **base_evidence(route),
        "parse_status": "not_implemented",
        "references": [],
    }


def plan_body(
    settings: Settings,
    prepared: PreparedQuery,
    route: RouteDecision,
    warnings: list[str],
    *,
    embedder=None,
    store=None,
) -> dict[str, Any]:
    documents = filter_documents_by_targets(load_chunk_documents(settings.paper_data_dir), route.target_papers)
    documents_by_id = {document.chunk_id: document for document in documents}
    try:
        dense_results = dense_search(settings, prepared.retrieval_query, embedder=embedder, store=store)
    except Exception as exc:
        warnings.append(f"dense retrieval failed: {exc}; using BM25 candidates only")
        dense_results = []
    bm25_results = bm25_chunk_search(documents, prepared.retrieval_query, settings.plan_bm25_top_k)
    fused = fuse_chunk_results(documents_by_id, dense_results, bm25_results)
    context_units = [
        context_unit(settings, candidate, settings.plan_block_window)
        for candidate in fused[:settings.plan_final_top_k]
    ]
    if not context_units:
        warnings.append("body route found no dense/BM25 candidates")
    return {
        **base_evidence(route),
        "context_units": context_units,
    }


def dense_search(settings: Settings, query: str, *, embedder=None, store=None) -> list[SearchResult]:
    embedder = embedder or build_embedder(settings)
    store = store or build_store(settings)
    query_vector = embedder.embed_texts([query])[0]
    return store.search(query_vector, settings.plan_dense_top_k)


def bm25_chunk_search(documents: list[ChunkDocument], query: str, top_k: int):
    bm25_documents = [
        BM25Document(
            document.chunk_id,
            f"{document.text}\n{document.embedding_text}",
            {"document": document},
        )
        for document in documents
    ]
    return BM25Index(bm25_documents).search(query, top_k)


def fuse_chunk_results(
    documents_by_id: dict[str, ChunkDocument],
    dense_results: list[SearchResult],
    bm25_results,
) -> list[FusedChunk]:
    by_id: dict[str, dict[str, Any]] = {}
    for rank, result in enumerate(dense_results, start=1):
        document = documents_by_id.get(result.chunk_id)
        if document is None:
            continue
        slot = by_id.setdefault(result.chunk_id, {"document": document, "score": 0.0, "sources": {}})
        slot["score"] += 1 / (RRF_K + rank)
        slot["sources"]["dense"] = {"rank": rank, "score": result.score}
        slot["dense_result"] = result
    for rank, hit in enumerate(bm25_results, start=1):
        document = hit.payload["document"]
        slot = by_id.setdefault(hit.doc_id, {"document": document, "score": 0.0, "sources": {}})
        slot["score"] += 1 / (RRF_K + rank)
        slot["sources"]["bm25"] = {"rank": rank, "score": hit.score}
    fused = [
        FusedChunk(
            document=value["document"],
            score=value["score"],
            sources=value["sources"],
            dense_result=value.get("dense_result"),
        )
        for value in by_id.values()
    ]
    fused.sort(key=lambda candidate: candidate.score, reverse=True)
    return fused


def filter_documents_by_targets(documents: list[ChunkDocument], target_papers: list[dict[str, Any]]) -> list[ChunkDocument]:
    if not target_papers:
        return documents
    paper_ids = {str(target.get("paper_id")) for target in target_papers if target.get("paper_id")}
    return [document for document in documents if document.paper_id in paper_ids]


def metadata_records_for_targets(target_papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "file_hash": target.get("file_hash"),
            "title": target.get("title"),
            "author": target.get("author"),
            "year": target.get("year"),
            "venue": target.get("venue"),
            "pdf_path": target.get("pdf_path"),
            "paper_data_path": target.get("paper_data_path"),
        }
        for target in target_papers
    ]


def metadata_lookup_records(
    settings: Settings,
    route: RouteDecision,
    warnings: list[str],
) -> list[dict[str, Any]]:
    return_field = route.return_field
    if return_field is None:
        warnings.append("metadata lookup missing return_field")
    records = metadata_records_for_targets(route.target_papers)
    if not records and route.target_query:
        records = match_manifest_records(settings, route.target_query)
    if return_field is None:
        return records
    return [
        {**record, "return_field": return_field, "value": record.get(str(return_field))}
        for record in records
    ]


def metadata_records_by_parser_filters(settings: Settings, filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in load_active_manifest_records(settings):
        if all(record_matches_filter(record, filter_item) for filter_item in filters):
            records.append(manifest_record_to_evidence(record))
    return records


def record_matches_filter(record, filter_item: dict[str, Any]) -> bool:
    matched = record_matches_positive_filter(record, filter_item)
    return not matched if filter_item.get("negated") else matched


def record_matches_positive_filter(record, filter_item: dict[str, Any]) -> bool:
    field = filter_item.get("field")
    op = filter_item.get("op")
    value = filter_item.get("value")
    if field == "year":
        return compare_number(record.year, op, value)
    if field == "author":
        return compare_authors(record.author, op, value)
    if field == "venue":
        return compare_text(record.venue, op, value)
    if field == "title":
        return compare_text(record.title, op, value)
    return False


def compare_number(actual: Any, op: str, expected: Any) -> bool:
    if actual is None:
        return False
    actual_number = int(actual)
    if op == "between":
        values = [int(item) for item in expected] if isinstance(expected, list) else []
        return len(values) >= 2 and min(values[0], values[1]) <= actual_number <= max(values[0], values[1])
    if op == "in":
        return isinstance(expected, list) and actual_number in {int(item) for item in expected}
    expected_number = int(expected)
    if op == "=":
        return actual_number == expected_number
    if op == ">":
        return actual_number > expected_number
    if op == ">=":
        return actual_number >= expected_number
    if op == "<":
        return actual_number < expected_number
    if op == "<=":
        return actual_number <= expected_number
    if op == "contains":
        return str(expected_number) in str(actual_number)
    return False


def compare_authors(authors: list[str], op: str, expected: Any) -> bool:
    values = flatten_filter_value(expected)
    if not values:
        return False
    if op in {"=", "contains"}:
        return any(matching_author(authors, value) for value in values)
    if op == "in":
        return any(matching_author(authors, value) for value in values)
    return False


def compare_text(actual: Any, op: str, expected: Any) -> bool:
    actual_text = str(actual or "")
    actual_key = normalized_text_key(actual_text)
    values = flatten_filter_value(expected)
    if not values:
        return False
    value_keys = [normalized_text_key(value) for value in values]
    if op == "=":
        return actual_key == value_keys[0]
    if op == "in":
        return actual_key in set(value_keys)
    if op == "contains":
        return any(value_key and value_key in actual_key for value_key in value_keys)
    return False


def normalized_text_key(value: str) -> str:
    return " ".join(route_tokens(value))


def matching_author(authors: list[str], author_query: str) -> str | None:
    query_tokens = route_tokens(author_query)
    if not query_tokens:
        return None
    for author in authors:
        if route_tokens(author) == query_tokens:
            return author
    return None
