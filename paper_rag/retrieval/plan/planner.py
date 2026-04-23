from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...config import Settings
from ..data.aliases import alias_match_to_dict, expand_query_with_aliases, resolve_target_papers
from ..data.chunks import ChunkDocument, load_chunk_documents
from ..data.manifest_lookup import load_active_manifest_records, manifest_record_to_evidence, match_manifest_records
from ..dense.milvus_store import SearchResult
from ..dense.service import build_embedder, build_store
from ..sparse.bm25 import BM25Document, BM25Index
from .context import context_unit
from .top_router import RouteDecision, build_route_decision, flatten_filter_value, route_tokens
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
        evidence = plan_reference(settings, route, warnings)
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


def plan_reference(settings: Settings, route: RouteDecision, warnings: list[str]) -> dict[str, Any]:
    if route.parse_status == "parse_failed":
        return {
            **base_evidence(route),
            "parse_status": "parse_failed",
            "parser_error": route.parser_error,
            "parser_result": None,
            "direction": route.direction,
            "anchor": route.anchor,
            "anchor_mode": route.anchor_mode,
            "references": [],
        }
    if route.parse_status == "unknown_direction":
        return {
            **base_evidence(route),
            "parse_status": "unknown_direction",
            "parser_result": route.parser_result,
            "direction": route.direction,
            "anchor": route.anchor,
            "anchor_mode": route.anchor_mode,
            "references": [],
        }
    if not route.anchor:
        warnings.append("reference route missing anchor")
        return reference_evidence(route, [], [], parse_status="missing_anchor")
    if route.direction == "cite":
        references, anchor_results = reference_cite_results(settings, route, warnings)
    elif route.direction == "cited_by":
        references, anchor_results = reference_cited_by_results(settings, route)
    else:
        warnings.append("reference route direction is unsupported")
        return reference_evidence(route, [], [], parse_status="unknown_direction")
    references = combine_reference_results(references, route.anchor_mode or "per", route.direction)
    if not references:
        warnings.append("reference route found no matching references")
    return reference_evidence(route, references, anchor_results, count=len(references))


def reference_evidence(
    route: RouteDecision,
    references: list[dict[str, Any]],
    anchor_results: list[dict[str, Any]],
    *,
    parse_status: str = "ok",
    count: int | None = None,
) -> dict[str, Any]:
    evidence = {
        **base_evidence(route),
        "parse_status": parse_status,
        "parser_result": route.parser_result,
        "direction": route.direction,
        "anchor": route.anchor,
        "anchor_mode": route.anchor_mode,
        "references": references,
        "anchor_results": anchor_results,
    }
    if route.intent == "count":
        evidence["count"] = count if count is not None else len(references)
    return evidence


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


def reference_cite_results(
    settings: Settings,
    route: RouteDecision,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    references: list[dict[str, Any]] = []
    anchor_results: list[dict[str, Any]] = []
    for anchor in route.anchor:
        anchor_value = str(anchor.get("value") or "").strip()
        target_papers, _ = resolve_target_papers(settings, [anchor_value])
        anchor_refs: list[dict[str, Any]] = []
        for target in target_papers:
            for ref in load_reference_rows(target):
                if reference_raw_matches_filters(ref, route.filters, warnings):
                    entry = {
                        "direction": "cite",
                        "anchor": anchor,
                        "anchor_paper": target,
                        "target_paper": None,
                        "reference": ref,
                    }
                    anchor_refs.append(entry)
                    references.append(entry)
        if not target_papers:
            warnings.append(f"reference anchor not found locally: {anchor_value}")
        anchor_results.append({
            "anchor": anchor,
            "target_papers": target_papers,
            "references": anchor_refs,
            "count": len(anchor_refs),
        })
    return references, anchor_results


def reference_cited_by_results(settings: Settings, route: RouteDecision) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    references: list[dict[str, Any]] = []
    anchor_results: list[dict[str, Any]] = []
    records = load_active_manifest_records(settings)
    for anchor in route.anchor:
        anchor_value = str(anchor.get("value") or "").strip()
        anchor_terms = reference_anchor_terms(settings, anchor_value)
        anchor_refs: list[dict[str, Any]] = []
        for record in records:
            if not all(record_matches_filter(record, filter_item) for filter_item in route.filters):
                continue
            paper = manifest_record_to_evidence(record)
            paper["paper_id"] = Path(str(paper.get("paper_data_path") or "")).name if paper.get("paper_data_path") else None
            for ref in load_reference_rows(paper):
                if reference_raw_matches_terms(ref.get("raw_text"), anchor_terms):
                    entry = {
                        "direction": "cited_by",
                        "anchor": anchor,
                        "anchor_terms": anchor_terms,
                        "citing_paper": paper,
                        "reference": ref,
                    }
                    anchor_refs.append(entry)
                    references.append(entry)
        target_papers, _ = resolve_target_papers(settings, [anchor_value])
        anchor_results.append({
            "anchor": anchor,
            "target_papers": target_papers,
            "references": anchor_refs,
            "count": len(anchor_refs),
        })
    return references, anchor_results


def load_reference_rows(paper: dict[str, Any]) -> list[dict[str, Any]]:
    paper_data_path = paper.get("paper_data_path")
    if not paper_data_path:
        return []
    path = Path(str(paper_data_path)) / "references.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append({
                "reference_id": row.get("reference_id"),
                "ref_index": row.get("ref_index"),
                "raw_text": row.get("raw_text"),
                "page": row.get("page"),
                "source_block_id": row.get("source_block_id"),
            })
    return rows


def reference_anchor_terms(settings: Settings, anchor_value: str) -> list[str]:
    expanded_query, matches = expand_query_with_aliases(settings, anchor_value)
    terms = [anchor_value, expanded_query]
    for match in matches:
        terms.extend(match.expanded_terms)
    return unique_reference_terms(terms)


def unique_reference_terms(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = normalized_text_key(text)
        if key and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def reference_raw_matches_terms(raw_text: Any, terms: list[str]) -> bool:
    raw_key = normalized_text_key(str(raw_text or ""))
    return any((term_key := normalized_text_key(term)) and term_key in raw_key for term in terms)


def reference_raw_matches_filters(ref: dict[str, Any], filters: list[dict[str, Any]], warnings: list[str]) -> bool:
    for filter_item in filters:
        field = filter_item.get("field")
        if field not in {"title", "year"}:
            warning = f"reference cite filters only support title/year; ignored {field}"
            if warning not in warnings:
                warnings.append(warning)
            continue
        matched = reference_raw_matches_positive_filter(ref, filter_item)
        if filter_item.get("negated"):
            matched = not matched
        if not matched:
            return False
    return True


def reference_raw_matches_positive_filter(ref: dict[str, Any], filter_item: dict[str, Any]) -> bool:
    raw_text = str(ref.get("raw_text") or "")
    field = filter_item.get("field")
    op = filter_item.get("op")
    expected = filter_item.get("value")
    if field == "title":
        values = flatten_filter_value(expected)
        if op in {"=", "contains", "in"}:
            return any(reference_raw_matches_terms(raw_text, [value]) for value in values)
    if field == "year":
        years = reference_years(raw_text)
        if op == "interval":
            return any(compare_number(year, "interval", expected) for year in years)
        if op == "in":
            return isinstance(expected, list) and any(year in {int(item) for item in expected} for year in years)
        if op == "=":
            return any(year == int(expected) for year in years)
        if op == "contains":
            return str(expected) in raw_text
    return False


def reference_years(raw_text: str) -> list[int]:
    return [int(match) for match in re.findall(r"\b(?:19|20)\d{2}\b", raw_text)]


def combine_reference_results(
    references: list[dict[str, Any]],
    anchor_mode: str,
    direction: str | None,
) -> list[dict[str, Any]]:
    if anchor_mode == "per":
        return references
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in references:
        grouped.setdefault(reference_result_key(entry, direction), []).append(entry)
    if anchor_mode == "or":
        return [items[0] for _, items in sorted(grouped.items())]
    if anchor_mode == "and":
        anchor_count = len({str(entry.get("anchor", {}).get("value") or "") for entry in references})
        return [items[0] for _, items in sorted(grouped.items()) if len({str(item.get("anchor", {}).get("value") or "") for item in items}) >= anchor_count]
    return references


def reference_result_key(entry: dict[str, Any], direction: str | None) -> str:
    if direction == "cited_by":
        paper = entry.get("citing_paper") or {}
        return str(paper.get("paper_id") or paper.get("title") or "")
    ref = entry.get("reference") or {}
    return normalized_text_key(str(ref.get("raw_text") or ""))


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
    if op == "interval":
        bounds = list(expected) if isinstance(expected, list) else []
        if len(bounds) != 2:
            return False
        lower_bound, upper_bound = bounds
        if not _is_negative_infinity(lower_bound):
            try:
                lower_number = int(lower_bound)
            except (TypeError, ValueError):
                return False
            if actual_number < lower_number:
                return False
        if not _is_positive_infinity(upper_bound):
            try:
                upper_number = int(upper_bound)
            except (TypeError, ValueError):
                return False
            if actual_number > upper_number:
                return False
        return True
    if op == "in":
        return isinstance(expected, list) and actual_number in {int(item) for item in expected}
    expected_number = int(expected)
    if op == "=":
        return actual_number == expected_number
    if op == "contains":
        return str(expected_number) in str(actual_number)
    return False


def _is_negative_infinity(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"-inf", "-infinity"}


def _is_positive_infinity(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"inf", "+inf", "infinity", "+infinity"}


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
