from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from langchain_core.documents import Document

from config import AppConfig
from generation.llm import build_llm_client
from generation.prompt import build_qa_prompt
from ingestion.chunking import split_documents_with_parents
from ingestion.embedding import build_embedding_model
from ingestion.pdf_loader import PdfLoadResult, load_documents_from_dir, load_pdf_pages
from retrieval.retriever import (
    clear_retrieval_cache,
)
from retrieval.vector_store import (
    load_vector_index,
    vector_index_exists,
)
from services import local_cache_store as cache_store
from services.retrieval_service import run_retrieval_flow
from services.sync_transaction import SyncPlan, execute_sync_plan
from services.telemetry import OperationTrace


INSUFFICIENT_EVIDENCE_ANSWER = (
    "根据当前检索到的资料，未找到充分证据支持回答该问题。"
    "请尝试换一种问法，或补充相关文档后再提问。"
)
_EMBEDDING_CACHE: dict[tuple, Any] = {}
_VECTOR_STORE_CACHE: dict[tuple, Any] = {}
_LLM_CACHE: dict[tuple, Any] = {}


@dataclass
class IngestResult:
    raw_documents: int
    chunks: int
    cache_dir: Path
    skipped: bool = False
    reference_chunks: int = 0
    suspicious_reference_chunks: int = 0


@dataclass
class DeleteResult:
    requested_doc_ids: list[str]
    deleted_doc_ids: list[str]
    removed_chunks: int
    removed_parents: int
    removed_reference_chunks: int
    removed_block_rows: int
    removed_structured_chunk_rows: int
    removed_reference_keyword_rows: int


@dataclass
class EvidenceRecord:
    citation_text: str
    citation_tag: str
    source: str
    page: str
    section_path: str
    doc_id: str
    block_id: str
    bbox: list[float]
    snippet: str


@dataclass
class QAResult:
    answer: str
    citations: list[str]
    contexts: list[str]
    evidences: list[EvidenceRecord]
    retrieval_scope: str = "main"


def _collect_pdf_paths(config: AppConfig, pdf_path: Path | None) -> list[Path]:
    if pdf_path:
        target = pdf_path.resolve()
        if not target.exists():
            raise FileNotFoundError(f"PDF not found: {target}")
        return [target]
    return sorted(config.data_pdf_dir.glob("*.pdf"))


def _build_ingest_signature(config: AppConfig, pdf_paths: list[Path]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in pdf_paths:
        stat = path.stat()
        files.append(
            {
                "name": path.name,
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {
        "mineru_api_base_url": config.mineru_api_base_url,
        "mineru_cloud_model_version": config.mineru_cloud_model_version,
        "mineru_cloud_poll_interval_sec": config.mineru_cloud_poll_interval_sec,
        "mineru_cloud_timeout_sec": config.mineru_cloud_timeout_sec,
        "mineru_output_dir": str(config.mineru_output_dir),
        "vector_backend": config.vector_backend,
        "embedding_provider": config.embedding_provider,
        "embedding_model": config.embedding_model,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "chunk_tokenizer_model": config.chunk_tokenizer_model,
        "chunk_strategy": config.chunk_strategy,
        "chunk_semantic_hard_max_chars": config.chunk_semantic_hard_max_chars,
        "chunk_use_structure_split": config.chunk_use_structure_split,
        "chunk_min_block_chars": config.chunk_min_block_chars,
        "chunk_quality_check_enabled": config.chunk_quality_check_enabled,
        "chunk_quality_header_footer_min_freq": config.chunk_quality_header_footer_min_freq,
        "ingestion_schema_version": 5,
        "milvus_uri": config.milvus_uri,
        "milvus_db_name": config.milvus_db_name,
        "milvus_collection": config.milvus_collection,
        "milvus_references_collection": config.milvus_references_collection,
        "references_strategy": config.references_strategy,
        "references_keyword_index_file": str(config.references_keyword_index_file),
        "files": files,
    }


def _vector_store_cache_key(
    config: AppConfig,
    *,
    collection_name: str | None = None,
) -> tuple:
    return (
        config.vector_backend,
        str(config.local_cache_dir.resolve()),
        config.milvus_uri,
        config.milvus_db_name,
        config.milvus_token,
        collection_name or config.milvus_collection,
    )


def _embedding_cache_key(config: AppConfig) -> tuple:
    return (
        config.embedding_provider,
        config.embedding_model,
        config.openai_base_url,
        config.aihubmix_base_url,
    )


def _llm_cache_key(config: AppConfig) -> tuple:
    return (
        config.llm_provider,
        config.llm_model,
        config.llm_temperature,
        config.aihubmix_base_url,
        config.aihubmix_api_mode,
        config.openai_base_url,
        config.openai_api_mode,
        config.ollama_base_url,
    )


def _get_embeddings(config: AppConfig):
    key = _embedding_cache_key(config)
    if key not in _EMBEDDING_CACHE:
        _EMBEDDING_CACHE[key] = build_embedding_model(
            config.embedding_provider,
            config.embedding_model,
            openai_api_key=config.openai_api_key,
            openai_base_url=config.openai_base_url,
            aihubmix_api_key=config.aihubmix_api_key,
            aihubmix_base_url=config.aihubmix_base_url,
        )
    return _EMBEDDING_CACHE[key]


def _get_vector_store(config: AppConfig, embeddings):
    key = _vector_store_cache_key(config)
    if key not in _VECTOR_STORE_CACHE:
        _VECTOR_STORE_CACHE[key] = load_vector_index(
            backend=config.vector_backend,
            embeddings=embeddings,
            persist_dir=config.local_cache_dir,
            milvus_uri=config.milvus_uri,
            milvus_token=config.milvus_token,
            milvus_db_name=config.milvus_db_name,
            milvus_collection=config.milvus_collection,
        )
    return _VECTOR_STORE_CACHE[key]


def _get_optional_vector_store(
    config: AppConfig,
    embeddings,
    *,
    collection_name: str,
):
    key = _vector_store_cache_key(config, collection_name=collection_name)
    if key in _VECTOR_STORE_CACHE:
        return _VECTOR_STORE_CACHE[key]

    if not vector_index_exists(
        config.vector_backend,
        config.local_cache_dir,
        milvus_uri=config.milvus_uri,
        milvus_token=config.milvus_token,
        milvus_db_name=config.milvus_db_name,
        milvus_collection=collection_name,
    ):
        return None

    _VECTOR_STORE_CACHE[key] = load_vector_index(
        backend=config.vector_backend,
        embeddings=embeddings,
        persist_dir=config.local_cache_dir,
        milvus_uri=config.milvus_uri,
        milvus_token=config.milvus_token,
        milvus_db_name=config.milvus_db_name,
        milvus_collection=collection_name,
    )
    return _VECTOR_STORE_CACHE[key]


def _set_vector_store_cache(
    config: AppConfig,
    vector_store,
    *,
    collection_name: str | None = None,
) -> None:
    _VECTOR_STORE_CACHE[_vector_store_cache_key(config, collection_name=collection_name)] = vector_store


def _normalize_retrieval_scope(scope: str) -> str:
    normalized = str(scope or "").strip().lower()
    if normalized in {"reference", "references", "ref", "refs"}:
        return "references"
    return "main"


def _to_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
        if "|" in text:
            return [part.strip() for part in text.split("|") if part.strip()]
        return [text]
    return [str(value)]


def _to_bbox_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                page = item.get("page")
                bbox = item.get("bbox")
                rows.append({"page": page, "bbox": bbox})
        return rows
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        return _to_bbox_list(parsed)
    return []


def _first_block_id(doc: Document) -> str:
    metadata = dict(doc.metadata or {})
    block_ids = _to_str_list(metadata.get("mineru_block_ids"))
    if block_ids:
        return block_ids[0]
    parent_id = str(metadata.get("parent_id", "")).strip()
    if parent_id:
        return parent_id
    return f"chunk_{metadata.get('chunk_id', '?')}"


def _normalize_bbox_numbers(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) < 4:
        return []
    bbox: list[float] = []
    for item in value[:4]:
        try:
            bbox.append(float(item))
        except (TypeError, ValueError):
            return []
    return bbox


def _pick_bbox(metadata: dict[str, Any], page: str) -> list[float]:
    rows = _to_bbox_list(metadata.get("mineru_bboxes"))
    for row in rows:
        row_page = str(row.get("page", "")).strip()
        bbox = _normalize_bbox_numbers(row.get("bbox"))
        if bbox and row_page == page:
            return bbox
    for row in rows:
        bbox = _normalize_bbox_numbers(row.get("bbox"))
        if bbox:
            return bbox
    return []


def _page_label(doc: Document) -> str:
    value = (doc.metadata or {}).get("page")
    try:
        page = int(value)
        return str(page) if page > 0 else "?"
    except (TypeError, ValueError):
        return "?"


def _section_label(doc: Document) -> str:
    section = str((doc.metadata or {}).get("section_path", "")).strip()
    if not section:
        return "正文"
    return section


def _paper_year(doc: Document) -> str:
    metadata = dict(doc.metadata or {})
    year = str(metadata.get("year", metadata.get("paper_year", ""))).strip()
    if re.fullmatch(r"\d{4}", year):
        return year
    return "n.d."


def _author_brief(doc: Document) -> str:
    metadata = dict(doc.metadata or {})
    authors = str(metadata.get("authors", metadata.get("paper_authors", ""))).strip()
    if authors:
        parts = [
            part.strip()
            for part in re.split(r"[;,，、]|(?:\band\b)", authors, flags=re.IGNORECASE)
            if part.strip()
        ]
        if parts:
            first = parts[0]
            has_multi = len(parts) > 1 or "et al" in authors.lower()
            brief = f"{first}等" if has_multi else first
            return " ".join(re.sub(r"[\[\],]+", " ", brief).split())

    return ""


def _title_or_source_brief(doc: Document) -> str:
    metadata = dict(doc.metadata or {})
    title = str(metadata.get("title", metadata.get("paper_title", ""))).strip()

    if title:
        clean_title = " ".join(re.sub(r"[\[\],]+", " ", title).split())
        if len(clean_title) > 72:
            return f"{clean_title[:69]}..."
        return clean_title

    source = str(metadata.get("source", "unknown")).strip() or "unknown"
    source_stem = Path(source).stem
    if source_stem:
        return source_stem

    author_brief = _author_brief(doc)
    if author_brief:
        return author_brief

    return "unknown"


def _citation_tag(doc: Document) -> str:
    metadata = dict(doc.metadata or {})
    doc_id = str(metadata.get("doc_id", "")).strip() or "unknown_doc"
    source = str(metadata.get("source", "")).strip() or "unknown"
    page = _page_label(doc)
    section = _section_label(doc)
    block = _first_block_id(doc)
    return (
        f"[doc:{doc_id}|source:{source}|p:{page}|"
        f"section:{section}|block:{block}]"
    )


def _citation_text(doc: Document) -> str:
    brief = _title_or_source_brief(doc)
    year = _paper_year(doc)
    page = _page_label(doc)
    section = _section_label(doc)
    return f"[{brief}, {year}, p.{page}, {section}]"


def _build_evidence_record(doc: Document) -> EvidenceRecord:
    metadata = dict(doc.metadata or {})
    page = _page_label(doc)
    snippet = " ".join(doc.page_content.strip().split())
    if len(snippet) > 1000:
        snippet = f"{snippet[:1000]} ..."
    return EvidenceRecord(
        citation_text=_citation_text(doc),
        citation_tag=_citation_tag(doc),
        source=str(metadata.get("source", "")).strip(),
        page=page,
        section_path=_section_label(doc),
        doc_id=str(metadata.get("doc_id", "")).strip(),
        block_id=_first_block_id(doc),
        bbox=_pick_bbox(metadata, page),
        snippet=snippet,
    )


def _strip_inline_citations(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return text

    # Strip common inline citation tags produced by LLM.
    patterns = [
        r"\[[^\[\]\n]{0,220}(?:doc:|source:|section:|block:)[^\[\]\n]*\]",
        r"\[[^\[\]\n]{0,220}p\.\s*\d+[^\[\]\n]*\]",
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    # Drop explicit citation/reference section in answer body.
    drop_headers = {"引用：", "引用:", "references:", "references：", "参考：", "参考:"}
    kept_lines: list[str] = []
    skipping = False
    for line in cleaned.splitlines():
        marker = line.strip().lower()
        if marker in drop_headers:
            skipping = True
            continue
        if skipping:
            if not line.strip():
                skipping = False
            continue
        kept_lines.append(line)

    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _extract_first_json_object(text: str) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    start = raw.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]
    return None


def _to_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(item).strip() for item in value]
        return [item for item in items if item]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return [str(value).strip()] if str(value).strip() else []


def _parse_structured_llm_answer(raw_answer: str) -> dict[str, Any] | None:
    json_text = _extract_first_json_object(raw_answer)
    if not json_text:
        return None
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    conclusion = str(payload.get("conclusion", "")).strip()
    evidence_points = _to_string_list(payload.get("evidence_points"))
    uncertainties = _to_string_list(payload.get("uncertainties"))
    return {
        "conclusion": conclusion,
        "evidence_points": evidence_points,
        "uncertainties": uncertainties,
    }


def _render_structured_answer(payload: dict[str, Any]) -> str:
    conclusion = str(payload.get("conclusion", "")).strip()
    evidence_points = _to_string_list(payload.get("evidence_points"))
    uncertainties = _to_string_list(payload.get("uncertainties"))

    parts: list[str] = []
    if conclusion:
        parts.append(f"结论：\n{conclusion}")

    if evidence_points:
        lines = [f"{idx}. {item}" for idx, item in enumerate(evidence_points, start=1)]
        parts.append("证据点：\n" + "\n".join(lines))

    if uncertainties:
        lines = [f"- {item}" for item in uncertainties]
        parts.append("不确定项：\n" + "\n".join(lines))

    return "\n\n".join(parts).strip()


def _normalize_answer_from_llm(raw_answer: str) -> str:
    parsed = _parse_structured_llm_answer(raw_answer)
    if parsed is not None:
        rendered = _render_structured_answer(parsed)
        if rendered:
            return rendered
    return _strip_inline_citations(raw_answer)


def _normalize_doc_id_list(doc_ids: list[str]) -> list[str]:
    return sorted({str(item).strip() for item in doc_ids if str(item).strip()})


def _get_llm_client(config: AppConfig):
    key = _llm_cache_key(config)
    if key not in _LLM_CACHE:
        _LLM_CACHE[key] = build_llm_client(
            provider=config.llm_provider,
            model=config.llm_model,
            temperature=config.llm_temperature,
            aihubmix_api_key=config.aihubmix_api_key,
            aihubmix_base_url=config.aihubmix_base_url,
            aihubmix_api_mode=config.aihubmix_api_mode,
            openai_api_key=config.openai_api_key,
            openai_base_url=config.openai_base_url,
            openai_api_mode=config.openai_api_mode,
            ollama_base_url=config.ollama_base_url,
        )
    return _LLM_CACHE[key]


def clear_runtime_cache() -> None:
    _EMBEDDING_CACHE.clear()
    _VECTOR_STORE_CACHE.clear()
    _LLM_CACHE.clear()
    cache_store.clear_local_cache_caches()
    clear_retrieval_cache()


def _load_question_answering_resources(
    config: AppConfig,
    embeddings,
    *,
    scope: str,
):
    normalized_scope = _normalize_retrieval_scope(scope)
    if normalized_scope == "references":
        reference_corpus, reference_corpus_key = cache_store.load_reference_chunk_corpus(config)
        vector_store = None
        retrieval_mode_override = "bm25"
        if config.references_strategy.strip().lower() == "separate_collection":
            vector_store = _get_optional_vector_store(
                config,
                embeddings,
                collection_name=config.milvus_references_collection,
            )
            if vector_store is not None:
                retrieval_mode_override = config.retrieval_mode
        return {
            "scope": "references",
            "vector_store": vector_store,
            "chunk_corpus": reference_corpus,
            "chunk_corpus_key": reference_corpus_key,
            "parent_map": None,
            "apply_metadata_filters": False,
            "retrieval_mode_override": retrieval_mode_override,
            "use_parent_context": False,
        }

    chunk_corpus, chunk_corpus_key = cache_store.load_chunk_corpus(config)
    return {
        "scope": "main",
        "vector_store": _get_vector_store(config, embeddings),
        "chunk_corpus": chunk_corpus,
        "chunk_corpus_key": chunk_corpus_key,
        "parent_map": cache_store.load_parent_corpus_map(config),
        "apply_metadata_filters": True,
        "retrieval_mode_override": None,
        "use_parent_context": True,
    }


def ingest_documents(
    config: AppConfig,
    pdf_path: Path | None = None,
    *,
    force: bool = False,
) -> IngestResult:
    trace = OperationTrace(
        "ingest",
        config.local_cache_dir / "observability.jsonl",
        metadata={"force": force, "single_pdf": bool(pdf_path)},
    )
    try:
        with trace.stage("collect_pdf_paths"):
            pdf_paths = _collect_pdf_paths(config, pdf_path)
        if not pdf_paths:
            raise ValueError("No PDF text found. Add PDFs to data/pdf and retry.")

        signature = _build_ingest_signature(config, pdf_paths)
        cached = cache_store.load_ingest_cache(config)
        chunk_corpus_exists = cache_store.chunk_corpus_path(config).exists()
        parent_corpus_exists = cache_store.parent_corpus_path(config).exists()
        if (
            not force
            and pdf_path is None
            and cached
            and cached.get("signature") == signature
            and chunk_corpus_exists
            and parent_corpus_exists
        ):
            result = IngestResult(
                raw_documents=int(cached.get("raw_documents", 0)),
                chunks=int(cached.get("chunks", 0)),
                cache_dir=config.local_cache_dir,
                skipped=True,
            )
            trace.set_field("skipped", True)
            trace.finish(status="ok")
            return result

        with trace.stage("parse_documents"):
            parse_result: PdfLoadResult
            if pdf_path:
                parse_result = load_pdf_pages(
                    pdf_path,
                    mineru_output_dir=config.mineru_output_dir,
                    mineru_api_token=config.mineru_api_token,
                    mineru_api_base_url=config.mineru_api_base_url,
                    mineru_cloud_model_version=config.mineru_cloud_model_version,
                    mineru_cloud_poll_interval_sec=config.mineru_cloud_poll_interval_sec,
                    mineru_cloud_timeout_sec=config.mineru_cloud_timeout_sec,
                )
            else:
                parse_result = load_documents_from_dir(
                    config.data_pdf_dir,
                    mineru_output_dir=config.mineru_output_dir,
                    mineru_api_token=config.mineru_api_token,
                    mineru_api_base_url=config.mineru_api_base_url,
                    mineru_cloud_model_version=config.mineru_cloud_model_version,
                    mineru_cloud_poll_interval_sec=config.mineru_cloud_poll_interval_sec,
                    mineru_cloud_timeout_sec=config.mineru_cloud_timeout_sec,
                )
        docs = parse_result.documents
        block_rows = parse_result.blocks
        if not docs:
            raise ValueError("No PDF text found. Add PDFs to data/pdf and retry.")

        with trace.stage("chunk_documents"):
            chunks, parent_docs = split_documents_with_parents(
                documents=docs,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                tokenizer_model=config.chunk_tokenizer_model,
                use_structure_split=config.chunk_use_structure_split,
                min_block_chars=config.chunk_min_block_chars,
                chunk_strategy=config.chunk_strategy,
                semantic_hard_max_chars=config.chunk_semantic_hard_max_chars,
            )
        with trace.stage("chunk_quality"):
            chunks, quality_summary = cache_store.run_chunk_quality_checks(
                chunks,
                enabled=config.chunk_quality_check_enabled,
                header_footer_min_freq=config.chunk_quality_header_footer_min_freq,
            )

        main_chunks, reference_chunks = cache_store.split_reference_docs(chunks)
        main_parent_docs, _ = cache_store.split_reference_docs(parent_docs)
        if not main_chunks and reference_chunks:
            main_chunks = reference_chunks
            main_parent_docs = parent_docs
            reference_chunks = []
        reference_purity_summary = cache_store.build_reference_purity_summary(reference_chunks)
        quality_summary["reference_purity"] = reference_purity_summary
        cache_store.save_chunk_quality_report(config, quality_summary)

        incoming_chunk_rows = [cache_store.chunk_to_structured_row(doc) for doc in chunks]
        incoming_reference_keyword_rows = cache_store.build_reference_keyword_rows(reference_chunks)
        incoming_doc_ids = cache_store.collect_doc_ids_from_docs(chunks)
        if not incoming_doc_ids:
            incoming_doc_ids = cache_store.collect_doc_ids_from_rows(block_rows)
        if not incoming_doc_ids:
            raise ValueError("Failed to extract doc_id from parsed documents.")

        with trace.stage("load_local_cache"):
            existing_state = cache_store.load_local_cache_state(config)

        existing_doc_ids = (
            cache_store.collect_doc_ids_from_docs(existing_state.main_chunks)
            | cache_store.collect_doc_ids_from_docs(existing_state.parent_docs)
            | cache_store.collect_doc_ids_from_docs(existing_state.reference_chunks)
            | cache_store.collect_doc_ids_from_rows(existing_state.block_rows)
            | cache_store.collect_doc_ids_from_rows(existing_state.chunk_rows)
            | cache_store.collect_doc_ids_from_rows(existing_state.reference_keyword_rows)
        )

        active_doc_ids = set(incoming_doc_ids)
        if not force:
            active_doc_ids -= existing_doc_ids

        if not active_doc_ids:
            result = IngestResult(
                raw_documents=0,
                chunks=0,
                cache_dir=config.local_cache_dir,
                skipped=True,
            )
            trace.set_field("skipped", True)
            trace.finish(status="ok")
            return result

        active_main_chunks = cache_store.filter_docs_by_doc_ids(main_chunks, active_doc_ids, keep=True)
        active_parent_docs = cache_store.filter_docs_by_doc_ids(
            main_parent_docs,
            active_doc_ids,
            keep=True,
        )
        active_reference_chunks = cache_store.filter_docs_by_doc_ids(
            reference_chunks,
            active_doc_ids,
            keep=True,
        )
        active_block_rows = cache_store.filter_rows_by_doc_ids(block_rows, active_doc_ids, keep=True)
        active_chunk_rows = cache_store.filter_rows_by_doc_ids(
            incoming_chunk_rows,
            active_doc_ids,
            keep=True,
        )
        active_reference_keyword_rows = cache_store.filter_rows_by_doc_ids(
            incoming_reference_keyword_rows,
            active_doc_ids,
            keep=True,
        )

        kept_existing_main = cache_store.filter_docs_by_doc_ids(
            existing_state.main_chunks,
            active_doc_ids,
            keep=False,
        )
        kept_existing_parents = cache_store.filter_docs_by_doc_ids(
            existing_state.parent_docs,
            active_doc_ids,
            keep=False,
        )
        kept_existing_reference = cache_store.filter_docs_by_doc_ids(
            existing_state.reference_chunks,
            active_doc_ids,
            keep=False,
        )
        kept_existing_blocks = cache_store.filter_rows_by_doc_ids(
            existing_state.block_rows,
            active_doc_ids,
            keep=False,
        )
        kept_existing_chunk_rows = cache_store.filter_rows_by_doc_ids(
            existing_state.chunk_rows,
            active_doc_ids,
            keep=False,
        )
        kept_existing_reference_keywords = cache_store.filter_rows_by_doc_ids(
            existing_state.reference_keyword_rows,
            active_doc_ids,
            keep=False,
        )

        merged_state = cache_store.LocalCacheState(
            main_chunks=[*kept_existing_main, *active_main_chunks],
            parent_docs=[*kept_existing_parents, *active_parent_docs],
            reference_chunks=[*kept_existing_reference, *active_reference_chunks],
            block_rows=[*kept_existing_blocks, *active_block_rows],
            chunk_rows=[*kept_existing_chunk_rows, *active_chunk_rows],
            reference_keyword_rows=[
                *kept_existing_reference_keywords,
                *active_reference_keyword_rows,
            ],
        )

        total_page_count = cache_store.count_pages_from_blocks(merged_state.block_rows)
        ingest_cache_payload = cache_store.build_ingest_cache_payload(
            config=config,
            signature=signature,
            raw_documents=(
                total_page_count if total_page_count > 0 else len(merged_state.parent_docs)
            ),
            chunks=len(merged_state.main_chunks),
        )

        with trace.stage("build_embeddings"):
            embeddings = _get_embeddings(config)
        with trace.stage("sync_remote_and_local"):
            vector_store = execute_sync_plan(
                config,
                SyncPlan(
                    operation="upsert",
                    doc_ids=sorted(active_doc_ids),
                    remote_payload={
                        "main_documents": cache_store.serialize_documents(active_main_chunks),
                        "reference_documents": cache_store.serialize_documents(active_reference_chunks),
                        "upsert_doc_ids": sorted(active_doc_ids),
                    },
                    target_state=merged_state,
                    ingest_cache_payload=ingest_cache_payload,
                ),
                embeddings,
            )

        if vector_store is not None:
            _set_vector_store_cache(config, vector_store)
        else:
            _VECTOR_STORE_CACHE.pop(_vector_store_cache_key(config), None)

        cache_store.clear_local_cache_caches()
        clear_retrieval_cache()

        active_page_count = cache_store.count_pages_from_blocks(active_block_rows)
        result = IngestResult(
            raw_documents=(
                active_page_count if active_page_count > 0 else len(active_parent_docs)
            ),
            chunks=len(active_main_chunks),
            cache_dir=config.local_cache_dir,
            skipped=False,
            reference_chunks=reference_purity_summary["total_reference_chunks"],
            suspicious_reference_chunks=reference_purity_summary["suspicious_reference_chunks"],
        )
        trace.set_field("active_doc_ids", len(active_doc_ids))
        trace.set_field("raw_documents", result.raw_documents)
        trace.set_field("chunks", result.chunks)
        trace.set_field("reference_chunks", result.reference_chunks)
        trace.set_field(
            "suspicious_reference_chunks",
            result.suspicious_reference_chunks,
        )
        trace.finish(status="ok")
        return result
    except Exception as exc:
        trace.finish(status="error", error=str(exc))
        raise


def delete_documents(config: AppConfig, doc_ids: list[str]) -> DeleteResult:
    trace = OperationTrace(
        "delete_documents",
        config.local_cache_dir / "observability.jsonl",
        metadata={"requested_doc_ids": len(doc_ids)},
    )
    try:
        requested_doc_ids = _normalize_doc_id_list(doc_ids)
        if not requested_doc_ids:
            raise ValueError("Please provide at least one valid doc_id.")
        target_doc_ids = set(requested_doc_ids)

        with trace.stage("load_local_cache"):
            existing_state = cache_store.load_local_cache_state(config)

        known_doc_ids = (
            cache_store.collect_doc_ids_from_docs(existing_state.main_chunks)
            | cache_store.collect_doc_ids_from_docs(existing_state.parent_docs)
            | cache_store.collect_doc_ids_from_docs(existing_state.reference_chunks)
            | cache_store.collect_doc_ids_from_rows(existing_state.block_rows)
            | cache_store.collect_doc_ids_from_rows(existing_state.chunk_rows)
            | cache_store.collect_doc_ids_from_rows(existing_state.reference_keyword_rows)
        )
        deleted_doc_ids = sorted(target_doc_ids & known_doc_ids)

        kept_state = cache_store.LocalCacheState(
            main_chunks=cache_store.filter_docs_by_doc_ids(
                existing_state.main_chunks,
                target_doc_ids,
                keep=False,
            ),
            parent_docs=cache_store.filter_docs_by_doc_ids(
                existing_state.parent_docs,
                target_doc_ids,
                keep=False,
            ),
            reference_chunks=cache_store.filter_docs_by_doc_ids(
                existing_state.reference_chunks,
                target_doc_ids,
                keep=False,
            ),
            block_rows=cache_store.filter_rows_by_doc_ids(
                existing_state.block_rows,
                target_doc_ids,
                keep=False,
            ),
            chunk_rows=cache_store.filter_rows_by_doc_ids(
                existing_state.chunk_rows,
                target_doc_ids,
                keep=False,
            ),
            reference_keyword_rows=cache_store.filter_rows_by_doc_ids(
                existing_state.reference_keyword_rows,
                target_doc_ids,
                keep=False,
            ),
        )

        with trace.stage("build_embeddings"):
            embeddings = _get_embeddings(config)
        with trace.stage("sync_remote_and_local"):
            vector_store = execute_sync_plan(
                config,
                SyncPlan(
                    operation="delete",
                    doc_ids=requested_doc_ids,
                    remote_payload={"delete_doc_ids": requested_doc_ids},
                    target_state=kept_state,
                    ingest_cache_payload=None,
                ),
                embeddings,
            )

        if vector_store is not None:
            _set_vector_store_cache(config, vector_store)
        else:
            _VECTOR_STORE_CACHE.pop(_vector_store_cache_key(config), None)

        cache_store.clear_local_cache_caches()
        clear_retrieval_cache()

        result = DeleteResult(
            requested_doc_ids=requested_doc_ids,
            deleted_doc_ids=deleted_doc_ids,
            removed_chunks=len(existing_state.main_chunks) - len(kept_state.main_chunks),
            removed_parents=len(existing_state.parent_docs) - len(kept_state.parent_docs),
            removed_reference_chunks=(
                len(existing_state.reference_chunks) - len(kept_state.reference_chunks)
            ),
            removed_block_rows=len(existing_state.block_rows) - len(kept_state.block_rows),
            removed_structured_chunk_rows=len(existing_state.chunk_rows) - len(kept_state.chunk_rows),
            removed_reference_keyword_rows=(
                len(existing_state.reference_keyword_rows)
                - len(kept_state.reference_keyword_rows)
            ),
        )
        trace.set_field("deleted_doc_ids", len(result.deleted_doc_ids))
        trace.finish(status="ok")
        return result
    except Exception as exc:
        trace.finish(status="error", error=str(exc))
        raise


def answer_question(
    config: AppConfig,
    question: str,
    *,
    scope: str = "main",
) -> QAResult:
    normalized_scope = _normalize_retrieval_scope(scope)
    trace = OperationTrace(
        "answer_question",
        config.local_cache_dir / "observability.jsonl",
        metadata={"question_length": len(question), "scope": normalized_scope},
    )
    try:
        with trace.stage("build_embeddings"):
            embeddings = _get_embeddings(config)
        with trace.stage("load_retrieval_resources"):
            resources = _load_question_answering_resources(
                config,
                embeddings,
                scope=normalized_scope,
            )
        with trace.stage("retrieve_evidence"):
            retrieval_result = run_retrieval_flow(
                config,
                question,
                resources["vector_store"],
                chunk_corpus=resources["chunk_corpus"],
                chunk_corpus_key=resources["chunk_corpus_key"],
                parent_map=resources["parent_map"],
                apply_metadata_filters=bool(resources["apply_metadata_filters"]),
                retrieval_mode_override=resources["retrieval_mode_override"],
                use_parent_context=bool(resources["use_parent_context"]),
            )

        docs = retrieval_result.evidence_docs
        if not docs:
            result = QAResult(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                citations=[],
                contexts=[],
                evidences=[],
                retrieval_scope=normalized_scope,
            )
            trace.set_field("evidence_docs", 0)
            trace.finish(status="ok")
            return result

        with trace.stage("generate_answer"):
            prompt = build_qa_prompt(
                question=question,
                documents=retrieval_result.generation_docs,
                context_label=(
                    "参考文献上下文" if normalized_scope == "references" else "上下文"
                ),
                scope_hint=(
                    "以下上下文全部来自论文的参考文献列表。回答时只依据这些参考文献条目，"
                    "优先回答是否引用、引用了哪些工作、涉及哪些作者或年份。"
                    if normalized_scope == "references"
                    else ""
                ),
            )
            llm = _get_llm_client(config)
            answer = _normalize_answer_from_llm(llm.generate(prompt))

        citations: list[str] = []
        evidences: list[EvidenceRecord] = []
        contexts = [doc.page_content for doc in retrieval_result.generation_docs]
        seen = set()
        for doc in docs:
            evidence = _build_evidence_record(doc)
            evidences.append(evidence)
            if evidence.citation_text not in seen:
                citations.append(evidence.citation_text)
                seen.add(evidence.citation_text)

        result = QAResult(
            answer=answer,
            citations=citations,
            contexts=contexts,
            evidences=evidences,
            retrieval_scope=normalized_scope,
        )
        trace.set_field("evidence_docs", len(docs))
        trace.set_field("generation_docs", len(retrieval_result.generation_docs))
        trace.finish(status="ok")
        return result
    except Exception as exc:
        trace.finish(status="error", error=str(exc))
        raise
