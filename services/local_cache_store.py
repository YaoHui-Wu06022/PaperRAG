from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from langchain_core.documents import Document

from config import AppConfig
from ingestion.reference_detection import (
    is_reference_heading,
    is_reference_section,
    looks_like_reference_chunk,
)

# 这个模块负责管理 `data/cache` 下的本地知识库状态。
# 它既保存检索直接使用的语料，也保存目录查询、诊断分析、
# 崩溃恢复所需的结构化行数据。


INGEST_CACHE_FILE = "ingest_cache.json"
CHUNK_CORPUS_FILE = "chunk_corpus.jsonl"
PARENT_CORPUS_FILE = "parent_corpus.jsonl"
PAPER_CORPUS_FILE = "paper_corpus.jsonl"
SECTION_SUMMARY_CORPUS_FILE = "section_summary_corpus.jsonl"
BLOCK_STRUCTURED_FILE = "block_structured.jsonl"
CHUNK_STRUCTURED_FILE = "chunk_structured.jsonl"
PAPER_CATALOG_FILE = "paper_catalog.jsonl"
PAPER_CATALOG_DB_FILE = "paper_catalog.sqlite3"
REFERENCE_CITATION_GRAPH_FILE = "citation_edges.jsonl"
REFERENCE_CHUNK_CORPUS_FILE = "reference_chunk_corpus.jsonl"
CHUNK_QUALITY_REPORT_FILE = "chunk_quality_report.json"
SYNC_JOURNAL_FILE = "sync_operation.json"

_CHUNK_CORPUS_CACHE: dict[tuple, list[Document]] = {}
_PARENT_CORPUS_CACHE: dict[tuple, dict[str, Document]] = {}
_PAPER_CORPUS_CACHE: dict[tuple, list[Document]] = {}
_SECTION_CORPUS_CACHE: dict[tuple, list[Document]] = {}


@dataclass
class LocalCacheState:
    # `main_chunks` / `parent_docs` / `paper_docs` 是检索主链路的几层语料。
    # 基于 row 的字段则是配套的结构化侧车数据，用于目录查询、诊断、
    # 引用网络和精确重建。
    main_chunks: list[Document]
    parent_docs: list[Document]
    paper_docs: list[Document]
    section_docs: list[Document]
    reference_chunks: list[Document]
    block_rows: list[dict[str, Any]]
    chunk_rows: list[dict[str, Any]]
    paper_rows: list[dict[str, Any]]
    citation_rows: list[dict[str, Any]]
    reference_keyword_rows: list[dict[str, Any]]


def clear_local_cache_caches() -> None:
    _CHUNK_CORPUS_CACHE.clear()
    _PARENT_CORPUS_CACHE.clear()
    _PAPER_CORPUS_CACHE.clear()
    _SECTION_CORPUS_CACHE.clear()


def ingest_cache_path(config: AppConfig) -> Path:
    return config.local_cache_dir / INGEST_CACHE_FILE


def chunk_corpus_path(config: AppConfig) -> Path:
    return config.local_cache_dir / CHUNK_CORPUS_FILE


def parent_corpus_path(config: AppConfig) -> Path:
    return config.local_cache_dir / PARENT_CORPUS_FILE


def paper_corpus_path(config: AppConfig) -> Path:
    return config.local_cache_dir / PAPER_CORPUS_FILE


def section_summary_corpus_path(config: AppConfig) -> Path:
    return config.local_cache_dir / SECTION_SUMMARY_CORPUS_FILE


def block_structured_path(config: AppConfig) -> Path:
    return config.local_cache_dir / BLOCK_STRUCTURED_FILE


def chunk_structured_path(config: AppConfig) -> Path:
    return config.local_cache_dir / CHUNK_STRUCTURED_FILE


def paper_catalog_path(config: AppConfig) -> Path:
    return config.local_cache_dir / PAPER_CATALOG_FILE


def paper_catalog_db_path(config: AppConfig) -> Path:
    return config.local_cache_dir / PAPER_CATALOG_DB_FILE


def citation_graph_path(config: AppConfig) -> Path:
    return config.local_cache_dir / REFERENCE_CITATION_GRAPH_FILE


def reference_chunk_corpus_path(config: AppConfig) -> Path:
    return config.local_cache_dir / REFERENCE_CHUNK_CORPUS_FILE


def chunk_quality_report_path(config: AppConfig) -> Path:
    return config.local_cache_dir / CHUNK_QUALITY_REPORT_FILE


def reference_keyword_index_path(config: AppConfig) -> Path:
    return config.references_keyword_index_file


def sync_journal_path(config: AppConfig) -> Path:
    return config.local_cache_dir / SYNC_JOURNAL_FILE


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def serialize_document(doc: Document) -> dict[str, Any]:
    return {
        "page_content": doc.page_content,
        "metadata": dict(doc.metadata or {}),
    }


def deserialize_document(row: dict[str, Any]) -> Document:
    return Document(
        page_content=str(row.get("page_content", "")),
        metadata=row.get("metadata") or {},
    )


def serialize_documents(docs: list[Document]) -> list[dict[str, Any]]:
    return [serialize_document(doc) for doc in docs]


def deserialize_documents(rows: list[dict[str, Any]]) -> list[Document]:
    return [deserialize_document(row) for row in rows if isinstance(row, dict)]


def _save_doc_corpus(path: Path, docs: list[Document]) -> None:
    lines = [
        json.dumps(serialize_document(doc), ensure_ascii=False, default=str)
        for doc in docs
    ]
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def save_rows_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False, default=str) for row in rows]
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def load_rows_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def doc_id_from_metadata(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    return str(metadata.get("doc_id", "")).strip()


def doc_id_from_doc(doc: Document) -> str:
    return doc_id_from_metadata(dict(doc.metadata or {}))


def collect_doc_ids_from_docs(docs: list[Document]) -> set[str]:
    return {doc_id for doc_id in (doc_id_from_doc(doc) for doc in docs) if doc_id}


def collect_doc_ids_from_rows(rows: list[dict[str, Any]]) -> set[str]:
    doc_ids: set[str] = set()
    for row in rows:
        doc_id = str(row.get("doc_id", "")).strip()
        if doc_id:
            doc_ids.add(doc_id)
    return doc_ids


def filter_docs_by_doc_ids(
    docs: list[Document],
    doc_ids: set[str],
    *,
    keep: bool,
) -> list[Document]:
    filtered: list[Document] = []
    for doc in docs:
        matched = doc_id_from_doc(doc) in doc_ids
        if (keep and matched) or (not keep and not matched):
            filtered.append(doc)
    return filtered


def filter_rows_by_doc_ids(
    rows: list[dict[str, Any]],
    doc_ids: set[str],
    *,
    keep: bool,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        matched = str(row.get("doc_id", "")).strip() in doc_ids
        if (keep and matched) or (not keep and not matched):
            filtered.append(row)
    return filtered


def to_str_list(value: Any) -> list[str]:
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
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        if "|" in text:
            return [part.strip() for part in text.split("|") if part.strip()]
        return [text]
    return [str(value)]


def to_bbox_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                rows.append({"page": item.get("page"), "bbox": item.get("bbox")})
        return rows
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        return to_bbox_list(parsed)
    return []


def is_reference_doc(doc: Document) -> bool:
    metadata = doc.metadata or {}
    value = metadata.get("is_reference", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def split_reference_docs(docs: list[Document]) -> tuple[list[Document], list[Document]]:
    """把一份语料拆成正文 chunk 和参考文献 chunk。"""
    main_docs: list[Document] = []
    reference_docs: list[Document] = []
    for doc in docs:
        if is_reference_doc(doc):
            reference_docs.append(doc)
        else:
            main_docs.append(doc)
    return main_docs, reference_docs


def count_pages_from_blocks(block_rows: list[dict[str, Any]]) -> int:
    page_keys = set()
    for row in block_rows:
        source = str(row.get("source", "")).strip()
        if not source:
            continue
        try:
            page = int(row.get("page", 0))
        except (TypeError, ValueError):
            continue
        if page > 0:
            page_keys.add((source, page))
    return len(page_keys)


def chunk_to_structured_row(doc: Document) -> dict[str, Any]:
    metadata = dict(doc.metadata or {})
    block_types = to_str_list(metadata.get("block_types"))
    figure_ids = to_str_list(metadata.get("figure_ids"))
    table_ids = to_str_list(metadata.get("table_ids"))
    section_paths = to_str_list(metadata.get("section_paths"))
    if not section_paths and str(metadata.get("section_path", "")).strip():
        section_paths = [str(metadata.get("section_path", "")).strip()]
    block_ids = to_str_list(metadata.get("mineru_block_ids"))
    bboxes = to_bbox_list(metadata.get("mineru_bboxes"))
    quality_flags = to_str_list(metadata.get("chunk_quality_flags"))
    return {
        "chunk_uid": str(
            metadata.get("chunk_uid")
            or metadata.get("chunk_id")
            or metadata.get("parent_id")
            or ""
        ),
        "chunk_id": metadata.get("chunk_id"),
        "doc_id": str(metadata.get("doc_id", "")),
        "source": str(metadata.get("source", "")),
        "title": str(metadata.get("title", metadata.get("paper_title", ""))),
        "section_path": str(metadata.get("section_path", "")),
        "section_paths": section_paths,
        "page": metadata.get("page"),
        "page_end": metadata.get("page_end", metadata.get("page")),
        "page_range": str(metadata.get("page_range", "")),
        "block_types": block_types,
        "figure_ids": figure_ids,
        "table_ids": table_ids,
        "year": str(metadata.get("year", metadata.get("paper_year", ""))),
        "authors": str(metadata.get("authors", metadata.get("paper_authors", ""))),
        "is_reference": is_reference_doc(doc),
        "parent_id": str(metadata.get("parent_id", "")),
        "block_ids": block_ids,
        "bboxes": bboxes,
        "quality_flags": quality_flags,
        "quality_pass": bool(metadata.get("chunk_quality_pass", not quality_flags)),
        "text": doc.page_content,
    }


def extract_reference_keywords(text: str, top_n: int = 20) -> list[str]:
    normalized = text.lower()
    en_tokens = re.findall(r"[a-z][a-z0-9\-]{2,}", normalized)
    zh_tokens = re.findall(r"[\u4e00-\u9fff]{2,6}", normalized)
    counter = Counter(en_tokens + zh_tokens)
    return [token for token, _ in counter.most_common(top_n)]


def build_reference_keyword_rows(chunks: list[Document]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for doc in chunks:
        metadata = dict(doc.metadata or {})
        rows.append(
            {
                "doc_id": str(metadata.get("doc_id", "")),
                "source": str(metadata.get("source", "")),
                "title": str(metadata.get("title", metadata.get("paper_title", ""))),
                "section_path": str(metadata.get("section_path", "")),
                "page_range": str(metadata.get("page_range", "")),
                "year": str(metadata.get("year", metadata.get("paper_year", ""))),
                "authors": str(metadata.get("authors", metadata.get("paper_authors", ""))),
                "keywords": extract_reference_keywords(doc.page_content),
                "text": doc.page_content,
            }
        )
    return rows


def build_reference_purity_summary(chunks: list[Document]) -> dict[str, Any]:
    """统计参考文献语料里可能存在的识别或切分异常。

    这是启发式质量信号，不是硬错误。它主要用来发现那些虽然被标成
    参考文献，但内容看起来仍像正文的 chunk。
    """
    suspicious_examples: list[dict[str, Any]] = []
    suspicious_count = 0

    for doc in chunks:
        metadata = dict(doc.metadata or {})
        text = " ".join(doc.page_content.split())
        section_path = str(metadata.get("section_path", "")).strip()
        section_ok = is_reference_section(section_path)
        text_ok = looks_like_reference_chunk(text)
        heading_ok = is_reference_heading(text)

        if heading_ok or (section_ok and text_ok):
            continue

        suspicious_count += 1
        if len(suspicious_examples) < 8:
            suspicious_examples.append(
                {
                    "source": str(metadata.get("source", "")),
                    "page": metadata.get("page"),
                    "section_path": section_path,
                    "preview": text[:180],
                    "reasons": [
                        reason
                        for reason, matched in (
                            ("section_path_not_reference", not section_ok),
                            ("text_not_reference_like", not text_ok),
                        )
                        if matched
                    ],
                }
            )

    total_reference_chunks = len(chunks)
    return {
        "total_reference_chunks": total_reference_chunks,
        "suspicious_reference_chunks": suspicious_count,
        "suspicious_rate": (
            suspicious_count / total_reference_chunks if total_reference_chunks else 0.0
        ),
        "examples": suspicious_examples,
    }


def _normalize_quality_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())[:160]


def _extract_edge_lines(text: str) -> tuple[str, str]:
    lines = [_normalize_quality_line(line) for line in text.splitlines() if line.strip()]
    if not lines:
        return "", ""
    return lines[0], lines[-1]


def _is_meaningful_edge_line(text: str) -> bool:
    if not text or len(text) < 4 or len(text) > 140:
        return False
    if re.fullmatch(r"[\d\W_]+", text):
        return False
    return bool(re.search(r"[A-Za-z\u4e00-\u9fff]{2,}", text))


def run_chunk_quality_checks(
    chunks: list[Document],
    *,
    enabled: bool,
    header_footer_min_freq: int,
) -> tuple[list[Document], dict[str, Any]]:
    """为 chunk 打上轻量质量标记，但不直接丢弃它们。

    这些结果仍然会保留给检索使用；标记的作用是暴露重复页眉、
    跨 section 合并、纯媒体块等解析风险。
    """
    if not chunks:
        return [], {
            "enabled": enabled,
            "total_chunks": 0,
            "failed_chunks": 0,
            "failed_rate": 0.0,
            "flag_counts": {},
            "repeated_edge_lines": [],
        }
    if not enabled:
        return chunks, {
            "enabled": False,
            "total_chunks": len(chunks),
            "failed_chunks": 0,
            "failed_rate": 0.0,
            "flag_counts": {},
            "repeated_edge_lines": [],
        }

    min_freq = max(2, int(header_footer_min_freq))
    edge_counter: Counter[str] = Counter()
    chunk_edges: list[tuple[str, str]] = []
    table_chunk_map: dict[str, list[int]] = {}

    for index, chunk in enumerate(chunks):
        first_line, last_line = _extract_edge_lines(chunk.page_content)
        chunk_edges.append((first_line, last_line))
        if _is_meaningful_edge_line(first_line):
            edge_counter[first_line] += 1
        if _is_meaningful_edge_line(last_line):
            edge_counter[last_line] += 1

        metadata = dict(chunk.metadata or {})
        for table_id in to_str_list(metadata.get("table_ids")):
            table_chunk_map.setdefault(table_id, []).append(index)

    repeated_edges = {line for line, freq in edge_counter.items() if freq >= min_freq}
    split_tables = {
        table_id for table_id, idx_list in table_chunk_map.items() if len(set(idx_list)) > 1
    }

    annotated: list[Document] = []
    flag_counts: Counter[str] = Counter()
    failed = 0

    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        flags: list[str] = []
        first_line, last_line = chunk_edges[index]
        if first_line in repeated_edges or last_line in repeated_edges:
            flags.append("header_footer_repetition")

        section_paths = to_str_list(metadata.get("section_paths"))
        if not section_paths:
            section = str(metadata.get("section_path", "")).strip()
            if section:
                section_paths = [section]
        if len({item for item in section_paths if item}) > 1:
            flags.append("cross_section_chunk")

        table_ids = to_str_list(metadata.get("table_ids"))
        if any(table_id in split_tables for table_id in table_ids):
            flags.append("table_split_risk")

        block_types = set(to_str_list(metadata.get("block_types")))
        normalized_text = " ".join(chunk.page_content.split())
        if "equation" in block_types and len(normalized_text) < 120:
            flags.append("equation_without_context")
        if ({"image", "table"} & block_types) and len(normalized_text) < 120:
            flags.append("media_without_context")

        if flags:
            failed += 1
            for flag in flags:
                flag_counts[flag] += 1

        metadata["chunk_quality_flags"] = flags
        metadata["chunk_quality_pass"] = not flags
        annotated.append(Document(page_content=chunk.page_content, metadata=metadata))

    summary = {
        "enabled": True,
        "total_chunks": len(chunks),
        "failed_chunks": failed,
        "failed_rate": failed / max(len(chunks), 1),
        "flag_counts": dict(flag_counts),
        "repeated_edge_lines": sorted(repeated_edges)[:30],
    }
    return annotated, summary


def save_chunk_quality_report(config: AppConfig, summary: dict[str, Any]) -> None:
    _atomic_write_text(
        chunk_quality_report_path(config),
        json.dumps(summary, ensure_ascii=False, indent=2),
    )


def _doc_corpus_cache_key(path: Path) -> tuple | None:
    if not path.exists():
        return None
    stat = path.stat()
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _load_doc_corpus(path: Path, cache: dict[tuple, list[Document]]) -> tuple[list[Document], str | None]:
    cache_key = _doc_corpus_cache_key(path)
    if cache_key is None:
        return [], None
    if cache_key in cache:
        return cache[cache_key], str(cache_key)

    docs: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                docs.append(deserialize_document(row))
    cache[cache_key] = docs
    return docs, str(cache_key)


def load_chunk_corpus(config: AppConfig) -> tuple[list[Document], str | None]:
    return _load_doc_corpus(chunk_corpus_path(config), _CHUNK_CORPUS_CACHE)


def load_reference_chunk_corpus(
    config: AppConfig,
) -> tuple[list[Document], str | None]:
    return _load_doc_corpus(reference_chunk_corpus_path(config), {})


def load_paper_corpus(config: AppConfig) -> tuple[list[Document], str | None]:
    return _load_doc_corpus(paper_corpus_path(config), _PAPER_CORPUS_CACHE)


def load_section_summary_corpus(config: AppConfig) -> tuple[list[Document], str | None]:
    return _load_doc_corpus(section_summary_corpus_path(config), _SECTION_CORPUS_CACHE)


def load_paper_catalog_rows(config: AppConfig) -> list[dict[str, Any]]:
    return load_rows_jsonl(paper_catalog_path(config))


def load_citation_graph_rows(config: AppConfig) -> list[dict[str, Any]]:
    return load_rows_jsonl(citation_graph_path(config))


def load_parent_corpus_map(config: AppConfig) -> dict[str, Document]:
    path = parent_corpus_path(config)
    cache_key = _doc_corpus_cache_key(path)
    if cache_key is None:
        return {}
    if cache_key in _PARENT_CORPUS_CACHE:
        return _PARENT_CORPUS_CACHE[cache_key]

    docs, _ = _load_doc_corpus(path, {})
    parent_map: dict[str, Document] = {}
    for index, doc in enumerate(docs):
        parent_id = str(doc.metadata.get("parent_id", f"parent_{index}"))
        parent_map[parent_id] = doc
    _PARENT_CORPUS_CACHE[cache_key] = parent_map
    return parent_map


def expand_to_parent_contexts(
    config: AppConfig,
    chunk_docs: list[Document],
    parent_map: dict[str, Document],
) -> list[Document]:
    """把证据 chunk 扩展成更大的生成上下文。

    检索阶段偏好更细粒度的 chunk，而生成阶段通常更适合看到这些
    chunk 周围稍大的 parent 段落。
    """
    if not chunk_docs:
        return []
    if not config.generation_use_parent_context:
        return chunk_docs[: config.final_top_k]

    top_n = max(1, config.generation_parent_top_n)
    max_chars = max(200, config.generation_parent_max_chars)
    expanded: list[Document] = []
    seen_parent = set()

    for chunk in chunk_docs:
        parent_id = str(chunk.metadata.get("parent_id", "")).strip()
        parent_doc = parent_map.get(parent_id) if parent_id else None
        base_doc = parent_doc or chunk
        unique_key = parent_id or str(base_doc.metadata.get("chunk_id", ""))
        if unique_key in seen_parent:
            continue
        seen_parent.add(unique_key)

        parent_text = base_doc.page_content
        if len(parent_text) > max_chars:
            parent_text = f"{parent_text[:max_chars]}\n...[truncated]"

        metadata = dict(base_doc.metadata or {})
        for key in (
            "retrieval_score",
            "retrieval_raw_score",
            "retrieval_rrf_score",
            "reranker_score",
        ):
            if key in chunk.metadata and key not in metadata:
                metadata[key] = chunk.metadata[key]
        expanded.append(Document(page_content=parent_text, metadata=metadata))
        if len(expanded) >= top_n:
            break

    return expanded if expanded else chunk_docs[: config.final_top_k]


def load_local_cache_state(config: AppConfig) -> LocalCacheState:
    """从 `data/cache` 加载完整的本地知识库状态。"""
    main_chunks, _ = _load_doc_corpus(chunk_corpus_path(config), {})
    parent_docs, _ = _load_doc_corpus(parent_corpus_path(config), {})
    paper_docs, _ = _load_doc_corpus(paper_corpus_path(config), {})
    section_docs, _ = _load_doc_corpus(section_summary_corpus_path(config), {})
    reference_chunks, _ = _load_doc_corpus(reference_chunk_corpus_path(config), {})
    return LocalCacheState(
        main_chunks=main_chunks,
        parent_docs=parent_docs,
        paper_docs=paper_docs,
        section_docs=section_docs,
        reference_chunks=reference_chunks,
        block_rows=load_rows_jsonl(block_structured_path(config)),
        chunk_rows=load_rows_jsonl(chunk_structured_path(config)),
        paper_rows=load_rows_jsonl(paper_catalog_path(config)),
        citation_rows=load_rows_jsonl(citation_graph_path(config)),
        reference_keyword_rows=load_rows_jsonl(reference_keyword_index_path(config)),
    )


def save_local_cache_state(config: AppConfig, state: LocalCacheState) -> None:
    """持久化完整本地知识库状态，并重建 SQLite catalog。"""
    _save_doc_corpus(chunk_corpus_path(config), state.main_chunks)
    _save_doc_corpus(parent_corpus_path(config), state.parent_docs)
    _save_doc_corpus(paper_corpus_path(config), state.paper_docs)
    _save_doc_corpus(section_summary_corpus_path(config), state.section_docs)
    _save_doc_corpus(reference_chunk_corpus_path(config), state.reference_chunks)
    save_rows_jsonl(block_structured_path(config), state.block_rows)
    save_rows_jsonl(chunk_structured_path(config), state.chunk_rows)
    save_rows_jsonl(paper_catalog_path(config), state.paper_rows)
    save_rows_jsonl(citation_graph_path(config), state.citation_rows)
    save_rows_jsonl(reference_keyword_index_path(config), state.reference_keyword_rows)
    from services.paper_catalog_store import rebuild_catalog_db

    rebuild_catalog_db(
        paper_catalog_db_path(config),
        paper_rows=state.paper_rows,
        citation_rows=state.citation_rows,
        section_docs=state.section_docs,
    )
    clear_local_cache_caches()


def local_cache_state_to_payload(state: LocalCacheState) -> dict[str, Any]:
    return {
        "main_chunks": serialize_documents(state.main_chunks),
        "parent_docs": serialize_documents(state.parent_docs),
        "paper_docs": serialize_documents(state.paper_docs),
        "section_docs": serialize_documents(state.section_docs),
        "reference_chunks": serialize_documents(state.reference_chunks),
        "block_rows": state.block_rows,
        "chunk_rows": state.chunk_rows,
        "paper_rows": state.paper_rows,
        "citation_rows": state.citation_rows,
        "reference_keyword_rows": state.reference_keyword_rows,
    }


def local_cache_state_from_payload(payload: dict[str, Any]) -> LocalCacheState:
    return LocalCacheState(
        main_chunks=deserialize_documents(payload.get("main_chunks", [])),
        parent_docs=deserialize_documents(payload.get("parent_docs", [])),
        paper_docs=deserialize_documents(payload.get("paper_docs", [])),
        section_docs=deserialize_documents(payload.get("section_docs", [])),
        reference_chunks=deserialize_documents(payload.get("reference_chunks", [])),
        block_rows=list(payload.get("block_rows", [])),
        chunk_rows=list(payload.get("chunk_rows", [])),
        paper_rows=list(payload.get("paper_rows", [])),
        citation_rows=list(payload.get("citation_rows", [])),
        reference_keyword_rows=list(payload.get("reference_keyword_rows", [])),
    )


def build_ingest_cache_payload(
    config: AppConfig,
    signature: dict[str, Any],
    raw_documents: int,
    chunks: int,
    *,
    main_chunk_count: int = 0,
    paper_doc_count: int = 0,
    reference_chunk_count: int = 0,
) -> dict[str, Any]:
    return {
        "signature": signature,
        "raw_documents": raw_documents,
        "chunks": chunks,
        "main_chunk_count": main_chunk_count,
        "paper_doc_count": paper_doc_count,
        "reference_chunk_count": reference_chunk_count,
        "cache_dir": str(config.local_cache_dir),
    }


def load_ingest_cache(config: AppConfig) -> dict[str, Any] | None:
    path = ingest_cache_path(config)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_ingest_cache_payload(config: AppConfig, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        ingest_cache_path(config),
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def invalidate_ingest_cache(config: AppConfig) -> None:
    path = ingest_cache_path(config)
    if path.exists():
        path.unlink()


def load_sync_journal(config: AppConfig) -> dict[str, Any] | None:
    path = sync_journal_path(config)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_sync_journal(config: AppConfig, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        sync_journal_path(config),
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
    )


def clear_sync_journal(config: AppConfig) -> None:
    path = sync_journal_path(config)
    if path.exists():
        path.unlink()
