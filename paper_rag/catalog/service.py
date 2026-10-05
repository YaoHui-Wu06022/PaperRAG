"""ArXiv 本地 Catalog 与 SQLite FTS5 派生索引。"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any
from uuid import uuid4

from paper_rag.config import Settings
from paper_rag.catalog.chunks import CHUNK_RULE_VERSION, Chunk, build_chunks, load_content_list
from paper_rag.catalog.references import extract_references, normalize_arxiv_id


class CatalogIndexNotReady(RuntimeError):
    """SQLite Catalog 尚未通过显式同步建立。"""


@dataclass(frozen=True)
class CatalogRecord:
    """一篇论文在本地 Catalog 中的只读记录。"""

    paper_id: str
    base_id: str
    canonical_id: str
    title: str
    authors: tuple[str, ...] = ()
    abstract: str = ""
    categories: tuple[str, ...] = ()
    published_at: str | None = None
    updated_at: str | None = None
    abs_url: str | None = None
    pdf_url: str | None = None
    metadata_path: Path | None = None
    pdf_path: Path | None = None
    mineru_dir: Path | None = None
    state: str = "missing"
    assets: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为 MCP 和 CLI 可安全序列化的结构。"""

        return {
            "paper_id": self.paper_id,
            "base_id": self.base_id,
            "canonical_id": self.canonical_id,
            "title": self.title,
            "authors": list(self.authors),
            "abstract": self.abstract,
            "categories": list(self.categories),
            "published_at": self.published_at,
            "updated_at": self.updated_at,
            "abs_url": self.abs_url,
            "pdf_url": self.pdf_url,
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "pdf_path": str(self.pdf_path) if self.pdf_path else None,
            "mineru_dir": str(self.mineru_dir) if self.mineru_dir else None,
            "state": self.state,
            "assets": self.assets,
        }


def _mineru_chunks(record: CatalogRecord, settings: Settings) -> tuple[list[Chunk], str | None]:
    """读取当前版本的 MinerU 结构结果；过期结果不参与索引。"""
    if not record.mineru_dir or not record.mineru_dir.is_dir():
        return [], None
    manifest_path = record.mineru_dir / "manifest.json"
    content_path = record.mineru_dir / "content_list.json"
    full_path = record.mineru_dir / "full.md"
    if not manifest_path.is_file() or not content_path.is_file() or not full_path.is_file():
        return [], "mineru output is incomplete"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_hash = _sha256(record.pdf_path) if record.pdf_path and record.pdf_path.is_file() else None
        if not isinstance(manifest, dict) or manifest.get("canonical_id") != record.canonical_id or (source_hash and manifest.get("source_sha256") != source_hash) or manifest.get("model_version") != settings.mineru_model_version or manifest.get("language") != settings.mineru_language:
            return [], "mineru output is stale"
        content = load_content_list(content_path)
        chunks, warnings = build_chunks(content, paper_id=record.paper_id, canonical_id=record.canonical_id, content_hash=_sha256(content_path))
        return chunks, "; ".join(warnings) if warnings else None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [], f"mineru output invalid: {exc}"


def scan_catalog(settings: Settings) -> list[CatalogRecord]:
    """扫描 ArXiv 目录，不读取旧的 ``data/mineru_output``。"""

    root = settings.arxiv_data_dir
    if not root.is_dir():
        return []
    records: list[CatalogRecord] = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(metadata, dict):
            records.append(_record_from_metadata(metadata_path, metadata))
    return records


def list_papers(settings: Settings, filters: dict[str, Any] | None = None) -> list[CatalogRecord]:
    """按结构化条件列出论文。"""

    filters = filters or {}
    records = scan_catalog(settings)
    return [record for record in records if _matches_filters(record, filters)]


def search_catalog(
    settings: Settings,
    query: str,
    filters: dict[str, Any] | None = None,
    limit: int = 50,
) -> list[CatalogRecord]:
    """使用已经同步的 SQLite FTS5 索引检索元数据。"""

    _require_index(settings)
    filters = filters or {}
    fts_query = _build_fts_query(str(query or "").strip())
    clauses = ["1=1"]
    parameters: list[Any] = []
    if fts_query:
        clauses.append("papers_fts MATCH ?")
        parameters.append(fts_query)
    _append_filter_sql(clauses, parameters, filters)
    sql = f"""
        SELECT p.paper_id, p.base_id, p.canonical_id, p.title, p.authors,
               p.abstract, p.categories, p.published_at, p.updated_at,
               p.state, p.metadata_path, p.pdf_path, p.mineru_dir,
               p.abs_url, p.pdf_url
        FROM papers AS p
        {"JOIN papers_fts ON papers_fts.paper_id = p.paper_id" if fts_query else ""}
        WHERE {' AND '.join(clauses)}
        ORDER BY {"bm25(papers_fts)," if fts_query else ""} lower(p.title), p.paper_id
        LIMIT ?
    """
    parameters.append(max(0, int(limit)))
    with sqlite3.connect(settings.paper_catalog_db_path) as connection:
        rows = connection.execute(sql, parameters).fetchall()
    return [_record_from_row(row) for row in rows]


def get_metadata(settings: Settings, paper_id: str) -> CatalogRecord | None:
    """按 base ID 或 canonical ID 读取论文元数据。"""

    wanted = str(paper_id).casefold()
    return next(
        (
            record
            for record in scan_catalog(settings)
            if record.base_id.casefold() == wanted or record.canonical_id.casefold() == wanted
        ),
        None,
    )


def get_assets(settings: Settings, paper_id: str) -> dict[str, Any]:
    """返回论文文件资产清单，不包含异步任务状态。"""

    record = get_metadata(settings, paper_id)
    if record is None:
        return {"paper_id": str(paper_id), "found": False, "assets": {}}
    return {"paper_id": record.paper_id, "found": True, "assets": record.assets}


def get_asset_status(settings: Settings, paper_id: str) -> dict[str, Any]:
    """返回论文持久化资产状态，与 JobManager 状态分离。"""

    record = get_metadata(settings, paper_id)
    if record is None:
        return {"paper_id": str(paper_id), "found": False, "state": "missing"}
    return {
        "paper_id": record.paper_id,
        "found": True,
        "state": record.state,
        "pdf": "present" if record.assets.get("pdf", {}).get("present") else "missing",
        "metadata": "present" if record.assets.get("metadata", {}).get("present") else "missing",
        "mineru": "present" if record.assets.get("mineru", {}).get("present") else "missing",
    }


def rebuild_catalog(settings: Settings) -> dict[str, Any]:
    """在临时数据库中重建索引并原子替换正式数据库。"""

    records = scan_catalog(settings)
    db_path = settings.paper_catalog_db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = db_path.with_name(f".{db_path.name}.{uuid4().hex}.tmp")
    chunk_json_temps: list[tuple[Path, Path]] = []
    issues: list[dict[str, str]] = []
    try:
        issues = _create_database(temporary_path, records, settings)
        chunk_json_temps = _prepare_chunk_json_exports(records, settings)
        os.replace(temporary_path, db_path)
        for staged, target in chunk_json_temps:
            os.replace(staged, target)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        for staged, _ in chunk_json_temps:
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    chunk_count = _count_chunks(db_path)
    return {
        "database": str(db_path),
        "papers": len(records),
        "status": "rebuilt",
        "indexed_at": _read_indexed_at(db_path),
        "chunks": chunk_count,
        "issues": issues,
    }


def _prepare_chunk_json_exports(records: list[CatalogRecord], settings: Settings) -> list[tuple[Path, Path]]:
    """为每篇有效论文准备可浏览的 Chunk JSON 临时文件。"""

    staged_files: list[tuple[Path, Path]] = []
    try:
        for record in records:
            if not record.mineru_dir or not record.mineru_dir.is_dir():
                continue
            content_path = record.mineru_dir / "content_list.json"
            if not content_path.is_file():
                continue
            chunks, _ = _mineru_chunks(record, settings)
            target = record.mineru_dir / "chunks.json"
            staged = record.mineru_dir / f".chunks-{uuid4().hex}.tmp"
            payload = {
                "schema_version": 1,
                "paper_id": record.paper_id,
                "canonical_id": record.canonical_id,
                "content_hash": _sha256(content_path),
                "chunk_rule_version": CHUNK_RULE_VERSION,
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "chunk_count": len(chunks),
                "chunks": [chunk.to_dict() for chunk in chunks],
            }
            staged.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            staged_files.append((staged, target))
    except Exception:
        for staged, _ in staged_files:
            staged.unlink(missing_ok=True)
        raise
    return staged_files


def catalog_status(settings: Settings) -> dict[str, Any]:
    """返回源目录和派生索引状态，不查询异步任务。"""

    source_records = scan_catalog(settings)
    result: dict[str, Any] = {
        "database": str(settings.paper_catalog_db_path),
        "database_exists": settings.paper_catalog_db_path.is_file(),
        "index_ready": False,
        "source_papers": len(source_records),
        "indexed_papers": 0,
        "indexed_at": None,
        "states": _state_counts(source_records),
    }
    if not settings.paper_catalog_db_path.is_file():
        return result
    try:
        with sqlite3.connect(settings.paper_catalog_db_path) as connection:
            paper_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(papers)").fetchall()
            }
            required_columns = {"paper_id", "base_id", "canonical_id", "abs_url", "pdf_url"}
            if not required_columns <= paper_columns:
                raise sqlite3.DatabaseError("catalog schema is outdated")
            indexed_papers = connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            connection.execute("SELECT 1 FROM papers_fts LIMIT 1")
            indexed_at = connection.execute(
                "SELECT value FROM catalog_meta WHERE key = 'indexed_at'"
            ).fetchone()
            indexed_chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] if _table_exists(connection, "chunks") else 0
        result.update(
            {
                "index_ready": True,
                "indexed_papers": int(indexed_papers),
                "indexed_chunks": int(indexed_chunks),
                "indexed_at": indexed_at[0] if indexed_at else None,
            }
        )
    except sqlite3.Error:
        result["index_error"] = "catalog database is invalid"
    return result


def _create_database(path: Path, records: list[CatalogRecord], settings: Settings) -> list[dict[str, str]]:
    """创建完整的临时 Catalog 数据库并执行完整性校验。"""

    indexed_at = dt.datetime.now(dt.timezone.utc).isoformat()
    issues: list[dict[str, str]] = []
    connection = sqlite3.connect(path)
    try:
        with connection:
            connection.executescript(
                """
                PRAGMA journal_mode = DELETE;
                CREATE TABLE papers (
                    paper_id TEXT PRIMARY KEY,
                    base_id TEXT NOT NULL,
                    canonical_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    authors TEXT NOT NULL,
                    abstract TEXT NOT NULL,
                    categories TEXT NOT NULL,
                    published_at TEXT,
                    updated_at TEXT,
                    state TEXT NOT NULL,
                    metadata_path TEXT,
                    pdf_path TEXT,
                    mineru_dir TEXT,
                    abs_url TEXT,
                    pdf_url TEXT
                );
                CREATE VIRTUAL TABLE papers_fts USING fts5(
                    paper_id UNINDEXED,
                    base_id,
                    canonical_id,
                    title,
                    abstract,
                    authors,
                    categories
                );
                CREATE TABLE chunks (
                    chunk_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    canonical_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    region TEXT NOT NULL,
                    chapter_number TEXT,
                    chapter_title TEXT,
                    section_path TEXT NOT NULL,
                    section_label TEXT,
                    type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    retrieval_text TEXT NOT NULL,
                    page_start INTEGER,
                    page_end INTEGER,
                    source_blocks TEXT NOT NULL,
                    asset_refs TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
                );
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    paper_id UNINDEXED,
                    canonical_id UNINDEXED,
                    section_path,
                    region,
                    chapter_title,
                    type UNINDEXED,
                    retrieval_text,
                    content='chunks', content_rowid='rowid'
                );
                CREATE TABLE "references" (
                    reference_id TEXT PRIMARY KEY,
                    source_paper_id TEXT NOT NULL,
                    source_canonical_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    raw_text TEXT NOT NULL,
                    page_start INTEGER,
                    page_end INTEGER,
                    target_arxiv_id TEXT,
                    target_doi TEXT,
                    resolution TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE references_fts USING fts5(reference_id UNINDEXED, source_paper_id UNINDEXED, raw_text);
                CREATE TABLE citation_edges (
                    source_paper_id TEXT NOT NULL,
                    target_arxiv_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    resolution TEXT NOT NULL,
                    PRIMARY KEY (source_paper_id, target_arxiv_id, relation)
                );
                CREATE TABLE catalog_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE embedding_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE embedding_items (
                    chunk_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    synced_at TEXT NOT NULL
                );
                """
            )
            for record in records:
                connection.execute(
                    """
                    INSERT INTO papers (
                        paper_id, base_id, canonical_id, title, authors, abstract,
                        categories, published_at, updated_at, state,
                        metadata_path, pdf_path, mineru_dir, abs_url, pdf_url
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    _record_values(record),
                )
                chunks, warning = _mineru_chunks(record, settings)
                if warning:
                    issues.append({"paper_id": record.paper_id, "error": warning})
                for chunk in chunks:
                    connection.execute(
                        "INSERT INTO chunks (chunk_id, paper_id, canonical_id, ordinal, region, chapter_number, chapter_title, section_path, section_label, type, text, retrieval_text, page_start, page_end, source_blocks, asset_refs, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (chunk.chunk_id, chunk.paper_id, chunk.canonical_id, chunk.ordinal, chunk.region, chunk.chapter_number, chunk.chapter_title, json.dumps(chunk.section_path, ensure_ascii=False), chunk.section_label, chunk.type, chunk.text, chunk.retrieval_text, chunk.page_start, chunk.page_end, json.dumps(chunk.source_blocks, ensure_ascii=False), json.dumps(chunk.asset_refs, ensure_ascii=False), chunk.content_hash),
                    )
                    connection.execute(
                        "INSERT INTO chunks_fts (rowid, chunk_id, paper_id, canonical_id, section_path, region, chapter_title, type, retrieval_text) SELECT rowid, chunk_id, paper_id, canonical_id, section_path, region, chapter_title, type, retrieval_text FROM chunks WHERE chunk_id = ?",
                        (chunk.chunk_id,),
                    )
                try:
                    content = load_content_list(record.mineru_dir / "content_list.json") if record.mineru_dir else []
                    refs, ref_warnings = extract_references(content, source_paper_id=record.paper_id, source_canonical_id=record.canonical_id, local_ids={r.base_id.casefold() for r in records})
                    for warning in ref_warnings:
                        issues.append({"paper_id": record.paper_id, "error": warning})
                    for ref in refs:
                        connection.execute("INSERT INTO \"references\" VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (ref.reference_id, ref.source_paper_id, ref.source_canonical_id, ref.ordinal, ref.raw_text, ref.page_start, ref.page_end, ref.target_arxiv_id, ref.target_doi, ref.resolution))
                        connection.execute("INSERT INTO references_fts (reference_id, source_paper_id, raw_text) VALUES (?, ?, ?)", (ref.reference_id, ref.source_paper_id, ref.raw_text))
                        if ref.target_arxiv_id:
                            connection.execute("INSERT OR IGNORE INTO citation_edges VALUES (?, ?, 'cites', ?)", (record.paper_id, normalize_arxiv_id(ref.target_arxiv_id), ref.resolution))
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    if record.mineru_dir and (record.mineru_dir / "content_list.json").is_file():
                        issues.append({"paper_id": record.paper_id, "error": f"reference parse failed: {exc}"})
                connection.execute(
                    """
                    INSERT INTO papers_fts (
                        paper_id, base_id, canonical_id, title, abstract, authors, categories
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.paper_id,
                        record.base_id,
                        record.canonical_id,
                        record.title,
                        record.abstract,
                        " ".join(record.authors),
                        " ".join(record.categories),
                    ),
                )
            connection.execute(
                "INSERT INTO catalog_meta (key, value) VALUES ('indexed_at', ?)",
                (indexed_at,),
            )
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise sqlite3.DatabaseError(f"catalog integrity check failed: {integrity}")
    finally:
        connection.close()
    return issues


def _require_index(settings: Settings) -> None:
    if not settings.paper_catalog_db_path.is_file():
        raise CatalogIndexNotReady("catalog index is not ready; run paper_catalog_sync first")
    try:
        with sqlite3.connect(settings.paper_catalog_db_path) as connection:
            paper_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(papers)").fetchall()
            }
            required_columns = {"paper_id", "base_id", "canonical_id", "abs_url", "pdf_url"}
            if not required_columns <= paper_columns:
                raise sqlite3.DatabaseError("catalog schema is outdated")
            connection.execute("SELECT 1 FROM papers_fts LIMIT 1")
            connection.execute("SELECT value FROM catalog_meta WHERE key = 'indexed_at'")
    except sqlite3.Error as exc:
        raise CatalogIndexNotReady("catalog index is not ready; run paper_catalog_sync first") from exc


def search_chunks(
    settings: Settings,
    query: str,
    paper_ids: list[str] | None = None,
    limit: int = 8,
    regions: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """使用 Chunk FTS5 返回正文证据。"""
    _require_index(settings)
    _require_chunk_index(settings)
    terms = _build_fts_query(str(query or "").strip())
    if not terms:
        return []
    clauses = ["chunks_fts MATCH ?"]
    params: list[Any] = [terms]
    if paper_ids:
        placeholders = ",".join("?" for _ in paper_ids)
        clauses.append(f"c.paper_id IN ({placeholders})")
        params.extend(paper_ids)
    selected_regions = [str(value) for value in (regions or ()) if str(value)]
    if selected_regions:
        placeholders = ",".join("?" for _ in selected_regions)
        clauses.append(f"c.region IN ({placeholders})")
        params.extend(selected_regions)
    params.append(max(1, min(int(limit), 50)))
    sql = f"""SELECT c.chunk_id, c.paper_id, c.canonical_id, c.ordinal, c.region,
                     c.chapter_number, c.chapter_title, c.section_path, c.section_label,
                     c.type, c.text, c.retrieval_text, c.page_start, c.page_end,
                     c.source_blocks, c.asset_refs, c.content_hash,
                     snippet(chunks_fts, 7, '[', ']', '…', 32), bm25(chunks_fts)
                FROM chunks AS c JOIN chunks_fts ON chunks_fts.rowid = c.rowid
               WHERE {' AND '.join(clauses)} ORDER BY bm25(chunks_fts) LIMIT ?"""
    with sqlite3.connect(settings.paper_catalog_db_path) as connection:
        rows = connection.execute(sql, params).fetchall()
    return [_chunk_row(row) for row in rows]


def get_chunk(settings: Settings, chunk_id: str) -> dict[str, Any] | None:
    _require_index(settings)
    _require_chunk_index(settings)
    with sqlite3.connect(settings.paper_catalog_db_path) as connection:
        row = connection.execute("SELECT chunk_id, paper_id, canonical_id, ordinal, region, chapter_number, chapter_title, section_path, section_label, type, text, retrieval_text, page_start, page_end, source_blocks, asset_refs, content_hash FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
    return _chunk_row(row) if row else None


def list_chunks(settings: Settings, paper_id: str, limit: int = 50) -> list[dict[str, Any]]:
    _require_index(settings)
    _require_chunk_index(settings)
    with sqlite3.connect(settings.paper_catalog_db_path) as connection:
        rows = connection.execute("SELECT chunk_id, paper_id, canonical_id, ordinal, region, chapter_number, chapter_title, section_path, section_label, type, text, retrieval_text, page_start, page_end, source_blocks, asset_refs, content_hash FROM chunks WHERE paper_id = ? ORDER BY ordinal LIMIT ?", (paper_id, max(1, int(limit)))).fetchall()
    return [_chunk_row(row) for row in rows]


def _chunk_row(row: tuple[Any, ...]) -> dict[str, Any]:
    result = {"chunk_id": row[0], "paper_id": row[1], "canonical_id": row[2], "ordinal": row[3], "region": row[4], "chapter_number": row[5], "chapter_title": row[6], "section_path": json.loads(row[7] or "[]"), "section_label": row[8], "type": row[9], "text": row[10], "retrieval_text": row[11], "page_start": row[12], "page_end": row[13], "page_start_display": row[12] + 1 if isinstance(row[12], int) else None, "page_end_display": row[13] + 1 if isinstance(row[13], int) else None, "source_blocks": json.loads(row[14] or "[]"), "asset_refs": json.loads(row[15] or "[]"), "content_hash": row[16]}
    if len(row) > 17:
        result.update({"excerpt": row[17], "score": row[18]})
    return result


def get_references(settings: Settings, paper_id: str) -> dict[str, Any]:
    _require_index(settings)
    source_id = _base_arxiv_id(paper_id)
    with sqlite3.connect(settings.paper_catalog_db_path) as connection:
        rows = connection.execute('SELECT reference_id, source_paper_id, source_canonical_id, ordinal, raw_text, page_start, page_end, target_arxiv_id, target_doi, resolution FROM "references" WHERE source_paper_id = ? ORDER BY ordinal', (source_id,)).fetchall()
    keys = ("reference_id", "source_paper_id", "source_canonical_id", "ordinal", "raw_text", "page_start", "page_end", "target_arxiv_id", "target_doi", "resolution")
    return {"paper_id": paper_id, "items": [dict(zip(keys, row)) for row in rows], "scope": "local_catalog"}


def get_citations(settings: Settings, paper_id: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    _require_index(settings)
    base = _base_arxiv_id(paper_id)
    with sqlite3.connect(settings.paper_catalog_db_path) as connection:
        rows = connection.execute("SELECT source_paper_id, target_arxiv_id, relation, resolution FROM citation_edges WHERE target_arxiv_id = ? ORDER BY source_paper_id", (base,)).fetchall()
        if filters:
            rows = [row for row in rows if _paper_row_matches(connection, row[0], filters)]
    return {"paper_id": paper_id, "items": [{"source_paper_id": r[0], "target_arxiv_id": r[1], "relation": r[2], "resolution": r[3]} for r in rows], "scope": "local_catalog"}


def citation_graph(settings: Settings, paper_id: str, direction: str = "both", depth: int = 1, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    if direction not in {"in", "out", "both"}:
        raise ValueError("direction must be in, out or both")
    max_depth = max(1, min(int(depth), 3))
    root = _base_arxiv_id(paper_id)
    nodes = {root}
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    frontier = {root}
    with sqlite3.connect(settings.paper_catalog_db_path) as connection:
        for level in range(max_depth):
            if not frontier:
                break
            next_frontier: set[str] = set()
            for current in frontier:
                rows: list[tuple[Any, ...]] = []
                if direction in {"out", "both"}:
                    rows.extend(connection.execute("SELECT source_paper_id, target_arxiv_id, relation, resolution FROM citation_edges WHERE source_paper_id = ?", (current,)).fetchall())
                if direction in {"in", "both"}:
                    rows.extend(connection.execute("SELECT source_paper_id, target_arxiv_id, relation, resolution FROM citation_edges WHERE target_arxiv_id = ?", (current,)).fetchall())
                for source, target, relation, resolution in rows:
                    if filters and not (_paper_row_matches(connection, source, filters) or _paper_row_matches(connection, target, filters)):
                        continue
                    edge_key = (str(source), str(target), str(relation))
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
                    edges.append({"source_paper_id": str(source), "target_arxiv_id": str(target), "relation": str(relation), "resolution": str(resolution), "depth": level + 1})
                    neighbor = str(target) if str(source) == current else str(source)
                    if neighbor not in nodes:
                        nodes.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
    return {"paper_id": paper_id, "direction": direction, "depth": max_depth, "nodes": sorted(nodes), "edges": edges, "scope": "local_catalog"}


def _base_arxiv_id(value: str) -> str:
    """兼容旧内部调用的统一 ArXiv ID 规范化入口。"""

    return normalize_arxiv_id(value)


def _paper_row_matches(connection: sqlite3.Connection, paper_id: str, filters: dict[str, Any]) -> bool:
    row = connection.execute("SELECT authors, categories, published_at, state FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    if not row:
        return False
    authors, categories, published_at, state = row
    if filters.get("author") and str(filters["author"]).casefold() not in str(authors).casefold():
        return False
    if filters.get("category") and str(filters["category"]).casefold() not in str(categories).casefold():
        return False
    if filters.get("year") and not str(published_at or "").startswith(str(filters["year"])):
        return False
    if filters.get("year_from") and str(published_at or "")[:4] < str(filters["year_from"]):
        return False
    if filters.get("year_to") and str(published_at or "")[:4] > str(filters["year_to"]):
        return False
    return not filters.get("state") or str(state) == str(filters["state"])


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _require_chunk_index(settings: Settings) -> None:
    try:
        with sqlite3.connect(settings.paper_catalog_db_path) as connection:
            if not _table_exists(connection, "chunks") or not _table_exists(connection, "chunks_fts"):
                raise sqlite3.DatabaseError("chunk schema is not ready")
            connection.execute("SELECT 1 FROM chunks_fts LIMIT 1")
    except sqlite3.Error as exc:
        raise CatalogIndexNotReady("chunk index is not ready; run paper_catalog_sync first") from exc


def _count_chunks(path: Path) -> int:
    with sqlite3.connect(path) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])


def _append_filter_sql(clauses: list[str], parameters: list[Any], filters: dict[str, Any]) -> None:
    if filters.get("author"):
        clauses.append("lower(p.authors) LIKE ?")
        parameters.append(f"%{str(filters['author']).casefold()}%")
    if filters.get("category"):
        clauses.append("lower(p.categories) LIKE ?")
        parameters.append(f"%{str(filters['category']).casefold()}%")
    if filters.get("state"):
        clauses.append("p.state = ?")
        parameters.append(str(filters["state"]))
    if filters.get("year"):
        clauses.append("p.published_at LIKE ?")
        parameters.append(f"{str(filters['year'])}%")
    if filters.get("year_from"):
        clauses.append("substr(p.published_at, 1, 4) >= ?")
        parameters.append(str(filters["year_from"]))
    if filters.get("year_to"):
        clauses.append("substr(p.published_at, 1, 4) <= ?")
        parameters.append(str(filters["year_to"]))


def _build_fts_query(value: str) -> str:
    terms = _terms(value)
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


def _terms(value: str) -> list[str]:
    compact = re.sub(r"\s+", "", value.casefold())
    terms = re.findall(r"[a-z0-9][a-z0-9._-]*|[\u4e00-\u9fff]+", value.casefold())
    if compact and compact not in terms:
        terms.append(compact)
    return [term for term in terms if len(term) > 1]


def _matches_filters(record: CatalogRecord, filters: dict[str, Any]) -> bool:
    if filters.get("author") and not _contains(record.authors, str(filters["author"])):
        return False
    if filters.get("category") and not _contains(record.categories, str(filters["category"])):
        return False
    if filters.get("state") and record.state != str(filters["state"]):
        return False
    if filters.get("year") and not str(record.published_at or "").startswith(str(filters["year"])):
        return False
    if filters.get("year_from") and str(record.published_at or "")[:4] < str(filters["year_from"]):
        return False
    if filters.get("year_to") and str(record.published_at or "")[:4] > str(filters["year_to"]):
        return False
    return True


def _record_values(record: CatalogRecord) -> tuple[Any, ...]:
    return (
        record.paper_id,
        record.base_id,
        record.canonical_id,
        record.title,
        json.dumps(record.authors, ensure_ascii=False),
        record.abstract,
        json.dumps(record.categories, ensure_ascii=False),
        record.published_at,
        record.updated_at,
        record.state,
        str(record.metadata_path) if record.metadata_path else None,
        str(record.pdf_path) if record.pdf_path else None,
        str(record.mineru_dir) if record.mineru_dir else None,
        record.abs_url,
        record.pdf_url,
    )


def _record_from_row(row: tuple[Any, ...]) -> CatalogRecord:
    (
        paper_id,
        base_id,
        canonical_id,
        title,
        authors,
        abstract,
        categories,
        published_at,
        updated_at,
        state,
        metadata_path,
        pdf_path,
        mineru_dir,
        abs_url,
        pdf_url,
    ) = row
    return CatalogRecord(
        paper_id=str(paper_id),
        base_id=str(base_id),
        canonical_id=str(canonical_id),
        title=str(title),
        authors=tuple(json.loads(authors or "[]")),
        abstract=str(abstract or ""),
        categories=tuple(json.loads(categories or "[]")),
        published_at=published_at,
        updated_at=updated_at,
        state=str(state),
        metadata_path=Path(metadata_path) if metadata_path else None,
        pdf_path=Path(pdf_path) if pdf_path else None,
        mineru_dir=Path(mineru_dir) if mineru_dir else None,
        abs_url=abs_url,
        pdf_url=pdf_url,
        assets=_assets(Path(metadata_path).parent) if metadata_path else {},
    )


def _record_from_metadata(metadata_path: Path, metadata: dict[str, Any]) -> CatalogRecord:
    source_dir = metadata_path.parent
    pdf_path = source_dir / "paper.pdf"
    mineru_dir = source_dir / "mineru"
    mineru_present = mineru_dir.is_dir() and (mineru_dir / "full.md").is_file()
    state = "missing"
    if pdf_path.is_file():
        state = "ingested" if mineru_present else "ready_for_ingest"
    base_id = str(metadata.get("base_id") or source_dir.name)
    canonical_id = str(metadata.get("canonical_id") or base_id)
    return CatalogRecord(
        paper_id=base_id,
        base_id=base_id,
        canonical_id=canonical_id,
        title=str(metadata.get("title") or base_id),
        authors=tuple(str(item) for item in metadata.get("authors", []) if item),
        abstract=str(metadata.get("abstract") or ""),
        categories=tuple(str(item) for item in metadata.get("categories", []) if item),
        published_at=_optional_string(metadata.get("published_at")),
        updated_at=_optional_string(metadata.get("updated_at")),
        abs_url=_optional_string(metadata.get("abs_url")),
        pdf_url=_optional_string(metadata.get("pdf_url")),
        metadata_path=metadata_path,
        pdf_path=pdf_path,
        mineru_dir=mineru_dir,
        state=state,
        assets=_assets(source_dir),
    )


def _assets(source_dir: Path) -> dict[str, Any]:
    metadata_path = source_dir / "metadata.json"
    pdf_path = source_dir / "paper.pdf"
    mineru_dir = source_dir / "mineru"
    mineru_present = mineru_dir.is_dir() and (mineru_dir / "full.md").is_file()
    return {
        "pdf": {"present": pdf_path.is_file(), "path": str(pdf_path)},
        "metadata": {"present": metadata_path.is_file(), "path": str(metadata_path)},
        "mineru": {"present": mineru_present, "path": str(mineru_dir)},
    }


def _state_counts(records: list[CatalogRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.state] = counts.get(record.state, 0) + 1
    return counts


def _read_indexed_at(path: Path) -> str | None:
    with sqlite3.connect(path) as connection:
        row = connection.execute("SELECT value FROM catalog_meta WHERE key = 'indexed_at'").fetchone()
    return row[0] if row else None


def _contains(values: tuple[str, ...], query: str) -> bool:
    wanted = query.casefold()
    return any(wanted in value.casefold() for value in values)


def _optional_string(value: Any) -> str | None:
    return str(value) if value else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "CatalogIndexNotReady",
    "CatalogRecord",
    "catalog_status",
    "get_asset_status",
    "get_assets",
    "get_metadata",
    "list_papers",
    "rebuild_catalog",
    "scan_catalog",
    "search_catalog",
    "search_chunks",
    "get_chunk",
    "list_chunks",
    "get_references",
    "get_citations",
    "citation_graph",
]
