from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

from langchain_core.documents import Document


# 这个模块会把论文 catalog 落到 SQLite，
# 让元数据类查询可以走结构化存储，而不是扫描原始 chunk。
def rebuild_catalog_db(
    db_path: Path,
    *,
    paper_rows: list[dict[str, Any]],
    citation_rows: list[dict[str, Any]],
    section_docs: list[Document],
) -> None:
    """根据最新的 JSONL 状态重建 SQLite catalog。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = db_path.with_suffix(f"{db_path.suffix}.tmp")
    if temp_path.exists():
        temp_path.unlink()

    conn = sqlite3.connect(temp_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        _create_schema(conn)
        _insert_papers(conn, paper_rows)
        _insert_citations(conn, citation_rows)
        _insert_sections(conn, section_docs)
        conn.commit()
    finally:
        conn.close()

    temp_path.replace(db_path)


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS papers;
        DROP TABLE IF EXISTS citations;
        DROP TABLE IF EXISTS sections;

        CREATE TABLE papers (
            doc_id TEXT PRIMARY KEY,
            source TEXT,
            source_path TEXT,
            title TEXT,
            abstract TEXT,
            year TEXT,
            venue TEXT,
            authors_text TEXT,
            normalized_authors_json TEXT,
            keywords_text TEXT,
            keywords_json TEXT,
            language TEXT,
            total_pages INTEGER,
            paper_summary TEXT,
            section_names_json TEXT,
            citation_count INTEGER,
            citation_titles_json TEXT
        );

        CREATE TABLE citations (
            edge_id TEXT PRIMARY KEY,
            source_doc_id TEXT,
            source_title TEXT,
            cited_title TEXT,
            cited_year TEXT,
            cited_authors TEXT,
            citation_text TEXT
        );

        CREATE TABLE sections (
            doc_id TEXT,
            section_name TEXT,
            page_range TEXT,
            summary TEXT
        );

        CREATE INDEX idx_papers_title ON papers(title);
        CREATE INDEX idx_papers_year ON papers(year);
        CREATE INDEX idx_papers_venue ON papers(venue);
        CREATE INDEX idx_citations_source_doc_id ON citations(source_doc_id);
        CREATE INDEX idx_citations_cited_title ON citations(cited_title);
        CREATE INDEX idx_sections_doc_id ON sections(doc_id);
        """
    )


def _insert_papers(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    payload = [
        (
            str(row.get("doc_id", "")),
            str(row.get("source", "")),
            str(row.get("source_path", "")),
            str(row.get("title", "")),
            str(row.get("abstract", "")),
            str(row.get("year", "")),
            str(row.get("venue", "")),
            str(row.get("authors_text", "")),
            json.dumps(row.get("normalized_authors", []), ensure_ascii=False),
            str(row.get("keywords_text", "")),
            json.dumps(row.get("keywords", []), ensure_ascii=False),
            str(row.get("language", "")),
            int(row.get("total_pages", 0) or 0),
            str(row.get("paper_summary", "")),
            json.dumps(row.get("section_names", []), ensure_ascii=False),
            int(row.get("citation_count", 0) or 0),
            json.dumps(row.get("citation_titles", []), ensure_ascii=False),
        )
        for row in rows
        if str(row.get("doc_id", "")).strip()
    ]
    conn.executemany(
        """
        INSERT INTO papers (
            doc_id, source, source_path, title, abstract, year, venue, authors_text,
            normalized_authors_json, keywords_text, keywords_json, language, total_pages,
            paper_summary, section_names_json, citation_count, citation_titles_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )


def _insert_citations(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    payload = [
        (
            str(row.get("edge_id", "")),
            str(row.get("source_doc_id", "")),
            str(row.get("source_title", "")),
            str(row.get("cited_title", "")),
            str(row.get("cited_year", "")),
            str(row.get("cited_authors", "")),
            str(row.get("citation_text", "")),
        )
        for row in rows
        if str(row.get("edge_id", "")).strip()
    ]
    conn.executemany(
        """
        INSERT INTO citations (
            edge_id, source_doc_id, source_title, cited_title, cited_year, cited_authors, citation_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )


def _insert_sections(conn: sqlite3.Connection, section_docs: list[Document]) -> None:
    payload = []
    for doc in section_docs:
        metadata = dict(doc.metadata or {})
        payload.append(
            (
                str(metadata.get("doc_id", "")),
                str(metadata.get("section_name", "")),
                str(metadata.get("page_range", "")),
                doc.page_content,
            )
        )
    conn.executemany(
        "INSERT INTO sections (doc_id, section_name, page_range, summary) VALUES (?, ?, ?, ?)",
        payload,
    )
