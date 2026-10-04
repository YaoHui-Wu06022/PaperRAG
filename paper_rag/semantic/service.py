"""显式 Embedding 同步和混合检索服务。"""

from __future__ import annotations

import datetime as dt
import sqlite3
from typing import Any

from paper_rag.catalog.service import CatalogIndexNotReady, _build_fts_query, search_chunks
from paper_rag.config import Settings
from paper_rag.semantic.embedding import DashScopeEmbeddingClient, EmbeddingError
from paper_rag.semantic.milvus import MilvusError, MilvusStore


def embedding_status(settings: Settings) -> dict[str, Any]:
    if not settings.paper_catalog_db_path.is_file():
        return {"status": "index_not_ready", "model": settings.embedding_model, "dimensions": settings.embedding_dimensions, "chunks": 0}
    try:
        with sqlite3.connect(settings.paper_catalog_db_path) as connection:
            row = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
            state = connection.execute("SELECT value FROM embedding_state WHERE key='last_sync' ").fetchone() if _table_exists(connection, "embedding_state") else None
        return {"status": "ok", "model": settings.embedding_model, "dimensions": settings.embedding_dimensions, "chunks": int(row[0]), "last_sync": state[0] if state else None, "configured": bool(settings.dashscope_api_key and settings.milvus_uri)}
    except sqlite3.Error:
        return {"status": "index_not_ready", "model": settings.embedding_model, "dimensions": settings.embedding_dimensions, "chunks": 0}


def rebuild_embeddings(settings: Settings) -> dict[str, Any]:
    if not settings.paper_catalog_db_path.is_file():
        return {"status": "index_not_ready", "chunks": 0}
    client = DashScopeEmbeddingClient(settings)
    store = MilvusStore(settings)
    with sqlite3.connect(settings.paper_catalog_db_path) as connection:
        rows = connection.execute("SELECT chunk_id, paper_id, canonical_id, region, chapter_number, section_path, content_hash, retrieval_text FROM chunks WHERE region IN ('abstract','content','appendix') ORDER BY ordinal").fetchall()
        previous = {row[0]: (row[1], row[2]) for row in connection.execute("SELECT chunk_id, content_hash, model FROM embedding_items")}
    pending = [row for row in rows if previous.get(row[0]) != (row[6], settings.embedding_model)]
    written = 0
    for start in range(0, len(pending), settings.embedding_batch_size):
        batch = pending[start:start + settings.embedding_batch_size]
        vectors = client.embed([row[7] for row in batch])
        payload = []
        for row, vector in zip(batch, vectors):
            payload.append({"chunk_id": row[0], "paper_id": row[1], "canonical_id": row[2], "region": row[3], "chapter_number": row[4] or "", "section_path": row[5], "content_hash": row[6], "embedding_model": settings.embedding_model, "embedding_version": "1", "embedding": vector})
        store.upsert(payload)
        written += len(payload)
    with sqlite3.connect(settings.paper_catalog_db_path) as connection:
        with connection:
            connection.execute("CREATE TABLE IF NOT EXISTS embedding_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS embedding_items (chunk_id TEXT PRIMARY KEY, content_hash TEXT NOT NULL, model TEXT NOT NULL, dimensions INTEGER NOT NULL, synced_at TEXT NOT NULL)")
            connection.execute("INSERT OR REPLACE INTO embedding_state VALUES ('last_sync', ?)", (dt.datetime.now(dt.timezone.utc).isoformat(),))
            connection.execute("INSERT OR REPLACE INTO embedding_state VALUES ('model', ?)", (settings.embedding_model,))
            stamp = dt.datetime.now(dt.timezone.utc).isoformat()
            connection.executemany("INSERT OR REPLACE INTO embedding_items VALUES (?, ?, ?, ?, ?)", [(row[0], row[6], settings.embedding_model, settings.embedding_dimensions, stamp) for row in pending])
    return {"status": "completed", "chunks": written, "model": settings.embedding_model, "dimensions": settings.embedding_dimensions}


def hybrid_search(settings: Settings, query: str, paper_ids: list[str] | None = None, limit: int = 8, mode: str = "hybrid") -> dict[str, Any]:
    mode = mode.casefold()
    if mode not in {"lexical", "semantic", "hybrid"}:
        raise ValueError("mode 必须是 lexical、semantic 或 hybrid")
    try:
        lexical = search_chunks(settings, query, paper_ids, max(limit, 50)) if mode in {"lexical", "hybrid"} else []
    except CatalogIndexNotReady:
        return {"status": "index_not_ready", "mode": mode, "items": []}
    if mode == "lexical":
        return {"status": "ok", "mode": mode, "items": lexical[:limit]}
    semantic: list[dict[str, Any]] = []
    try:
        vector = DashScopeEmbeddingClient(settings).embed([query])[0]
        semantic = MilvusStore(settings).search(vector, limit=max(limit, 50), paper_ids=paper_ids)
    except (EmbeddingError, MilvusError):
        if mode == "semantic":
            return {"status": "semantic_unavailable", "mode": mode, "items": []}
    if mode == "semantic":
        return {"status": "ok", "mode": mode, "items": semantic[:limit]}
    merged: dict[str, dict[str, Any]] = {}
    for rank, item in enumerate(lexical, 1):
        merged.setdefault(item["chunk_id"], dict(item)).update({"lexical_rank": rank})
    for rank, item in enumerate(semantic, 1):
        merged.setdefault(item.get("chunk_id"), dict(item)).update({"semantic_rank": rank})
    for item in merged.values():
        item["rrf_score"] = (1 / (60 + item.get("lexical_rank", 10_000))) + (1 / (60 + item.get("semantic_rank", 10_000)))
    return {"status": "ok", "mode": mode, "items": sorted(merged.values(), key=lambda item: item["rrf_score"], reverse=True)[:limit]}


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


__all__ = ["embedding_status", "hybrid_search", "rebuild_embeddings"]
