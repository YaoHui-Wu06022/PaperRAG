"""LlamaIndex Milvus 索引生命周期管理。"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import uuid
from typing import Any

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore

from paper_rag.catalog.service import CatalogIndexNotReady, catalog_status
from paper_rag.config import Settings
from paper_rag.llamaindex.embedding import DashScopeEmbedding
from paper_rag.llamaindex.nodes import load_nodes


class LlamaIndexError(RuntimeError):
    """LlamaIndex 索引不可用或重建失败。"""


class IndexService:
    """延迟加载的 LlamaIndex IndexService。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._index: VectorStoreIndex | None = None
        self._collection: str | None = None

    def status(self) -> dict[str, Any]:
        manifest = _read_manifest(self.settings)
        try:
            catalog = catalog_status(self.settings)
        except Exception:
            catalog = {"status": "index_not_ready", "papers": 0, "chunks": 0}
        result = {
            "status": "ok" if manifest and manifest.get("status") == "ready" else "index_not_ready",
            "catalog_ready": bool(catalog.get("index_ready", catalog.get("status") == "ok")),
            "index_ready": bool(manifest and manifest.get("status") == "ready"),
            "chunk_count": int((manifest or {}).get("chunk_count", catalog.get("chunks", 0) or 0)),
            "indexed_count": int((manifest or {}).get("indexed_count", 0)),
            "embedding_model": self.settings.embedding_model,
            "embedding_dimensions": self.settings.embedding_dimensions,
            "milvus_collection": (manifest or {}).get("milvus_collection", self.settings.llamaindex_milvus_collection),
            "built_at": (manifest or {}).get("built_at"),
        }
        if manifest:
            result["manifest"] = manifest
        return result

    def load(self) -> VectorStoreIndex:
        if self._index is not None:
            return self._index
        manifest = _read_manifest(self.settings)
        if not manifest or manifest.get("status") != "ready":
            raise LlamaIndexError("LlamaIndex index is not ready; run library_index_rebuild first")
        collection = str(manifest.get("milvus_collection") or self.settings.llamaindex_milvus_collection)
        vector_store = self._vector_store(collection, overwrite=False)
        try:
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=DashScopeEmbedding(self.settings),
            )
        except Exception as exc:
            raise LlamaIndexError(f"无法加载 Milvus LlamaIndex：{type(exc).__name__}") from exc
        self._collection = collection
        return self._index

    def rebuild(self) -> dict[str, Any]:
        """先构建临时 Collection，验证后再切换 active Manifest。"""

        nodes = load_nodes(self.settings, regions=("abstract", "content", "appendix"))
        if not nodes:
            raise LlamaIndexError("Catalog 中没有可建立向量索引的正文 Chunk")
        previous_manifest = _read_manifest(self.settings)
        previous = str(previous_manifest.get("milvus_collection")) if previous_manifest else None
        staging = f"{self.settings.llamaindex_milvus_collection}__build_{uuid.uuid4().hex[:12]}"
        try:
            vector_store = self._vector_store(staging, overwrite=True)
            storage = StorageContext.from_defaults(vector_store=vector_store)
            built_index = VectorStoreIndex(
                nodes,
                storage_context=storage,
                embed_model=DashScopeEmbedding(self.settings),
                insert_batch_size=self.settings.embedding_batch_size,
                show_progress=False,
            )
            probe = built_index.as_retriever(similarity_top_k=1).retrieve(nodes[0].get_content())
            if not probe:
                raise LlamaIndexError("LlamaIndex 索引探测查询没有返回结果")
            manifest = {
                "schema_version": 1,
                "catalog_indexed_at": _catalog_indexed_at(self.settings),
                "chunk_count": len(nodes),
                "indexed_count": len(nodes),
                "chunk_rule_version": "content-list-regions-v2-1200",
                "embedding_model": self.settings.embedding_model,
                "embedding_dimensions": self.settings.embedding_dimensions,
                "milvus_collection": staging,
                "built_at": _utc_now(),
                "status": "ready",
            }
            _write_manifest(self.settings, manifest)
            self._index = None
            self._collection = staging
            if previous and previous != staging:
                _drop_collection(self.settings, previous)
            return {"status": "completed", **manifest}
        except Exception:
            _drop_collection(self.settings, staging)
            raise

    def _vector_store(self, collection: str, *, overwrite: bool) -> MilvusVectorStore:
        if not self.settings.milvus_uri:
            raise LlamaIndexError("MILVUS_URI 未配置")
        kwargs: dict[str, Any] = {
            "uri": self.settings.milvus_uri,
            "collection_name": collection,
            "overwrite": overwrite,
            "dim": self.settings.embedding_dimensions,
            "text_key": "text",
            "similarity_metric": "COSINE",
            "batch_size": self.settings.embedding_batch_size,
            "use_async_client": False,
        }
        if self.settings.milvus_token:
            kwargs["token"] = self.settings.milvus_token
        kwargs["db_name"] = self.settings.milvus_db_name or "default"
        try:
            return MilvusVectorStore(**kwargs)
        except Exception as exc:
            raise LlamaIndexError(f"Milvus Collection 初始化失败：{type(exc).__name__}") from exc


def _manifest_path(settings: Settings) -> Path:
    return settings.llamaindex_index_dir / "manifest.json"


def _read_manifest(settings: Settings) -> dict[str, Any] | None:
    path = _manifest_path(settings)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_manifest(settings: Settings, manifest: dict[str, Any]) -> None:
    directory = settings.llamaindex_index_dir
    directory.mkdir(parents=True, exist_ok=True)
    staged = directory / f".manifest-{uuid.uuid4().hex}.tmp"
    staged.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    staged.replace(_manifest_path(settings))


def _drop_collection(settings: Settings, collection: str) -> None:
    if not settings.milvus_uri:
        return
    try:
        from pymilvus import MilvusClient

        kwargs: dict[str, Any] = {"uri": settings.milvus_uri}
        if settings.milvus_token:
            kwargs["token"] = settings.milvus_token
        kwargs["db_name"] = settings.milvus_db_name or "default"
        client = MilvusClient(**kwargs)
        if client.has_collection(collection_name=collection):
            client.drop_collection(collection_name=collection)
    except Exception:
        # 切换已完成后，旧 Collection 清理失败不应让新索引变成失败状态。
        return


def _catalog_indexed_at(settings: Settings) -> str | None:
    try:
        import sqlite3

        with sqlite3.connect(settings.paper_catalog_db_path) as connection:
            row = connection.execute("SELECT value FROM catalog_meta WHERE key='indexed_at'").fetchone()
        return str(row[0]) if row else None
    except sqlite3.Error:
        return None


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


__all__ = ["IndexService", "LlamaIndexError"]
