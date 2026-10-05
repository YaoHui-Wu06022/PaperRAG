"""LlamaIndex Milvus 索引生命周期与增量 Embedding 缓存。"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sqlite3
import uuid
from typing import Any, Iterable

from llama_index.core.schema import TextNode
from llama_index.vector_stores.milvus import MilvusVectorStore

from paper_rag.catalog.chunks import CHUNK_RULE_VERSION
from paper_rag.catalog.service import catalog_status
from paper_rag.config import Settings
from paper_rag.llamaindex.embedding import DashScopeEmbedding
from paper_rag.llamaindex.nodes import load_nodes


INDEX_SCHEMA_VERSION = 2
_CACHE_COLUMNS = (
    "chunk_id",
    "retrieval_text_hash",
    "embedding_model",
    "embedding_dimensions",
    "chunk_rule_version",
    "milvus_collection",
    "synced_at",
)


class LlamaIndexError(RuntimeError):
    """LlamaIndex 索引不可用或重建失败。"""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class IndexService:
    """延迟加载的 LlamaIndex IndexService。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._index: Any | None = None
        self._collection: str | None = None

    def status(self) -> dict[str, Any]:
        """返回索引状态以及可操作的 stale 原因。"""

        manifest = _read_manifest(self.settings)
        try:
            catalog = catalog_status(self.settings)
        except Exception:
            catalog = {"status": "index_not_ready", "indexed_chunks": 0}
        catalog_indexed_at = _catalog_indexed_at(self.settings)
        catalog_count = _catalog_chunk_count(self.settings)
        catalog_ids = _catalog_chunk_ids(self.settings)
        state, items, cache_error = _read_embedding_cache(self.settings)
        reasons = _stale_reasons(self.settings, manifest, catalog_indexed_at, catalog_count, catalog_ids, state, items, cache_error)
        manifest_ready = bool(manifest and manifest.get("status") == "ready")
        index_ready = manifest_ready and not reasons
        result = {
            "status": "ok" if index_ready else "index_not_ready",
            "catalog_ready": bool(catalog.get("index_ready")),
            "index_ready": index_ready,
            "index_stale": bool(reasons),
            "stale_reasons": sorted(reasons),
            "cache_complete": not cache_error and _cache_matches_catalog(items, catalog_ids, manifest),
            "chunk_count": int((manifest or {}).get("chunk_count", catalog_count)),
            "indexed_count": int((manifest or {}).get("indexed_count", 0)),
            "embedding_model": self.settings.embedding_model,
            "embedding_dimensions": self.settings.embedding_dimensions,
            "milvus_collection": (manifest or {}).get("milvus_collection", self.settings.llamaindex_milvus_collection),
            "built_at": (manifest or {}).get("built_at"),
            "last_sync_mode": state.get("last_sync_mode"),
            "last_sync_status": state.get("last_sync_status"),
            "sync_stats": _manifest_stats(manifest),
        }
        if manifest:
            result["manifest"] = manifest
        if cache_error:
            result["cache_error"] = cache_error
        return result

    def load(self) -> Any:
        """加载 active Collection；过期索引不能被查询。"""

        if self._index is not None:
            return self._index
        manifest = _read_manifest(self.settings)
        state, items, cache_error = _read_embedding_cache(self.settings)
        reasons = _stale_reasons(self.settings, manifest, _catalog_indexed_at(self.settings), _catalog_chunk_count(self.settings), _catalog_chunk_ids(self.settings), state, items, cache_error)
        if reasons:
            raise LlamaIndexError(
                "LlamaIndex index is not ready or stale; run library_index_rebuild first",
                details={"stale_reasons": sorted(reasons)},
            )
        collection = str(manifest.get("milvus_collection") or self.settings.llamaindex_milvus_collection)
        vector_store = self._vector_store(collection, overwrite=False)
        try:
            from llama_index.core import VectorStoreIndex

            self._index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=DashScopeEmbedding(self.settings))
        except Exception as exc:
            raise LlamaIndexError(f"无法加载 Milvus LlamaIndex：{type(exc).__name__}") from exc
        self._collection = collection
        return self._index

    def rebuild(self, mode: str = "auto") -> dict[str, Any]:
        """使用临时 Collection 执行全量或增量索引同步。"""

        mode = str(mode or "auto").casefold()
        if mode not in {"auto", "incremental", "full"}:
            raise LlamaIndexError("mode must be auto, incremental or full")
        nodes = load_nodes(self.settings, regions=("abstract", "content", "appendix"))
        if not nodes:
            raise LlamaIndexError("Catalog 中没有可建立向量索引的正文 Chunk")
        previous_manifest = _read_manifest(self.settings)
        previous_collection = str(previous_manifest.get("milvus_collection")) if previous_manifest else None
        state, cached_items, cache_error = _read_embedding_cache(self.settings)
        current = {node.node_id: node for node in nodes}
        reasons = _stale_reasons(self.settings, previous_manifest, _catalog_indexed_at(self.settings), len(nodes), set(current), state, cached_items, cache_error)
        # Catalog 新增或删除 Chunk 时，旧缓存自然会暂时不完整；这正是增量同步的正常输入。
        # 只有索引契约、缓存结构或 active Collection 不可复用时，才需要全量重建。
        boundary_reasons = {"manifest_missing", "manifest_schema_mismatch", "chunk_rule_version_mismatch", "embedding_model_mismatch", "embedding_dimensions_mismatch", "collection_missing", "cache_missing", "cache_schema_mismatch"}
        if mode == "incremental" and reasons.intersection(boundary_reasons):
            raise LlamaIndexError("incremental rebuild requires a compatible active index", details={"status": "rebuild_required", "stale_reasons": sorted(reasons)})
        effective_mode = "full" if mode == "full" or (mode == "auto" and reasons.intersection(boundary_reasons)) else "incremental"
        cached = {str(item["chunk_id"]): item for item in cached_items}
        reused_ids: set[str] = set()
        added_ids: set[str] = set()
        updated_ids: set[str] = set()
        deleted_ids = set(cached) - set(current)
        if effective_mode == "incremental":
            for chunk_id, node in current.items():
                item = cached.get(chunk_id)
                if item and _cache_key_matches(item, node, self.settings):
                    reused_ids.add(chunk_id)
                elif item:
                    updated_ids.add(chunk_id)
                else:
                    added_ids.add(chunk_id)
        else:
            for chunk_id in current:
                if chunk_id in cached:
                    updated_ids.add(chunk_id)
                else:
                    added_ids.add(chunk_id)
        stats = {"reused": len(reused_ids), "added": len(added_ids), "updated": len(updated_ids), "deleted": len(deleted_ids), "failed": 0}
        if effective_mode == "incremental" and previous_collection:
            return self._rebuild_active_collection(
                nodes,
                current,
                previous_collection,
                added_ids,
                updated_ids,
                deleted_ids,
                stats,
            )
        staging = f"{self.settings.llamaindex_milvus_collection}__build_{uuid.uuid4().hex[:12]}"
        old_cache_snapshot = _snapshot_embedding_cache(self.settings)
        failed_ids: list[str] = []
        try:
            vector_store = self._vector_store(staging, overwrite=True)
            old_vectors = _read_old_vectors(self.settings, previous_collection, reused_ids)
            missing_reused = reused_ids - set(old_vectors)
            if missing_reused:
                stats["failed"] = len(missing_reused)
                failed_ids.extend(sorted(missing_reused))
                raise LlamaIndexError("active Collection 缺少需要复用的向量", details={"stats": stats, "failed_chunk_ids": failed_ids})
            if old_vectors:
                _add_nodes(vector_store, [_node_with_embedding(current[chunk_id], old_vectors[chunk_id]) for chunk_id in sorted(old_vectors)])
            changed_ids = sorted(added_ids | updated_ids)
            if changed_ids:
                embed_model = DashScopeEmbedding(self.settings)
                for batch in _batches([current[chunk_id] for chunk_id in changed_ids], self.settings.embedding_batch_size):
                    batch_ids = [node.node_id for node in batch]
                    try:
                        vectors = embed_model.get_text_embedding_batch([node.text for node in batch])
                        if len(vectors) != len(batch) or any(len(vector) != self.settings.embedding_dimensions for vector in vectors):
                            raise LlamaIndexError("Embedding 返回数量或维度不匹配")
                        _add_nodes(vector_store, [_node_with_embedding(node, vector) for node, vector in zip(batch, vectors)])
                    except Exception:
                        stats["failed"] += len(batch)
                        failed_ids.extend(batch_ids)
                        raise
            _verify_collection(vector_store, current)
            manifest = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "catalog_indexed_at": _catalog_indexed_at(self.settings),
                "chunk_count": len(nodes),
                "indexed_count": len(nodes),
                "chunk_rule_version": CHUNK_RULE_VERSION,
                "embedding_model": self.settings.embedding_model,
                "embedding_dimensions": self.settings.embedding_dimensions,
                "milvus_collection": staging,
                "built_at": _utc_now(),
                "status": "ready",
                "last_sync_mode": effective_mode,
                "sync_stats": stats,
            }
            cache_rows = [_cache_row(node, staging, self.settings) for node in nodes]
            _replace_embedding_cache(self.settings, cache_rows, manifest, stats)
            try:
                _write_manifest(self.settings, manifest)
            except Exception:
                _restore_embedding_cache(self.settings, old_cache_snapshot)
                raise
            self._index = None
            self._collection = staging
            if previous_collection and previous_collection != staging:
                _drop_collection(self.settings, previous_collection)
            return {"status": "completed", **manifest}
        except LlamaIndexError:
            _restore_embedding_cache(self.settings, old_cache_snapshot)
            _drop_collection(self.settings, staging)
            raise
        except Exception as exc:
            _restore_embedding_cache(self.settings, old_cache_snapshot)
            _drop_collection(self.settings, staging)
            raise LlamaIndexError(f"LlamaIndex 索引同步失败：{type(exc).__name__}", details={"stats": stats, "failed_chunk_ids": failed_ids}) from exc
        except BaseException:
            # Ctrl+C 等中断也必须清理 staging Collection，避免占用 Zilliz Collection 配额。
            _restore_embedding_cache(self.settings, old_cache_snapshot)
            _drop_collection(self.settings, staging)
            raise

    def _rebuild_active_collection(
        self,
        nodes: list[TextNode],
        current: dict[str, TextNode],
        collection: str,
        added_ids: set[str],
        updated_ids: set[str],
        deleted_ids: set[str],
        stats: dict[str, int],
    ) -> dict[str, Any]:
        """只在 active Collection 上写入新增、变更和删除的 Chunk。"""

        old_cache_snapshot = _snapshot_embedding_cache(self.settings)
        failed_ids: list[str] = []
        try:
            vector_store = self._vector_store(collection, overwrite=False, upsert_mode=True)
            changed_ids = sorted(added_ids | updated_ids)
            if changed_ids:
                embed_model = DashScopeEmbedding(self.settings)
                for batch in _batches([current[chunk_id] for chunk_id in changed_ids], self.settings.embedding_batch_size):
                    batch_ids = [node.node_id for node in batch]
                    try:
                        vectors = embed_model.get_text_embedding_batch([node.text for node in batch])
                        if len(vectors) != len(batch) or any(len(vector) != self.settings.embedding_dimensions for vector in vectors):
                            raise LlamaIndexError("Embedding 返回数量或维度不匹配")
                        _add_nodes(vector_store, [_node_with_embedding(node, vector) for node, vector in zip(batch, vectors)])
                    except Exception:
                        stats["failed"] += len(batch)
                        failed_ids.extend(batch_ids)
                        raise
            if deleted_ids:
                vector_store.delete_nodes(node_ids=sorted(deleted_ids))
                vector_store.client.flush(collection)
            _verify_collection(vector_store, current)
            manifest = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "catalog_indexed_at": _catalog_indexed_at(self.settings),
                "chunk_count": len(nodes),
                "indexed_count": len(nodes),
                "chunk_rule_version": CHUNK_RULE_VERSION,
                "embedding_model": self.settings.embedding_model,
                "embedding_dimensions": self.settings.embedding_dimensions,
                "milvus_collection": collection,
                "built_at": _utc_now(),
                "status": "ready",
                "last_sync_mode": "incremental",
                "sync_stats": stats,
            }
            cache_rows = [_cache_row(node, collection, self.settings) for node in nodes]
            _replace_embedding_cache(self.settings, cache_rows, manifest, stats)
            try:
                _write_manifest(self.settings, manifest)
            except Exception:
                _restore_embedding_cache(self.settings, old_cache_snapshot)
                raise
            self._index = None
            self._collection = collection
            return {"status": "completed", **manifest}
        except LlamaIndexError:
            _restore_embedding_cache(self.settings, old_cache_snapshot)
            raise
        except Exception as exc:
            _restore_embedding_cache(self.settings, old_cache_snapshot)
            raise LlamaIndexError(
                f"LlamaIndex 增量同步失败：{type(exc).__name__}",
                details={"stats": stats, "failed_chunk_ids": failed_ids},
            ) from exc
        except BaseException:
            _restore_embedding_cache(self.settings, old_cache_snapshot)
            raise

    def _vector_store(self, collection: str, *, overwrite: bool, upsert_mode: bool = False) -> MilvusVectorStore:
        if not self.settings.milvus_uri:
            raise LlamaIndexError("MILVUS_URI 未配置")
        kwargs: dict[str, Any] = {"uri": self.settings.milvus_uri, "collection_name": collection, "overwrite": overwrite, "upsert_mode": upsert_mode, "dim": self.settings.embedding_dimensions, "text_key": "text", "similarity_metric": "COSINE", "batch_size": self.settings.embedding_batch_size, "use_async_client": False}
        if self.settings.milvus_token:
            kwargs["token"] = self.settings.milvus_token
        kwargs["db_name"] = self.settings.milvus_db_name or "default"
        try:
            return MilvusVectorStore(**kwargs)
        except Exception as exc:
            raise LlamaIndexError(f"Milvus Collection 初始化失败：{type(exc).__name__}") from exc


def _add_nodes(vector_store: MilvusVectorStore, nodes: list[TextNode]) -> None:
    if nodes:
        vector_store.add(nodes, force_flush=True)


def _node_with_embedding(node: TextNode, embedding: list[float]) -> TextNode:
    return TextNode(id_=node.node_id, text=node.text, metadata=dict(node.metadata), embedding=list(embedding), excluded_embed_metadata_keys=list(node.metadata))


def _batches(values: list[TextNode], size: int) -> Iterable[list[TextNode]]:
    for start in range(0, len(values), max(1, size)):
        yield values[start : start + max(1, size)]


def _read_old_vectors(settings: Settings, collection: str | None, chunk_ids: set[str]) -> dict[str, list[float]]:
    if not collection or not chunk_ids:
        return {}
    client = _milvus_client(settings)
    if not client.has_collection(collection_name=collection):
        return {}
    result: dict[str, list[float]] = {}
    for batch in _id_batches(sorted(chunk_ids)):
        values = ",".join(f'"{value}"' for value in batch)
        rows = client.query(collection_name=collection, filter=f"chunk_id in [{values}]", output_fields=["chunk_id", "embedding"], limit=len(batch))
        for row in rows:
            if row.get("chunk_id") and isinstance(row.get("embedding"), list):
                result[str(row["chunk_id"])] = [float(value) for value in row["embedding"]]
    return result


def _verify_collection(vector_store: MilvusVectorStore, current: dict[str, TextNode]) -> None:
    # Zilliz 的 get_collection_stats 可能在删除后继续返回旧 row_count，改用强一致查询核对主键集合。
    rows = vector_store.client.query(
        collection_name=vector_store.collection_name,
        filter='id != ""',
        output_fields=["id"],
        limit=max(1, len(current) * 2),
        consistency_level="Strong",
    )
    actual_ids = {str(row.get("id")) for row in rows if row.get("id")}
    if actual_ids != set(current):
        raise LlamaIndexError("Milvus 写入数量与 Catalog Chunk 数量不一致")
    first = next(iter(current))
    rows = vector_store.client.query(collection_name=vector_store.collection_name, filter=f'chunk_id == "{first}"', output_fields=["chunk_id", "embedding"], limit=1)
    if not rows or len(rows[0].get("embedding", [])) != vector_store.dim:
        raise LlamaIndexError("Milvus 索引探测查询失败")


def _cache_row(node: TextNode, collection: str, settings: Settings) -> tuple[Any, ...]:
    return (node.node_id, str(node.metadata.get("retrieval_text_hash") or ""), settings.embedding_model, settings.embedding_dimensions, CHUNK_RULE_VERSION, collection, _utc_now())


def _cache_key_matches(item: dict[str, Any], node: TextNode, settings: Settings) -> bool:
    return str(item.get("chunk_id")) == node.node_id and str(item.get("retrieval_text_hash")) == str(node.metadata.get("retrieval_text_hash")) and str(item.get("embedding_model")) == settings.embedding_model and int(item.get("embedding_dimensions", -1)) == settings.embedding_dimensions and str(item.get("chunk_rule_version")) == CHUNK_RULE_VERSION


def _read_embedding_cache(settings: Settings) -> tuple[dict[str, str], list[dict[str, Any]], str | None]:
    connection = None
    try:
        connection = sqlite3.connect(settings.paper_catalog_db_path)
        state = {str(key): str(value) for key, value in connection.execute("SELECT key, value FROM embedding_state").fetchall()}
        rows = connection.execute("SELECT chunk_id, retrieval_text_hash, embedding_model, embedding_dimensions, chunk_rule_version, milvus_collection, synced_at FROM embedding_items").fetchall()
        return state, [dict(zip(_CACHE_COLUMNS, row)) for row in rows], state.get("cache_status")
    except sqlite3.Error:
        return {}, [], "cache_schema_mismatch"
    finally:
        if connection is not None:
            connection.close()


def _snapshot_embedding_cache(settings: Settings) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    connection = None
    try:
        connection = sqlite3.connect(settings.paper_catalog_db_path)
        state = connection.execute("SELECT key, value FROM embedding_state").fetchall()
        items = connection.execute("SELECT chunk_id, retrieval_text_hash, embedding_model, embedding_dimensions, chunk_rule_version, milvus_collection, synced_at FROM embedding_items").fetchall()
        return state, items
    except sqlite3.Error:
        return [], []
    finally:
        if connection is not None:
            connection.close()


def _replace_embedding_cache(settings: Settings, rows: list[tuple[Any, ...]], manifest: dict[str, Any], stats: dict[str, int]) -> None:
    connection = None
    try:
        connection = sqlite3.connect(settings.paper_catalog_db_path)
        with connection:
            connection.execute("DELETE FROM embedding_items")
            connection.executemany("INSERT INTO embedding_items (chunk_id, retrieval_text_hash, embedding_model, embedding_dimensions, chunk_rule_version, milvus_collection, synced_at) VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
            connection.execute("DELETE FROM embedding_state")
            values = {"active_collection": manifest["milvus_collection"], "catalog_indexed_at": manifest["catalog_indexed_at"], "embedding_model": manifest["embedding_model"], "embedding_dimensions": str(manifest["embedding_dimensions"]), "chunk_rule_version": manifest["chunk_rule_version"], "last_sync_at": manifest["built_at"], "last_sync_mode": manifest["last_sync_mode"], "last_sync_status": "completed", **{f"last_{key}": str(value) for key, value in stats.items()}, "stale_reasons": "[]"}
            connection.executemany("INSERT INTO embedding_state (key, value) VALUES (?, ?)", values.items())
    except sqlite3.Error as exc:
        raise LlamaIndexError(f"Embedding 缓存更新失败：{type(exc).__name__}") from exc
    finally:
        if connection is not None:
            connection.close()


def _restore_embedding_cache(settings: Settings, snapshot: tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]) -> None:
    state, items = snapshot
    connection = None
    try:
        connection = sqlite3.connect(settings.paper_catalog_db_path)
        with connection:
            connection.execute("DELETE FROM embedding_items")
            connection.execute("DELETE FROM embedding_state")
            if items:
                connection.executemany("INSERT INTO embedding_items (chunk_id, retrieval_text_hash, embedding_model, embedding_dimensions, chunk_rule_version, milvus_collection, synced_at) VALUES (?, ?, ?, ?, ?, ?, ?)", items)
            if state:
                connection.executemany("INSERT INTO embedding_state (key, value) VALUES (?, ?)", state)
    except sqlite3.Error:
        return
    finally:
        if connection is not None:
            connection.close()


def _stale_reasons(settings: Settings, manifest: dict[str, Any] | None, catalog_indexed_at: str | None, catalog_count: int, catalog_ids: set[str], state: dict[str, str], items: list[dict[str, Any]], cache_error: str | None) -> set[str]:
    reasons: set[str] = set()
    if not manifest:
        reasons.add("manifest_missing")
        return reasons
    if int(manifest.get("schema_version", 0)) < INDEX_SCHEMA_VERSION:
        reasons.add("manifest_schema_mismatch")
    if manifest.get("status") != "ready":
        reasons.add("last_sync_failed")
    if manifest.get("catalog_indexed_at") != catalog_indexed_at:
        reasons.add("catalog_timestamp_mismatch")
    if int(manifest.get("chunk_count", -1)) != catalog_count:
        reasons.add("chunk_count_mismatch")
    if manifest.get("chunk_rule_version") != CHUNK_RULE_VERSION:
        reasons.add("chunk_rule_version_mismatch")
    if manifest.get("embedding_model") != settings.embedding_model:
        reasons.add("embedding_model_mismatch")
    if int(manifest.get("embedding_dimensions", -1)) != settings.embedding_dimensions:
        reasons.add("embedding_dimensions_mismatch")
    collection = str(manifest.get("milvus_collection") or "")
    if collection and settings.milvus_uri and not _collection_exists(settings, collection):
        reasons.add("collection_missing")
    if cache_error:
        reasons.add(cache_error)
    elif not items and catalog_count:
        reasons.add("cache_missing")
    elif not _cache_matches_catalog(items, catalog_ids, manifest):
        reasons.add("cache_incomplete")
    if state.get("last_sync_status") == "failed":
        reasons.add("last_sync_failed")
    return reasons


def _cache_matches_catalog(items: list[dict[str, Any]], catalog_ids: set[str], manifest: dict[str, Any] | None) -> bool:
    if not manifest or {str(item.get("chunk_id")) for item in items} != catalog_ids:
        return False
    collection = str(manifest.get("milvus_collection") or "")
    return all(str(item.get("milvus_collection")) == collection for item in items)


def _manifest_stats(manifest: dict[str, Any] | None) -> dict[str, int]:
    value = (manifest or {}).get("sync_stats")
    return value if isinstance(value, dict) else {"reused": 0, "added": 0, "updated": 0, "deleted": 0, "failed": 0}


def _collection_exists(settings: Settings, collection: str) -> bool:
    try:
        return _milvus_client(settings).has_collection(collection_name=collection)
    except Exception:
        return False


def _milvus_client(settings: Settings) -> Any:
    from pymilvus import MilvusClient

    kwargs: dict[str, Any] = {"uri": settings.milvus_uri, "db_name": settings.milvus_db_name or "default"}
    if settings.milvus_token:
        kwargs["token"] = settings.milvus_token
    return MilvusClient(**kwargs)


def _id_batches(values: list[str], size: int = 100) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


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
        client = _milvus_client(settings)
        if client.has_collection(collection_name=collection):
            client.drop_collection(collection_name=collection)
    except Exception:
        return


def _catalog_indexed_at(settings: Settings) -> str | None:
    connection = None
    try:
        connection = sqlite3.connect(settings.paper_catalog_db_path)
        row = connection.execute("SELECT value FROM catalog_meta WHERE key='indexed_at'").fetchone()
        return str(row[0]) if row else None
    except sqlite3.Error:
        return None
    finally:
        if connection is not None:
            connection.close()


def _catalog_chunk_count(settings: Settings) -> int:
    connection = None
    try:
        connection = sqlite3.connect(settings.paper_catalog_db_path)
        row = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0
    finally:
        if connection is not None:
            connection.close()


def _catalog_chunk_ids(settings: Settings) -> set[str]:
    connection = None
    try:
        connection = sqlite3.connect(settings.paper_catalog_db_path)
        rows = connection.execute("SELECT chunk_id FROM chunks").fetchall()
        return {str(row[0]) for row in rows}
    except sqlite3.Error:
        return set()
    finally:
        if connection is not None:
            connection.close()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


__all__ = ["IndexService", "LlamaIndexError"]
