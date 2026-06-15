"""Milvus/Zilliz 向量库读写封装。"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from paper_rag.corpus.chunks import ChunkDocument


@dataclass(frozen=True)
class SearchResult:
    """Milvus chunk 命中的对外轻量结果。"""

    score: float
    chunk_id: str
    paper_id: str
    chunk_index: int
    title: str
    section_path_text: str
    pages_text: str
    text: str

    @property
    def snippet(self) -> str:
        """给 CLI search 展示用的短文本片段。"""
        return snippet(self.text)


class MilvusStore:
    """封装 Milvus collection 的重建、插入和向量搜索。"""

    vector_field = "vector"
    output_fields = [
        "chunk_id",
        "paper_id",
        "chunk_index",
        "title",
        "section_path_text",
        "pages_text",
        "text",
    ]

    def __init__(
        self,
        *,
        uri: str,
        token: str | None,
        db_name: str | None,
        collection_name: str,
        dimensions: int,
        client: Any | None = None,
    ) -> None:
        if client is None:
            # 延迟导入 pymilvus，避免只跑 schema/router 测试时要求安装 Milvus 依赖。
            from pymilvus import MilvusClient
            client = MilvusClient(uri=uri, token=token or "", db_name=db_name or "")
        self.client = client
        self.collection_name = collection_name
        self.dimensions = dimensions

    def recreate_collection(self) -> None:
        """删除旧 collection 并按当前 embedding 维度重建。"""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        self.create_collection(self.collection_name)

    def rebuild_collection(self, chunk_documents: list[ChunkDocument], vectors: list[list[float]]) -> int:
        """在 staging collection 中完整构建索引，成功后再切换正式 alias。"""
        self.require_alias_support()
        staging_name = f"{self.collection_name}__staging_{uuid.uuid4().hex[:12]}"
        self.create_collection(staging_name)
        try:
            inserted = self.insert_chunk_documents(chunk_documents, vectors, collection_name=staging_name)
            if inserted != len(chunk_documents):
                raise RuntimeError(f"staging collection 写入数量不一致：{inserted}/{len(chunk_documents)}")
            self.switch_alias_to(staging_name)
        except Exception:
            self.drop_collection_if_exists(staging_name)
            raise
        return inserted

    def create_collection(self, collection_name: str) -> None:
        """按当前 embedding 维度创建 collection。"""
        self.client.create_collection(
            collection_name=collection_name,
            dimension=self.dimensions,
            primary_field_name="chunk_id",
            id_type="string",
            vector_field_name=self.vector_field,
            metric_type="COSINE",
            auto_id=False,
            max_length=512,
            enable_dynamic_field=True,
        )

    def insert_chunk_documents(
        self,
        chunk_documents: list[ChunkDocument],
        vectors: list[list[float]],
        batch_size: int = 100,
        *,
        collection_name: str | None = None,
    ) -> int:
        """把 chunk 元数据和向量成批写入 Milvus。"""
        if len(chunk_documents) != len(vectors):
            raise ValueError(f"chunk 数和向量数不一致：{len(chunk_documents)} != {len(vectors)}")
        target_collection = collection_name or self.collection_name
        rows = [
            build_chunk_row(chunk_document, vector, self.vector_field)
            for chunk_document, vector in zip(chunk_documents, vectors)
        ]
        for start in range(0, len(rows), batch_size):
            self.client.insert(target_collection, rows[start:start + batch_size])
        if hasattr(self.client, "load_collection"):
            self.client.load_collection(target_collection)
        return len(rows)

    def require_alias_support(self) -> None:
        """staging rebuild 依赖 alias API；缺失时拒绝破坏性重建。"""
        missing = [
            name
            for name in ["create_alias", "alter_alias", "describe_alias"]
            if not hasattr(self.client, name)
        ]
        if missing:
            raise RuntimeError(f"Milvus client 缺少 alias API：{', '.join(missing)}")

    def switch_alias_to(self, collection_name: str) -> None:
        """把正式 collection 名作为 alias 指向新的 staging collection。"""
        old_target = self.alias_target(self.collection_name)
        if old_target:
            self.client.alter_alias(collection_name=collection_name, alias=self.collection_name)
            if old_target != collection_name:
                self.drop_collection_if_exists(old_target)
            return
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        self.client.create_alias(collection_name=collection_name, alias=self.collection_name)

    def alias_target(self, alias: str) -> str | None:
        try:
            payload = self.client.describe_alias(alias)
        except Exception:
            return None
        if isinstance(payload, dict):
            target = payload.get("collection_name") or payload.get("collection")
            return str(target) if target else None
        target = getattr(payload, "collection_name", None) or getattr(payload, "collection", None)
        return str(target) if target else None

    def drop_collection_if_exists(self, collection_name: str) -> None:
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)

    def search(self, query_vector: list[float], top_k: int, *, paper_ids: list[str] | None = None) -> list[SearchResult]:
        """用 query 向量检索最相近的 chunk。"""
        filter_expr = paper_id_filter_expr(paper_ids)
        if paper_ids is not None and not filter_expr:
            return []
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            filter=filter_expr,
            limit=top_k,
            output_fields=self.output_fields,
            search_params={"metric_type": "COSINE"},
            anns_field=self.vector_field,
        )
        hits = results[0] if results else []
        return [parse_search_hit(hit) for hit in hits]


def build_chunk_row(chunk_document: ChunkDocument, vector: list[float], vector_field: str) -> dict[str, Any]:
    """把 ChunkDocument 投影成 Milvus 可插入的行。"""
    return {
        "chunk_id": chunk_document.chunk_id,
        "paper_id": chunk_document.paper_id,
        "chunk_index": chunk_document.chunk_index,
        "title": chunk_document.title,
        "section_path_text": chunk_document.section_path_text,
        "pages_text": chunk_document.pages_text,
        "text": chunk_document.text,
        vector_field: vector,
    }


def parse_search_hit(hit: dict[str, Any]) -> SearchResult:
    """兼容 pymilvus hit/entity 结构并转成 SearchResult。"""
    entity = hit.get("entity") if isinstance(hit.get("entity"), dict) else hit
    score = hit.get("distance", hit.get("score", 0.0))
    return SearchResult(
        score=float(score),
        chunk_id=str(entity.get("chunk_id") or hit.get("id") or ""),
        paper_id=str(entity.get("paper_id") or ""),
        chunk_index=int(entity.get("chunk_index") or 0),
        title=str(entity.get("title") or ""),
        section_path_text=str(entity.get("section_path_text") or ""),
        pages_text=str(entity.get("pages_text") or ""),
        text=str(entity.get("text") or ""),
    )


def paper_id_filter_expr(paper_ids: list[str] | None) -> str:
    """把 scope paper_id 列表转成 Milvus 标量过滤表达式。"""
    if paper_ids is None:
        return ""
    values = []
    seen: set[str] = set()
    for paper_id in paper_ids:
        text = str(paper_id or "").strip()
        if text and text not in seen:
            seen.add(text)
            values.append(text)
    if not values:
        return ""
    return "paper_id in " + json.dumps(values, ensure_ascii=False)


def snippet(text: str, limit: int = 320) -> str:
    """压缩空白并截断为 CLI 可读片段。"""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."
