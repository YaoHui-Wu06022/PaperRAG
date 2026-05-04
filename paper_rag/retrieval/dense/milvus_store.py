"""Milvus/Zilliz 向量库读写封装。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..data.chunks import ChunkDocument


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
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimensions,
            primary_field_name="chunk_id",
            id_type="string",
            vector_field_name=self.vector_field,
            metric_type="COSINE",
            auto_id=False,
            max_length=512,
            enable_dynamic_field=True,
        )

    def insert_documents(self, documents: list[ChunkDocument], vectors: list[list[float]], batch_size: int = 100) -> int:
        """把 chunk 元数据和向量成批写入 Milvus。"""
        rows = [
            build_document_row(document, vector, self.vector_field)
            for document, vector in zip(documents, vectors)
        ]
        for start in range(0, len(rows), batch_size):
            self.client.insert(self.collection_name, rows[start:start + batch_size])
        if hasattr(self.client, "load_collection"):
            self.client.load_collection(self.collection_name)
        return len(rows)

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        """用 query 向量检索最相近的 chunk。"""
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=self.output_fields,
            search_params={"metric_type": "COSINE"},
            anns_field=self.vector_field,
        )
        hits = results[0] if results else []
        return [parse_search_hit(hit) for hit in hits]


def build_document_row(document: ChunkDocument, vector: list[float], vector_field: str) -> dict[str, Any]:
    """把 ChunkDocument 投影成 Milvus 可插入的行。"""
    return {
        "chunk_id": document.chunk_id,
        "paper_id": document.paper_id,
        "chunk_index": document.chunk_index,
        "title": document.title,
        "section_path_text": document.section_path_text,
        "pages_text": document.pages_text,
        "text": document.text,
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


def snippet(text: str, limit: int = 320) -> str:
    """压缩空白并截断为 CLI 可读片段。"""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."
