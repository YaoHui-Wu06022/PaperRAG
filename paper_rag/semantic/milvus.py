"""Milvus 的延迟加载适配层；未安装依赖时基础服务仍可运行。"""

from __future__ import annotations

from typing import Any

from paper_rag.config import Settings


class MilvusError(RuntimeError):
    pass


class MilvusStore:
    def __init__(self, settings: Settings):
        if not settings.milvus_uri:
            raise MilvusError("MILVUS_URI 未配置")
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as exc:
            raise MilvusError("pymilvus 未安装") from exc
        self.settings = settings
        kwargs: dict[str, Any] = {"uri": settings.milvus_uri}
        if settings.milvus_token:
            kwargs["token"] = settings.milvus_token
        if settings.milvus_db_name:
            kwargs["db_name"] = settings.milvus_db_name
        self.client = MilvusClient(**kwargs)

    def ensure_collection(self, name: str | None = None) -> str:
        collection = name or self.settings.milvus_collection
        if not self.client.has_collection(collection_name=collection):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("paper_id", datatype=DataType.VARCHAR, max_length=128)
            schema.add_field("canonical_id", datatype=DataType.VARCHAR, max_length=128)
            schema.add_field("region", datatype=DataType.VARCHAR, max_length=32)
            schema.add_field("chapter_number", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field("section_path", datatype=DataType.VARCHAR, max_length=2048)
            schema.add_field("content_hash", datatype=DataType.VARCHAR, max_length=128)
            schema.add_field("embedding_model", datatype=DataType.VARCHAR, max_length=128)
            schema.add_field("embedding_version", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field("embedding", datatype=DataType.FLOAT_VECTOR, dim=self.settings.milvus_dimension)
            self.client.create_collection(collection_name=collection, schema=schema)
        return collection

    def upsert(self, rows: list[dict[str, Any]], collection: str | None = None) -> None:
        name = self.ensure_collection(collection)
        if rows:
            self.client.upsert(collection_name=name, data=rows)

    def search(self, vector: list[float], limit: int, paper_ids: list[str] | None = None, collection: str | None = None) -> list[dict[str, Any]]:
        name = self.ensure_collection(collection)
        expr = None
        if paper_ids:
            values = ",".join(repr(value) for value in paper_ids)
            expr = f"paper_id in [{values}]"
        result = self.client.search(collection_name=name, data=[vector], anns_field="embedding", limit=limit, filter=expr, output_fields=["chunk_id", "paper_id", "canonical_id", "region", "chapter_number", "section_path", "content_hash"])
        return list(result[0] if result else [])


__all__ = ["MilvusError", "MilvusStore"]
