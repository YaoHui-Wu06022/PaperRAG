from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..data.chunks import ChunkDocument


@dataclass(frozen=True)
class SearchResult:
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
        return snippet(self.text)


class MilvusStore:
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
            from pymilvus import MilvusClient
            client = MilvusClient(uri=uri, token=token or "", db_name=db_name or "")
        self.client = client
        self.collection_name = collection_name
        self.dimensions = dimensions

    def recreate_collection(self) -> None:
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
        rows = [
            document_to_row(document, vector, self.vector_field)
            for document, vector in zip(documents, vectors)
        ]
        for start in range(0, len(rows), batch_size):
            self.client.insert(self.collection_name, rows[start:start + batch_size])
        if hasattr(self.client, "load_collection"):
            self.client.load_collection(self.collection_name)
        return len(rows)

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=self.output_fields,
            search_params={"metric_type": "COSINE"},
            anns_field=self.vector_field,
        )
        hits = results[0] if results else []
        return [search_result_from_hit(hit) for hit in hits]


def document_to_row(document: ChunkDocument, vector: list[float], vector_field: str) -> dict[str, Any]:
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


def search_result_from_hit(hit: dict[str, Any]) -> SearchResult:
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
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."
