"""一次 plan/ask 内复用的本地语料运行期上下文。"""

from __future__ import annotations

from typing import Any

from paper_rag.config import Settings
from paper_rag.corpus.annotation_index import PaperAnnotationEntry, PaperTags, load_paper_annotation_entries
from paper_rag.corpus.chunks import (
    ChunkDocument,
    filter_chunks_by_paper_records,
    filter_content_retrieval_chunks,
    load_chunk_documents,
)
from paper_rag.corpus.citation_index import load_citation_graph
from paper_rag.corpus.records import load_active_manifest_records
from paper_rag.ingest.manifest import ManifestRecord
from paper_rag.retrieval.sparse.bm25 import BM25CorpusIndex


class CorpusContext:
    """懒加载 manifest、chunks、citation graph 和 BM25 索引。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._active_manifest_records: list[ManifestRecord] | None = None
        self._annotation_entries: list[PaperAnnotationEntry] | None = None
        self._annotation_tag_index: dict[str, PaperTags] | None = None
        self._chunk_documents: list[ChunkDocument] | None = None
        self._content_chunk_documents: list[ChunkDocument] | None = None
        self._citation_graph: dict[str, Any] | None | bool = False
        self._bm25_index: BM25CorpusIndex | None = None

    @property
    def active_manifest_records(self) -> list[ManifestRecord]:
        if self._active_manifest_records is None:
            self._active_manifest_records = load_active_manifest_records(self.settings)
        return self._active_manifest_records

    @property
    def annotation_entries(self) -> list[PaperAnnotationEntry]:
        if self._annotation_entries is None:
            self._annotation_entries = load_paper_annotation_entries(self.settings)
        return self._annotation_entries

    @property
    def annotation_tag_index(self) -> dict[str, PaperTags]:
        if self._annotation_tag_index is None:
            self._annotation_tag_index = {
                entry.paper_title_key: entry.tags
                for entry in self.annotation_entries
                if entry.tags["zh"] or entry.tags["en"]
            }
        return self._annotation_tag_index

    @property
    def chunk_documents(self) -> list[ChunkDocument]:
        if self._chunk_documents is None:
            self._chunk_documents = load_chunk_documents(self.settings.paper_data_dir)
        return self._chunk_documents

    @property
    def content_chunk_documents(self) -> list[ChunkDocument]:
        if self._content_chunk_documents is None:
            self._content_chunk_documents = filter_content_retrieval_chunks(self.chunk_documents)
        return self._content_chunk_documents

    @property
    def citation_graph(self) -> dict[str, Any] | None:
        if self._citation_graph is False:
            self._citation_graph = load_citation_graph(self.settings)
        return self._citation_graph

    @property
    def bm25_index(self) -> BM25CorpusIndex:
        if self._bm25_index is None:
            loaded = BM25CorpusIndex.load(self.settings.bm25_index_path, self.content_chunk_documents)
            self._bm25_index = loaded or BM25CorpusIndex.from_chunks(self.content_chunk_documents)
        return self._bm25_index

    def content_chunks_for_records(self, records: list[dict[str, Any]]) -> list[ChunkDocument]:
        return filter_chunks_by_paper_records(self.content_chunk_documents, records)
