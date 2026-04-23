from __future__ import annotations

from dataclasses import dataclass

from ...config import Settings
from ..data.chunks import ChunkDocument, load_chunk_documents
from .cache import CachedEmbedder, EmbeddingCache
from .embedding import EmbeddingClient
from .milvus_store import MilvusStore, SearchResult


@dataclass(frozen=True)
class IndexSummary:
    chunk_count: int
    collection_name: str


def build_embedder(settings: Settings) -> CachedEmbedder:
    client = EmbeddingClient(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
    )
    cache = EmbeddingCache(settings.embedding_cache_path)
    return CachedEmbedder(
        client,
        cache,
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
        batch_size=settings.embedding_batch_size,
    )


def build_store(settings: Settings) -> MilvusStore:
    if not settings.milvus_uri:
        raise ValueError("MILVUS_URI is missing in .env")
    return MilvusStore(
        uri=settings.milvus_uri,
        token=settings.milvus_token,
        db_name=settings.milvus_db_name,
        collection_name=settings.milvus_collection,
        dimensions=settings.embedding_dim,
    )


def run_index(settings: Settings, *, reporter=print, embedder=None, store=None) -> IndexSummary:
    documents = load_chunk_documents(settings.paper_data_dir)
    if not documents:
        raise ValueError(f"No chunks.jsonl found in {settings.paper_data_dir}")
    reporter(f"[index] Loaded {len(documents)} chunk(s)")
    embedder = embedder or build_embedder(settings)
    store = store or build_store(settings)
    reporter("[index] Embedding chunks")
    vectors = embedder.embed_texts([document.embedding_text for document in documents])
    reporter(f"[index] Recreating Milvus collection: {settings.milvus_collection}")
    store.recreate_collection()
    inserted = store.insert_documents(documents, vectors)
    reporter(f"[index] Inserted {inserted} vector(s)")
    return IndexSummary(chunk_count=inserted, collection_name=settings.milvus_collection)


def run_search(settings: Settings, query: str, *, top_k: int = 5, embedder=None, store=None) -> list[SearchResult]:
    embedder = embedder or build_embedder(settings)
    store = store or build_store(settings)
    query_vector = embedder.embed_texts([query])[0]
    return store.search(query_vector, top_k)
