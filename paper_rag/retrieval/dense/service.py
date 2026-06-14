"""dense retrieval 高层服务：index/search/content dense search。"""

from __future__ import annotations

from dataclasses import dataclass

from paper_rag.config import Settings
from paper_rag.corpus.chunks import ChunkDocument, filter_content_retrieval_chunks, load_chunk_documents
from paper_rag.retrieval.dense.cache import CachedEmbedder, EmbeddingCache
from paper_rag.retrieval.dense.embedding import EmbeddingClient
from paper_rag.retrieval.dense.milvus_store import MilvusStore, SearchResult
from paper_rag.retrieval.sparse.bm25 import BM25CorpusIndex


@dataclass(frozen=True)
class IndexSummary:
    """一次向量索引构建的简要结果。"""

    chunk_count: int
    collection_name: str


def build_embedder(settings: Settings, *, cache_path=None, store_cache_text: bool = False) -> CachedEmbedder:
    """按配置组装带本地缓存的 embedding 客户端。"""
    client = EmbeddingClient(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
    )
    cache = EmbeddingCache(cache_path or settings.embedding_cache_path, store_text=store_cache_text)
    return CachedEmbedder(
        client,
        cache,
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
        batch_size=settings.embedding_batch_size,
    )


def build_query_embedder(settings: Settings) -> CachedEmbedder:
    """为用户 query 使用独立 embedding cache。"""
    return build_embedder(
        settings,
        cache_path=settings.query_embedding_cache_path,
        store_cache_text=True,
    )


def build_store(settings: Settings) -> MilvusStore:
    """按配置创建 Milvus/Zilliz collection 访问对象。"""
    if not settings.milvus_uri:
        raise ValueError(".env 中缺少 MILVUS_URI")
    return MilvusStore(
        uri=settings.milvus_uri,
        token=settings.milvus_token,
        db_name=settings.milvus_db_name,
        collection_name=settings.milvus_collection,
        dimensions=settings.embedding_dim,
    )


def run_index(settings: Settings, *, reporter=print, embedder=None, store=None) -> IndexSummary:
    """读取正文 chunks，生成向量并重建 Milvus collection。"""
    chunk_documents = filter_content_retrieval_chunks(load_chunk_documents(settings.paper_data_dir))
    if not chunk_documents:
        raise ValueError(f"在 {settings.paper_data_dir} 中没有找到 abstract/body chunks")
    reporter(f"[index] 已加载 {len(chunk_documents)} 个 chunk")
    embedder = embedder or build_embedder(settings)
    store = store or build_store(settings)
    reporter("[index] 正在生成 chunk embedding")
    # index 使用 chunk.embedding_text，里面通常包含标题/section/text 的稳定组合。
    vectors = embedder.embed_texts([chunk_document.embedding_text for chunk_document in chunk_documents])
    reporter(f"[index] 正在重建 Milvus collection：{settings.milvus_collection}")
    store.recreate_collection()
    inserted = store.insert_chunk_documents(chunk_documents, vectors)
    reporter(f"[index] 已写入 {inserted} 个向量")
    reporter(f"[index] 正在写入 BM25 索引：{settings.bm25_index_path}")
    BM25CorpusIndex.from_chunks(chunk_documents).save(settings.bm25_index_path)
    return IndexSummary(chunk_count=inserted, collection_name=settings.milvus_collection)


def run_search(settings: Settings, query: str, *, top_k: int = 5, embedder=None, store=None) -> list[SearchResult]:
    """把用户 query 向量化后在 Milvus 中召回 chunk。"""
    embedder = embedder or build_query_embedder(settings)
    store = store or build_store(settings)
    query_vector = embedder.embed_texts([query])[0]
    return store.search(query_vector, top_k)


def search_dense_chunks(
    settings: Settings,
    query: str,
    *,
    paper_ids: list[str] | None = None,
    embedder=None,
    store=None,
) -> list[SearchResult]:
    """content planner 用的 dense chunk 检索薄封装。"""
    embedder = embedder or build_query_embedder(settings)
    store = store or build_store(settings)
    query_vector = embedder.embed_texts([query])[0]
    return store.search(query_vector, settings.plan_dense_top_k, paper_ids=paper_ids)
