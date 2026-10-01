"""dense retrieval 高层服务：index/search/content dense search。"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

from paper_rag.config import Settings
from paper_rag.corpus.chunks import ChunkDocument, filter_content_retrieval_chunks, load_chunk_documents
from paper_rag.retrieval.dense.cache import CachedEmbedder, EmbeddingCache
from paper_rag.retrieval.dense.embedding import EmbeddingClient, EmbeddingError
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
        # profile 只用于本地缓存和 collection 命名空间，发送给 provider 的仍是原始模型 ID。
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
    )
    cache = EmbeddingCache(cache_path or settings.embedding_cache_path, store_text=store_cache_text)
    primary = CachedEmbedder(
        client,
        cache,
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
        batch_size=settings.embedding_batch_size,
    )
    fallback_url = getattr(settings, "embedding_fallback_base_url", "")
    fallback_key = getattr(settings, "embedding_fallback_api_key", None)
    fallback_model = getattr(settings, "embedding_fallback_model", "")
    fallback_dim = int(getattr(settings, "embedding_fallback_dim", 0) or 0)
    # 只有维度完全一致才允许同一 collection 自动切换；否则由 planner 降级 BM25，避免向量维度混用。
    if fallback_url and fallback_key and fallback_model and fallback_dim == settings.embedding_dim:
        fallback_cache_path = cache_path or settings.embedding_cache_path
        fallback_cache = EmbeddingCache(fallback_cache_path, store_text=store_cache_text)
        fallback = CachedEmbedder(
            EmbeddingClient(fallback_url, fallback_key, fallback_model, fallback_dim),
            fallback_cache,
            model=f"fallback:{fallback_model}",
            dimensions=fallback_dim,
            batch_size=settings.embedding_batch_size,
        )
        return FailoverEmbedder(primary, fallback)
    return primary


class FailoverEmbedder:
    """同维度 embedding profile 的主备切换；不同维度不进入此类。"""

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self.model = primary.model
        self.dimensions = primary.dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.primary.embed_texts(texts)
        except (EmbeddingError, OSError, ValueError):
            return self.fallback.embed_texts(texts)


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
    profile = getattr(settings, "embedding_profile", "qwen_v4")
    collection_name = settings.milvus_collection if profile == "qwen_v4" else f"{settings.milvus_collection}__{profile}"
    return MilvusStore(
        uri=settings.milvus_uri,
        token=settings.milvus_token,
        db_name=settings.milvus_db_name,
        collection_name=collection_name,
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
    collection_name = store.collection_name if hasattr(store, "collection_name") else settings.milvus_collection
    reporter(f"[index] 正在通过 staging collection 重建 Milvus alias：{collection_name}")
    inserted = store.rebuild_collection(chunk_documents, vectors)
    reporter(f"[index] 已写入 {inserted} 个向量")
    reporter(f"[index] 正在写入 BM25 索引：{settings.bm25_index_path}")
    BM25CorpusIndex.from_chunks(chunk_documents).save(settings.bm25_index_path)
    summary = IndexSummary(chunk_count=inserted, collection_name=collection_name)
    save_index_state(settings, summary)
    return summary


def save_index_state(settings: Settings, summary: IndexSummary) -> None:
    """原子保存当前 Dense/BM25 索引的可观测状态。"""
    path = getattr(settings, "mcp_index_state_path", None) or settings.data_dir / "index" / "index_state.json"
    payload = {
        "collection_name": summary.collection_name,
        "embedding_profile": getattr(settings, "embedding_profile", "qwen_v4"),
        "model": settings.embedding_model,
        "dimensions": settings.embedding_dim,
        "chunk_count": summary.chunk_count,
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "ok",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


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
