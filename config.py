from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent

# 这个模块是运行时配置的唯一入口。
# `load_config()` 会一次性解析环境变量，并返回不可变的 AppConfig，
# 这样下游模块基本可以保持无状态。


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_optional_float(name: str, default: float | None) -> float | None:
    value = os.getenv(name)
    if value is None:
        return default
    text = value.strip()
    if not text:
        return default
    if text.lower() in {"none", "null", "off", "disable", "disabled"}:
        return None
    try:
        return float(text)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_milvus_settings() -> tuple[str, str, str]:
    """把云端 Milvus 和 Milvus Lite 的配置整理成统一格式。"""
    mode = os.getenv("MILVUS_MODE", "").strip().lower()
    if mode == "lite":
        lite_uri = os.getenv("MILVUS_LITE_URI", "").strip()
        if lite_uri:
            return lite_uri, "", ""
    return (
        os.getenv("MILVUS_URI", "").strip(),
        os.getenv("MILVUS_TOKEN", "").strip(),
        os.getenv("MILVUS_DB_NAME", "").strip(),
    )


@dataclass(frozen=True)
class AppConfig:
    # 配置项按职责大致分成：
    # 输入与缓存路径 -> 解析 -> 切块 -> 检索 -> 生成。
    data_pdf_dir: Path
    local_cache_dir: Path
    mineru_api_token: str
    mineru_api_base_url: str
    mineru_cloud_model_version: str
    mineru_cloud_poll_interval_sec: int
    mineru_cloud_timeout_sec: int
    mineru_output_dir: Path
    embedding_provider: str
    embedding_model: str
    vector_backend: str
    milvus_uri: str
    milvus_token: str
    milvus_db_name: str
    milvus_collection: str
    milvus_papers_collection: str
    milvus_references_collection: str
    references_strategy: str
    references_keyword_index_file: Path
    chunk_size: int
    chunk_overlap: int
    chunk_tokenizer_model: str
    chunk_strategy: str
    chunk_semantic_hard_max_chars: int
    chunk_use_structure_split: bool
    chunk_min_block_chars: int
    chunk_quality_check_enabled: bool
    chunk_quality_header_footer_min_freq: int
    retrieval_mode: str
    hybrid_dense_top_k: int
    hybrid_bm25_top_k: int
    hybrid_rrf_k: int
    metadata_filter_enabled: bool
    query_rewrite_enabled: bool
    query_rewrite_max_variants: int
    retriever_top_k: int
    final_top_k: int
    retrieval_score_threshold: float | None
    retrieval_score_threshold_mode: str
    retrieval_score_relative_ratio: float
    retrieval_score_quantile: float
    use_reranker: bool
    rerank_conditional_enabled: bool
    rerank_skip_min_top1_score: float
    rerank_skip_min_score_gap: float
    rerank_skip_min_rel_gap: float
    reranker_model: str
    reranker_score_threshold: float | None
    diversify_by_source: bool
    max_chunks_per_source: int
    generation_use_parent_context: bool
    generation_parent_top_n: int
    generation_parent_max_chars: int
    paper_summary_max_chars: int
    section_summary_max_chars: int
    llm_provider: str
    llm_model: str
    llm_temperature: float
    aihubmix_api_key: str
    aihubmix_base_url: str
    aihubmix_api_mode: str
    openai_api_key: str
    openai_base_url: str
    openai_api_mode: str
    ollama_base_url: str


def load_config() -> AppConfig:
    """加载环境变量并构建不可变的运行时配置对象。"""
    load_dotenv()
    milvus_uri, milvus_token, milvus_db_name = _resolve_milvus_settings()
    data_pdf_dir = BASE_DIR / "data" / "pdf"
    local_cache_dir = BASE_DIR / "data" / "cache"
    mineru_output_dir = BASE_DIR / "data" / "mineru_output"
    data_pdf_dir.mkdir(parents=True, exist_ok=True)
    local_cache_dir.mkdir(parents=True, exist_ok=True)
    mineru_output_dir.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        data_pdf_dir=data_pdf_dir,
        local_cache_dir=local_cache_dir,
        mineru_api_token=os.getenv("MINERU_API_TOKEN", "").strip(),
        mineru_api_base_url=os.getenv(
            "MINERU_API_BASE_URL",
            "https://mineru.net/api/v4",
        )
        .strip()
        .rstrip("/"),
        mineru_cloud_model_version=os.getenv(
            "MINERU_CLOUD_MODEL_VERSION",
            "pipeline",
        ).strip(),
        mineru_cloud_poll_interval_sec=_env_int(
            "MINERU_CLOUD_POLL_INTERVAL_SEC",
            5,
        ),
        mineru_cloud_timeout_sec=_env_int(
            "MINERU_CLOUD_TIMEOUT_SEC",
            900,
        ),
        mineru_output_dir=Path(
            os.getenv("MINERU_OUTPUT_DIR", str(mineru_output_dir))
        ).resolve(),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "huggingface"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        vector_backend=os.getenv("VECTOR_BACKEND", "milvus"),
        milvus_uri=milvus_uri,
        milvus_token=milvus_token,
        milvus_db_name=milvus_db_name,
        milvus_collection=os.getenv("MILVUS_COLLECTION", "rag_pdf_chunks"),
        milvus_papers_collection=os.getenv(
            "MILVUS_PAPERS_COLLECTION",
            "rag_pdf_papers",
        ),
        milvus_references_collection=os.getenv(
            "MILVUS_REFERENCES_COLLECTION",
            "rag_pdf_references",
        ),
        references_strategy=os.getenv(
            "REFERENCES_STRATEGY",
            "keyword_only",
        )
        .strip()
        .lower(),
        references_keyword_index_file=Path(
            os.getenv(
                "REFERENCES_KEYWORD_INDEX_FILE",
                str(local_cache_dir / "references_keyword_index.jsonl"),
            )
        ).resolve(),
        chunk_size=_env_int("CHUNK_SIZE", 500),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 100),
        chunk_tokenizer_model=os.getenv(
            "CHUNK_TOKENIZER_MODEL",
            os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        ),
        chunk_strategy=os.getenv("CHUNK_STRATEGY", "semantic_paper").strip().lower(),
        chunk_semantic_hard_max_chars=_env_int(
            "CHUNK_SEMANTIC_HARD_MAX_CHARS",
            2400,
        ),
        chunk_use_structure_split=_env_bool("CHUNK_USE_STRUCTURE_SPLIT", True),
        chunk_min_block_chars=_env_int("CHUNK_MIN_BLOCK_CHARS", 120),
        chunk_quality_check_enabled=_env_bool("CHUNK_QUALITY_CHECK_ENABLED", True),
        chunk_quality_header_footer_min_freq=_env_int(
            "CHUNK_QUALITY_HEADER_FOOTER_MIN_FREQ",
            3,
        ),
        retrieval_mode=os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower(),
        hybrid_dense_top_k=_env_int("HYBRID_DENSE_TOP_K", 20),
        hybrid_bm25_top_k=_env_int("HYBRID_BM25_TOP_K", 20),
        hybrid_rrf_k=_env_int("HYBRID_RRF_K", 60),
        metadata_filter_enabled=_env_bool("METADATA_FILTER_ENABLED", True),
        query_rewrite_enabled=_env_bool("QUERY_REWRITE_ENABLED", True),
        query_rewrite_max_variants=_env_int("QUERY_REWRITE_MAX_VARIANTS", 3),
        retriever_top_k=_env_int("RETRIEVER_TOP_K", 8),
        final_top_k=_env_int("FINAL_TOP_K", 5),
        retrieval_score_threshold=_env_optional_float(
            "RETRIEVAL_SCORE_THRESHOLD", 0.35
        ),
        retrieval_score_threshold_mode=os.getenv(
            "RETRIEVAL_SCORE_THRESHOLD_MODE",
            "relative",
        )
        .strip()
        .lower(),
        retrieval_score_relative_ratio=_env_float(
            "RETRIEVAL_SCORE_RELATIVE_RATIO",
            0.7,
        ),
        retrieval_score_quantile=_env_float(
            "RETRIEVAL_SCORE_QUANTILE",
            0.6,
        ),
        use_reranker=_env_bool("USE_RERANKER", False),
        rerank_conditional_enabled=_env_bool("RERANK_CONDITIONAL_ENABLED", True),
        rerank_skip_min_top1_score=_env_float("RERANK_SKIP_MIN_TOP1_SCORE", 0.55),
        rerank_skip_min_score_gap=_env_float("RERANK_SKIP_MIN_SCORE_GAP", 0.08),
        rerank_skip_min_rel_gap=_env_float("RERANK_SKIP_MIN_REL_GAP", 0.2),
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"),
        reranker_score_threshold=_env_optional_float(
            "RERANKER_SCORE_THRESHOLD", 0.0005
        ),
        diversify_by_source=_env_bool("DIVERSIFY_BY_SOURCE", True),
        max_chunks_per_source=_env_int("MAX_CHUNKS_PER_SOURCE", 2),
        generation_use_parent_context=_env_bool(
            "GENERATION_USE_PARENT_CONTEXT",
            True,
        ),
        generation_parent_top_n=_env_int("GENERATION_PARENT_TOP_N", 3),
        generation_parent_max_chars=_env_int("GENERATION_PARENT_MAX_CHARS", 2200),
        paper_summary_max_chars=_env_int("PAPER_SUMMARY_MAX_CHARS", 2400),
        section_summary_max_chars=_env_int("SECTION_SUMMARY_MAX_CHARS", 700),
        llm_provider=os.getenv("LLM_PROVIDER", "aihubmix"),
        llm_model=os.getenv("LLM_MODEL", "your-free-model-id"),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.1),
        aihubmix_api_key=os.getenv(
            "AIHUBMIX_API_KEY",
            os.getenv("OPENAI_API_KEY", ""),
        ),
        aihubmix_base_url=os.getenv(
            "AIHUBMIX_BASE_URL",
            os.getenv("OPENAI_BASE_URL", "https://aihubmix.com/v1"),
        ),
        aihubmix_api_mode=os.getenv(
            "AIHUBMIX_API_MODE",
            os.getenv("OPENAI_API_MODE", "chat"),
        ),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
        openai_api_mode=os.getenv("OPENAI_API_MODE", "chat"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
