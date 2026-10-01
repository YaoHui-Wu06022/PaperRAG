from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    pdf_dir: Path
    mineru_output_dir: Path
    paper_data_dir: Path
    archive_dir: Path
    manifest_path: Path
    mineru_api_key: str | None
    mineru_api_base_url: str
    mineru_model_version: str
    mineru_language: str
    semantic_scholar_delay_seconds: float
    semantic_scholar_api_key: str | None
    arxiv_delay_seconds: float
    chunk_target_chars: int
    chunk_overlap_chars: int
    milvus_uri: str
    milvus_token: str | None
    milvus_db_name: str | None
    milvus_collection: str
    embedding_base_url: str
    embedding_api_key: str | None
    embedding_model: str
    embedding_dim: int
    embedding_batch_size: int
    embedding_cache_path: Path
    query_embedding_cache_path: Path
    bm25_index_path: Path
    plan_dense_top_k: int
    plan_bm25_top_k: int
    plan_final_top_k: int
    plan_block_window: int
    plan_bm25_translate_providers: list[str]
    plan_bm25_translate_timeout_seconds: int
    tencent_translate_secret_id: str | None
    tencent_translate_secret_key: str | None
    tencent_translate_region: str
    tencent_translate_endpoint: str
    aliyun_translate_access_key_id: str | None
    aliyun_translate_access_key_secret: str | None
    aliyun_translate_region: str
    aliyun_translate_endpoint: str
    aliyun_translate_version: str
    # 第二版运行时配置。
    jev_base_url: str = "https://jevmodel.org/v1/systemone"
    jev_api_key: str | None = None
    jev_model: str = "jev-1.13.0"
    jev_timeout_seconds: int = 20
    jev_min_confidence: float = 0.55
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-flash"
    deepseek_timeout_seconds: int = 60
    extraction_cache_path: Path | None = None
    crossref_delay_seconds: float = 1.0
    crossref_mailto: str | None = None
    embedding_profile: str = "qwen_v4"
    embedding_fallback_base_url: str = ""
    embedding_fallback_api_key: str | None = None
    embedding_fallback_model: str = ""
    embedding_fallback_dim: int = 0
    mcp_job_log_path: Path | None = None
    mcp_index_state_path: Path | None = None
    mcp_input_roots: list[Path] = field(default_factory=list)
    mcp_max_download_mb: int = 100

    @classmethod
    def load(cls, project_root: Path | None = None) -> "Settings":
        root = (project_root or Path.cwd()).resolve()
        env = load_dotenv(root / ".env")
        # Conda 或系统环境变量覆盖 .env，部署密钥无需写入仓库，CI 也可按进程注入。
        env.update({key: value for key, value in os.environ.items() if value is not None})
        data_dir = root / "data"
        pdf_dir = resolve_config_path(root, env.get("PDF_DIR"), data_dir / "pdf")
        mineru_output_dir = resolve_config_path(root, env.get("MINERU_DIR"), data_dir / "mineru_output")
        paper_data_dir = resolve_config_path(root, env.get("PAPER_DIR"), data_dir / "paper_data")
        api_key = env.get("MINERU_API_KEY") or env.get("MINERU_API_TOKEN")
        return cls(
            project_root=root,
            data_dir=data_dir,
            pdf_dir=pdf_dir,
            mineru_output_dir=mineru_output_dir,
            paper_data_dir=paper_data_dir,
            archive_dir=data_dir / "archive",
            manifest_path=data_dir / "manifest.jsonl",
            mineru_api_key=api_key,
            mineru_api_base_url=env.get("MINERU_API_BASE_URL", "https://mineru.net/api/v4").rstrip("/"),
            mineru_model_version=env.get("MINERU_MODEL_VERSION", "vlm"),
            mineru_language=env.get("MINERU_LANGUAGE", "en"),
            semantic_scholar_delay_seconds=float(env.get("SEMANTIC_SCHOLAR_DELAY_SECONDS", "5.0")),
            semantic_scholar_api_key=env.get("SEMANTIC_SCHOLAR_API_KEY") or None,
            arxiv_delay_seconds=float(env.get("ARXIV_DELAY_SECONDS", "3.0")),
            chunk_target_chars=int(env.get("CHUNK_TARGET_CHARS", "1400")),
            chunk_overlap_chars=int(env.get("CHUNK_OVERLAP_CHARS", "200")),
            milvus_uri=env.get("MILVUS_URI", "").strip(),
            milvus_token=env.get("MILVUS_TOKEN") or None,
            milvus_db_name=env.get("MILVUS_DB_NAME") or None,
            milvus_collection=env.get("MILVUS_COLLECTION", "paper_rag_chunks"),
            embedding_base_url=env.get(
                "EMBEDDING_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ).rstrip("/"),
            embedding_api_key=env.get("EMBEDDING_API_KEY") or None,
            embedding_model=env.get("EMBEDDING_MODEL", "text-embedding-v4"),
            embedding_dim=int(env.get("EMBEDDING_DIM", "1024")),
            embedding_batch_size=int(env.get("EMBEDDING_BATCH_SIZE", "10")),
            embedding_cache_path=resolve_config_path(
                root,
                env.get("EMBEDDING_CACHE_PATH"),
                data_dir / "index" / "embedding_cache.jsonl",
            ),
            query_embedding_cache_path=resolve_config_path(
                root,
                env.get("QUERY_EMBEDDING_CACHE_PATH"),
                data_dir / "index" / "query_embedding_cache.jsonl",
            ),
            bm25_index_path=data_dir / "index" / "bm25_chunks.json",
            plan_dense_top_k=int(env.get("PLAN_DENSE_TOP_K", "20")),
            plan_bm25_top_k=int(env.get("PLAN_BM25_TOP_K", "20")),
            plan_final_top_k=int(env.get("PLAN_FINAL_TOP_K", "8")),
            plan_block_window=int(env.get("PLAN_BLOCK_WINDOW", "2")),
            plan_bm25_translate_providers=parse_csv(env.get("PLAN_BM25_TRANSLATE_PROVIDERS", "tencent,aliyun")),
            plan_bm25_translate_timeout_seconds=int(env.get("PLAN_BM25_TRANSLATE_TIMEOUT_SECONDS", "10")),
            tencent_translate_secret_id=env.get("TENCENT_TRANSLATE_SECRET_ID") or None,
            tencent_translate_secret_key=env.get("TENCENT_TRANSLATE_SECRET_KEY") or None,
            tencent_translate_region=env.get("TENCENT_TRANSLATE_REGION", "ap-shanghai"),
            tencent_translate_endpoint=env.get("TENCENT_TRANSLATE_ENDPOINT", "tmt.tencentcloudapi.com"),
            aliyun_translate_access_key_id=env.get("ALIYUN_TRANSLATE_ACCESS_KEY_ID") or None,
            aliyun_translate_access_key_secret=env.get("ALIYUN_TRANSLATE_ACCESS_KEY_SECRET") or None,
            aliyun_translate_region=env.get("ALIYUN_TRANSLATE_REGION", "cn-hangzhou"),
            aliyun_translate_endpoint=env.get("ALIYUN_TRANSLATE_ENDPOINT", "mt.aliyuncs.com"),
            aliyun_translate_version=env.get("ALIYUN_TRANSLATE_VERSION", "2018-10-12"),
            jev_base_url=env.get("JEV_BASE_URL", "https://jevmodel.org/v1/systemone").rstrip("/"),
            jev_api_key=env.get("JEV_API_KEY") or None,
            jev_model=env.get("JEV_MODEL", "jev-1.13.0"),
            jev_timeout_seconds=int(env.get("JEV_TIMEOUT_SECONDS", "20")),
            jev_min_confidence=float(env.get("JEV_MIN_CONFIDENCE", "0.55")),
            deepseek_base_url=env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            deepseek_api_key=env.get("DEEPSEEK_API_KEY") or None,
            deepseek_model=env.get("DEEPSEEK_MODEL", "deepseek-flash"),
            deepseek_timeout_seconds=int(env.get("DEEPSEEK_TIMEOUT_SECONDS", "60")),
            extraction_cache_path=resolve_config_path(
                root,
                env.get("EXTRACTION_CACHE_PATH"),
                data_dir / "index" / "extraction_cache.jsonl",
            ),
            crossref_delay_seconds=float(env.get("CROSSREF_DELAY_SECONDS", "1.0")),
            crossref_mailto=env.get("CROSSREF_MAILTO") or None,
            embedding_profile=env.get("EMBEDDING_PROFILE", "qwen_v4"),
            embedding_fallback_base_url=env.get("EMBEDDING_FALLBACK_BASE_URL", "").rstrip("/"),
            embedding_fallback_api_key=env.get("EMBEDDING_FALLBACK_API_KEY") or None,
            embedding_fallback_model=env.get("EMBEDDING_FALLBACK_MODEL", ""),
            embedding_fallback_dim=int(env.get("EMBEDDING_FALLBACK_DIM", "0")),
            mcp_job_log_path=resolve_config_path(
                root,
                env.get("MCP_JOB_LOG_PATH"),
                data_dir / "index" / "mcp_jobs.jsonl",
            ),
            mcp_index_state_path=resolve_config_path(
                root,
                env.get("MCP_INDEX_STATE_PATH"),
                data_dir / "index" / "index_state.json",
            ),
            mcp_input_roots=parse_path_list(
                root,
                env.get("PAPER_RAG_INPUT_ROOTS"),
                [root, pdf_dir],
            ),
            mcp_max_download_mb=int(env.get("MCP_MAX_DOWNLOAD_MB", "100")),
        )


def resolve_config_path(root: Path, value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value.strip())
    if path.is_absolute():
        return path
    return root / path


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_path_list(root: Path, value: str | None, defaults: list[Path]) -> list[Path]:
    values = parse_csv(value)
    paths = [resolve_config_path(root, item, root) for item in values]
    return list(dict.fromkeys(path.resolve() for path in (paths or defaults)))
