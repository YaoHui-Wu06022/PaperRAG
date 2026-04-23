from __future__ import annotations

from dataclasses import dataclass
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
    dblp_delay_seconds: float
    dblp_candidate_limit: int
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
    baidu_translate_app_id: str | None
    baidu_translate_secret_key: str | None
    baidu_translate_endpoint: str
    baidu_translate_domain: str | None
    plan_dense_top_k: int
    plan_bm25_top_k: int
    plan_final_top_k: int
    plan_block_window: int
    plan_parser_base_url: str
    plan_parser_api_key: str | None
    plan_parser_model: str
    plan_parser_timeout_seconds: int

    @classmethod
    def load(cls, project_root: Path | None = None) -> "Settings":
        root = (project_root or Path.cwd()).resolve()
        env = load_dotenv(root / ".env")
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
            dblp_delay_seconds=float(env.get("DBLP_DELAY_SECONDS", "1.0")),
            dblp_candidate_limit=int(env.get("DBLP_CANDIDATE_LIMIT", "20")),
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
            baidu_translate_app_id=env.get("BAIDU_TRANSLATE_APP_ID") or None,
            baidu_translate_secret_key=env.get("BAIDU_TRANSLATE_SECRET_KEY") or None,
            baidu_translate_endpoint=env.get(
                "BAIDU_TRANSLATE_ENDPOINT",
                "https://fanyi-api.baidu.com/api/trans/vip/fieldtranslate",
            ),
            baidu_translate_domain=env.get("BAIDU_TRANSLATE_DOMAIN") or "academic",
            plan_dense_top_k=int(env.get("PLAN_DENSE_TOP_K", "20")),
            plan_bm25_top_k=int(env.get("PLAN_BM25_TOP_K", "20")),
            plan_final_top_k=int(env.get("PLAN_FINAL_TOP_K", "8")),
            plan_block_window=int(env.get("PLAN_BLOCK_WINDOW", "2")),
            plan_parser_base_url=env.get("PLAN_PARSER_BASE_URL", "").rstrip("/"),
            plan_parser_api_key=env.get("PLAN_PARSER_API_KEY") or None,
            plan_parser_model=env.get("PLAN_PARSER_MODEL", "").strip(),
            plan_parser_timeout_seconds=int(env.get("PLAN_PARSER_TIMEOUT_SECONDS", "30")),
        )


def resolve_config_path(root: Path, value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value.strip())
    if path.is_absolute():
        return path
    return root / path
