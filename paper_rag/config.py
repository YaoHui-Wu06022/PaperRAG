from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


def load_dotenv(path: Path) -> dict[str, str]:
    """读取简单的 UTF-8 ``.env`` 文件。"""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_path(root: Path, value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value.strip())
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    arxiv_data_dir: Path
    arxiv_api_base_url: str
    arxiv_request_delay_seconds: float
    arxiv_timeout_seconds: int
    arxiv_download_timeout_seconds: int
    arxiv_max_download_mb: int
    arxiv_user_agent: str
    mineru_api_key: str = field(repr=False)
    mineru_api_base_url: str
    mineru_model_version: str
    mineru_language: str
    mineru_request_timeout_seconds: int
    mineru_upload_timeout_seconds: int
    mineru_poll_interval_seconds: float
    mineru_poll_timeout_seconds: int
    jev_enabled: bool
    jev_api_key: str = field(repr=False)
    jev_base_url: str
    jev_model: str
    jev_timeout_seconds: int
    jev_retry_count: int
    jev_route_probability_threshold: float
    paper_rag_toolsets: str
    paper_catalog_db_path: Path
    mcp_job_log_path: Path

    @classmethod
    def load(cls, project_root: Path | None = None) -> "Settings":
        root = (project_root or Path.cwd()).resolve()
        values = load_dotenv(root / ".env")
        values.update({key: value for key, value in os.environ.items() if value is not None})
        data_dir = root / "data"
        jev_api_key = values.get("JEV_API_KEY", "")
        return cls(
            project_root=root,
            arxiv_data_dir=resolve_path(root, values.get("ARXIV_DATA_DIR"), data_dir / "sources" / "arxiv"),
            arxiv_api_base_url=values.get(
                "ARXIV_API_BASE_URL", "https://export.arxiv.org/api/query"
            ).rstrip("/"),
            arxiv_request_delay_seconds=float(values.get("ARXIV_REQUEST_DELAY_SECONDS", "3")),
            arxiv_timeout_seconds=int(values.get("ARXIV_TIMEOUT_SECONDS", "30")),
            arxiv_download_timeout_seconds=int(values.get("ARXIV_DOWNLOAD_TIMEOUT_SECONDS", "300")),
            arxiv_max_download_mb=int(values.get("ARXIV_MAX_DOWNLOAD_MB", "100")),
            arxiv_user_agent=values.get(
                "ARXIV_USER_AGENT",
                "paper-rag/0.1 (https://github.com/YaoHui-Wu06022/PaperRAG)",
            ),
            mineru_api_key=values.get("MINERU_API_KEY", ""),
            mineru_api_base_url=values.get("MINERU_API_BASE_URL", "https://mineru.net/api/v4").rstrip("/"),
            mineru_model_version=values.get("MINERU_MODEL_VERSION", "vlm"),
            mineru_language=values.get("MINERU_LANGUAGE", "en"),
            mineru_request_timeout_seconds=int(values.get("MINERU_REQUEST_TIMEOUT_SECONDS", "60")),
            mineru_upload_timeout_seconds=int(values.get("MINERU_UPLOAD_TIMEOUT_SECONDS", "300")),
            mineru_poll_interval_seconds=float(values.get("MINERU_POLL_INTERVAL_SECONDS", "10")),
            mineru_poll_timeout_seconds=int(values.get("MINERU_POLL_TIMEOUT_SECONDS", "1800")),
            jev_enabled=_parse_bool(values.get("JEV_ENABLED", "true")),
            jev_api_key=jev_api_key,
            jev_base_url=values.get("JEV_BASE_URL", "https://jevmodel.org/v1/systemone").strip(),
            jev_model=values.get("JEV_MODEL", "jev-1.13.0").strip(),
            jev_timeout_seconds=int(values.get("JEV_TIMEOUT_SECONDS", "15")),
            jev_retry_count=int(values.get("JEV_RETRY_COUNT", "2")),
            jev_route_probability_threshold=float(
                values.get("JEV_ROUTE_PROBABILITY_THRESHOLD", "0.65")
            ),
            paper_rag_toolsets=values.get("PAPER_RAG_TOOLSETS", "").strip(),
            paper_catalog_db_path=resolve_path(
                root,
                values.get("PAPER_CATALOG_DB_PATH"),
                data_dir / "index" / "paper_catalog.sqlite3",
            ),
            mcp_job_log_path=resolve_path(
                root,
                values.get("MCP_JOB_LOG_PATH"),
                data_dir / "index" / "mcp_jobs.jsonl",
            ),
        )


def _parse_bool(value: str) -> bool:
    """解析环境变量中的布尔值。"""

    return str(value).strip().casefold() in {"1", "true", "yes", "on"}

