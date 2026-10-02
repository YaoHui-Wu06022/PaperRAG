"""MCP 运行时共享状态。"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from paper_rag.config import Settings
from paper_rag.mcp.jobs import JobManager


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """从环境变量解析项目根目录，并缓存当前进程配置。"""

    configured_root = os.environ.get("PAPER_RAG_PROJECT_ROOT")
    root = Path(configured_root).resolve() if configured_root else Path.cwd().resolve()
    return Settings.load(root)


@lru_cache(maxsize=1)
def get_jobs() -> JobManager:
    """返回复用同一任务日志的单线程任务管理器。"""

    return JobManager(get_settings())

