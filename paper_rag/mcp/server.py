"""Paper RAG MCP 启动入口与公共工具导出。"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from paper_rag.acquisition.arxiv import download_arxiv_inputs, preview_arxiv_inputs
from paper_rag.mcp._app import mcp
from paper_rag.mcp import tools  # noqa: F401 - 导入副作用负责注册工具
from paper_rag.config import Settings
from paper_rag.mcp.jobs import JobManager
from paper_rag.mcp.runtime import get_settings
from paper_rag.mcp.toolsets import TOOLSETS, validate_toolsets
from paper_rag.mcp.tools.ingestion import paper_arxiv_ingest


# 保留旧入口的可替换钩子，便于已有调用方在迁移期间注入测试配置。
_settings = get_settings


@lru_cache(maxsize=1)
def _jobs() -> JobManager:
    return JobManager(_settings())


def paper_arxiv_download(inputs: list[str], confirm: bool = False) -> dict[str, Any]:
    """兼容旧导出入口，并复用与注册工具相同的确认语义。"""

    settings: Settings = _settings()
    previews = preview_arxiv_inputs(settings, inputs)
    if not confirm:
        return {
            "status": "confirmation_required",
            "items": previews,
            "planned_action": "写入 ArXiv 原始 PDF、metadata.json 和 manifest.jsonl",
        }

    def worker(report):
        result = download_arxiv_inputs(settings, inputs, reporter=report)
        return result.to_dict()

    return {"status": "queued", "job": _jobs().submit("arxiv_download", worker), "items": previews}


def paper_job_status(job_id: str) -> dict[str, Any]:
    """兼容旧导出入口，返回异步任务状态。"""

    return _jobs().status(job_id)


def _registered_tool_names() -> set[str]:
    """读取 FastMCP 当前注册的工具名称，兼容不同 FastMCP 版本。"""

    registry = getattr(mcp, "_tool_manager", None)
    tools_by_name = getattr(registry, "_tools", None)
    if isinstance(tools_by_name, dict):
        return set(tools_by_name)
    return {"paper_arxiv_download", "paper_arxiv_ingest", "paper_job_status"}


validate_toolsets(_registered_tool_names())

__all__ = [
    "TOOLSETS",
    "mcp",
    "paper_arxiv_download",
    "paper_arxiv_ingest",
    "paper_job_status",
]


def main() -> None:
    """以 stdio 传输启动 MCP 服务。"""

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

