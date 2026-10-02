"""Paper_RAG ArXiv 原始资料 MCP 服务。"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from paper_rag.acquisition.arxiv import download_arxiv_inputs, preview_arxiv_inputs
from paper_rag.config import Settings
from paper_rag.mcp.jobs import JobManager


server = FastMCP("paper-rag")


@lru_cache(maxsize=1)
def _settings() -> Settings:
    configured_root = os.environ.get("PAPER_RAG_PROJECT_ROOT")
    root = Path(configured_root).resolve() if configured_root else Path.cwd().resolve()
    return Settings.load(root)


@lru_cache(maxsize=1)
def _jobs() -> JobManager:
    return JobManager(_settings())


@server.tool()
def paper_arxiv_download(inputs: list[str], confirm: bool = False) -> dict[str, Any]:
    """预览或异步下载 ArXiv 当前版本的 PDF 与元数据。"""
    settings = _settings()
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


@server.tool()
def paper_job_status(job_id: str) -> dict[str, Any]:
    """查询 ArXiv 下载任务状态。"""
    return _jobs().status(job_id)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()

