"""ArXiv 原始资料获取 MCP 工具。"""

from __future__ import annotations

from typing import Any

from paper_rag.acquisition.arxiv import download_arxiv_inputs, preview_arxiv_inputs
from paper_rag.mcp._app import mcp
from paper_rag.mcp.runtime import get_jobs, get_settings


@mcp.tool(name="library_acquire_arxiv", description="预览或异步下载 ArXiv 当前版本的 PDF 与元数据。")
def library_acquire_arxiv(inputs: list[str], confirm: bool = False) -> dict[str, Any]:
    settings = get_settings()
    previews = preview_arxiv_inputs(settings, inputs)
    if not confirm:
        return {"status": "confirmation_required", "data": {"items": previews}, "warnings": [], "read_only": True}

    def worker(report):
        result = download_arxiv_inputs(settings, inputs, reporter=report)
        return result.to_dict()

    return {"status": "queued", "job": get_jobs().submit("arxiv_download", worker), "data": {"items": previews}, "warnings": [], "read_only": False}


@mcp.tool(name="library_job_status", description="查询论文下载、MinerU 和索引重建异步任务状态。")
def library_job_status(job_id: str) -> dict[str, Any]:
    return get_jobs().status(job_id)


__all__ = ["library_acquire_arxiv", "library_job_status"]
