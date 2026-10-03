"""ArXiv 原始资料获取 MCP 工具。"""

from __future__ import annotations

from typing import Any

from paper_rag.acquisition.arxiv import download_arxiv_inputs, preview_arxiv_inputs
from paper_rag.mcp._app import mcp
from paper_rag.mcp.runtime import get_jobs, get_settings


@mcp.tool(
    name="paper_arxiv_download",
    description="预览或异步下载 ArXiv 当前版本的 PDF 与元数据。",
)
def paper_arxiv_download(inputs: list[str], confirm: bool = False) -> dict[str, Any]:
    """先返回计划，确认后创建 ArXiv 下载任务。"""

    settings = get_settings()
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

    return {"status": "queued", "job": get_jobs().submit("arxiv_download", worker), "items": previews}


@mcp.tool(
    name="paper_job_status",
    description="查询 ArXiv 下载或 MinerU 入库异步任务状态。",
)
def paper_job_status(job_id: str) -> dict[str, Any]:
    """返回 JobManager 中异步任务的最新状态。"""

    return get_jobs().status(job_id)


__all__ = ["paper_arxiv_download", "paper_job_status"]

