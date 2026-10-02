"""ArXiv MinerU 入库 MCP 工具。"""

from __future__ import annotations

from typing import Any

from paper_rag.ingest.mineru import ingest_arxiv_inputs, preview_arxiv_ingest
from paper_rag.mcp._app import mcp
from paper_rag.mcp.runtime import get_jobs, get_settings


@mcp.tool(
    name="paper_arxiv_ingest",
    description="预览或异步使用 MinerU 解析已下载的 ArXiv PDF。",
)
def paper_arxiv_ingest(inputs: list[str], confirm: bool = False) -> dict[str, Any]:
    """先检查本地 ArXiv 资产，确认后创建 MinerU 入库任务。"""

    settings = get_settings()
    previews = preview_arxiv_ingest(settings, inputs)
    if not confirm:
        return {
            "status": "confirmation_required",
            "items": previews,
            "planned_action": "上传本地 ArXiv PDF，写入每篇论文目录下的 mineru/",
        }

    def worker(report):
        result = ingest_arxiv_inputs(settings, inputs, reporter=report)
        return result.to_dict()

    return {"status": "queued", "job": get_jobs().submit("arxiv_ingest", worker), "items": previews}


__all__ = ["paper_arxiv_ingest"]

