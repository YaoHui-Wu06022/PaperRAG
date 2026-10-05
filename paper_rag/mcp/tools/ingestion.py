"""ArXiv MinerU 入库 MCP 工具。"""

from __future__ import annotations

from typing import Any

from paper_rag.ingest.mineru import ingest_arxiv_inputs, preview_arxiv_ingest
from paper_rag.mcp._app import mcp
from paper_rag.mcp.runtime import get_jobs, get_settings


@mcp.tool(name="library_ingest_mineru", description="预览或异步使用 MinerU 解析已下载的 ArXiv PDF。")
def library_ingest_mineru(inputs: list[str], confirm: bool = False) -> dict[str, Any]:
    settings = get_settings()
    previews = preview_arxiv_ingest(settings, inputs)
    if not confirm:
        return {"status": "confirmation_required", "data": {"items": previews}, "warnings": [], "read_only": True}

    def worker(report):
        result = ingest_arxiv_inputs(settings, inputs, reporter=report)
        return result.to_dict()

    return {"status": "queued", "job": get_jobs().submit("arxiv_ingest", worker), "data": {"items": previews}, "warnings": [], "read_only": False}


__all__ = ["library_ingest_mineru"]
