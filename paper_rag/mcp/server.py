"""Paper_RAG 的本地 stdio MCP 服务入口。

服务只暴露检索证据，不生成最终自然语言答案。连接它的 Agent 负责
组合多个工具结果并回答用户问题。
"""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from paper_rag.config import Settings
from paper_rag.corpus.chunks import load_chunk_documents
from paper_rag.ingest.manifest import Manifest
from paper_rag.ingest.pipeline import run_ingest
from paper_rag.mcp.jobs import JobManager
from paper_rag.mcp.sources import preview_sources, stage_sources
from paper_rag.retrieval.plan import run_plan
from paper_rag.retrieval.dense.service import run_index


server = FastMCP("paper-rag")


@lru_cache(maxsize=1)
def _settings() -> Settings:
    configured_root = os.environ.get("PAPER_RAG_PROJECT_ROOT")
    root = Path(configured_root).resolve() if configured_root else Path.cwd().resolve()
    return Settings.load(root)


@lru_cache(maxsize=1)
def _jobs() -> JobManager:
    return JobManager(_settings())


def _summary_dict(summary: Any) -> dict[str, Any]:
    if hasattr(summary, "__dict__"):
        return {key: value for key, value in vars(summary).items()}
    return dict(summary) if isinstance(summary, dict) else {"value": str(summary)}


def _target_collection(settings: Settings) -> str:
    profile = getattr(settings, "embedding_profile", "qwen_v4")
    return settings.milvus_collection if profile == "qwen_v4" else f"{settings.milvus_collection}__{profile}"


def _run_evidence(query: str) -> dict[str, Any]:
    text = str(query or "").strip()
    if not text:
        return {
            "route": "unclear",
            "status": "unclear",
            "intent": None,
            "plan": {},
            "resolved": {},
            "decision": {"backend": "rules", "route": "unclear", "fallback": True},
            "retrieval": {},
            "results": {},
            "warnings": ["query 不能为空"],
        }
    evidence = run_plan(_settings(), text, debug=False)
    retrieval = evidence.get("retrieval") or {}
    status = evidence.get("status", "unclear")
    if evidence.get("route") == "content" and retrieval.get("status") in {"insufficient", "backend_unavailable"}:
        status = "insufficient"
    # MCP 工具使用稳定的 evidence envelope，便于 Agent 组合多个路由结果。
    return {
        "route": evidence.get("route", "unclear"),
        "status": status,
        "intent": evidence.get("intent"),
        "plan": evidence.get("plan") or {},
        "resolved": evidence.get("resolved") or {},
        "decision": evidence.get("decision") or {},
        "retrieval": retrieval,
        "results": evidence.get("results") or {},
        "warnings": evidence.get("warnings") or [],
        **({"parser_error": evidence["parser_error"]} if evidence.get("parser_error") else {}),
    }


@server.tool()
def paper_metadata(query: str) -> dict[str, Any]:
    """查询论文标题、作者、年份、venue 及 metadata 列表/计数。"""
    return _run_evidence(query)


@server.tool()
def paper_references(query: str) -> dict[str, Any]:
    """查询本地 citation graph 中的引用关系、引用论文和引用数量。"""
    return _run_evidence(query)


@server.tool()
def paper_content(query: str, top_k: int | None = None) -> dict[str, Any]:
    """检索论文正文，并返回 Dense、BM25、RRF 及 contexts 证据。"""
    settings = _settings()
    if top_k is not None and (top_k < 1 or top_k > settings.plan_final_top_k):
        raise ValueError(f"top_k 必须位于 1 到 {settings.plan_final_top_k} 之间")
    evidence = _run_evidence(query)
    if top_k is not None:
        results = evidence.get("results")
        if isinstance(results, dict) and isinstance(results.get("contexts"), list):
            results["contexts"] = results["contexts"][:top_k]
    return evidence


@server.tool()
def paper_library_status() -> dict[str, Any]:
    """查看论文库、引用图和当前索引状态，不修改本地数据。"""
    settings = _settings()
    manifest = Manifest.load(settings.manifest_path)
    records = list(manifest.records.values())
    active = [record for record in records if record.status == "active"]
    deleted = [record for record in records if record.status == "deleted"]
    errors = [record for record in records if record.status == "error"]
    chunks = load_chunk_documents(settings.paper_data_dir) if settings.paper_data_dir.exists() else []
    graph_path = settings.paper_data_dir / "citation_graph.json"
    graph: dict[str, Any] = {}
    if graph_path.exists():
        try:
            loaded = json.loads(graph_path.read_text(encoding="utf-8"))
            graph = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            graph = {}
    state_path = getattr(settings, "mcp_index_state_path", None) or settings.data_dir / "index" / "index_state.json"
    index_state: dict[str, Any] = {}
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            index_state = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            index_state = {}
    return {
        "status": "ok",
        "manifest": {"active": len(active), "deleted": len(deleted), "errors": len(errors)},
        "paper_data": {"papers": len(active), "chunks": len(chunks)},
        "citation_graph": {
            "exists": graph_path.exists(),
            "nodes": len(graph.get("nodes") or []),
            "edges": len(graph.get("edges") or []),
        },
        "index": {
            "collection": index_state.get("collection_name", _target_collection(settings)),
            "embedding_profile": index_state.get("embedding_profile", settings.embedding_profile),
            "dimensions": index_state.get("dimensions", settings.embedding_dim),
            "bm25_exists": settings.bm25_index_path.exists(),
            "chunk_count": index_state.get("chunk_count", len(chunks)),
            **({"status": index_state.get("status")} if index_state.get("status") else {}),
        },
        "active_jobs": _jobs().active_jobs(),
    }


@server.tool()
def paper_ingest(
    sources: list[str],
    refresh_metadata: bool = False,
    confirm: bool = False,
) -> dict[str, Any]:
    """接收本地或 HTTPS PDF，并异步执行入库。"""
    settings = _settings()
    previews = preview_sources(settings, sources)
    if not confirm:
        return {
            "status": "confirmation_required",
            "sources": previews,
            "planned_action": "写入 data/pdf 并执行 ingest",
        }
    manager = _jobs()

    def worker(report, job_id="pending"):
        report("download", "正在接收并校验 PDF", 0, len(sources))
        staged = stage_sources(settings, sources, job_id, reporter=lambda message: report("download", message, None, len(sources)))
        report("ingest", "正在执行本地入库", 0, len(staged))
        summary = run_ingest(settings, refresh_metadata=refresh_metadata, reporter=lambda message: report("ingest", message, None, len(staged)))
        return {"sources": staged, "ingest": _summary_dict(summary)}

    record = manager.submit("ingest", worker)
    return {"status": "queued", "job": record, "sources": previews}


@server.tool()
def paper_index(confirm: bool = False) -> dict[str, Any]:
    """异步重建 Dense、BM25 和索引状态。"""
    settings = _settings()
    chunks = load_chunk_documents(settings.paper_data_dir) if settings.paper_data_dir.exists() else []
    preview = {
        "status": "confirmation_required",
        "plan": {
            "chunk_count": len(chunks),
            "collection": _target_collection(settings),
            "embedding_profile": settings.embedding_profile,
            "dimensions": settings.embedding_dim,
        },
        "planned_action": "重建 Dense/BM25 索引并切换 collection",
    }
    if not confirm:
        return preview
    manager = _jobs()

    def worker(report):
        report("index", "正在重建 Dense/BM25 索引", 0, len(chunks))
        summary = run_index(settings, reporter=lambda message: report("index", message, None, len(chunks)))
        report("index", "索引重建完成", len(chunks), len(chunks))
        return {"index": _summary_dict(summary)}

    return {"status": "queued", "job": manager.submit("index", worker)}


@server.tool()
def paper_full_sync(
    sources: list[str],
    refresh_metadata: bool = False,
    confirm: bool = False,
) -> dict[str, Any]:
    """异步执行 PDF staging、ingest 和 index 的完整流程。"""
    settings = _settings()
    previews = preview_sources(settings, sources)
    if not confirm:
        return {
            "status": "confirmation_required",
            "sources": previews,
            "planned_action": "写入 data/pdf，执行 ingest，随后重建 Dense/BM25 索引",
        }
    manager = _jobs()
    def worker(report, job_id="pending"):
        report("download", "正在接收并校验 PDF", 0, len(sources))
        staged = stage_sources(settings, sources, job_id, reporter=lambda message: report("download", message, None, len(sources)))
        report("ingest", "正在执行本地入库", 0, len(staged))
        ingest_summary = run_ingest(settings, refresh_metadata=refresh_metadata, reporter=lambda message: report("ingest", message, None, len(staged)))
        chunks = load_chunk_documents(settings.paper_data_dir) if settings.paper_data_dir.exists() else []
        report("index", "正在重建 Dense/BM25 索引", 0, len(chunks))
        index_summary = run_index(settings, reporter=lambda message: report("index", message, None, len(chunks)))
        report("index", "完整同步完成", len(chunks), len(chunks))
        return {"sources": staged, "ingest": _summary_dict(ingest_summary), "index": _summary_dict(index_summary)}

    record = manager.submit("full_sync", worker)
    return {"status": "queued", "job": record, "sources": previews}


@server.tool()
def paper_job_status(job_id: str) -> dict[str, Any]:
    """查询异步入库、索引或完整同步任务的持久化状态。"""
    return _jobs().status(job_id)


def main() -> None:
    """启动 stdio MCP 服务；协议消息之外不向 stdout 写日志。"""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
