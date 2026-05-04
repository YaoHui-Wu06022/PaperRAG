"""绕过 content prompt，用手写 parser JSON 测试 dense/BM25 召回。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def find_project_root() -> Path:
    """向上寻找项目根目录，支持直接运行本文件。"""
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".env").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_rag.config import Settings
from paper_rag.retrieval.chunk_fusion import fuse_chunk_hits
from paper_rag.corpus.chunks import filter_chunks_by_paper_records, load_chunk_documents
from paper_rag.corpus.records import dedupe_paper_records
from paper_rag.corpus.scope import combined_semantic, records_for_scope
from paper_rag.retrieval.dense.service import search_dense_chunks
from paper_rag.retrieval.routes.content.context import context_unit
from paper_rag.retrieval.routes.content.retrieval_query import build_content_retrieval_query
from paper_rag.retrieval.routes.content.router import build_content_decision
from paper_rag.retrieval.routes.content.schema import validate_content_parse
from paper_rag.retrieval.sparse.bm25 import search_bm25_chunks
from paper_rag.retrieval.route import RouteDecision


DEFAULT_CASES_PATH = Path(__file__).with_name("retrieval_probe_cases.json")


class StaticContentParser:
    """把手写 JSON 伪装成 content parser，绕过 prompt/LLM。"""

    def __init__(self, parser_result: dict[str, Any]) -> None:
        self.parser_result = parser_result

    def parse_content(self, query: str) -> dict[str, Any]:
        return validate_content_parse(self.parser_result, query)


def main() -> int:
    """读取案例 JSON，输出每个案例的 query、scope、dense/BM25/fused 命中。"""
    parser = argparse.ArgumentParser(description="Probe content dense/BM25 retrieval with handcrafted parser JSON.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help="Project root containing .env")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH, help="JSON file containing probe cases")
    parser.add_argument("--case", dest="case_query", help="Run cases whose query contains this text")
    parser.add_argument("--top-k", type=int, default=5, help="Number of hits to show for dense/BM25/fused")
    parser.add_argument("--no-dense", action="store_true", help="Skip dense retrieval and only run BM25")
    args = parser.parse_args()

    settings = Settings.load(args.project_root)
    cases = load_cases(args.cases)
    if args.case_query:
        cases = [case for case in cases if args.case_query in str(case.get("query") or "")]
    if not cases:
        print(json.dumps({"error": "No matching cases", "cases": str(args.cases)}, ensure_ascii=False, indent=2))
        return 1

    payload = [run_case(settings, case, top_k=args.top_k, run_dense=not args.no_dense) for case in cases]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def load_cases(path: Path) -> list[dict[str, Any]]:
    """读取手写案例；根可以是数组，也可以是 {"cases": [...]}。"""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        payload = payload.get("cases") or []
    if not isinstance(payload, list):
        raise ValueError("cases file must contain a list or {'cases': [...]}")
    return [case for case in payload if isinstance(case, dict)]


def run_case(settings: Settings, case: dict[str, Any], *, top_k: int, run_dense: bool) -> dict[str, Any]:
    """运行单个 probe case，分开展示 dense、BM25 和融合结果。"""
    warnings: list[str] = []
    query = str(case.get("query") or "").strip()
    parser_result = case.get("parser_result")
    if not query or not isinstance(parser_result, dict):
        return {
            "status": "invalid_case",
            "warnings": ["case requires query and parser_result"],
        }

    try:
        route = build_content_decision(
            settings,
            RouteDecision(route="content", query=query, parse_status="ok"),
            warnings,
            plan_parser=StaticContentParser(parser_result),
        )
        retrieval_query = build_content_retrieval_query(settings, route, warnings)
        scope_records = content_scope_records(settings, route)
        documents = filter_chunks_by_paper_records(load_chunk_documents(settings.paper_data_dir), scope_records)
        documents_by_id = {document.chunk_id: document for document in documents}
        dense_hits = dense_results(settings, retrieval_query["dense_query"], run_dense, warnings)
        bm25_hits = search_bm25_chunks(documents, retrieval_query["bm25_queries"], settings.plan_bm25_top_k)
        fused_hits = fuse_chunk_hits(documents_by_id, dense_hits, bm25_hits)
        contexts = [context_unit(settings, hit, settings.plan_block_window) for hit in fused_hits[:top_k]]
        return {
            "query": query,
            "status": "ok",
            "parser_result": parser_result,
            "resolved_scope": {
                "paper_semantic": route.paper_semantic,
                "filters": route.filters,
                "paper_groups": route.paper_groups,
                "papers": [record.get("title") for record in scope_records],
                "chunk_count": len(documents),
            },
            "retrieval_query": {
                "dense_query": retrieval_query["dense_query"],
                "bm25_queries": retrieval_query["bm25_queries"],
                "source_terms": retrieval_query.get("source_terms") or {},
            },
            "dense_hits": [compact_dense_hit(hit) for hit in dense_hits[:top_k]],
            "bm25_hits": [compact_bm25_hit(hit) for hit in bm25_hits[:top_k]],
            "fused_contexts": [compact_context(context) for context in contexts],
            "expected_terms": case.get("expected_terms") or [],
            "warnings": warnings,
        }
    except Exception as exc:
        return {
            "query": query,
            "status": "probe_failed",
            "error": str(exc),
            "warnings": warnings,
        }


def content_scope_records(settings: Settings, route: RouteDecision) -> list[dict[str, Any]]:
    """复用 content planner 的 scope 语义，得到候选论文 records。"""
    if route.group_mode in {"per", "or", "and"}:
        return dedupe_paper_records([
            record
            for group in route.paper_groups
            for record in records_for_scope(
                settings,
                combined_semantic(route.paper_semantic, group.get("semantic") or ""),
                [*route.filters, *(group.get("filters") or [])],
                route.group_mode,
            )
        ])
    return records_for_scope(settings, route.paper_semantic, route.filters, route.group_mode)


def dense_results(settings: Settings, dense_query: str, run_dense: bool, warnings: list[str]) -> list[Any]:
    """执行 dense 检索；不可用时只记录 warning，BM25 仍可继续测。"""
    if not run_dense:
        warnings.append("dense retrieval skipped by --no-dense")
        return []
    try:
        return search_dense_chunks(settings, dense_query)
    except Exception as exc:
        warnings.append(f"dense retrieval failed: {exc}; BM25 results are still shown")
        return []


def compact_dense_hit(hit: Any) -> dict[str, Any]:
    """压缩 dense hit，便于比较命中的论文、section 和文本片段。"""
    return {
        "score": getattr(hit, "score", None),
        "chunk_id": getattr(hit, "chunk_id", ""),
        "title": getattr(hit, "title", ""),
        "section": getattr(hit, "section_path_text", ""),
        "pages": getattr(hit, "pages_text", ""),
        "text": snippet(getattr(hit, "text", "")),
    }


def compact_bm25_hit(hit: Any) -> dict[str, Any]:
    """压缩 BM25 hit；真实 ChunkDocument 放在 payload.document 里。"""
    document = (hit.payload or {}).get("document")
    return {
        "score": hit.score,
        "chunk_id": hit.doc_id,
        "title": getattr(document, "title", ""),
        "section": getattr(document, "section_path_text", ""),
        "pages": getattr(document, "pages_text", ""),
        "text": snippet(hit.text),
    }


def compact_context(context: dict[str, Any]) -> dict[str, Any]:
    """压缩融合后的 context，接近最终 answer LLM 会看到的证据。"""
    return {
        "chunk_id": context.get("chunk_id"),
        "title": context.get("title"),
        "section_path": context.get("section_path"),
        "pages": context.get("pages"),
        "text": snippet(context.get("chunk_text") or context.get("text") or "", limit=700),
    }


def snippet(text: str, limit: int = 360) -> str:
    """压缩空白并截断长文本。"""
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


if __name__ == "__main__":
    sys.exit(main())
